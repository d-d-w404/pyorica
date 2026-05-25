"""Offline ICA analysis: IC source MS energy across pipeline stages."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def _fit_ica(raw, *, random_state: int, max_iter: int):
    """Fit MNE extended-Infomax ICA on a Raw object. Returns fitted ICA."""
    from mne.preprocessing import ICA

    n_components = len(raw.ch_names)
    ica = ICA(
        n_components=n_components,
        method="infomax",
        random_state=random_state,
        max_iter=max_iter,
        fit_params={"extended": True},
    )
    ica.fit(raw, verbose=False, reject_by_annotation=False)
    return ica


def _label_ica(raw, ica) -> List[str]:
    """Return per-IC ICLabel class strings (e.g. 'brain', 'muscle', 'eye', ...)."""
    from mne_icalabel import label_components

    labels_out = label_components(raw, ica, method="iclabel")
    return list(labels_out["labels"])


def _make_raw(data: np.ndarray, ch_names: Sequence[str], sfreq: float):
    """Wrap a (channels × samples) array in an MNE Raw object with standard_1020 montage.

    Channel names are matched case-insensitively to the montage so that datasets
    using all-caps conventions (e.g. FP1, CZ, OZ) are handled correctly.
    """
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    montage_lookup = {name.lower(): name for name in montage.ch_names}
    normalized = [montage_lookup.get(ch.lower(), ch) for ch in ch_names]

    info = mne.create_info(normalized, sfreq, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    return raw


def _project_sources(ica, data: np.ndarray) -> np.ndarray:
    """Project (channels × samples) data through the ICA unmixing matrix.

    Returns sources with shape (n_components, n_samples).
    The unmixing is: W @ sphere @ (data - pca_mean[:, None]).
    We replicate what MNE does internally without going through Raw objects.
    """
    # PCA whitening: sphere = pca_components_ @ (1 / sqrt(pca_explained_variance_))
    # MNE's unmixing_matrix_ already combines W and sphere, so:
    # sources = unmixing_matrix_ @ pca_components_ @ (x - pca_mean)
    pca_mean = ica.pca_mean_  # (n_channels,)
    pca_comps = ica.pca_components_  # (n_components, n_channels)
    unmixing = ica.unmixing_matrix_  # (n_components, n_components)

    centered = data - pca_mean[:, np.newaxis]  # (n_channels, n_samples)
    pca_proj = pca_comps @ centered             # (n_components, n_samples)
    sources = unmixing @ pca_proj               # (n_components, n_samples)
    return sources


def ic_source_energy(
    iir: np.ndarray,
    asr: np.ndarray,
    orica: np.ndarray,
    ch_names: Sequence[str],
    sfreq: float,
    *,
    random_state: int = 42,
    max_iter: int = 500,
) -> List[dict]:
    """Compute per-IC source mean-square energy across pipeline stages.

    Fits MNE extended-Infomax ICA on the IIR stage array, applies the resulting
    unmixing matrix to all three stage arrays, and computes per-IC MS energy.
    ICs are classified by ICLabel.

    Parameters
    ----------
    iir : ndarray, shape (n_channels, n_samples)
        IIR-filtered stage array.
    asr : ndarray, shape (n_channels, n_samples)
        ASR-cleaned stage array.
    orica : ndarray, shape (n_channels, n_samples)
        ORICA-reconstructed stage array (final pipeline output).
    ch_names : sequence of str
        Channel names matching the standard_1020 montage (used for ICLabel).
    sfreq : float
        Sampling frequency in Hz.
    random_state : int
        Random seed passed to MNE ICA.
    max_iter : int
        Maximum ICA iterations.

    Returns
    -------
    list of dict, one per IC, with keys:
        ``ic``, ``label``, ``ms_iir``, ``ms_asr``, ``ms_orica``,
        ``pct_asr``, ``pct_orica``
    """
    raw_iir = _make_raw(iir, ch_names, sfreq)
    ica = _fit_ica(raw_iir, random_state=random_state, max_iter=max_iter)
    labels = _label_ica(raw_iir, ica)

    src_iir = _project_sources(ica, iir)
    src_asr = _project_sources(ica, asr)
    src_orica = _project_sources(ica, orica)

    results = []
    for i, label in enumerate(labels):
        ms_iir = float(np.mean(src_iir[i] ** 2))
        ms_asr = float(np.mean(src_asr[i] ** 2))
        ms_orica = float(np.mean(src_orica[i] ** 2))
        denom = ms_iir if ms_iir > 0 else 1.0
        results.append({
            "ic": i,
            "label": label,
            "ms_iir": ms_iir,
            "ms_asr": ms_asr,
            "ms_orica": ms_orica,
            "pct_asr": float(ms_asr / denom * 100.0),
            "pct_orica": float(ms_orica / denom * 100.0),
        })

    return results
