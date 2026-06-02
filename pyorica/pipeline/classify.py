"""IC artifact classifiers — legacy ORICA-compatible ICLabel path."""

from __future__ import annotations

import numpy as np

ICLABEL_CLASS_NAMES = (
    "brain",
    "muscle",
    "eye",
    "heart",
    "line_noise",
    "channel_noise",
    "other",
)

_LABEL_ALIASES = {
    "muscle_artifact": "muscle",
    "eye_blink": "eye",
    "eog": "eye",
    "ecg": "heart",
    "heart_beat": "heart",
    "ch_noise": "channel_noise",
}


def _label_to_str(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    return str(val).strip().lower().replace(" ", "_")


def _canonical_icalabel_label(label: str) -> str:
    return _LABEL_ALIASES.get(label, label)


def _canonical_names_for_standard_montage(
    ch_names: list,
    montage_name: str = "standard_1020",
):
    """Map FP1/FZ/CZ-style labels to MNE standard_1020 spellings (Fp1/Fz/Cz)."""
    import mne

    montage_obj = mne.channels.make_standard_montage(montage_name)
    lookup = {name.upper().replace(" ", ""): name for name in montage_obj.ch_names}
    out = []
    for ch in ch_names:
        key = str(ch).strip().upper().replace(" ", "")
        out.append(lookup.get(key, str(ch).strip()))
    return out, montage_obj


def _pack_icalabel_per_ic(ic_labels, ic_probs, n_components: int) -> dict:
    """Align label_components output to per-IC strings and top-1 probabilities."""
    n_ic = int(n_components)
    label_list = ["other"] * n_ic
    if ic_labels is not None:
        raw_labels = np.asarray(ic_labels).ravel()
        for i in range(min(n_ic, len(raw_labels))):
            label_list[i] = _canonical_icalabel_label(_label_to_str(raw_labels[i]))

    prob_top1 = np.full(n_ic, np.nan, dtype=np.float64)

    if ic_probs is not None:
        p = np.asarray(ic_probs, dtype=np.float64)
        if p.ndim == 1 and p.size == n_ic:
            prob_top1 = p.astype(np.float64, copy=False)
        elif p.ndim == 2:
            if p.shape[1] == n_ic and p.shape[0] != n_ic:
                p = p.T
            if p.shape[0] == n_ic:
                for i in range(n_ic):
                    prob_top1[i] = float(np.max(p[i, :min(p.shape[1], len(ICLABEL_CLASS_NAMES))]))

    return {
        "ic_labels": np.asarray(label_list, dtype=object),
        "ic_prob_top1": prob_top1,
    }


class ICLabelClassifier:
    """Artifact classifier using ICLabel, matching legacy ORICA ``use_icalabel_online``.

    Differences from the previous pyorica implementation (now aligned with legacy):

    * Feed **ASR-cleaned EEG** (``data``) directly to ``label_components``, not a
      reconstructed/scaled copy from sources.
    * Inject ORICA **unmixing (W)** and **mixing (A)** matrices separately into an
      MNE ICA container (Picard extended).
    * Never remove ICs whose label is ``brain`` or ``other``; remove all other
      classes when top-1 probability ≥ ``threshold``.

    Parameters
    ----------
    info : mne.Info
        MNE channel info (names used for montage matching).
    threshold : float
        Minimum top-1 probability to reject a non-protected IC (default 0.7).
    protect_labels : set of str, optional
        Labels never zeroed regardless of probability. Defaults to
        ``{'brain', 'other'}`` (legacy ORICA behaviour).
    montage : str
        MNE montage name for electrode positions (default ``standard_1020``).

    Usage
    -----
    Pass as the ``classifier`` argument to ``EEGPipeline``::

        clf = ICLabelClassifier(raw.info, threshold=0.7)
        pipeline = EEGPipeline(n_channels=n_ch, sfreq=sfreq, classifier=clf)
    """

    def __init__(self, info, threshold=0.7, protect_labels=None, montage="standard_1020"):
        self._info = info
        self._ch_names = list(info["ch_names"])
        self._threshold = threshold
        self._montage = montage
        self._protect_labels = frozenset(
            protect_labels if protect_labels is not None else ("brain", "other")
        )

    def __call__(self, data, sources, unmixing, mixing, sfreq):
        """Classify ICs and return an artifact mask.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            ASR-cleaned EEG chunk (legacy ``use_icalabel_online`` input).
        sources : ndarray, shape (n_components, n_samples)
            ORICA source activations for the same chunk.
        unmixing : ndarray, shape (n_components, n_channels)
            ORICA unmixing matrix W @ sphere.
        mixing : ndarray, shape (n_channels, n_components)
            Mixing matrix A = pinv(unmixing).
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        mask : ndarray of bool, shape (n_components,)
            True where a component should be zeroed as an artifact.
        """
        n_components = sources.shape[0]
        if data.shape[1] < int(sfreq * 3.5):
            return np.zeros(n_components, dtype=bool)

        label_strings, prob_top1 = self._run_icalabel(
            data, unmixing, mixing, sfreq, n_components
        )
        return self._artifact_mask(label_strings, prob_top1)

    def _artifact_mask(self, label_strings, prob_top1) -> np.ndarray:
        """Legacy rule: skip brain/other; reject everything else above threshold."""
        mask = np.zeros(len(label_strings), dtype=bool)
        for i, label in enumerate(label_strings):
            label_str = str(label)
            if label_str in self._protect_labels:
                continue
            prob = float(prob_top1[i]) if np.isfinite(prob_top1[i]) else None
            if prob is not None and prob >= self._threshold:
                mask[i] = True
        return mask

    def _run_icalabel(self, data, unmixing, mixing, sfreq, n_components):
        """Port of ORICA/code/orica_processor.py ``use_icalabel_online``."""
        import mne
        from mne.preprocessing import ICA
        from mne_icalabel import label_components

        n_channels = len(self._ch_names)
        ch_names_mne, montage_obj = _canonical_names_for_standard_montage(
            self._ch_names, self._montage
        )

        info = mne.create_info(
            ch_names=ch_names_mne, sfreq=float(sfreq), ch_types="eeg", verbose=False
        )
        raw = mne.io.RawArray(np.asarray(data, dtype=np.float64), info, verbose=False)
        try:
            raw.set_montage(montage_obj, on_missing="ignore", verbose=False)
        except TypeError:
            raw.set_montage(montage_obj, verbose=False)

        W_use = np.asarray(unmixing, dtype=float)
        if W_use.shape == (n_channels, n_channels):
            W_use = W_use[:n_components, :]
        A_use = np.asarray(mixing, dtype=float)

        ica = ICA(
            n_components=n_components,
            method="picard",
            fit_params={"extended": True, "ortho": False},
            random_state=97,
            verbose=False,
        )
        ica.mixing_matrix_ = A_use.copy()
        ica.unmixing_matrix_ = W_use.copy()
        ica._mixing = A_use.copy()
        ica._unmixing = W_use.copy()
        ica.n_components_ = n_components
        ica.ch_names = list(ch_names_mne)
        ica._ica_names = [f"IC {k:03d}" for k in range(n_components)]
        ica.picks_ = np.arange(n_channels)
        ica._ica_channel_names = list(ch_names_mne)
        ica.current_fit = "raw"
        ica.pca_mean_ = np.zeros(n_channels)
        ica.pca_components_ = np.eye(n_channels)
        ica.pca_explained_variance_ = np.ones(n_channels)
        ica.pca_explained_variance_ratio_ = (
            ica.pca_explained_variance_ / ica.pca_explained_variance_.sum()
        )
        ica._pre_whitener = np.ones((n_channels, 1))
        ica._whitener = np.eye(n_channels)

        labels_out = label_components(raw, ica, method="iclabel")
        ic_probs = labels_out.get("y_pred_proba", None)
        ic_labels = labels_out.get("labels", None)
        meta = _pack_icalabel_per_ic(ic_labels, ic_probs, n_components)
        return meta["ic_labels"], meta["ic_prob_top1"]
