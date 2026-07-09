"""IC artifact classifiers — legacy ORICA-compatible ICLabel path.

Requires ``pip install pyorica[pipeline]``.
"""

import warnings

import numpy as np

LABEL_NAMES = [
    'brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise',
    'channel noise', 'other',
]

_DEFAULT_ARTIFACT_LABELS = frozenset(
    ['muscle', 'eye', 'heart', 'line_noise', 'channel_noise']
)

# Aliases map every spelling ICLabelClassifier may see to the internal
# canonical short names used by _DEFAULT_ARTIFACT_LABELS / artifact_labels.
# This includes mne_icalabel's own multi-word label strings
# (e.g. 'muscle artifact', as returned by label_components()) alongside
# legacy/alternate spellings ('eog', 'ch_noise', 'muscle_artifact', ...).
_LABEL_ALIASES = {
    'muscle artifact': 'muscle',
    'muscle_artifact': 'muscle',
    'eye blink': 'eye',
    'eye_blink': 'eye',
    'eog': 'eye',
    'heart beat': 'heart',
    'heart_beat': 'heart',
    'ecg': 'heart',
    'line noise': 'line_noise',
    'channel noise': 'channel_noise',
    'ch_noise': 'channel_noise',
}


def _canonical_icalabel_label(label):
    lbl = str(label).strip().lower()
    return _LABEL_ALIASES.get(lbl, lbl)


class ICLabelClassifier:
    """Artifact classifier using ICLabel, matching legacy ORICA ``use_icalabel_online``.

    Differences from the previous pyorica implementation:

    * Feed **ASR-cleaned EEG** (``data``) directly to ``label_components``, not a
      reconstructed/scaled copy from sources.
    * Inject ORICA **unmixing (W)** and **mixing (A)** matrices separately into an
      MNE ICA container (Picard extended).
    * ``apply_car_bandpass`` controls whether common-average-reference and a
      1-100 Hz bandpass are applied before classification, matching ICLabel's
      documented training assumptions — off by default to match the legacy
      reference, but exposed so both settings can be benchmarked. Benchmarking
      confirmed applying both (not CAR alone) slightly improves ICLabel
      accuracy (fewer brain ICs misclassified/reduced as artifacts), at the
      cost of extra per-chunk compute for the bandpass filter.

    Parameters
    ----------
    info : mne.Info
        MNE channel info (names used for montage matching).
    threshold : float
        Minimum top-1 probability required to reject an IC (default 0.7).
    artifact_labels : set of str, optional
        Labels rejected once above ``threshold``. Defaults to
        ``{'muscle', 'eye', 'heart', 'line_noise', 'channel_noise'}``. Accepts
        legacy spellings (``'eog'``, ``'ecg'``, ``'ch_noise'``).
    apply_car_bandpass : bool
        See above (default False). ``True`` slightly improves classification
        accuracy (retains more brain-labeled ICs) but adds the runtime cost
        of an extra bandpass filter on every classified chunk.
    montage : str
        MNE montage name for electrode positions (default ``standard_1020``).
    record_snapshots : bool
        If True, append ``(seq, top1_labels, top1_probs, mask)`` to
        ``self.snapshots`` on each classification call (for benchmark timeline
        plots). Does not affect the returned artifact mask.

    Usage
    -----
    Pass as the ``classifier`` argument to ``EEGPipeline``::

        clf = ICLabelClassifier(raw.info, threshold=0.7)
        pipeline = EEGPipeline(n_channels=n_ch, sfreq=sfreq, classifier=clf)
    """

    def __init__(self, info, threshold=0.7, artifact_labels=None,
                 record_snapshots=False, apply_car_bandpass=False,
                 montage='standard_1020'):
        self._info = info
        self._ch_names = list(info['ch_names'])
        self._montage = montage
        self._apply_car_bandpass = apply_car_bandpass
        self._threshold = threshold
        self._record_snapshots = record_snapshots
        self.snapshots = []
        self._artifact_labels = frozenset(
            _canonical_icalabel_label(lbl)
            for lbl in (artifact_labels if artifact_labels is not None
                        else _DEFAULT_ARTIFACT_LABELS)
        )

    def __call__(self, data, sources, unmixing, mixing, sfreq):
        n_components = sources.shape[0]
        # ICLabel's FIR filter needs ~3.5 x sfreq samples (825 at 250 Hz, 845 at
        # 256 Hz). Short final chunks cannot be classified — treat as artifact-free.
        if data.shape[1] < int(sfreq * 3.5):
            return np.zeros(n_components, dtype=bool)

        label_strings, prob_top1 = self._run_icalabel(
            data, unmixing, mixing, sfreq, n_components
        )
        mask = self._artifact_mask(label_strings, prob_top1)

        if self._record_snapshots:
            self.snapshots.append((
                len(self.snapshots),
                list(label_strings),
                np.asarray(prob_top1, dtype=np.float64).copy(),
                mask.copy(),
            ))

        return mask

    def _artifact_mask(self, label_strings, prob_top1):
        mask = np.zeros(len(label_strings), dtype=bool)
        for i, label in enumerate(label_strings):
            prob = float(prob_top1[i]) if np.isfinite(prob_top1[i]) else None
            if prob is None or prob < self._threshold:
                continue
            canonical = _canonical_icalabel_label(label)
            mask[i] = canonical in self._artifact_labels
        return mask

    def _run_icalabel(self, data, unmixing, mixing, sfreq, n_components):
        """Port of ORICA/code/orica_processor.py ``use_icalabel_online``."""
        import mne
        from mne.preprocessing import ICA
        from mne_icalabel import label_components

        n_channels = len(self._ch_names)
        ch_names_mne, montage_obj = self._canonical_names_for_montage()

        info = mne.create_info(
            ch_names=ch_names_mne, sfreq=float(sfreq), ch_types='eeg', verbose=False
        )
        # copy=True: RawArray otherwise reuses `data`'s buffer when it's already
        # float64, and set_eeg_reference/filter below mutate in place — without
        # this copy, apply_car_bandpass=True silently rewrites the caller's
        # ASR-cleaned chunk (and anything aliasing it, e.g. verbose _last_asr).
        raw = mne.io.RawArray(np.array(data, dtype=np.float64, copy=True), info,
                               verbose=False)
        raw.set_montage(montage_obj, on_missing='ignore', verbose=False)

        if self._apply_car_bandpass:
            raw.set_eeg_reference('average', projection=False, verbose=False)
            raw.filter(1.0, 100.0, verbose=False)

        unmixing = np.asarray(unmixing, dtype=float)
        if unmixing.shape == (n_channels, n_channels):
            unmixing = unmixing[:n_components, :]
        mixing = np.asarray(mixing, dtype=float)

        ica = ICA(
            n_components=n_components,
            method='picard',
            fit_params={'extended': True, 'ortho': False},
            random_state=97,
            verbose=False,
        )
        ica.unmixing_matrix_ = unmixing.copy()
        ica.mixing_matrix_ = mixing.copy()
        ica._unmixing = unmixing.copy()
        ica._mixing = mixing.copy()
        ica.n_components_ = n_components
        ica.ch_names = list(ch_names_mne)
        ica._ica_names = [f'IC{k:03d}' for k in range(n_components)]
        ica.current_fit = 'raw'
        ica.pca_mean_ = np.zeros(n_channels)
        ica.pca_components_ = np.eye(n_channels)
        ica.pca_explained_variance_ = np.ones(n_channels)

        labels_out = self._label_components(label_components, raw, ica)

        ic_labels = np.asarray(labels_out['labels'], dtype=object)
        ic_probs = np.asarray(labels_out['y_pred_proba'], dtype=np.float64)
        return ic_labels, ic_probs

    def _label_components(self, label_components, raw, ica):
        if self._apply_car_bandpass:
            return label_components(raw, ica, method='iclabel')
        # Data is already IIR-filtered upstream; ORICA matrices match that
        # reference. mne-icalabel only warns about Raw.info metadata (CAR,
        # 1-100 Hz) here — suppress that noise since it's expected.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='.*common average reference.*',
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                'ignore', message='.*not filtered between 1 and 100 Hz.*',
                category=RuntimeWarning,
            )
            return label_components(raw, ica, method='iclabel')

    def _canonical_names_for_montage(self):
        """Map FP1/FZ/CZ-style labels to MNE standard_1020 spellings (Fp1/Fz/Cz).

        Assumes the caller's channel names already correspond to a standard
        10-20 layout (true for the NCTU Quick-30 dataset this pipeline targets
        today, see benchmarks/run_validation.py). Revisit if/when a genuinely
        custom montage with real digitized positions needs to flow through
        this classifier instead of being forced onto standard_1020.
        """
        import mne

        montage_obj = mne.channels.make_standard_montage(self._montage)
        lookup = {name.upper().replace(' ', ''): name for name in montage_obj.ch_names}
        out = []
        for ch in self._ch_names:
            key = str(ch).strip().upper().replace(' ', '')
            out.append(lookup.get(key, str(ch).strip()))
        return out, montage_obj
