"""IC artifact classifiers."""

import numpy as np

LABEL_NAMES = ['brain', 'muscle', 'eog', 'ecg', 'line_noise', 'ch_noise', 'other']
_DEFAULT_ARTIFACT_LABELS = frozenset(['muscle', 'eog', 'ecg', 'line_noise', 'ch_noise'])

# Aliases for alternate spellings across mne-icalabel versions and user convention.
# All map to the internal LABEL_NAMES entries used by run_iclabel column order.
_LABEL_ALIASES = {
    'eye': 'eog',
    'eye_blink': 'eog',
    'heart': 'ecg',
    'heart_beat': 'ecg',
    'channel_noise': 'ch_noise',
    'muscle_artifact': 'muscle',
}


def _normalise_label(label: str) -> str:
    lbl = label.strip().lower()
    return _LABEL_ALIASES.get(lbl, lbl)


def _normalise_channel_names(info):
    """Return a copy of info with channel names remapped to MNE standard_1020 spelling.

    Handles EEGLAB-style all-caps names (FP1, FZ, CZ) that mne-icalabel's montage
    lookup requires in mixed-case form (Fp1, Fz, Cz). Channels not found in
    standard_1020 are left unchanged.
    """
    import mne
    montage = mne.channels.make_standard_montage('standard_1020')
    lookup = {n.upper().replace(' ', ''): n for n in montage.ch_names}
    rename = {
        ch: lookup[ch.strip().upper().replace(' ', '')]
        for ch in info['ch_names']
        if ch.strip().upper().replace(' ', '') in lookup and
           ch != lookup[ch.strip().upper().replace(' ', '')]
    }
    if not rename:
        return info
    info = info.copy()
    info.rename_channels(rename)
    return info


class ICLabelClassifier:
    """Artifact classifier using the ICLabel neural network.

    Parameters
    ----------
    info : mne.Info
        MNE channel info with electrode positions (required for topographic maps).
        Channel names are automatically normalised to MNE standard_1020 spelling
        (e.g. ``FP1`` → ``Fp1``) so EEGLAB-sourced infos work without manual
        renaming.
    artifact_labels : set of str, optional
        IC labels treated as artifacts. Defaults to
        ``{'muscle', 'eog', 'ecg', 'line_noise', 'ch_noise'}``.
        Accepts both legacy spellings (``'eog'``, ``'ecg'``, ``'ch_noise'``) and
        newer mne-icalabel aliases (``'eye'``, ``'heart'``, ``'channel_noise'``).
    threshold : float
        Probability threshold for the top-predicted label to be accepted as an
        artifact (default 0.5).

    Usage
    -----
    Pass as the ``classifier`` argument to ``EEGPipeline``::

        clf = ICLabelClassifier(raw.info)
        pipeline = EEGPipeline(n_channels=n_ch, sfreq=sfreq, classifier=clf)

    Requires ``pip install pyorica[pipeline]``.
    """

    def __init__(self, info, artifact_labels=None, threshold=0.5,
                 record_snapshots=False):
        self._info = _normalise_channel_names(info)
        self._artifact_labels = frozenset(
            _normalise_label(lbl)
            for lbl in (artifact_labels if artifact_labels is not None
                        else _DEFAULT_ARTIFACT_LABELS)
        )
        self._threshold = threshold
        self._record_snapshots = record_snapshots
        self.snapshots: list = []  # (seq_num, top1_labels, top1_probs) per call

    def __call__(self, sources, mixing_matrix, sfreq):
        """Classify ICs and return an artifact mask.

        Parameters
        ----------
        sources : ndarray, shape (n_components, n_samples)
            IC activation time series.
        mixing_matrix : ndarray, shape (n_channels, n_components)
            Mixing matrix A = pinv(W @ sphere).
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        mask : ndarray of bool, shape (n_components,)
            True where a component is classified as an artifact.
        """
        # ICLabel's FIR filter needs ~3.5 × sfreq samples (825 at 250 Hz, 845 at
        # 256 Hz). Short final chunks cannot be classified — treat as artifact-free.
        if sources.shape[1] < int(sfreq * 3.5):
            return np.zeros(sources.shape[0], dtype=bool)

        proba = self._get_probabilities(sources, mixing_matrix, sfreq)
        argmax_idx = np.argmax(proba, axis=1)
        pred_labels = [LABEL_NAMES[i] for i in argmax_idx]
        pred_proba = proba[np.arange(len(pred_labels)), argmax_idx]

        mask = np.array(
            [label in self._artifact_labels and prob >= self._threshold
             for label, prob in zip(pred_labels, pred_proba)],
            dtype=bool,
        )

        if self._record_snapshots:
            self.snapshots.append((
                len(self.snapshots),
                pred_labels,
                pred_proba.copy(),
                mask.copy(),
            ))

        return mask

    def _get_probabilities(self, sources, mixing_matrix, sfreq):
        """Return ICLabel probability matrix of shape (n_components, 7).

        Columns: brain, muscle, eog, ecg, line_noise, ch_noise, other.
        """
        import mne
        from mne_icalabel.iclabel import get_iclabel_features, run_iclabel

        n_components, n_samples = sources.shape
        n_channels = mixing_matrix.shape[0]

        # Reconstruct EEG in volts (scale to ~10 µV range for ICLabel)
        eeg = (mixing_matrix @ sources) * 1e-5

        raw = mne.io.RawArray(eeg, self._info, verbose=False)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        raw.filter(1.0, 100.0, verbose=False)

        ica = self._make_ica(mixing_matrix, n_components, n_channels, n_samples)
        features = get_iclabel_features(raw, ica)
        return run_iclabel(*features, backend=None)

    def _make_ica(self, mixing_matrix, n_components, n_channels, n_samples):
        """Construct a fitted-looking MNE ICA from the ORICA mixing matrix."""
        import mne

        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method='infomax',
            fit_params={'extended': True},
            verbose=False,
        )
        ica.current_fit = 'raw'
        ica.n_components_ = n_components
        ica._max_pca_components = n_components
        ica.n_pca_components = n_components
        ica.pca_mean_ = np.zeros(n_channels)
        ica.pca_components_ = np.eye(n_channels)
        ica.pca_explained_variance_ = np.ones(n_channels)
        # unmixing = pinv(A); then get_components() = pinv(unmixing) = A
        ica.unmixing_matrix_ = np.linalg.pinv(mixing_matrix)
        ica.mixing_matrix_ = mixing_matrix
        ica._ica_names = [f'ICA{i:03d}' for i in range(n_components)]
        ica.n_samples_ = n_samples
        ica.n_iter_ = 1
        ica.reject_ = None
        ica.pre_whitener_ = np.ones((n_channels, 1))
        ica.ch_names = self._info['ch_names']
        ica.exclude = []
        ica.info = self._info.copy()
        return ica
