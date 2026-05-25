"""IC artifact classifiers."""

import numpy as np

LABEL_NAMES = ['brain', 'muscle', 'eog', 'ecg', 'line_noise', 'ch_noise', 'other']
_DEFAULT_ARTIFACT_LABELS = frozenset(['muscle', 'eog', 'ecg', 'line_noise', 'ch_noise'])


class ICLabelClassifier:
    """Artifact classifier using the ICLabel neural network.

    Parameters
    ----------
    info : mne.Info
        MNE channel info with electrode positions (required for topographic maps).
    artifact_labels : set of str, optional
        IC labels treated as artifacts. Defaults to
        ``{'muscle', 'eog', 'ecg', 'line_noise', 'ch_noise'}``.
        Valid values: ``'brain'``, ``'muscle'``, ``'eog'``, ``'ecg'``,
        ``'line_noise'``, ``'ch_noise'``, ``'other'``.
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

    def __init__(self, info, artifact_labels=None, threshold=0.5):
        self._info = info
        self._artifact_labels = (
            set(artifact_labels) if artifact_labels is not None
            else set(_DEFAULT_ARTIFACT_LABELS)
        )
        self._threshold = threshold

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
        proba = self._get_probabilities(sources, mixing_matrix, sfreq)
        argmax_idx = np.argmax(proba, axis=1)
        pred_labels = [LABEL_NAMES[i] for i in argmax_idx]
        pred_proba = proba[np.arange(len(pred_labels)), argmax_idx]
        return np.array(
            [label in self._artifact_labels and prob >= self._threshold
             for label, prob in zip(pred_labels, pred_proba)],
            dtype=bool,
        )

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
