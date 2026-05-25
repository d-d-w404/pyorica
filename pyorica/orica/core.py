"""Online Recursive ICA decomposer (ORICA).

Reference
---------
Hsu, S.-H., Mullen, T., Jung, T.-P., & Cauwenberghs, G. (2016).
Real-time adaptive EEG source separation using online recursive independent
component analysis. IEEE Transactions on Neural Systems and Rehabilitation
Engineering, 24(3), 309-319.
"""

import numpy as np


class ORICAFilter:
    """Online Recursive ICA with RLS whitening.

    Parameters
    ----------
    n_components : int
        Number of EEG channels / independent components.
    sfreq : float
        Sampling frequency in Hz.
    ff_profile : {'cooling', 'constant', 'adaptive'}
        Forgetting factor profile.
    block_size_white : int
        Block size for RLS whitening updates.
    block_size_ica : int
        Block size for ICA weight updates.
    tau_const : float
        Local stationarity parameter (samples). Controls steady-state λ.
    gamma : float
        Decay rate for cooling forgetting factor.
    lambda_0 : float
        Initial forgetting factor.
    num_subgaussian : int
        Number of sub-Gaussian sources (default 0; EEG brain sources are
        typically super-Gaussian).
    """

    def __init__(
        self,
        n_components: int,
        sfreq: float,
        ff_profile: str = "cooling",
        block_size_white: int = 8,
        block_size_ica: int = 8,
        tau_const: float = np.inf,
        gamma: float = 0.6,
        lambda_0: float = 0.995,
        num_subgaussian: int = 0,
    ) -> None:
        self.n_components = n_components
        self.sfreq = sfreq
        self.ff_profile = ff_profile
        self.block_size_white = block_size_white
        self.block_size_ica = block_size_ica
        self.tau_const = tau_const
        self.gamma = gamma
        self.lambda_0 = lambda_0

        # steady-state lambda
        if np.isfinite(tau_const):
            self._lambda_const = 1.0 - np.exp(-1.0 / (tau_const * sfreq))
        else:
            self._lambda_const = 0.0

        # kurtosis sign: True = super-Gaussian, False = sub-Gaussian
        self._kurtosis_sign = np.ones(n_components, dtype=bool)
        if num_subgaussian > 0:
            self._kurtosis_sign[:num_subgaussian] = False

        # state
        self._W = np.eye(n_components)          # ICA weight matrix
        self._sphere = np.eye(n_components)     # whitening matrix
        self._counter = 0
        self._Rn = None                         # leaky average for NSI

    # ------------------------------------------------------------------
    # Public attributes
    # ------------------------------------------------------------------

    @property
    def weights_(self) -> np.ndarray:
        """ICA weight matrix W, shape (n_components, n_components)."""
        return self._W

    @property
    def sphere_(self) -> np.ndarray:
        """Whitening matrix (sphere), shape (n_components, n_components)."""
        return self._sphere

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, calibration_data: np.ndarray) -> "ORICAFilter":
        """Warm-start weights from a calibration recording.

        Parameters
        ----------
        calibration_data : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        self
        """
        if calibration_data.shape[0] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} channels, "
                f"got {calibration_data.shape[0]}"
            )
        self._run_orica(calibration_data)
        return self

    def update(self, chunk: np.ndarray) -> None:
        """Incrementally update W and sphere from one chunk.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_samples)
        """
        if chunk.shape[0] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} channels, "
                f"got {chunk.shape[0]}"
            )
        self._run_orica(chunk)

    def transform(self, chunk: np.ndarray) -> np.ndarray:
        """Project EEG channels to source space.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        np.ndarray, shape (n_components, n_samples)
        """
        return self._W @ (self._sphere @ chunk)

    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        """Reconstruct sensor-space signals from (modified) sources.

        Parameters
        ----------
        sources : np.ndarray, shape (n_components, n_samples)

        Returns
        -------
        np.ndarray, shape (n_channels, n_samples)
        """
        A = np.linalg.pinv(self._W @ self._sphere)
        return A @ sources

    # ------------------------------------------------------------------
    # Internal algorithm
    # ------------------------------------------------------------------

    def _gen_cooling_ff(self, t: np.ndarray) -> np.ndarray:
        return self.lambda_0 / np.power(np.maximum(t, 1e-10), self.gamma)

    def _forgetting_factor(self, data_range: np.ndarray) -> np.ndarray:
        if self.ff_profile in ("cooling", "constant"):
            lam = self._gen_cooling_ff(self._counter + data_range)
            if self._lambda_const > 0:
                lam = np.where(lam < self._lambda_const, self._lambda_const, lam)
            return lam
        # adaptive falls back to constant for now
        return np.full(len(data_range), self._lambda_const)

    def _dynamic_whitening(self, block: np.ndarray, data_range: np.ndarray) -> None:
        n_pts = block.shape[1]
        lam = self._forgetting_factor(data_range)
        lam_avg = 1.0 - lam[int(np.ceil(len(lam) / 2)) - 1]
        v = self._sphere @ block
        Q = lam_avg / (1.0 - lam_avg) + np.linalg.norm(v, "fro") ** 2 / n_pts
        self._sphere = (1.0 / lam_avg) * (
            self._sphere - (v @ v.T) / n_pts / Q @ self._sphere
        )

    def _dynamic_orica(self, block: np.ndarray, data_range: np.ndarray) -> None:
        n_pts = block.shape[1]
        Y = self._W @ block
        F = np.empty_like(Y)
        F[self._kurtosis_sign] = -2.0 * np.tanh(Y[self._kurtosis_sign])
        F[~self._kurtosis_sign] = np.tanh(Y[~self._kurtosis_sign]) - Y[~self._kurtosis_sign]

        # non-stationarity index
        model_fitness = np.eye(self.n_components) + (Y @ F.T) / n_pts
        if self._Rn is None:
            self._Rn = model_fitness
        else:
            self._Rn = 0.99 * self._Rn + 0.01 * model_fitness

        lam = self._forgetting_factor(data_range)
        self._counter += n_pts

        lam_prod = np.prod(1.0 / (1.0 - lam))
        Q = 1.0 + lam * (np.sum(F * Y, axis=0) - 1.0)
        self._W = lam_prod * (self._W - Y @ np.diag(lam / Q) @ F.T @ self._W)

        # orthogonalise
        D, V = np.linalg.eigh(self._W @ self._W.T)
        D = np.maximum(D, 1e-12)
        self._W = (V @ np.diag(1.0 / np.sqrt(D)) @ V.T) @ self._W

    def _run_orica(self, data: np.ndarray) -> None:
        """Run whitening + ICA block updates over data (n_channels, n_samples)."""
        n_pts = data.shape[1]
        block_size = min(self.block_size_white, self.block_size_ica)
        n_blocks = n_pts // block_size

        # centre data for whitening
        data_c = data - data.mean(axis=1, keepdims=True)
        # apply current sphere to get mixtures
        mixtures = self._sphere @ data_c

        for bi in range(n_blocks):
            start = bi * block_size
            end = min(n_pts, (bi + 1) * block_size)
            data_range = np.arange(start, end) + 1  # 1-indexed, matches MATLAB
            block = data_c[:, start:end]

            self._dynamic_whitening(block, data_range)
            mixtures[:, start:end] = self._sphere @ data_c[:, start:end]
            self._dynamic_orica(mixtures[:, start:end], data_range)
