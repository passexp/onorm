import numpy as np

from .normalization_base import Normalizer


class StandardScaler(Normalizer):
    """Standard Scaling (z-score normalization) for each feature

    The class performs online standardization so that each element of the
    vector $x_t$ has mean 0 and standard deviation 1. This implementation uses
    Welford's online algorithm for numerically stable computation of variance.

    The online mean update at time $t$ for feature $i$ is:

    $$\\mu_{ti} = \\mu_{(t-1)i} + \\frac{x_{ti} - \\mu_{(t-1)i}}{t}$$

    The online variance uses Welford's algorithm:

    $$M_{ti} = M_{(t-1)i} + (x_{ti} - \\mu_{(t-1)i})(x_{ti} - \\mu_{ti})$$

    $$\\sigma^2_{ti} = \\frac{M_{ti}}{t - 1}$$

    Then standardization is:

    $$z_{ti} = \\frac{x_{ti} - \\mu_{ti}}{\\sigma_{ti}}$$

    References:
        Welford's online algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(
        self, n_dim: int, with_mean: bool = True, with_std: bool = True, ddof: int = 1
    ) -> None:
        """Initialize the StandardScaler.

        Args:
            n_dim: Number of dimensions/features in the data.
            with_mean: If True, center the data by subtracting the mean.
            with_std: If True, scale the data to unit variance.
            ddof: Degrees of freedom for variance calculation (Bessel's correction).
                  ddof=1 (default) uses sample variance, ddof=0 uses population variance.
        """
        self.n_dim = n_dim
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof
        self.reset()

    def _update_mean(self, x: np.ndarray) -> np.ndarray:
        """Update running mean using Welford's algorithm.

        Args:
            x: A 1d array representing a new observation.

        Returns:
            The difference between x and the previous mean.
        """
        delta = x - self.mean
        self.mean += delta / self.n
        return delta

    def _update_variance(self, x: np.ndarray, delta_old: np.ndarray) -> None:
        """Update running variance using Welford's algorithm.

        Args:
            x: A 1d array representing a new observation.
            delta_old: The difference between x and the previous mean.
        """
        delta_new = x - self.mean
        self.M += delta_old * delta_new

    def partial_fit(self, x: np.ndarray) -> None:
        """Update the mean and variance estimates for each feature.

        Uses Welford's online algorithm for numerical stability when
        computing variance incrementally.

        Args:
            x: A 1d array representing a new observation.
        """
        self.n += 1
        delta_old = self._update_mean(x)
        self._update_variance(x, delta_old)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the feature vector to have zero mean and unit variance.

        If n=1, returns zeros (no variance information yet).
        If variance is near zero, only centers the data (no scaling).

        Args:
            x: A 1d array representing an observation to normalize.

        Returns:
            Standardized array with mean 0 and std 1 (when with_mean and with_std are True).
        """
        if self.n <= self.ddof:
            # Not enough observations for variance estimate
            return np.zeros_like(x)

        result = x.copy()

        # Center the data
        if self.with_mean:
            result = result - self.mean

        # Scale to unit variance
        if self.with_std:
            # Calculate standard deviation from Welford's M
            variance = self.M / (self.n - self.ddof)
            std = np.sqrt(variance)

            # Avoid division by zero - only scale features with non-zero variance
            mask = std > np.finfo(np.float64).eps
            result[mask] = result[mask] / std[mask]

        return result

    def reset(self) -> None:
        """Reset the scaler to its initial state."""
        self.n = 0
        self.mean = np.zeros(self.n_dim)
        self.M = np.zeros(self.n_dim)  # Welford's M for variance calculation
