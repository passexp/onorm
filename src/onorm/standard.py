import numpy as np

from .normalization_base import Normalizer


class StandardScaler(Normalizer):
    """
    Online standardization (z-score normalization) using Welford's algorithm.

    Transforms features to have zero mean and unit variance using an online
    algorithm that is numerically stable and memory-efficient. This implementation
    uses Welford's algorithm to compute mean and variance incrementally without
    storing historical observations.

    For each feature i at time t:
        - Mean update: μ_ti = μ_(t-1)i + (x_ti - μ_(t-1)i) / t
        - Variance (Welford's M): M_ti = M_(t-1)i + (x_ti - μ_(t-1)i)(x_ti - μ_ti)
        - Sample variance: σ²_ti = M_ti / (t - ddof)
        - Standardization: z_ti = (x_ti - μ_ti) / σ_ti

    Parameters
    ----------
    n_dim : int
        Number of dimensions/features to normalize.
    with_mean : bool, default=True
        If True, center the data by subtracting the mean before scaling.
    with_std : bool, default=True
        If True, scale the data to unit standard deviation.
    ddof : int, default=1
        Degrees of freedom for variance calculation (Bessel's correction).
        - ddof=1 (default) uses sample variance (divide by n-1)
        - ddof=0 uses population variance (divide by n)

    Attributes
    ----------
    n : int
        Number of observations seen so far.
    mean : np.ndarray
        Running mean for each feature, shape (n_dim,).
    M : np.ndarray
        Welford's M statistic for variance calculation, shape (n_dim,).
    variance : np.ndarray
        Computed variance for each feature, shape (n_dim,). This is a property
        that calculates variance as M / (n - ddof).

    Examples
    --------
    >>> from onorm import StandardScaler
    >>> import numpy as np
    >>> scaler = StandardScaler(n_dim=3)
    >>> X = np.random.normal(loc=5, scale=2, size=(100, 3))
    >>> for x in X:
    ...     scaler.partial_fit(x)
    >>> x_new = np.array([5.0, 5.0, 5.0])
    >>> x_normalized = scaler.transform(x_new.copy())
    >>> # x_normalized will be close to [0, 0, 0] since x_new is near the mean

    >>> # Standardize without mean centering
    >>> scaler2 = StandardScaler(n_dim=2, with_mean=False)

    >>> # Use population variance instead of sample variance
    >>> scaler3 = StandardScaler(n_dim=2, ddof=0)

    References
    ----------
    Welford's online algorithm:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    Notes
    -----
    - If fewer than (ddof + 1) observations have been seen, transform returns zeros
    - For features with near-zero variance, only centering is applied to avoid
      division by zero
    """

    def __init__(
        self, n_dim: int, with_mean: bool = True, with_std: bool = True, ddof: int = 1
    ) -> None:
        self.n_dim = n_dim
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof
        self.reset()

    def _update_mean(self, x: np.ndarray) -> np.ndarray:
        """
        Update running mean using Welford's algorithm.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array representing a new observation.

        Returns
        -------
        np.ndarray
            The difference between x and the previous mean (delta_old).
        """
        delta = x - self.mean
        self.mean += delta / self.n
        return delta

    def _update_variance(self, x: np.ndarray, delta_old: np.ndarray) -> None:
        """
        Update Welford's M statistic for variance calculation.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array representing a new observation.
        delta_old : np.ndarray
            The difference between x and the previous mean.
        """
        delta_new = x - self.mean
        self.M += delta_old * delta_new

    @property
    def variance(self) -> np.ndarray:
        """
        Calculate the variance from Welford's M statistic.

        Returns
        -------
        np.ndarray
            Variance for each feature, shape (n_dim,). If n <= ddof, returns zeros.

        Notes
        -----
        The variance is computed as M / (n - ddof), where ddof is the degrees
        of freedom correction (Bessel's correction).
        """
        if self.n <= self.ddof:
            return np.zeros(self.n_dim)
        return self.M / (self.n - self.ddof)

    def partial_fit(self, x: np.ndarray) -> None:
        """
        Update mean and variance estimates using Welford's algorithm.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) representing a new observation.
        """
        self.n += 1
        delta_old = self._update_mean(x)
        self._update_variance(x, delta_old)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize features to zero mean and unit variance.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) to normalize.

        Returns
        -------
        np.ndarray
            Standardized array of shape (n_dim,). If with_mean and with_std are
            both True, features will have approximately mean=0 and std=1.

        Notes
        -----
        - Returns zeros if n <= ddof (insufficient observations)
        - For constant features (zero variance), only centering is applied
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
            # Calculate standard deviation from variance
            std = np.sqrt(self.variance)

            # Avoid division by zero - only scale features with non-zero variance
            mask = std > np.finfo(np.float64).eps
            result[mask] = result[mask] / std[mask]

        return result

    def reset(self) -> None:
        """
        Reset the scaler to initial state.

        Resets observation count to 0 and reinitializes mean and variance
        statistics to zeros.
        """
        self.n = 0
        self.mean = np.zeros(self.n_dim)
        self.M = np.zeros(self.n_dim)  # Welford's M for variance calculation
