import base64
import json
from typing import Any, Dict

import numpy as np

from .normalization_base import Normalizer


class StandardScaler(Normalizer):
    r"""
    Online standard scaler for z-score normalization using Welford's algorithm.

    Transforms features to have zero mean and unit variance using a numerically
    stable and memory-efficient online algorithm. Uses Welford's algorithm to
    compute mean and variance incrementally without storing historical observations.

    For each feature $i$ at time $t$:

    - Mean update: $\mu_{t,i} = \mu_{t-1,i} + \frac{x_{t,i} - \mu_{t-1,i}}{t}$
    - Variance (Welford's M): $M_{t,i} = M_{t-1,i} + (x_{t,i} - \mu_{t-1,i})(x_{t,i} - \mu_{t,i})$
    - Sample variance: $\sigma^2_{t,i} = \frac{M_{t,i}}{t - \text{ddof}}$
    - Standardization: $z_{t,i} = \frac{x_{t,i} - \mu_{t,i}}{\sigma_{t,i}}$

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
        that calculates variance as `M / (n - ddof)`.

    Examples
    --------
    ```{python}
    from onorm import StandardScaler
    import numpy as np
    scaler = StandardScaler(n_dim=3)
    X = np.random.normal(loc=5, scale=2, size=(100, 3))
    for x in X:
        scaler.partial_fit(x)
    x_new = np.array([5.0, 5.0, 5.0])
    x_normalized = scaler.transform(x_new.copy())
    # x_normalized will be close to [0, 0, 0] since x_new is near the mean

    # Standardize without mean centering
    scaler2 = StandardScaler(n_dim=2, with_mean=False)

    # Use population variance instead of sample variance
    scaler3 = StandardScaler(n_dim=2, ddof=0)
    ```

    References
    ----------
    [Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

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
        The variance is computed as `M / (n - ddof)`, where ddof is the degrees
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the scaler state to a dictionary.

        Returns
        -------
        dict
            Dictionary with JSON-serializable metadata and base64-encoded arrays.
        """
        return {
            "version": "1.0",
            "class": "StandardScaler",
            "config": {
                "n_dim": self.n_dim,
                "with_mean": self.with_mean,
                "with_std": self.with_std,
                "ddof": self.ddof,
            },
            "state": {
                "n": self.n,
                "mean": base64.b64encode(self.mean.tobytes()).decode("ascii"),
                "M": base64.b64encode(self.M.tobytes()).decode("ascii"),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardScaler":
        """
        Deserialize a scaler from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary created by to_dict().

        Returns
        -------
        StandardScaler
            Deserialized scaler instance.
        """
        if data.get("class") != "StandardScaler":
            raise ValueError(f"Cannot deserialize {data.get('class')} as StandardScaler")

        config = data["config"]
        instance = cls(
            n_dim=config["n_dim"],
            with_mean=config["with_mean"],
            with_std=config["with_std"],
            ddof=config["ddof"],
        )

        state = data["state"]
        instance.n = state["n"]
        instance.mean = np.frombuffer(base64.b64decode(state["mean"]), dtype=np.float64)
        instance.M = np.frombuffer(base64.b64decode(state["M"]), dtype=np.float64)

        return instance

    def to_json(self) -> str:
        """Serialize the scaler to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "StandardScaler":
        """Deserialize a scaler from a JSON string."""
        return cls.from_dict(json.loads(json_str))
