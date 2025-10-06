import base64
import json
from typing import Any, Dict

import numpy as np
from scipy.linalg import cholesky, inv

from .normalization_base import Normalizer


class MultivariateNormalizer(Normalizer):
    r"""
    Online multivariate normalizer with decorrelation via covariance estimation.

    Transforms multivariate data to have zero mean, unit variance, and zero
    correlation (decorrelation). Uses online estimation of the covariance matrix
    and computes the inverse square root for transformation.

    The transformation is:

    $$\Sigma^{-1/2} (x - \mu)$$

    where $\mu$ is the estimated mean and $\Sigma$ is the estimated covariance matrix.

    Parameters
    ----------
    n_dim : int
        Number of dimensions/features in the data.

    Attributes
    ----------
    n : int
        Number of observations seen so far.
    muhat : np.ndarray
        Estimated mean vector, shape (n_dim,).
    Sigmahat : np.ndarray
        Estimated covariance matrix (computed from _M), shape (n_dim, n_dim).
    invsqrtSigmahat : np.ndarray
        Inverse square root of covariance matrix ($\Sigma^{-1/2}$), shape (n_dim, n_dim).

    Examples
    --------
    ```{python}
    from onorm import MultivariateNormalizer
    import numpy as np
    normalizer = MultivariateNormalizer(n_dim=3)
    # Generate correlated data
    cov = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    X = np.random.multivariate_normal([0, 0, 0], cov, size=100)
    for x in X:
        normalizer.partial_fit(x)
    x_new = np.array([1.0, 1.0, 1.0])
    x_normalized = normalizer.transform(x_new.copy())
    # x_normalized will be decorrelated with zero mean and unit variance
    ```

    Notes
    -----
    - Time complexity: O(d²) per observation for fitting, O(d²) for transformation
    - Space complexity: O(d²) for storing covariance matrix
    - The covariance matrix is computed using Welford's online algorithm
    - Transformation uses Cholesky decomposition of the inverse covariance

    References
    ----------
    [Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
    """

    def __init__(self, n_dim: int) -> None:
        self.n_dim = n_dim
        self.reset()

    def partial_fit(self, x: np.ndarray) -> None:
        """
        Update mean and covariance estimates with a new observation.

        Uses Welford's online algorithm to incrementally update the mean
        and covariance matrix.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) representing a new observation.

        See Also
        --------
        Normalizer.partial_fit : Base class method for incremental fitting.

        References
        ----------
        [Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
        """
        delta = x - self.muhat
        self.n += 1
        self._update_muhat(delta)
        self._update_Sigmahat(delta)

    def _update_muhat(self, delta: np.ndarray) -> None:
        """Update running mean estimate using Welford's algorithm."""
        self.muhat += delta / self.n

    def _update_Sigmahat(self, delta: np.ndarray) -> None:
        """
        Update covariance matrix using Welford's algorithm.

        Accumulates the sum of outer products in _M, which is used to
        compute the covariance matrix.
        """
        delta = delta.reshape(-1, 1)
        frac = (self.n - 1) / self.n
        self._M += frac * delta @ delta.T

    @property
    def Sigmahat(self) -> np.ndarray:
        """
        Compute the sample covariance matrix.

        Returns
        -------
        np.ndarray
            Sample covariance matrix, shape (n_dim, n_dim). Returns identity
            matrix if n <= 1.
        """
        if self.n <= 1:
            return np.eye(self.n_dim, dtype=np.float64)
        return self._M / (self.n - 1)

    @property
    def invsqrtSigmahat(self) -> np.ndarray:
        r"""
        Compute the inverse square root of the covariance matrix.

        Returns:

        $$\Sigma^{-1/2}$$

        Computed via Cholesky decomposition of the inverse covariance matrix.

        Returns
        -------
        np.ndarray
            Inverse square root of covariance matrix, shape (n_dim, n_dim).
            Returns identity matrix if covariance is not positive definite.
        """
        try:
            sqrtinvSigma = cholesky(inv(self.Sigmahat, check_finite=False), check_finite=False)
        except np.linalg.LinAlgError:
            sqrtinvSigma = np.eye(self.n_dim, dtype=np.float64)
        return sqrtinvSigma

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply multivariate normalization (decorrelation and standardization).

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) to normalize.

        Returns
        -------
        np.ndarray
            Decorrelated and standardized array of shape (n_dim,).
        """
        return (self.invsqrtSigmahat @ (x - self.muhat)).reshape(-1)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the normalizer state to a dictionary.

        Returns
        -------
        dict
            Dictionary with JSON-serializable metadata and base64-encoded arrays.
        """
        return {
            "version": "1.0",
            "class": "MultivariateNormalizer",
            "config": {"n_dim": self.n_dim},
            "state": {
                "n": self.n,
                "muhat": base64.b64encode(self.muhat.tobytes()).decode("ascii"),
                "_M": base64.b64encode(self._M.tobytes()).decode("ascii"),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultivariateNormalizer":
        """
        Deserialize a normalizer from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary created by to_dict().

        Returns
        -------
        MultivariateNormalizer
            Deserialized normalizer instance.
        """
        if data.get("class") != "MultivariateNormalizer":
            raise ValueError(f"Cannot deserialize {data.get('class')} as MultivariateNormalizer")

        config = data["config"]
        instance = cls(n_dim=config["n_dim"])

        state = data["state"]
        instance.n = state["n"]
        instance.muhat = np.frombuffer(base64.b64decode(state["muhat"]), dtype=np.float64)
        instance._M = np.frombuffer(base64.b64decode(state["_M"]), dtype=np.float64).reshape(
            (config["n_dim"], config["n_dim"])
        )

        return instance

    def to_json(self) -> str:
        """Serialize the normalizer to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MultivariateNormalizer":
        """Deserialize a normalizer from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def reset(self) -> None:
        """
        Reset the normalizer to initial state.

        Reinitializes observation count, mean, and covariance accumulator (_M)
        to zeros.
        """
        self.n = 0
        self.muhat = np.zeros(self.n_dim, dtype=np.float64)
        self._M = np.zeros((self.n_dim, self.n_dim), dtype=np.float64)
