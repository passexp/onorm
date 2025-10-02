import numpy as np
from scipy.linalg import cholesky

from .normalization_base import Normalizer


def downdate_cholesky(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Perform Cholesky downdate (rank-one downdate of Cholesky factorization).

    Given a Cholesky factor R such that R^T R = A, computes the updated
    Cholesky factor R' such that R'^T R' = A - xx^T.

    Parameters
    ----------
    R : np.ndarray
        Upper triangular Cholesky factor, shape (p, p).
    x : np.ndarray
        Vector for rank-one downdate, shape (p,).

    Returns
    -------
    np.ndarray
        Updated Cholesky factor R', shape (p, p).

    Notes
    -----
    This function modifies R in-place and also modifies x during computation.
    The algorithm uses Givens rotations to maintain the triangular structure.

    References
    ----------
    Gill, Golub, Murray, and Saunders (1974). Methods for modifying matrix
    factorizations. Mathematics of Computation, 28(126), 505-535.
    """
    p = np.size(x)
    for k in range(p):
        r = np.sqrt(np.power(R[k, k], 2) - np.power(x[k], 2))
        c = r / R[k, k].item()
        s = (x[k] / R[k, k]).item()
        R[k, k] = r
        for i in range(k + 1, p):
            R[k, i] = (R[k, i] - s * x[i]) / c
            x[i] = c * x[i] - s * R[k, i]
    return R


class MultivariateNormalizer(Normalizer):
    """
    Online multivariate normalization using inverse covariance matrix estimation.

    This normalizer transforms multivariate data to have zero mean, unit variance,
    and zero correlation (decorrelation). It uses online estimation of the inverse
    covariance matrix via rank-one updates to Cholesky decomposition, achieving
    O(d²) time complexity per observation.

    The transformation is: Σ^(-1/2) (x - μ)
    where μ is the mean and Σ is the covariance matrix.

    Parameters
    ----------
    n_dim : int
        Number of dimensions/features in the data.
    warmup_period : int
        Number of initial observations before switching to efficient Cholesky
        updates. During warmup, full Cholesky decomposition is computed.

    Attributes
    ----------
    n : int
        Number of observations seen so far.
    muhat : np.ndarray
        Estimated mean vector, shape (n_dim,).
    invSigmahat : np.ndarray
        Estimated inverse covariance matrix, shape (n_dim, n_dim).
    invsqrtSigmahat : np.ndarray
        Cholesky factor of inverse covariance (Σ^(-1/2)), shape (n_dim, n_dim).

    Examples
    --------
    >>> from onorm import MultivariateNormalizer
    >>> import numpy as np
    >>> normalizer = MultivariateNormalizer(n_dim=3, warmup_period=10)
    >>> # Generate correlated data
    >>> cov = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    >>> X = np.random.multivariate_normal([0, 0, 0], cov, size=100)
    >>> for x in X:
    ...     normalizer.partial_fit(x)
    >>> x_new = np.array([1.0, 1.0, 1.0])
    >>> x_normalized = normalizer.transform(x_new.copy())
    >>> # x_normalized will be decorrelated

    Notes
    -----
    - Time complexity: O(d²) per observation for both fitting and transformation
    - Space complexity: O(d²) for storing covariance matrix
    - WARNING: Current implementation has a critical bug with invSigmahat
      initialization using max float values, causing numerical overflow. This
      class may not function correctly until fixed.

    References
    ----------
    Rank-one updates to Cholesky decomposition:
    https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
    """

    def __init__(self, n_dim: int, warmup_period: int) -> None:
        self.n_dim = n_dim
        self.warmup_period = warmup_period
        self.reset()

    def partial_fit(self, x: np.ndarray) -> None:
        """
        Update mean and inverse covariance estimates with a new observation.

        Uses rank-one update formulas to incrementally update the inverse
        covariance matrix and its Cholesky decomposition.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) representing a new observation.

        References
        ----------
        https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
        """
        delta = x - self.muhat
        self.n += 1
        self._update_muhat(delta)
        self._update_invSigmahat(delta)
        self._update_invsqrtSigmahat(delta)

    def _update_muhat(self, delta: np.ndarray) -> None:
        """Update running mean estimate."""
        self.muhat += delta / self.n

    def _update_invSigmahat(self, delta: np.ndarray) -> None:
        """Update inverse covariance matrix using Sherman-Morrison formula."""
        frac = (self.n - 1) / self.n
        Mnum = frac * (self.invSigmahat @ delta) @ (delta @ self.invSigmahat)
        Mden = 1.0 + frac * delta @ self.invSigmahat @ delta
        self.invSigmahat = self.invSigmahat - Mnum / Mden.item()

    def _update_invsqrtSigmahat(self, delta: np.ndarray) -> None:
        """Update Cholesky factor of inverse covariance."""
        if self.n < self.warmup_period:
            cholesky(self.invSigmahat, check_finite=False)
        else:
            frac = np.sqrt((self.n - 1) / self.n)
            v = self.invsqrtSigmahat @ delta
            norm = frac / np.sqrt(1 + (v.T @ v).item())
            self.invsqrtSigmahat = downdate_cholesky(
                self.invsqrtSigmahat, (norm * self.invsqrtSigmahat @ v).reshape(-1)
            )

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

    def reset(self) -> None:
        """
        Reset the normalizer to initial state.

        Reinitializes observation count, mean, inverse covariance, and its
        Cholesky factor.

        Notes
        -----
        WARNING: The invSigmahat initialization uses max float values which
        can cause numerical overflow. This is a known bug.
        """
        self.n = 0
        self.muhat = np.array([0.0] * self.n_dim)
        self.invSigmahat = np.eye(self.n_dim, dtype=np.float64) * np.finfo(np.float64).max
        self.invsqrtSigmahat = np.eye(self.n_dim, dtype=np.float64)
