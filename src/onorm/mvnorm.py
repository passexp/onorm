import numpy as np
from scipy.linalg import cholesky

from .normalization_base import Normalizer


def downdate_cholesky(R, x):
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
    """Multivariate normalization

    The class performs normalization so that elements of the resulting matrix have zero
    correlation, zero mean and standard deviation of one.

    Define the multivariate mean and covariance to be $\\mu$ and $\\Sigma$.

    If we knew both of these quantities, then normalization would be:

    $$\\Sigma^{-\\frac{1}{2}}(x_t - \\mu)$$

    Instead, we make (online) estimates of these quantities ($\\Sigma^{-\\frac{1}{2}}$ and $\\mu$)
    and normalize using those estimates.

    In particular, these online estimates rely on the ability to perform rank one updates to a
    Cholesky decomposition, which is an $O(d^2)$ operation. Application of normalization can
    also be accomplished in $O(d^2)$. Without sparsity, this matches the optimal time complexity.
    """

    def __init__(self, n_dim: int, warmup_period: int) -> None:
        self.n_dim = n_dim
        self.warmup_period = warmup_period
        self.reset()

    def partial_fit(self, x: np.ndarray) -> None:
        """Update the estimates for the normalization model.

        Updates the estimates of the mean, inverse covariance and Cholesky decomposition of the
        inverse covariance.

        https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update

        Args:
            x: A 1d array representing a new observation.
        """
        delta = x - self.muhat
        self.n += 1
        self._update_muhat(delta)
        self._update_invSigmahat(delta)
        self._update_invsqrtSigmahat(delta)

    def _update_muhat(self, delta):
        self.muhat += delta / self.n

    def _update_invSigmahat(self, delta):
        frac = (self.n - 1) / self.n
        Mnum = frac * (self.invSigmahat @ delta) @ (delta @ self.invSigmahat)
        Mden = 1.0 + frac * delta @ self.invSigmahat @ delta
        self.invSigmahat = self.invSigmahat - Mnum / Mden.item()

    def _update_invsqrtSigmahat(self, delta):
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
        """Transform the feature vector according to the current state

        If the current minimum and maximum are equal, then the transformation
        returns $x_{ti} - \\textrm{mn}_{ti}$.

        Args:
            x: A 1d array representing an observation to normalize.
        """
        return (self.invsqrtSigmahat @ (x - self.muhat)).reshape(-1)

    def reset(self):
        self.muhat = np.array([0.0] * self.n_dim)
        self.invSigmahat = np.eye(self.n_dim, dtype=np.float64) * np.finfo(np.float64).max
        self.invsqrtSigmahat = np.eye(self.n_dim, dtype=np.float64)
