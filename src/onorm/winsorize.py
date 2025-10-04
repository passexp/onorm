from typing import List, Tuple

import numpy as np
from fastdigest import TDigest

from .normalization_base import Normalizer


class Winsorizer(Normalizer):
    """
    Online winsorization normalizer using TDigest for quantile estimation.

    Clips extreme values to specified quantiles, replacing outliers with the
    values at the quantile boundaries. Uses TDigest for efficient online
    quantile estimation without storing all historical data.

    Parameters
    ----------
    n_dim : int
        Number of dimensions/features to normalize
    clip_q : tuple of float, default=(0, 1)
        Lower and upper quantiles for clipping, in range [0, 1].
        For example, (0.1, 0.9) clips values below the 10th quantile
        and above the 90th quantile.
    max_centroids : int, default=1000
        Maximum number of centroids for TDigest. Higher values increase precision
        but use more memory.

    Attributes
    ----------
    digests : List[TDigest]
        List of TDigest objects for tracking quantiles per feature.

    Examples
    --------
    >>> from onorm import Winsorizer
    >>> winsorizer = Winsorizer(n_dim=3, clip_q=(0.1, 0.9))
    >>> import numpy as np
    >>> X = np.random.normal(size=(100, 3))
    >>> for x in X:
    ...     winsorizer.partial_fit(x)
    >>> x_new = np.array([10.0, 10.0, 10.0])  # Outlier
    >>> x_clipped = winsorizer.transform(x_new.copy())  # Clips to 90th quantile

    Notes
    -----
    - Winsorization is robust to outliers, unlike min-max scaling
    - TDigest provides approximate quantiles with bounded memory
    - Clipping is applied independently to each feature
    """

    def __init__(
        self, n_dim: int, clip_q: Tuple[float, float] = (0, 1), max_centroids: int = 1000
    ) -> None:
        self.clip_q = clip_q
        self.n_dim = n_dim
        self.max_centroids = max_centroids
        self.reset()

    def partial_fit(self, x: np.ndarray) -> None:
        """
        Update quantile estimates for each feature.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) representing a new observation.
        """
        for i, xi in enumerate(x):
            self.digests[i].update(xi.item())

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Clip extreme values to learned quantile boundaries.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) to clip.

        Returns
        -------
        np.ndarray
            Clipped array where values below the lower quantile are set to the
            lower quantile value, and values above the upper quantile are set to
            the upper quantile value.
        """
        for i in range(self.n_dim):
            x[i] = np.clip(
                x[i],
                self.digests[i].quantile(self.clip_q[0]),
                self.digests[i].quantile(self.clip_q[1]),
            )
        return x

    def reset(self) -> None:
        """
        Reset the winsorizer to initial state.

        Reinitializes TDigest objects for all features, clearing quantile estimates.
        """
        self.digests: List[TDigest] = [
            TDigest(max_centroids=self.max_centroids) for _ in range(self.n_dim)
        ]
