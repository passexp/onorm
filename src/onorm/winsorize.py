from typing import List, Tuple

import numpy as np
from tdigest import TDigest

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
        For example, (0.1, 0.9) clips values below the 10th percentile
        and above the 90th percentile.
    tdigest_delta : float, default=0.01
        Compression parameter for TDigest. Smaller values increase precision
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
    >>> x_clipped = winsorizer.transform(x_new.copy())  # Clips to 90th percentile

    Notes
    -----
    - Winsorization is robust to outliers, unlike min-max scaling
    - TDigest provides approximate quantiles with bounded memory
    - Clipping is applied independently to each feature
    """

    def __init__(
        self, n_dim: int, clip_q: Tuple[float, float] = (0, 1), tdigest_delta: float = 0.01
    ) -> None:
        self.clip_q = clip_q
        self.n_dim = n_dim
        self.delta = tdigest_delta
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
            self.digests[i].update(xi)

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
                self.digests[i].percentile(self.clip_q[0] * 100),
                self.digests[i].percentile(self.clip_q[1] * 100),
            )
        return x

    def reset(self) -> None:
        """
        Reset the winsorizer to initial state.

        Reinitializes TDigest objects for all features, clearing quantile estimates.
        """
        self.digests: List[TDigest] = [TDigest(delta=self.delta) for _ in range(self.n_dim)]
