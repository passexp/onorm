import json
from typing import Any, Dict, List, Tuple

import numpy as np
from fastdigest import TDigest

from .normalization_base import Normalizer


class Winsorizer(Normalizer):
    r"""
    Online winsorizer for robust outlier clipping using TDigest quantiles.

    Clips extreme values to specified quantiles, replacing outliers with the
    values at the quantile boundaries. Uses TDigest for efficient online
    quantile estimation without storing all historical data.

    For each feature $i$, the transformation is:

    $$x_{\text{clip},i} = \begin{cases}
    Q_{\text{lower},i} & \text{if } x_i < Q_{\text{lower},i} \\
    x_i & \text{if } Q_{\text{lower},i} \leq x_i \leq Q_{\text{upper},i} \\
    Q_{\text{upper},i} & \text{if } x_i > Q_{\text{upper},i}
    \end{cases}$$

    where $Q_{\text{lower},i}$ and $Q_{\text{upper},i}$ are the estimated quantiles.

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
    ```{python}
    from onorm import Winsorizer
    import numpy as np
    winsorizer = Winsorizer(n_dim=3, clip_q=(0.1, 0.9))
    X = np.random.normal(size=(100, 3))
    for x in X:
        winsorizer.partial_fit(x)
    x_new = np.array([10.0, 10.0, 10.0])  # Outlier
    x_clipped = winsorizer.transform(x_new.copy())  # Clips to 90th quantile
    ```

    References
    ----------
    [Computing Extremely Accurate Quantiles Using t-Digests](https://arxiv.org/abs/1902.04023)

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the winsorizer state to a dictionary.

        Returns
        -------
        dict
            Dictionary with JSON-serializable metadata and TDigest states.

        Notes
        -----
        TDigest objects are serialized using their native to_dict() method,
        which returns a JSON-serializable dictionary containing centroids,
        min/max values, and max_centroids configuration.
        """
        return {
            "version": "1.0",
            "class": "Winsorizer",
            "config": {
                "n_dim": self.n_dim,
                "clip_q": list(self.clip_q),
                "max_centroids": self.max_centroids,
            },
            "state": {"digests": [digest.to_dict() for digest in self.digests]},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Winsorizer":
        """
        Deserialize a winsorizer from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary created by to_dict().

        Returns
        -------
        Winsorizer
            Deserialized winsorizer instance.
        """
        if data.get("class") != "Winsorizer":
            raise ValueError(f"Cannot deserialize {data.get('class')} as Winsorizer")

        config = data["config"]
        instance = cls(
            n_dim=config["n_dim"],
            clip_q=tuple(config["clip_q"]),
            max_centroids=config["max_centroids"],
        )

        state = data["state"]
        instance.digests = [TDigest.from_dict(digest_dict) for digest_dict in state["digests"]]

        return instance

    def to_json(self) -> str:
        """Serialize the winsorizer to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Winsorizer":
        """Deserialize a winsorizer from a JSON string."""
        return cls.from_dict(json.loads(json_str))
