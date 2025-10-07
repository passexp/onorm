"""
Quantile-based normalization using TDigest for online CDF estimation.
"""

import json
from typing import Any, Dict, List

import numpy as np
from fastdigest import TDigest

from .normalization_base import Normalizer


class QuantileTransformer(Normalizer):
    r"""
    Online quantile-based normalization using marginal CDF estimation.

    Transforms features to follow a uniform distribution on [0, 1] by mapping
    each value to its empirical cumulative distribution function (CDF) value.
    Uses TDigest for efficient online quantile estimation without storing all
    historical data.

    For each feature $i$, the transformation is:

    $$x_{\text{norm},i} = F_i(x_i)$$

    where $F_i$ is the estimated cumulative distribution function for feature $i$.

    Parameters
    ----------
    n_dim : int
        Number of dimensions/features to normalize
    max_centroids : int, default=1000
        Maximum number of centroids for TDigest. Higher values increase precision
        but use more memory. Typically 100-1000 is sufficient.
    output_distribution : str, default='uniform'
        Target output distribution:
        - 'uniform': Output in [0, 1] (raw CDF values)
        - 'normal': Apply inverse normal CDF to get standard normal output

    Attributes
    ----------
    digests : List[TDigest]
        List of TDigest objects for tracking marginal distributions per feature.

    Examples
    --------
    ```{python}
    from onorm import QuantileTransformer
    import numpy as np

    # Create transformer
    qt = QuantileTransformer(n_dim=3)

    # Fit on skewed data
    X = np.random.exponential(scale=2.0, size=(1000, 3))
    for x in X:
        qt.partial_fit(x)

    # Transform maps to uniform [0, 1]
    x_new = np.array([0.5, 1.0, 5.0])
    x_uniform = qt.transform(x_new.copy())  # Values close to 0, 0.4, 0.9
    ```

    Notes
    -----
    - Transforms arbitrary distributions to uniform [0, 1]
    - Robust to outliers and heavy-tailed distributions
    - TDigest provides approximate quantiles with bounded memory
    - Transformation is monotonic within each feature
    - Features are transformed independently (marginal CDFs)
    - Values outside the observed range are clipped to [0, 1]

    References
    ----------
    [Computing Extremely Accurate Quantiles Using t-Digests](https://arxiv.org/abs/1902.04023)

    See Also
    --------
    Winsorizer : For robust outlier clipping at quantiles
    StandardScaler : For Gaussian-based normalization
    """

    def __init__(
        self,
        n_dim: int,
        max_centroids: int = 1000,
        output_distribution: str = "uniform",
    ) -> None:
        if output_distribution not in ("uniform", "normal"):
            raise ValueError(
                f"output_distribution must be 'uniform' or 'normal', got '{output_distribution}'"
            )

        self.n_dim = n_dim
        self.max_centroids = max_centroids
        self.output_distribution = output_distribution
        self.reset()

    def partial_fit(self, x: np.ndarray) -> None:
        """
        Update CDF estimates for each feature.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) representing a new observation.
        """
        for i, xi in enumerate(x):
            self.digests[i].update(xi.item())

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform data to target distribution using estimated CDFs.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) to transform.

        Returns
        -------
        np.ndarray
            Transformed array where each value is mapped to its CDF value.
            If output_distribution='uniform': values in [0, 1]
            If output_distribution='normal': values are standard normal

        Notes
        -----
        Values smaller than all observed data receive CDF ≈ 0.
        Values larger than all observed data receive CDF ≈ 1.
        """
        for i in range(self.n_dim):
            # Get CDF value for this observation
            cdf_value = self.digests[i].cdf(x[i].item())

            # Clip to [0, 1] in case of numerical issues
            cdf_value = np.clip(cdf_value, 0.0, 1.0)

            if self.output_distribution == "uniform":
                x[i] = cdf_value
            else:  # normal
                # Apply inverse normal CDF (probit function)
                # Handle edge cases to avoid inf
                if cdf_value <= 0.0:
                    x[i] = -8.0  # Roughly norm.ppf(1e-15)
                elif cdf_value >= 1.0:
                    x[i] = 8.0  # Roughly norm.ppf(1 - 1e-15)
                else:
                    # Clip to safe range for scipy
                    from scipy.stats import norm

                    cdf_value = np.clip(cdf_value, 1e-15, 1 - 1e-15)
                    x[i] = norm.ppf(cdf_value)

        return x

    def reset(self) -> None:
        """
        Reset the transformer to initial state.

        Reinitializes TDigest objects for all features, clearing CDF estimates.
        """
        self.digests: List[TDigest] = [
            TDigest(max_centroids=self.max_centroids) for _ in range(self.n_dim)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the transformer state to a dictionary.

        Returns
        -------
        dict
            Dictionary with JSON-serializable metadata and TDigest states.

        Notes
        -----
        Uses TDigest's native to_dict() method - fully JSON-serializable.
        """
        return {
            "version": "1.0",
            "class": "QuantileTransformer",
            "config": {
                "n_dim": self.n_dim,
                "max_centroids": self.max_centroids,
                "output_distribution": self.output_distribution,
            },
            "state": {"digests": [digest.to_dict() for digest in self.digests]},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantileTransformer":
        """
        Deserialize a transformer from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary created by to_dict().

        Returns
        -------
        QuantileTransformer
            Deserialized transformer instance.
        """
        if data.get("class") != "QuantileTransformer":
            raise ValueError(f"Cannot deserialize {data.get('class')} as QuantileTransformer")

        config = data["config"]
        instance = cls(
            n_dim=config["n_dim"],
            max_centroids=config["max_centroids"],
            output_distribution=config["output_distribution"],
        )

        state = data["state"]
        instance.digests = [TDigest.from_dict(digest_dict) for digest_dict in state["digests"]]

        return instance

    def to_json(self) -> str:
        """Serialize the transformer to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "QuantileTransformer":
        """Deserialize a transformer from a JSON string."""
        return cls.from_dict(json.loads(json_str))
