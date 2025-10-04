import numpy as np

from .normalization_base import Normalizer


class MinMaxScaler(Normalizer):
    r"""
    Online min-max scaler for feature normalization to [0, 1] range.

    Tracks the running minimum and maximum for each feature and scales values
    to the range [0, 1] based on these statistics. The normalization is updated
    incrementally as new observations arrive.

    For each feature $i$ at time $t$, tracks:

    $$\text{min}_i = \min\{x_{1,i}, \ldots, x_{t,i}\}$$
    $$\text{max}_i = \max\{x_{1,i}, \ldots, x_{t,i}\}$$

    And transforms values as:

    $$x_{\text{norm},i} = \frac{x_i - \text{min}_i}{\text{max}_i - \text{min}_i}$$

    Parameters
    ----------
    n_dim : int
        Number of dimensions/features to normalize.

    Attributes
    ----------
    min : np.ndarray
        Running minimum for each feature, shape (n_dim,).
    max : np.ndarray
        Running maximum for each feature, shape (n_dim,).

    Examples
    --------
    ```{python}
    from onorm import MinMaxScaler
    import numpy as np
    scaler = MinMaxScaler(n_dim=3)
    X = np.random.uniform(-5, 5, size=(100, 3))
    for x in X:
        scaler.partial_fit(x)
    x_new = np.array([2.0, -1.0, 3.0])
    x_normalized = scaler.transform(x_new.copy())
    assert np.all((x_normalized >= 0) & (x_normalized <= 1))
    ```

    Notes
    -----
    - If a feature has constant values (min == max), the transformed value
      will be 0 to avoid division by zero.
    - This scaler is sensitive to outliers since min/max can be heavily
      influenced by extreme values.
    """

    def __init__(self, n_dim: int) -> None:
        self.n_dim = n_dim
        self.reset()

    def _update_min(self, x: np.ndarray) -> None:
        """Update running minimum for each feature."""
        self.min = np.fmin(self.min, x)

    def _update_max(self, x: np.ndarray) -> None:
        """Update running maximum for each feature."""
        self.max = np.fmax(self.max, x)

    def partial_fit(self, x: np.ndarray) -> None:
        """
        Update the minimum and maximum for each feature.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) representing a new observation.
        """
        self._update_min(x)
        self._update_max(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform features to [0, 1] range using current min/max statistics.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of shape (n_dim,) to normalize.

        Returns
        -------
        np.ndarray
            Normalized array of shape (n_dim,) with values in [0, 1].

        Notes
        -----
        If min == max for a feature (constant feature), returns 0 for that
        feature to avoid division by zero.
        """
        denom = self.max - self.min
        if np.linalg.norm(denom) <= np.finfo(np.float64).eps:
            denom = 1
        return (x - self.min) / denom

    def reset(self) -> None:
        """
        Reset the scaler to initial state.

        Reinitializes min to positive infinity and max to negative infinity
        so that the first observation will set both values.
        """
        self.min = np.array([np.inf] * self.n_dim)
        self.max = np.array([-np.inf] * self.n_dim)
