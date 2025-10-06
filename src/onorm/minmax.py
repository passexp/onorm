import base64
import json
from typing import Any, Dict

import numpy as np

from .normalization_base import Normalizer


class MinMaxScaler(Normalizer):
    r"""
    Online min-max scaler for feature normalization to [0, 1] range.

    Tracks the running minimum and maximum for each feature and scales values
    to the range [0, 1] based on these statistics. The normalization is updated
    incrementally as new observations arrive.

    For each feature $i$ at time $t$, tracks:

    $$\begin{aligned}\text{min}_i &= \min\{x_{1,i}, \ldots, x_{t,i}\}\\
    \text{max}_i = \max\{x_{1,i}, \ldots, x_{t,i}\}\end{aligned}$$

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the scaler state to a dictionary.

        Returns a dictionary with JSON-serializable metadata and base64-encoded
        numpy arrays for efficient storage and database compatibility.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'version': str, serialization format version
            - 'class': str, class name
            - 'config': dict, configuration parameters
            - 'state': dict, serialized state arrays (base64-encoded)

        Examples
        --------
        ```{python}
        from onorm import MinMaxScaler
        import numpy as np

        scaler = MinMaxScaler(n_dim=3)
        X = np.random.uniform(-5, 5, size=(100, 3))
        for x in X:
            scaler.partial_fit(x)

        # Serialize
        data = scaler.to_dict()

        # Could save to database
        # db.execute("INSERT INTO models (config, state) VALUES (%s, %s)",
        #            (json.dumps(data['config']), data['state']))
        ```
        """
        return {
            "version": "1.0",
            "class": "MinMaxScaler",
            "config": {"n_dim": self.n_dim},
            "state": {
                "min": base64.b64encode(self.min.tobytes()).decode("ascii"),
                "max": base64.b64encode(self.max.tobytes()).decode("ascii"),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinMaxScaler":
        """
        Deserialize a scaler from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary created by to_dict() containing:
            - 'version': serialization format version
            - 'class': class name (must be 'MinMaxScaler')
            - 'config': configuration parameters
            - 'state': serialized state arrays

        Returns
        -------
        MinMaxScaler
            Deserialized scaler instance with restored state.

        Raises
        ------
        ValueError
            If the data format is invalid or class name doesn't match.

        Examples
        --------
        ```{python}
        from onorm import MinMaxScaler

        # Deserialize from saved data
        data = {
            "version": "1.0",
            "class": "MinMaxScaler",
            "config": {"n_dim": 3},
            "state": {"min": "...", "max": "..."}
        }
        scaler = MinMaxScaler.from_dict(data)
        ```
        """
        if data.get("class") != "MinMaxScaler":
            raise ValueError(f"Cannot deserialize {data.get('class')} as MinMaxScaler")

        # Create instance with config
        config = data["config"]
        instance = cls(n_dim=config["n_dim"])

        # Restore state arrays
        state = data["state"]
        instance.min = np.frombuffer(base64.b64decode(state["min"]), dtype=np.float64)
        instance.max = np.frombuffer(base64.b64decode(state["max"]), dtype=np.float64)

        return instance

    def to_json(self) -> str:
        """
        Serialize the scaler to a JSON string.

        Returns
        -------
        str
            JSON string representation of the scaler state.

        Examples
        --------
        ```{python}
        from onorm import MinMaxScaler

        scaler = MinMaxScaler(n_dim=3)
        # ... train scaler ...
        json_str = scaler.to_json()
        ```
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MinMaxScaler":
        """
        Deserialize a scaler from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON string created by to_json().

        Returns
        -------
        MinMaxScaler
            Deserialized scaler instance.

        Examples
        --------
        ```{python}
        from onorm import MinMaxScaler

        # Deserialize from JSON string
        scaler = MinMaxScaler.from_json(json_str)
        ```
        """
        return cls.from_dict(json.loads(json_str))
