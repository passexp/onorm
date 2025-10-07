import json
from typing import Any, Dict, List

import numpy as np

from .normalization_base import Normalizer


class Pipeline(Normalizer):
    """
    Pipeline for chaining multiple normalizers sequentially.

    Applies a sequence of normalizers in order, where each normalizer's output
    becomes the input to the next normalizer. This allows combining different
    normalization strategies (e.g., winsorization followed by standardization).

    During partial_fit, each normalizer is fitted with the transformed output
    from the previous normalizer, ensuring that each stage learns from the
    distribution it will actually see during inference.

    Parameters
    ----------
    normalizers : List[Normalizer]
        List of normalizer instances to apply in sequence. Can be empty.

    Attributes
    ----------
    normalizers : List[Normalizer]
        The list of normalizers in the pipeline.

    Examples
    --------
    ```{python}
    from onorm import Pipeline, Winsorizer, StandardScaler, MinMaxScaler
    import numpy as np
    # Create a pipeline: clip outliers, then standardize, then scale to [0,1]
    pipeline = Pipeline(
        [
            Winsorizer(n_dim=3, clip_q=(0.05, 0.95)),
            StandardScaler(n_dim=3),
            MinMaxScaler(n_dim=3),
        ]
    )
    X = np.random.normal(size=(100, 3))
    for x in X:
        pipeline.partial_fit(x)
    x_new = np.array([2.0, -1.0, 0.5])
    x_normalized = pipeline.transform(x_new.copy())

    # Empty pipeline (identity transformation)
    identity_pipeline = Pipeline([])
    ```

    Notes
    -----
    - Order matters: Pipeline([A, B]) produces different results than Pipeline([B, A])
    - Each normalizer is fitted with the transformed output from the previous stage
    - Both fitting and transformation are applied sequentially through the chain
    - Empty pipeline returns input unchanged
    """

    def __init__(self, normalizers: List[Normalizer]) -> None:
        self.normalizers = normalizers

    def partial_fit(self, x: np.ndarray) -> None:
        """
        Update all normalizers sequentially with transformed observations.

        Each normalizer is fitted with the transformed output from the previous
        normalizer in the pipeline. This ensures each stage learns from the
        distribution it will encounter during transformation.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array representing a new observation. The shape should match
            the n_dim parameter used by the normalizers.

        Notes
        -----
        The data is transformed sequentially: normalizer 1 fits on x, normalizer 2
        fits on transform_1(x), normalizer 3 fits on transform_2(transform_1(x)),
        and so on.
        """
        x_transformed = x.copy()
        for normalizer in self.normalizers:
            normalizer.partial_fit(x_transformed)
            x_transformed = normalizer.transform(x_transformed)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply all normalizers sequentially to transform the data.

        The output of each normalizer becomes the input to the next normalizer
        in the pipeline.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array to normalize.

        Returns
        -------
        np.ndarray
            The transformed array after applying all normalizers in sequence.
            If the pipeline is empty, returns the input unchanged.

        Notes
        -----
        This method modifies the input array in-place for efficiency. Pass a
        copy if you need to preserve the original: transform(x.copy()).
        """
        for normalizer in self.normalizers:
            x = normalizer.transform(x)
        return x

    def reset(self) -> None:
        """
        Reset all normalizers in the pipeline to their initial state.

        Calls reset() on each normalizer, clearing all learned statistics.
        """
        for normalizer in self.normalizers:
            normalizer.reset()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the pipeline to a dictionary.

        Returns
        -------
        dict
            Dictionary with recursively serialized normalizers.

        Notes
        -----
        Each normalizer in the pipeline is serialized using its own to_dict() method.
        """
        return {
            "version": "1.0",
            "class": "Pipeline",
            "config": {},
            "state": {"normalizers": [norm.to_dict() for norm in self.normalizers]},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """
        Deserialize a pipeline from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary created by to_dict().

        Returns
        -------
        Pipeline
            Deserialized pipeline instance.

        Notes
        -----
        Automatically detects the class of each normalizer and deserializes accordingly.
        """
        if data.get("class") != "Pipeline":
            raise ValueError(f"Cannot deserialize {data.get('class')} as Pipeline")

        # Import normalizer classes (avoid circular imports)
        from . import (
            MinMaxScaler,
            MultivariateNormalizer,
            QuantileTransformer,
            StandardScaler,
            Winsorizer,
        )

        class_map = {
            "MinMaxScaler": MinMaxScaler,
            "StandardScaler": StandardScaler,
            "MultivariateNormalizer": MultivariateNormalizer,
            "Winsorizer": Winsorizer,
            "QuantileTransformer": QuantileTransformer,
            "Pipeline": cls,
        }

        state = data["state"]
        normalizers = []
        for norm_data in state["normalizers"]:
            norm_class = class_map.get(norm_data["class"])
            if norm_class is None:
                raise ValueError(f"Unknown normalizer class: {norm_data['class']}")
            normalizers.append(norm_class.from_dict(norm_data))

        return cls(normalizers)

    def to_json(self) -> str:
        """Serialize the pipeline to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Pipeline":
        """Deserialize a pipeline from a JSON string."""
        return cls.from_dict(json.loads(json_str))
