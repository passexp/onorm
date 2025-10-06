from typing import Any, Dict

import numpy as np
import pytest
from onorm import Normalizer


def test_instantiate():
    """Test that abstract Normalizer class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Normalizer()


def test_normalizer_init_called():
    """Test that Normalizer.__init__ is called when creating concrete instances."""

    class ConcreteNormalizer(Normalizer):
        """Minimal concrete implementation for testing."""

        def __init__(self, n_dim):
            super().__init__()  # Explicitly call base class __init__
            self.n_dim = n_dim

        def partial_fit(self, x: np.ndarray) -> None:
            pass

        def transform(self, x: np.ndarray) -> np.ndarray:
            return x

        def reset(self) -> None:
            pass

        def to_dict(self) -> Dict[str, Any]:
            return {
                "version": "1.0",
                "class": "ConcreteNormalizer",
                "config": {"n_dim": self.n_dim},
                "state": {},
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "ConcreteNormalizer":
            return cls(n_dim=data["config"]["n_dim"])

    # Create instance - this should call Normalizer.__init__
    normalizer = ConcreteNormalizer(n_dim=3)
    assert normalizer.n_dim == 3
