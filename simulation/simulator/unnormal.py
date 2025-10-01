import numpy as np
from onorm import Normalizer


class Unnormalizer(Normalizer):
    def partial_fit(self, x: np.ndarray) -> None:
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def reset(self):
        pass
