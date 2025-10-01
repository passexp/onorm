import numpy as np
from tdigest import TDigest

from .normalization_base import Normalizer


class Winsorizer(Normalizer):
    def __init__(self, n_dim: int, clip_q=(0, 1), tdigest_delta = 0.01):
        self.clip_q = clip_q
        self.n_dim = n_dim
        self.delta = tdigest_delta
        self.reset()

    def partial_fit(self, x):
        for i, xi in enumerate(x):
            self.digests[i].update(xi)

    def transform(self, x):
        for i in range(self.n_dim):
            x[i] = np.clip(
                x,
                self.digests[i].percentile(self.clip_q[0]),
                self.digests[i].percentile(self.clip_q[1]),
            )
        return x

    def reset(self):
        self.digests = [TDigest(q = self.delta) for _ in range(self.n_dim)]
