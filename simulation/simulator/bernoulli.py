from abc import ABCMeta

import numpy as np


class Bernoulli(metaclass=ABCMeta):
    def __init__(self, q=0.5, **kwargs):
        self.q = q

    def assign_next(self, x):
        return np.random.binomial(n=1, p=self.q, size=1).item()

    def assign_all(self, X):
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])

    def reset(self):
        pass
