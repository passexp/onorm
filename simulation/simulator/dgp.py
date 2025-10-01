from abc import ABCMeta, abstractmethod

import numpy as np


class XDGP(metaclass=ABCMeta):
    @abstractmethod
    def setup(self, n):
        pass

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        pass


class YDGP(metaclass=ABCMeta):
    @abstractmethod
    def setup(self, X):
        self.d = X.shape[1]
        self.n = X.shape[0]

    @abstractmethod
    def Y(self, A) -> np.ndarray:
        pass

    @abstractmethod
    def Yi(self, idx, a):
        pass

    @property
    def ITE(self):
        return self.Y(np.array([1] * self.n)) - self.Y(np.array([0] * self.n))

    @property
    def ATE(self):
        return np.average(self.ITE)


class Normal(XDGP):
    def __init__(self, d=4):
        self.d = d

    def setup(self, n):
        self.n = n
        self._X = np.random.normal(size=(self.n, self.d))

    @property
    def X(self):
        return self._X


class NoisedNormal(XDGP):
    def __init__(self, d=4, frac_cauchy=0.05):
        self.d = d
        self.frac_cauchy = frac_cauchy

    def setup(self, n):
        self.n = n
        n_normal = np.floor((1 - self.frac_cauchy) * self.n).astype(int)
        n_cauchy = np.floor(self.frac_cauchy * self.n).astype(int)
        n_extra = self.n - n_normal - n_cauchy
        if n_extra > 0:
            extra_cauchy = np.random.binomial(n_extra, self.frac_cauchy, size=1)
            n_cauchy += extra_cauchy
            n_normal += n_extra - extra_cauchy
        self._X = np.concatenate(
            (
                np.random.normal(size=(n_normal, self.d)),
                np.random.standard_cauchy(size=(n_cauchy, self.d)),
            )
        )

    @property
    def X(self):
        return self._X


class Uniform(XDGP):
    def __init__(self, d=4):
        self.d = d

    def setup(self, n):
        self.n = n
        self._X = np.random.uniform(size=(self.n, self.d))

    @property
    def X(self):
        return self._X


class Linear(YDGP):
    def setup(self, X):
        self.d = X.shape[1]
        self.n = X.shape[0]
        beta0 = np.random.normal(size=self.d)
        eps0 = np.random.normal(size=self.n)
        self.mu0 = (X @ beta0).flatten()
        self.Y0 = (self.mu0 + eps0).flatten()
        beta1 = np.random.normal(size=self.d)
        eps1 = np.random.normal(size=self.n)
        self.mu1 = (X @ beta1).flatten()
        self.Y1 = (self.mu1 + eps1).flatten()

    def Y(self, A):
        return A * self.Y1 + (1 - A) * self.Y0

    def Yi(self, idx, a):
        return a * self.Y1[idx] + (1 - a) * self.Y0[idx]
