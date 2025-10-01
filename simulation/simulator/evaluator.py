import time
from abc import ABCMeta
from typing import List

import numpy as np

from .dgp import XDGP, YDGP
from .model import Model


class Evaluator(metaclass=ABCMeta):
    def __init__(self, n, xdgp: XDGP, ydgp: YDGP, models: List[Model]) -> None:
        self.n = n
        self.xdgp = xdgp
        self.xdgp.setup(self.n)
        self.ydgp = ydgp
        self.ydgp.setup(self.xdgp.X)
        self.models = models
        self.results = []

    def log(self, model, metric, value) -> None:
        self.results.append(
            {
                "xdgp": type(self.xdgp).__name__,
                "ydgp": type(self.ydgp).__name__,
                "model": model,
                "metric": metric,
                "value": value,
            }
        )

    def imbalance(self, A, X):
        return np.linalg.norm(
            np.average(X, weights=A, axis=0) - np.average(X, weights=1 - A, axis=0)
        )

    def causal_error(self, Y, A, ATE):
        return (np.average(Y, weights=A) - np.average(Y, weights=1 - A)) - ATE

    def treatment_imbalance(self, A, pr):
        return np.average(A) - pr

    def evaluate(self) -> None:
        for model in self.models:
            model.reset()
            time_start = time.time()
            A = np.zeros(self.n)
            Y = np.zeros(self.n)
            for idx, x in enumerate(self.xdgp.X):
                x_trans = model.normalize(x)
                A[idx] = model.assign(x_trans)
                Y[idx] = self.ydgp.Yi(idx, A[idx])
            time_end = time.time()
            time_elapsed = time_end - time_start
            self.log(model.name, "elapsed_time", time_elapsed)
            self.log(
                model.name, "treatment_imbalance", self.treatment_imbalance(A, model.assigner.q)
            )
            self.log(model.name, "imbalance", self.imbalance(A, self.xdgp.X))
            self.log(model.name, "causal_error", self.causal_error(Y, A, self.ydgp.ATE))
