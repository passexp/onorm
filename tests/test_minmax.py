import pytest

from onorm import MinMaxScaler

rng = default_rng(2022)

n = 10000
d = 5
ate = 1
beta = rng.normal(size=d)

X = rng.normal(size=(n, d))

balancer = bwd.BWD(N=n, D=d)
normalizer = MinMaxScaler(n_dim=d)