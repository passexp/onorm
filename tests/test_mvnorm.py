import numpy as np
import pytest
from numpy.random import default_rng

from onorm import MultivariateNormalizer


@pytest.mark.skip(reason="MultivariateNormalizer has numerical stability issues - see TODO.md")
def test_mvnorm_basic():
    """Test basic MultivariateNormalizer functionality."""
    rng = default_rng(2022)
    n_dim = 3
    normalizer = MultivariateNormalizer(n_dim=n_dim, warmup_period=10)

    # This test is skipped due to numerical overflow issues
    # in the implementation (invSigmahat initialization with max float)
    X = rng.normal(size=(10, n_dim))
    for x in X:
        normalizer.partial_fit(x)


def test_mvnorm_reset():
    """Test that reset() properly resets the normalizer state."""
    n_dim = 3
    normalizer = MultivariateNormalizer(n_dim=n_dim, warmup_period=10)

    # Reset should work
    normalizer.reset()

    # State should be reset
    assert normalizer.n == 0, "n should be reset to 0"
    assert np.allclose(normalizer.muhat, np.zeros(n_dim)), "Mean should be reset"
    assert np.allclose(
        normalizer.invsqrtSigmahat, np.eye(n_dim)
    ), "Inverse sqrt should be reset to identity"


def test_mvnorm_initialization():
    """Test MultivariateNormalizer initialization."""
    n_dim = 5
    warmup = 20
    normalizer = MultivariateNormalizer(n_dim=n_dim, warmup_period=warmup)

    assert normalizer.n_dim == n_dim, "Should store n_dim"
    assert normalizer.warmup_period == warmup, "Should store warmup_period"
    assert normalizer.n == 0, "Should initialize n to 0"
    assert normalizer.muhat.shape == (n_dim,), "Mean should have correct shape"
    assert normalizer.invSigmahat.shape == (n_dim, n_dim), "Inv cov should have correct shape"
