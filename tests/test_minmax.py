import numpy as np
from numpy.random import default_rng
from onorm import MinMaxScaler


def test_minmax_basic():
    """Test basic MinMaxScaler functionality."""
    rng = default_rng(2022)
    n_dim = 5
    normalizer = MinMaxScaler(n_dim=n_dim)

    # Test single observation
    x = rng.normal(size=n_dim)
    normalizer.partial_fit(x)
    x_norm = normalizer.transform(x)

    # First observation should be all zeros (min == max == x)
    assert np.allclose(x_norm, np.zeros(n_dim)), "First observation should normalize to zeros"


def test_minmax_range():
    """Test that MinMaxScaler produces values in [0, 1] range."""
    rng = default_rng(2022)
    n = 100
    n_dim = 5
    normalizer = MinMaxScaler(n_dim=n_dim)

    X = rng.normal(size=(n, n_dim))

    # Fit normalizer
    for x in X:
        normalizer.partial_fit(x)

    # Transform last observation
    x_norm = normalizer.transform(X[-1])

    # Check range
    assert np.all(x_norm >= 0), "Normalized values should be >= 0"
    assert np.all(x_norm <= 1), "Normalized values should be <= 1"


def test_minmax_reset():
    """Test that reset() properly resets the normalizer state."""
    rng = default_rng(2022)
    n_dim = 3
    normalizer = MinMaxScaler(n_dim=n_dim)

    # Fit with some data
    X = rng.normal(size=(10, n_dim))
    for x in X:
        normalizer.partial_fit(x)

    # Reset
    normalizer.reset()

    # Min and max should be reset
    assert np.all(np.isinf(normalizer.min)), "Min should be reset to inf"
    assert np.all(np.isneginf(normalizer.max)), "Max should be reset to -inf"
