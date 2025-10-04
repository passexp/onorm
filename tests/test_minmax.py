import numpy as np
import pytest
from numpy.random import default_rng
from onorm import MinMaxScaler


@pytest.fixture
def rng():
    """Provide a consistent random number generator."""
    return default_rng(2022)


@pytest.fixture
def sample_data(rng):
    """Generate sample data for testing."""
    n_dim = 5
    n_samples = 100
    X = rng.normal(size=(n_samples, n_dim))
    return X, n_dim


@pytest.fixture
def fitted_scaler(sample_data):
    """Create and fit a MinMaxScaler on sample data."""
    X, n_dim = sample_data
    scaler = MinMaxScaler(n_dim=n_dim)
    for x in X:
        scaler.partial_fit(x)
    return scaler


def test_minmax_basic(rng):
    """Test basic MinMaxScaler functionality."""
    n_dim = 5
    scaler = MinMaxScaler(n_dim=n_dim)

    # First observation should be all zeros (min == max == x)
    x = rng.normal(size=n_dim)
    scaler.partial_fit(x)
    x_norm = scaler.transform(x)

    assert np.allclose(x_norm, np.zeros(n_dim)), "First observation should normalize to zeros"


def test_minmax_range(sample_data, fitted_scaler):
    """Test that MinMaxScaler produces values in [0, 1] range."""
    X, _ = sample_data
    scaler = fitted_scaler

    # Transform last observation
    x_norm = scaler.transform(X[-1])

    # Check range
    assert np.all(x_norm >= 0), "Normalized values should be >= 0"
    assert np.all(x_norm <= 1), "Normalized values should be <= 1"


def test_minmax_reset(rng):
    """Test that reset() properly resets the scaler state."""
    n_dim = 3
    scaler = MinMaxScaler(n_dim=n_dim)

    # Fit with some data
    X = rng.normal(size=(10, n_dim))
    for x in X:
        scaler.partial_fit(x)

    # Reset and verify
    scaler.reset()

    assert np.all(np.isinf(scaler.min)), "Min should be reset to inf"
    assert np.all(np.isneginf(scaler.max)), "Max should be reset to -inf"
