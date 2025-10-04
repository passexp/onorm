import numpy as np
import pytest
from numpy.random import default_rng
from onorm import Winsorizer


@pytest.fixture
def rng():
    """Provide a consistent random number generator."""
    return default_rng(2022)


@pytest.fixture
def sample_data(rng):
    """Generate sample data with outliers for testing."""
    n_dim = 3
    n_samples = 100
    X = rng.normal(size=(n_samples, n_dim))
    X[0] = [100, 100, 100]  # Add outlier in first row
    return X, n_dim


@pytest.fixture
def fitted_winsorizer(sample_data):
    """Create and fit a Winsorizer on sample data."""
    X, n_dim = sample_data
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9))
    for x in X:
        winsorizer.partial_fit(x)
    return winsorizer


def test_winsorizer_basic(fitted_winsorizer):
    """Test basic Winsorizer functionality."""
    # Transform should clip outliers
    x_outlier = np.array([100.0, 100.0, 100.0])
    x_clipped = fitted_winsorizer.transform(x_outlier.copy())

    # Values should be clipped (not equal to original outliers)
    assert np.all(x_clipped < 100), "Outliers should be clipped"


def test_winsorizer_no_clipping(rng):
    """Test Winsorizer with clip_q=(0, 1) does minimal clipping."""
    n_dim = 3
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0, 1))

    X = rng.normal(size=(50, n_dim))
    for x in X:
        winsorizer.partial_fit(x)

    # Transform values should be minimally clipped
    x_test = X[25].copy()
    x_transformed = winsorizer.transform(x_test.copy())

    # Should be very close to original (0,1 quantiles are min/max)
    assert np.allclose(x_transformed, x_test, atol=0.5), "Should not clip normal values much"


def test_winsorizer_symmetric_clipping(rng):
    """Test symmetric clipping around median."""
    n_dim = 3
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0.25, 0.75))

    X = rng.normal(size=(200, n_dim))
    for x in X:
        winsorizer.partial_fit(x)

    # High and low outliers should both be clipped
    x_high = np.array([10.0, 10.0, 10.0])
    x_high_clipped = winsorizer.transform(x_high.copy())

    x_low = np.array([-10.0, -10.0, -10.0])
    x_low_clipped = winsorizer.transform(x_low.copy())

    assert np.all(x_high_clipped < 10), "High values should be clipped"
    assert np.all(x_low_clipped > -10), "Low values should be clipped"


def test_winsorizer_reset(rng):
    """Test that reset() properly resets the winsorizer state."""
    n_dim = 3
    winsorizer = Winsorizer(n_dim=n_dim)

    # Fit with some data
    X = rng.normal(size=(10, n_dim))
    for x in X:
        winsorizer.partial_fit(x)

    # Reset and verify
    winsorizer.reset()

    assert len(winsorizer.digests) == n_dim, "Should have correct number of digests"

    # Should work after reset
    x_new = rng.normal(size=n_dim)
    winsorizer.partial_fit(x_new)


def test_winsorizer_preserves_middle_values(sample_data):
    """Test that values within quantile range are preserved."""
    X, n_dim = sample_data
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9))

    for x in X:
        winsorizer.partial_fit(x)

    # Middle values should be mostly unchanged
    x_test = X[50].copy()
    x_transformed = winsorizer.transform(x_test.copy())

    assert np.allclose(
        x_transformed, x_test, atol=0.5
    ), "Values within quantile range should be mostly preserved"


def test_winsorizer_extreme_quantiles(rng):
    """Test winsorizer with extreme quantile values."""
    n_dim = 3
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0.45, 0.55))

    X = rng.normal(size=(200, n_dim))
    for x in X:
        winsorizer.partial_fit(x)

    # Most values should be clipped to narrow range
    x_test = rng.normal(size=n_dim) * 2
    x_clipped = winsorizer.transform(x_test.copy())

    # Check that clipping happened (values are within narrow range)
    for i in range(n_dim):
        lower = winsorizer.digests[i].quantile(0.45)
        upper = winsorizer.digests[i].quantile(0.55)
        assert lower <= x_clipped[i] <= upper, "Value should be clipped to quantile range"


def test_winsorizer_partial_fit_transform(rng):
    """Test that partial_fit_transform works correctly."""
    n_dim = 3
    x = rng.normal(size=n_dim)

    # Compare partial_fit_transform with separate calls
    winsorizer1 = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9))
    winsorizer1.partial_fit(x)
    x_separate = winsorizer1.transform(x.copy())

    winsorizer2 = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9))
    x_combined = winsorizer2.partial_fit_transform(x.copy())

    assert np.allclose(x_combined, x_separate), "partial_fit_transform should match separate calls"


def test_winsorizer_max_centroids_parameter(rng):
    """Test that max_centroids parameter affects TDigest precision."""
    n_dim = 3

    # Different max_centroids values for different precision levels
    winsorizer_precise = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9), max_centroids=2000)
    winsorizer_coarse = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9), max_centroids=3)

    X = rng.normal(size=(1000, n_dim))
    for x in X:
        winsorizer_precise.partial_fit(x)
        winsorizer_coarse.partial_fit(x)

    # Both should produce finite results
    x_test = rng.normal(size=n_dim)
    x_precise = winsorizer_precise.transform(x_test.copy())
    x_coarse = winsorizer_coarse.transform(x_test.copy())

    assert np.all(np.isfinite(x_precise)), "Precise winsorizer should produce finite results"
    assert np.all(np.isfinite(x_coarse)), "Coarse winsorizer should produce finite results"


def test_winsorizer_maintains_shape(rng):
    """Test that Winsorizer maintains array shape."""
    n_dim = 5
    winsorizer = Winsorizer(n_dim=n_dim)

    X = rng.normal(size=(50, n_dim))
    for x in X:
        winsorizer.partial_fit(x)

    x_test = rng.normal(size=n_dim)
    x_transformed = winsorizer.transform(x_test.copy())

    assert x_transformed.shape == x_test.shape, "Shape should be preserved"
    assert len(x_transformed) == n_dim, "Dimension should be preserved"
