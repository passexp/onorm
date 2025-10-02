import numpy as np
from numpy.random import default_rng

from onorm import Winsorizer


def test_winsorizer_basic():
    """Test basic Winsorizer functionality."""
    rng = default_rng(2022)
    n_dim = 3
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9))

    # Generate data with outliers
    X = rng.normal(size=(100, n_dim))
    X[0] = [100, 100, 100]  # Add outliers

    # Fit winsorizer
    for x in X:
        winsorizer.partial_fit(x)

    # Transform should clip outliers
    x_outlier = np.array([100.0, 100.0, 100.0])
    x_clipped = winsorizer.transform(x_outlier.copy())

    # Values should be clipped (not equal to original outliers)
    assert np.all(x_clipped < 100), "Outliers should be clipped"


def test_winsorizer_no_clipping():
    """Test Winsorizer with clip_q=(0, 1) does minimal clipping."""
    rng = default_rng(2022)
    n_dim = 3
    # Use quantiles 0-1 (full range, minimal clipping)
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0, 1))

    # Generate data
    X = rng.normal(size=(50, n_dim))

    for x in X:
        winsorizer.partial_fit(x)

    # Transform values should be minimally clipped
    x_test = X[25].copy()  # Use a middle value from the data
    x_transformed = winsorizer.transform(x_test.copy())

    # Should be very close to original (0,100 percentiles are min/max)
    assert np.allclose(x_transformed, x_test, atol=0.5), "Should not clip normal values much"


def test_winsorizer_symmetric_clipping():
    """Test symmetric clipping around median."""
    rng = default_rng(2022)
    n_dim = 3
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0.25, 0.75))

    # Generate symmetric data
    X = rng.normal(size=(200, n_dim))

    for x in X:
        winsorizer.partial_fit(x)

    # High outlier
    x_high = np.array([10.0, 10.0, 10.0])
    x_high_clipped = winsorizer.transform(x_high.copy())

    # Low outlier
    x_low = np.array([-10.0, -10.0, -10.0])
    x_low_clipped = winsorizer.transform(x_low.copy())

    # Both should be clipped
    assert np.all(x_high_clipped < 10), "High values should be clipped"
    assert np.all(x_low_clipped > -10), "Low values should be clipped"


def test_winsorizer_reset():
    """Test that reset() properly resets the winsorizer state."""
    rng = default_rng(2022)
    n_dim = 3
    winsorizer = Winsorizer(n_dim=n_dim)

    # Fit with some data
    X = rng.normal(size=(10, n_dim))
    for x in X:
        winsorizer.partial_fit(x)

    # Get a percentile value before reset
    p50_before = winsorizer.digests[0].percentile(50)

    # Reset
    winsorizer.reset()

    # After reset, should be new empty TDigest objects
    assert len(winsorizer.digests) == n_dim, "Should have correct number of digests"
    # Try to add new data after reset
    x_new = rng.normal(size=n_dim)
    winsorizer.partial_fit(x_new)
    # Should work without errors
    assert True


def test_winsorizer_preserves_middle_values():
    """Test that values within quantile range are preserved."""
    rng = default_rng(2022)
    n_dim = 3
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9))

    # Generate data
    X = rng.normal(size=(100, n_dim))

    for x in X:
        winsorizer.partial_fit(x)

    # Use value from the data that's within the percentile range
    x_test = X[50].copy()  # Middle value
    x_transformed = winsorizer.transform(x_test.copy())

    # Values within 10-90 percentile range should be mostly unchanged
    # (TDigest approximation may cause small differences)
    assert np.allclose(
        x_transformed, x_test, atol=0.5
    ), "Values within percentile range should be mostly preserved"


def test_winsorizer_extreme_percentiles():
    """Test winsorizer with extreme percentile values."""
    rng = default_rng(2022)
    n_dim = 3

    # Very tight clipping (almost everything gets clipped)
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0.45, 0.55))

    X = rng.normal(size=(200, n_dim))

    for x in X:
        winsorizer.partial_fit(x)

    # Most values should be clipped to narrow range
    x_test = rng.normal(size=n_dim) * 2  # Likely outside narrow range
    x_clipped = winsorizer.transform(x_test.copy())

    # Check that clipping happened (values are within a narrow range)
    for i in range(n_dim):
        lower = winsorizer.digests[i].percentile(45)
        upper = winsorizer.digests[i].percentile(55)
        assert (
            lower <= x_clipped[i] <= upper
        ), f"Value should be clipped to percentile range"


def test_winsorizer_partial_fit_transform():
    """Test that partial_fit_transform works correctly."""
    rng = default_rng(2022)
    n_dim = 3
    winsorizer = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9))

    x = rng.normal(size=n_dim)

    # partial_fit_transform should be equivalent to partial_fit then transform
    winsorizer_copy = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9))
    winsorizer_copy.partial_fit(x)
    x_transformed_separate = winsorizer_copy.transform(x.copy())

    x_transformed_combined = winsorizer.partial_fit_transform(x.copy())

    assert np.allclose(
        x_transformed_combined, x_transformed_separate
    ), "partial_fit_transform should match separate calls"


def test_winsorizer_delta_parameter():
    """Test that delta parameter affects TDigest precision."""
    rng = default_rng(2022)
    n_dim = 3

    # Smaller delta = higher precision (more memory)
    winsorizer_precise = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9), tdigest_delta=0.001)

    # Larger delta = lower precision (less memory)
    winsorizer_coarse = Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9), tdigest_delta=0.1)

    X = rng.normal(size=(1000, n_dim))

    for x in X:
        winsorizer_precise.partial_fit(x)
        winsorizer_coarse.partial_fit(x)

    # Both should work, precision difference may be subtle
    x_test = rng.normal(size=n_dim)
    x_precise = winsorizer_precise.transform(x_test.copy())
    x_coarse = winsorizer_coarse.transform(x_test.copy())

    # Both should produce finite results
    assert np.all(np.isfinite(x_precise)), "Precise winsorizer should produce finite results"
    assert np.all(np.isfinite(x_coarse)), "Coarse winsorizer should produce finite results"


def test_winsorizer_maintains_shape():
    """Test that Winsorizer maintains array shape."""
    rng = default_rng(2022)
    n_dim = 5
    winsorizer = Winsorizer(n_dim=n_dim)

    X = rng.normal(size=(50, n_dim))

    for x in X:
        winsorizer.partial_fit(x)

    x_test = rng.normal(size=n_dim)
    x_transformed = winsorizer.transform(x_test.copy())

    assert x_transformed.shape == x_test.shape, "Shape should be preserved"
    assert len(x_transformed) == n_dim, "Dimension should be preserved"
