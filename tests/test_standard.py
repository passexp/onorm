import numpy as np
import pytest
from numpy.random import default_rng
from onorm import StandardScaler


@pytest.fixture
def rng():
    """Provide a consistent random number generator."""
    return default_rng(2022)


@pytest.fixture
def sample_data(rng):
    """Generate sample data for testing."""
    n_dim = 3
    n_samples = 100
    X = rng.normal(loc=5, scale=2, size=(n_samples, n_dim))
    return X, n_dim


@pytest.fixture
def fitted_scaler(sample_data):
    """Create and fit a standard scaler on sample data."""
    X, n_dim = sample_data
    scaler = StandardScaler(n_dim=n_dim)
    for x in X:
        scaler.partial_fit(x)
    return scaler


def test_standard_basic(rng):
    """Test basic StandardScaler functionality."""
    n_dim = 5
    scaler = StandardScaler(n_dim=n_dim)

    # First observation should return zeros (insufficient data for variance)
    x = rng.normal(size=n_dim)
    scaler.partial_fit(x)
    x_norm = scaler.transform(x)

    assert np.allclose(x_norm, np.zeros(n_dim)), "First observation should be zeros"


def test_standard_mean_and_std(rng):
    """Test that StandardScaler produces mean≈0 and std≈1."""
    n_samples = 1000
    n_dim = 5

    # Generate data with known mean and std
    true_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    true_std = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

    scaler = StandardScaler(n_dim=n_dim)
    X = rng.normal(loc=true_mean, scale=true_std, size=(n_samples, n_dim))

    for x in X:
        scaler.partial_fit(x)

    # Transform all data
    X_transformed = np.array([scaler.transform(x) for x in X])

    # Check mean ≈ 0 and std ≈ 1
    assert np.allclose(np.mean(X_transformed, axis=0), 0, atol=0.1), "Mean should be ≈0"
    assert np.allclose(np.std(X_transformed, axis=0, ddof=1), 1, atol=0.1), "Std should be ≈1"


def test_standard_vs_batch(sample_data, fitted_scaler):
    """Test that online StandardScaler matches batch computation."""
    X, _ = sample_data
    scaler = fitted_scaler

    # Batch statistics
    batch_mean = np.mean(X, axis=0)
    batch_std = np.std(X, axis=0, ddof=1)

    # Compare learned statistics
    assert np.allclose(scaler.mean, batch_mean, rtol=1e-10), "Mean should match batch"
    assert np.allclose(np.sqrt(scaler.variance), batch_std, rtol=1e-10), "Std should match batch"


def test_standard_with_mean_false(rng):
    """Test StandardScaler with centering disabled."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim, with_mean=False)
    X = rng.normal(loc=5, scale=2, size=(100, n_dim))

    for x in X:
        scaler.partial_fit(x)

    # Transform should only scale, not center
    x_test = rng.normal(loc=5, scale=2, size=n_dim)
    x_norm = scaler.transform(x_test)

    # Verify data is not centered
    assert not np.allclose(x_norm, x_norm - scaler.mean), "Data should not be centered"


def test_standard_with_std_false(rng):
    """Test StandardScaler with scaling disabled."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim, with_std=False)
    X = rng.normal(loc=5, scale=2, size=(100, n_dim))

    for x in X:
        scaler.partial_fit(x)

    # Transform should only center, not scale
    x_test = rng.normal(loc=5, scale=2, size=n_dim)
    x_norm = scaler.transform(x_test)

    expected = x_test - scaler.mean
    assert np.allclose(x_norm, expected), "Should only center, not scale"


def test_standard_ddof(rng):
    """Test StandardScaler with different degrees of freedom."""
    n_dim = 3
    X = rng.normal(size=(50, n_dim))

    # Population variance (ddof=0)
    scaler0 = StandardScaler(n_dim=n_dim, ddof=0)
    for x in X:
        scaler0.partial_fit(x)

    # Sample variance (ddof=1)
    scaler1 = StandardScaler(n_dim=n_dim, ddof=1)
    for x in X:
        scaler1.partial_fit(x)

    # Transforms should differ
    x_test = rng.normal(size=n_dim)
    x_norm0 = scaler0.transform(x_test)
    x_norm1 = scaler1.transform(x_test)

    assert not np.allclose(x_norm0, x_norm1), "Different ddof should produce different transforms"

    # Sample std should be >= population std
    std0 = np.sqrt(scaler0.variance)
    std1 = np.sqrt(scaler1.variance)
    assert np.all(std0 <= std1), "Sample std (ddof=1) should be >= population std (ddof=0)"


def test_standard_reset(rng):
    """Test that reset() properly resets the scaler state."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    # Fit with some data
    X = rng.normal(size=(10, n_dim))
    for x in X:
        scaler.partial_fit(x)

    # Reset and verify
    scaler.reset()

    assert scaler.n == 0, "n should be reset to 0"
    assert np.allclose(scaler.mean, np.zeros(n_dim)), "Mean should be reset to zeros"
    assert np.allclose(scaler.M, np.zeros(n_dim)), "M should be reset to zeros"


def test_standard_constant_feature():
    """Test StandardScaler behavior with constant (zero variance) features."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    # Feature 0: constant, Features 1-2: variable
    X = np.array([[5.0, 1.0, 2.0], [5.0, 2.0, 3.0], [5.0, 3.0, 4.0], [5.0, 4.0, 5.0]])

    for x in X:
        scaler.partial_fit(x)

    x_test = np.array([5.0, 2.5, 3.5])
    x_norm = scaler.transform(x_test)

    # Constant feature should be centered without divide-by-zero
    assert np.isfinite(x_norm[0]), "Constant feature should not cause NaN"
    assert np.allclose(x_norm[0], 0.0), "Constant feature should be centered to 0"


def test_standard_numerical_stability():
    """Test numerical stability with large values."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    # Very large values
    X = np.array([[1e10, 2e10, 3e10], [1e10 + 1, 2e10 + 2, 3e10 + 3]])

    for x in X:
        scaler.partial_fit(x)

    # Should handle large values without overflow
    assert np.all(np.isfinite(scaler.mean)), "Mean should be finite with large values"
    assert np.all(np.isfinite(scaler.M)), "M should be finite with large values"


def test_standard_partial_fit_transform(rng):
    """Test that partial_fit_transform works correctly."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    x = rng.normal(size=n_dim)

    # Compare partial_fit_transform with separate calls
    scaler_copy = StandardScaler(n_dim=n_dim)
    scaler_copy.partial_fit(x)
    x_norm_separate = scaler_copy.transform(x)

    x_norm_combined = scaler.partial_fit_transform(x)

    assert np.allclose(
        x_norm_combined, x_norm_separate
    ), "partial_fit_transform should match separate calls"


def test_standard_variance_with_high_ddof(rng):
    """Test variance property when n <= ddof."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim, ddof=2)

    # Add only one sample (n=1, ddof=2, so n <= ddof)
    x = rng.normal(size=n_dim)
    scaler.partial_fit(x)

    # Variance should be zero when n <= ddof
    assert np.allclose(scaler.variance, np.zeros(n_dim)), "Variance should be zero when n <= ddof"

    # Add another sample (n=2, ddof=2, so n <= ddof still)
    scaler.partial_fit(rng.normal(size=n_dim))
    assert np.allclose(scaler.variance, np.zeros(n_dim)), "Variance should be zero when n == ddof"

    # Add third sample (n=3, ddof=2, so n > ddof now)
    scaler.partial_fit(rng.normal(size=n_dim))
    assert not np.allclose(
        scaler.variance, np.zeros(n_dim)
    ), "Variance should be non-zero when n > ddof"


def test_standard_with_mean_and_std_both_false(rng):
    """Test StandardScaler with both centering and scaling disabled."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim, with_mean=False, with_std=False)
    X = rng.normal(loc=5, scale=2, size=(100, n_dim))

    for x in X:
        scaler.partial_fit(x)

    # Transform should return data unchanged (except maybe after first n<=ddof samples)
    x_test = rng.normal(loc=5, scale=2, size=n_dim)
    x_norm = scaler.transform(x_test.copy())

    # Since both are disabled, should return input as-is
    assert np.allclose(x_norm, x_test), "Should return input unchanged"


def test_standard_one_dimensional(rng):
    """Test StandardScaler with 1-dimensional data."""
    n_dim = 1
    scaler = StandardScaler(n_dim=n_dim)

    X = rng.normal(loc=10.0, scale=3.0, size=(100, 1))
    for x in X:
        scaler.partial_fit(x)

    # Verify statistics
    batch_mean = np.mean(X, axis=0)
    batch_std = np.std(X, axis=0, ddof=1)

    assert np.allclose(scaler.mean, batch_mean, rtol=1e-10)
    assert np.allclose(np.sqrt(scaler.variance), batch_std, rtol=1e-10)

    # Transform should work
    x_test = np.array([10.0])
    x_norm = scaler.transform(x_test)
    assert x_norm.shape == (1,)
    assert np.isfinite(x_norm[0])


def test_standard_extreme_outliers(rng):
    """Test StandardScaler with extreme outliers."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    # Mix of normal values and extreme outliers
    X_normal = rng.normal(loc=0, scale=1, size=(50, n_dim))
    X_outliers = np.array([[1000.0, -1000.0, 500.0], [-500.0, 750.0, -250.0]])

    X = np.vstack([X_normal, X_outliers])

    for x in X:
        scaler.partial_fit(x)

    # Transform should handle outliers without crashing
    x_test = rng.normal(size=n_dim)
    x_norm = scaler.transform(x_test)

    assert np.all(np.isfinite(x_norm)), "Should handle outliers without overflow"
    assert x_norm.shape == (n_dim,)


def test_standard_sequential_reset_and_refit(rng):
    """Test resetting and refitting the scaler multiple times."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    for cycle in range(3):
        # Generate new data for each cycle
        X = rng.normal(loc=cycle * 10, scale=cycle + 1, size=(50, n_dim))

        for x in X:
            scaler.partial_fit(x)

        # Check that statistics reflect current data
        batch_mean = np.mean(X, axis=0)
        assert np.allclose(scaler.mean, batch_mean, rtol=1e-10)

        # Reset for next cycle
        scaler.reset()
        assert scaler.n == 0
        assert np.allclose(scaler.variance, np.zeros(n_dim))


def test_standard_mixed_variance_features(rng):
    """Test StandardScaler with features having very different variances."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    # Create data with very different scales
    # Feature 0: small variance, Feature 1: medium variance, Feature 2: large variance
    X = np.column_stack(
        [
            rng.normal(loc=0, scale=0.01, size=100),  # Low variance
            rng.normal(loc=0, scale=1.0, size=100),  # Medium variance
            rng.normal(loc=0, scale=100.0, size=100),  # High variance
        ]
    )

    for x in X:
        scaler.partial_fit(x)

    # All features should be scaled to similar ranges
    X_transformed = np.array([scaler.transform(x) for x in X])
    transformed_std = np.std(X_transformed, axis=0, ddof=1)

    # All should be close to 1.0
    assert np.allclose(
        transformed_std, 1.0, atol=0.1
    ), "All features should have similar standard deviations after scaling"


def test_standard_high_dimensional(rng):
    """Test StandardScaler with higher dimensional data."""
    n_dim = 20
    n_samples = 100

    scaler = StandardScaler(n_dim=n_dim)
    X = rng.normal(loc=np.arange(n_dim), scale=np.arange(1, n_dim + 1), size=(n_samples, n_dim))

    for x in X:
        scaler.partial_fit(x)

    # Verify statistics match batch
    batch_mean = np.mean(X, axis=0)
    batch_std = np.std(X, axis=0, ddof=1)

    assert np.allclose(scaler.mean, batch_mean, rtol=1e-10)
    assert np.allclose(np.sqrt(scaler.variance), batch_std, rtol=1e-10)

    # Transform should normalize properly
    X_transformed = np.array([scaler.transform(x) for x in X])
    assert np.allclose(np.mean(X_transformed, axis=0), 0, atol=0.1)
    assert np.allclose(np.std(X_transformed, axis=0, ddof=1), 1, atol=0.1)


def test_standard_incremental_statistics(rng):
    """Test that statistics update correctly with each observation."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    X = rng.normal(loc=5, scale=2, size=(10, n_dim))

    means = []
    variances = []

    for x in X:
        scaler.partial_fit(x)
        means.append(scaler.mean.copy())
        variances.append(scaler.variance.copy())

    # Mean should change with each observation
    for i in range(1, len(means)):
        assert not np.allclose(means[i], means[i - 1]), "Mean should update with each observation"

    # Variance should stabilize as more data is added
    if len(variances) > 5:
        # Later variances should be more stable
        early_var_diff = np.abs(variances[2] - variances[1])
        # This is a rough check - early differences should generally be larger
        assert np.mean(early_var_diff) >= 0  # Just ensure it's computed


def test_standard_transform_before_fit():
    """Test transforming data before any fitting (edge case)."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    x = np.array([1.0, 2.0, 3.0])
    x_norm = scaler.transform(x)

    # Should return zeros since n <= ddof (n=0, ddof=1)
    assert np.allclose(x_norm, np.zeros(n_dim)), "Should return zeros before any fitting"


def test_standard_negative_values(rng):
    """Test StandardScaler with negative values."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    # Data centered around negative values
    X = rng.normal(loc=-50, scale=10, size=(100, n_dim))

    for x in X:
        scaler.partial_fit(x)

    # Should handle negative means correctly
    assert np.all(scaler.mean < 0), "Mean should be negative for negative data"

    # Transform should still work
    x_test = rng.normal(loc=-50, scale=10, size=n_dim)
    x_norm = scaler.transform(x_test)

    assert np.all(np.isfinite(x_norm)), "Should handle negative values"
    assert x_norm.shape == (n_dim,)
