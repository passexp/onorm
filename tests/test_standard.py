import numpy as np
from numpy.random import default_rng
from onorm import StandardScaler


def test_standard_basic():
    """Test basic StandardScaler functionality."""
    rng = default_rng(2022)
    n_dim = 5
    scaler = StandardScaler(n_dim=n_dim)

    # Test single observation
    x = rng.normal(size=n_dim)
    scaler.partial_fit(x)
    x_norm = scaler.transform(x)

    # First observation should be zeros (not enough data for variance)
    assert np.allclose(x_norm, np.zeros(n_dim)), "First observation should be zeros"


def test_standard_mean_and_std():
    """Test that StandardScaler produces mean=0 and std=1."""
    rng = default_rng(2022)
    n = 1000
    n_dim = 5

    # Generate data with known mean and std
    true_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    true_std = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

    scaler = StandardScaler(n_dim=n_dim)
    X = rng.normal(loc=true_mean, scale=true_std, size=(n, n_dim))

    # Fit scaler
    for x in X:
        scaler.partial_fit(x)

    # Transform all data
    X_transformed = np.array([scaler.transform(x) for x in X])

    # Check mean ≈ 0 and std ≈ 1
    assert np.allclose(
        np.mean(X_transformed, axis=0), 0, atol=0.1
    ), "Mean should be close to 0"
    assert np.allclose(
        np.std(X_transformed, axis=0, ddof=1), 1, atol=0.1
    ), "Std should be close to 1"


def test_standard_vs_batch():
    """Test that online StandardScaler matches batch computation."""
    rng = default_rng(2022)
    n = 100
    n_dim = 3

    X = rng.normal(loc=5, scale=2, size=(n, n_dim))

    # Online scaler
    scaler = StandardScaler(n_dim=n_dim)
    for x in X:
        scaler.partial_fit(x)

    # Batch statistics
    batch_mean = np.mean(X, axis=0)
    batch_std = np.std(X, axis=0, ddof=1)

    # Compare
    assert np.allclose(scaler.mean, batch_mean, rtol=1e-10), "Mean should match batch"
    variance = scaler.M / (scaler.n - 1)
    assert np.allclose(
        np.sqrt(variance), batch_std, rtol=1e-10
    ), "Std should match batch"


def test_standard_with_mean_false():
    """Test StandardScaler with centering disabled."""
    rng = default_rng(2022)
    n = 100
    n_dim = 3

    scaler = StandardScaler(n_dim=n_dim, with_mean=False)
    X = rng.normal(loc=5, scale=2, size=(n, n_dim))

    for x in X:
        scaler.partial_fit(x)

    # Transform should only scale, not center
    x_test = rng.normal(loc=5, scale=2, size=n_dim)
    x_norm = scaler.transform(x_test)

    # Should not be centered (mean preserved but scaled)
    assert not np.allclose(
        x_norm, x_norm - scaler.mean
    ), "Data should not be centered"


def test_standard_with_std_false():
    """Test StandardScaler with scaling disabled."""
    rng = default_rng(2022)
    n = 100
    n_dim = 3

    scaler = StandardScaler(n_dim=n_dim, with_std=False)
    X = rng.normal(loc=5, scale=2, size=(n, n_dim))

    for x in X:
        scaler.partial_fit(x)

    # Transform should only center, not scale
    x_test = rng.normal(loc=5, scale=2, size=n_dim)
    x_norm = scaler.transform(x_test)

    # Should be centered (mean=0) but not scaled
    expected = x_test - scaler.mean
    assert np.allclose(x_norm, expected), "Should only center, not scale"


def test_standard_ddof():
    """Test StandardScaler with different degrees of freedom."""
    rng = default_rng(2022)
    n = 50
    n_dim = 3
    X = rng.normal(size=(n, n_dim))

    # Test with ddof=0 (population variance)
    scaler0 = StandardScaler(n_dim=n_dim, ddof=0)
    for x in X:
        scaler0.partial_fit(x)

    # Test with ddof=1 (sample variance)
    scaler1 = StandardScaler(n_dim=n_dim, ddof=1)
    for x in X:
        scaler1.partial_fit(x)

    # Transform the same test point
    x_test = rng.normal(size=n_dim)
    x_norm0 = scaler0.transform(x_test)
    x_norm1 = scaler1.transform(x_test)

    # Transforms should be different due to different variance calculations
    assert not np.allclose(
        x_norm0, x_norm1
    ), "Different ddof should produce different transforms"

    # ddof=1 uses n-1 denominator, resulting in larger std
    std0 = np.sqrt(scaler0.variance)
    std1 = np.sqrt(scaler1.variance)
    assert np.all(
        std0 <= std1
    ), "Sample std (ddof=1) should be >= population std (ddof=0)"


def test_standard_reset():
    """Test that reset() properly resets the scaler state."""
    rng = default_rng(2022)
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    # Fit with some data
    X = rng.normal(size=(10, n_dim))
    for x in X:
        scaler.partial_fit(x)

    # Reset
    scaler.reset()

    # State should be reset
    assert scaler.n == 0, "n should be reset to 0"
    assert np.allclose(scaler.mean, np.zeros(n_dim)), "Mean should be reset to zeros"
    assert np.allclose(scaler.M, np.zeros(n_dim)), "M should be reset to zeros"


def test_standard_constant_feature():
    """Test StandardScaler behavior with constant (zero variance) features."""
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    # Feature 1: constant, Features 2-3: variable
    X = np.array(
        [[5.0, 1.0, 2.0], [5.0, 2.0, 3.0], [5.0, 3.0, 4.0], [5.0, 4.0, 5.0]]
    )

    for x in X:
        scaler.partial_fit(x)

    x_test = np.array([5.0, 2.5, 3.5])
    x_norm = scaler.transform(x_test)

    # Constant feature should be centered but not cause divide by zero
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


def test_standard_partial_fit_transform():
    """Test that partial_fit_transform works correctly."""
    rng = default_rng(2022)
    n_dim = 3
    scaler = StandardScaler(n_dim=n_dim)

    x = rng.normal(size=n_dim)

    # partial_fit_transform should be equivalent to partial_fit then transform
    scaler_copy = StandardScaler(n_dim=n_dim)
    scaler_copy.partial_fit(x)
    x_norm_separate = scaler_copy.transform(x)

    x_norm_combined = scaler.partial_fit_transform(x)

    assert np.allclose(
        x_norm_combined, x_norm_separate
    ), "partial_fit_transform should match separate calls"
