import numpy as np
import pytest
from numpy.random import default_rng
from onorm import MultivariateNormalizer


@pytest.fixture
def simple_data():
    """Generate simple uncorrelated data for testing."""
    rng = default_rng(2022)
    n_dim = 3
    n_samples = 50
    X = rng.normal(size=(n_samples, n_dim))
    return X, n_dim


@pytest.fixture
def correlated_data():
    """Generate correlated data with known mean and covariance."""
    rng = default_rng(42)
    n_dim = 3
    n_samples = 50

    true_mean = np.array([1.0, 2.0, 3.0])
    true_cov = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.4], [0.3, 0.4, 1.0]])

    X = rng.multivariate_normal(true_mean, true_cov, size=n_samples)
    return X, n_dim, true_mean, true_cov


@pytest.fixture
def fitted_normalizer(correlated_data):
    """Create and fit a normalizer on correlated data."""
    X, n_dim, _, _ = correlated_data
    normalizer = MultivariateNormalizer(n_dim=n_dim)
    for x in X:
        normalizer.partial_fit(x)
    return normalizer


def test_mvnorm_basic(simple_data):
    """Test basic MultivariateNormalizer functionality."""
    X, n_dim = simple_data
    normalizer = MultivariateNormalizer(n_dim=n_dim)

    for x in X:
        normalizer.partial_fit(x)

    # Should be able to transform data
    rng = default_rng(2022)
    x_test = rng.normal(size=n_dim)
    x_transformed = normalizer.transform(x_test.copy())

    assert x_transformed.shape == (n_dim,), "Should preserve shape"
    assert np.all(np.isfinite(x_transformed)), "Should produce finite results"


def test_mvnorm_reset():
    """Test that reset() properly resets the normalizer state."""
    n_dim = 3
    normalizer = MultivariateNormalizer(n_dim=n_dim)

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
    normalizer = MultivariateNormalizer(n_dim=n_dim)

    assert normalizer.n_dim == n_dim, "Should store n_dim"
    assert normalizer.n == 0, "Should initialize n to 0"
    assert normalizer.muhat.shape == (n_dim,), "Mean should have correct shape"
    assert normalizer.Sigmahat.shape == (n_dim, n_dim), "Covariance should have correct shape"
    assert normalizer.invsqrtSigmahat.shape == (
        n_dim,
        n_dim,
    ), "Inv sqrt cov should have correct shape"


def test_mvnorm_mean_estimation(correlated_data, fitted_normalizer):
    """Test that online mean estimation matches batch calculation."""
    X, n_dim, _, _ = correlated_data
    normalizer = fitted_normalizer

    batch_mean = np.mean(X, axis=0)

    assert np.allclose(normalizer.muhat, batch_mean, rtol=1e-10), (
        f"Learned mean differs from batch mean:\n"
        f"  Online: {normalizer.muhat}\n"
        f"  Batch:  {batch_mean}"
    )


def test_mvnorm_covariance_estimation(correlated_data, fitted_normalizer):
    """Test that online covariance estimation matches batch calculation."""
    X, n_dim, _, _ = correlated_data
    normalizer = fitted_normalizer

    batch_cov = np.cov(X.T, ddof=1)

    assert np.allclose(normalizer.Sigmahat, batch_cov, rtol=1e-10), (
        f"Learned covariance differs from batch:\n"
        f"  Online:\n{normalizer.Sigmahat}\n"
        f"  Batch:\n{batch_cov}"
    )


def test_mvnorm_transformation_vs_batch(correlated_data, fitted_normalizer):
    """Test that transformation matches batch calculation."""
    X, n_dim, true_mean, true_cov = correlated_data
    normalizer = fitted_normalizer

    # Batch calculation
    batch_mean = np.mean(X, axis=0)
    batch_cov = np.cov(X.T, ddof=1)
    batch_cov_inv = np.linalg.inv(batch_cov)
    batch_cov_invsqrt = np.linalg.cholesky(batch_cov_inv).T

    # Test on new data points
    rng = default_rng(42)
    x_test = rng.multivariate_normal(true_mean, true_cov, size=5)

    for x in x_test:
        # Online transformation
        x_online = normalizer.transform(x.copy())

        # Batch transformation
        x_batch = batch_cov_invsqrt @ (x - batch_mean)

        # Should be very close (within numerical precision)
        assert np.allclose(x_online, x_batch, rtol=1e-10, atol=1e-10), (
            f"Online and batch transformations differ:\n"
            f"  Online: {x_online}\n"
            f"  Batch:  {x_batch}\n"
            f"  Diff:   {x_online - x_batch}"
        )


def test_mvnorm_decorrelation(correlated_data, fitted_normalizer):
    """Test that transformed data has zero mean and identity covariance."""
    X, n_dim, _, _ = correlated_data
    normalizer = fitted_normalizer

    # Transform all data
    X_transformed = np.array([normalizer.transform(x.copy()) for x in X])

    # Check mean
    transformed_mean = np.mean(X_transformed, axis=0)
    assert np.allclose(
        transformed_mean, 0, atol=1e-10
    ), f"Transformed data should have zero mean, got {transformed_mean}"

    # Check covariance
    transformed_cov = np.cov(X_transformed.T, ddof=1)
    assert np.allclose(
        transformed_cov, np.eye(n_dim), atol=1e-10
    ), f"Transformed data should have identity covariance, got:\n{transformed_cov}"


def test_mvnorm_singular_covariance():
    """Test that MultivariateNormalizer handles singular covariance matrices."""
    n_dim = 3
    normalizer = MultivariateNormalizer(n_dim=n_dim)

    # Create data where all points are identical (singular covariance)
    x_constant = np.array([1.0, 2.0, 3.0])
    for _ in range(10):
        normalizer.partial_fit(x_constant.copy())

    # Covariance should be zero matrix (singular)
    assert np.allclose(normalizer.Sigmahat, 0), "Covariance should be zero for constant data"

    # invsqrtSigmahat should fall back to identity when Cholesky fails
    assert np.allclose(
        normalizer.invsqrtSigmahat, np.eye(n_dim)
    ), "Should return identity matrix for singular covariance"

    # Transform should still work (using identity)
    x_test = np.array([4.0, 5.0, 6.0])
    x_transformed = normalizer.transform(x_test.copy())
    assert np.all(np.isfinite(x_transformed)), "Should handle singular covariance gracefully"


def test_mvnorm_one_dimensional():
    """Test MultivariateNormalizer with 1-dimensional data."""
    n_dim = 1
    normalizer = MultivariateNormalizer(n_dim=n_dim)

    rng = default_rng(123)
    X = rng.normal(loc=5.0, scale=2.0, size=(50, 1))

    for x in X:
        normalizer.partial_fit(x)

    # Check mean estimation
    batch_mean = np.mean(X, axis=0)
    assert np.allclose(normalizer.muhat, batch_mean, rtol=1e-10)

    # Transform should work
    x_test = np.array([5.0])
    x_norm = normalizer.transform(x_test.copy())
    assert x_norm.shape == (1,)
    assert np.isfinite(x_norm[0])


def test_mvnorm_partial_fit_transform(correlated_data):
    """Test partial_fit_transform method."""
    X, n_dim, _, _ = correlated_data
    normalizer = MultivariateNormalizer(n_dim=n_dim)

    # Use partial_fit_transform for all but last point
    for x in X[:-1]:
        normalizer.partial_fit(x)

    # Test partial_fit_transform on last point
    x_last = X[-1].copy()
    x_transformed = normalizer.partial_fit_transform(x_last)

    # Should have updated the model
    assert normalizer.n == len(X)

    # Transform should produce valid output
    assert x_transformed.shape == (n_dim,)
    assert np.all(np.isfinite(x_transformed))


def test_mvnorm_small_sample_sizes():
    """Test MultivariateNormalizer with very small sample sizes."""
    n_dim = 3
    normalizer = MultivariateNormalizer(n_dim=n_dim)
    rng = default_rng(456)

    # Test with n=0 (before any fitting)
    assert normalizer.n == 0
    assert np.allclose(normalizer.Sigmahat, np.eye(n_dim))

    # Test with n=1
    x1 = rng.normal(size=n_dim)
    normalizer.partial_fit(x1)
    assert normalizer.n == 1
    # Should return identity for n <= 1
    assert np.allclose(normalizer.Sigmahat, np.eye(n_dim))

    # Transform should still work
    x_test = rng.normal(size=n_dim)
    x_norm = normalizer.transform(x_test.copy())
    assert np.all(np.isfinite(x_norm))

    # Test with n=2
    x2 = rng.normal(size=n_dim)
    normalizer.partial_fit(x2)
    assert normalizer.n == 2
    # Now should have actual covariance
    assert not np.allclose(normalizer.Sigmahat, np.eye(n_dim))


def test_mvnorm_negative_correlation():
    """Test MultivariateNormalizer with negatively correlated data."""
    rng = default_rng(789)
    n_dim = 2
    n_samples = 100

    # Create negatively correlated data
    true_cov = np.array([[1.0, -0.7], [-0.7, 1.0]])
    X = rng.multivariate_normal([0, 0], true_cov, size=n_samples)

    normalizer = MultivariateNormalizer(n_dim=n_dim)
    for x in X:
        normalizer.partial_fit(x)

    # Verify covariance is learned
    batch_cov = np.cov(X.T, ddof=1)
    assert np.allclose(normalizer.Sigmahat, batch_cov, rtol=1e-10)

    # Check that negative correlation is preserved in learned covariance
    assert normalizer.Sigmahat[0, 1] < 0
    assert normalizer.Sigmahat[1, 0] < 0


def test_mvnorm_properties_before_fitting():
    """Test accessing properties before any data is fitted."""
    n_dim = 4
    normalizer = MultivariateNormalizer(n_dim=n_dim)

    # Should have default values
    assert normalizer.n == 0
    assert np.allclose(normalizer.muhat, np.zeros(n_dim))
    assert np.allclose(normalizer.Sigmahat, np.eye(n_dim))
    assert np.allclose(normalizer.invsqrtSigmahat, np.eye(n_dim))

    # Transform with default state should just center (subtract zero mean)
    rng = default_rng(999)
    x = rng.normal(size=n_dim)
    x_copy = x.copy()
    x_transformed = normalizer.transform(x)
    assert np.allclose(x_transformed, x_copy)


def test_mvnorm_sequential_reset_and_refit():
    """Test resetting and refitting the normalizer multiple times."""
    n_dim = 3
    normalizer = MultivariateNormalizer(n_dim=n_dim)
    rng = default_rng(111)

    for cycle in range(3):
        # Generate new data for each cycle
        X = rng.normal(loc=cycle, scale=1.0, size=(20, n_dim))

        for x in X:
            normalizer.partial_fit(x)

        # Check that mean reflects current data
        assert np.allclose(normalizer.muhat, np.mean(X, axis=0), rtol=1e-10)

        # Reset for next cycle
        normalizer.reset()
        assert normalizer.n == 0


def test_mvnorm_high_dimensional():
    """Test MultivariateNormalizer with higher dimensional data."""
    n_dim = 10
    n_samples = 50
    rng = default_rng(222)

    X = rng.normal(size=(n_samples, n_dim))
    normalizer = MultivariateNormalizer(n_dim=n_dim)

    for x in X:
        normalizer.partial_fit(x)

    # Verify statistics match batch
    batch_mean = np.mean(X, axis=0)
    batch_cov = np.cov(X.T, ddof=1)

    assert np.allclose(normalizer.muhat, batch_mean, rtol=1e-10)
    assert np.allclose(normalizer.Sigmahat, batch_cov, rtol=1e-10)

    # Transform should decorrelate
    X_transformed = np.array([normalizer.transform(x.copy()) for x in X])
    transformed_cov = np.cov(X_transformed.T, ddof=1)
    assert np.allclose(transformed_cov, np.eye(n_dim), atol=1e-10)
