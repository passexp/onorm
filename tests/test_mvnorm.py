import numpy as np
import pytest
from numpy.random import default_rng
from scipy.linalg import sqrtm

from onorm import MultivariateNormalizer



def test_mvnorm_basic():
    """Test basic MultivariateNormalizer functionality."""
    rng = default_rng(2022)
    n_dim = 3
    normalizer = MultivariateNormalizer(n_dim=n_dim)

    X = rng.normal(size=(50, n_dim))
    for x in X:
        normalizer.partial_fit(x)

    # Should be able to transform data
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
    assert normalizer.invsqrtSigmahat.shape == (n_dim, n_dim), "Inv sqrt cov should have correct shape"


def test_mvnorm_correctness_vs_batch():
    """Test that MultivariateNormalizer produces same results as batch calculation.

    After fitting on data, the online normalizer should produce the same
    transformation as computing mean and covariance in batch mode.
    """
    rng = default_rng(42)
    n_dim = 3
    n_samples = 50

    # Generate correlated data
    true_mean = np.array([1.0, 2.0, 3.0])
    true_cov = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 1.5, 0.4],
        [0.3, 0.4, 1.0]
    ])
    X = rng.multivariate_normal(true_mean, true_cov, size=n_samples)

    # Fit online normalizer
    normalizer = MultivariateNormalizer(n_dim=n_dim)
    for x in X:
        normalizer.partial_fit(x)

    # Batch calculation: compute sample mean and covariance
    batch_mean = np.mean(X, axis=0)
    batch_cov = np.cov(X.T, ddof=1)  # Sample covariance (divide by n-1)

    # Batch transformation: Sigma^(-1/2) @ (x - mu)
    # We use Cholesky decomposition of the inverse covariance
    batch_cov_inv = np.linalg.inv(batch_cov)
    batch_cov_invsqrt = np.linalg.cholesky(batch_cov_inv).T

    # Test on new data points
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

    # Also verify the learned statistics match
    assert np.allclose(normalizer.muhat, batch_mean, rtol=1e-10), (
        f"Learned mean differs from batch mean:\n"
        f"  Online: {normalizer.muhat}\n"
        f"  Batch:  {batch_mean}"
    )

    # The covariance should match
    assert np.allclose(normalizer.Sigmahat, batch_cov, rtol=1e-10), (
        f"Learned covariance differs from batch:\n"
        f"  Online:\n{normalizer.Sigmahat}\n"
        f"  Batch:\n{batch_cov}"
    )

    # Test that transformed data has identity covariance
    X_transformed = np.array([normalizer.transform(x.copy()) for x in X])
    transformed_mean = np.mean(X_transformed, axis=0)
    transformed_cov = np.cov(X_transformed.T, ddof=1)

    # Mean should be near zero
    assert np.allclose(transformed_mean, 0, atol=1e-10), (
        f"Transformed data should have zero mean, got {transformed_mean}"
    )

    # Covariance should be near identity
    assert np.allclose(transformed_cov, np.eye(n_dim), atol=1e-10), (
        f"Transformed data should have identity covariance, got:\n{transformed_cov}"
    )
