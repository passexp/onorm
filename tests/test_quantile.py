import json

import numpy as np
import pytest
from numpy.random import default_rng
from onorm import QuantileTransformer


@pytest.fixture
def rng():
    """Provide a consistent random number generator."""
    return default_rng(2022)


@pytest.fixture
def sample_data(rng):
    """Generate sample data for testing."""
    n_dim = 3
    n_samples = 200
    # Use exponential distribution (skewed) to test quantile transformation
    X = rng.exponential(scale=2.0, size=(n_samples, n_dim))
    return X, n_dim


@pytest.fixture
def fitted_transformer(sample_data):
    """Create and fit a QuantileTransformer on sample data."""
    X, n_dim = sample_data
    qt = QuantileTransformer(n_dim=n_dim)
    for x in X:
        qt.partial_fit(x)
    return qt


def test_quantile_basic(rng):
    """Test basic QuantileTransformer functionality."""
    n_dim = 3
    qt = QuantileTransformer(n_dim=n_dim)

    # Fit on uniform data
    X = rng.uniform(low=0, high=10, size=(100, n_dim))
    for x in X:
        qt.partial_fit(x)

    # Transform should map to approximately uniform [0, 1]
    x_test = np.array([5.0, 5.0, 5.0])  # Middle values
    x_transformed = qt.transform(x_test.copy())

    # Should be around 0.5 for uniform data
    assert np.all(x_transformed >= 0.0)
    assert np.all(x_transformed <= 1.0)
    assert np.all(np.abs(x_transformed - 0.5) < 0.3)  # Roughly middle


def test_quantile_uniform_output(sample_data, fitted_transformer):
    """Test that output follows uniform distribution."""
    X, _ = sample_data
    qt = fitted_transformer

    # Transform all data
    X_transformed = np.array([qt.transform(x.copy()) for x in X])

    # Check that values are in [0, 1]
    assert np.all(X_transformed >= 0.0)
    assert np.all(X_transformed <= 1.0)

    # Check that distribution is roughly uniform (using mean and std)
    # Uniform[0,1] has mean=0.5, std=1/sqrt(12)≈0.289
    for i in range(X_transformed.shape[1]):
        assert np.abs(np.mean(X_transformed[:, i]) - 0.5) < 0.1
        assert np.abs(np.std(X_transformed[:, i]) - 0.289) < 0.1


def test_quantile_monotonic(rng):
    """Test that transformation is monotonic."""
    n_dim = 2
    qt = QuantileTransformer(n_dim=n_dim)

    # Fit on normal data
    X = rng.normal(loc=5, scale=2, size=(100, n_dim))
    for x in X:
        qt.partial_fit(x)

    # Test monotonicity: larger values should have larger CDF values
    x1 = np.array([3.0, 3.0])
    x2 = np.array([5.0, 5.0])
    x3 = np.array([7.0, 7.0])

    y1 = qt.transform(x1.copy())
    y2 = qt.transform(x2.copy())
    y3 = qt.transform(x3.copy())

    assert np.all(y1 < y2)
    assert np.all(y2 < y3)


def test_quantile_extreme_values(rng):
    """Test handling of extreme values."""
    n_dim = 2
    qt = QuantileTransformer(n_dim=n_dim)

    # Fit on data in range [0, 10]
    X = rng.uniform(low=0, high=10, size=(100, n_dim))
    for x in X:
        qt.partial_fit(x)

    # Very small value should map close to 0
    x_small = np.array([-100.0, -100.0])
    y_small = qt.transform(x_small.copy())
    assert np.all(y_small >= 0.0)
    assert np.all(y_small < 0.1)

    # Very large value should map close to 1
    x_large = np.array([100.0, 100.0])
    y_large = qt.transform(x_large.copy())
    assert np.all(y_large > 0.9)
    assert np.all(y_large <= 1.0)


def test_quantile_reset(rng):
    """Test that reset() clears the transformer state."""
    n_dim = 3
    qt = QuantileTransformer(n_dim=n_dim)

    # Fit with some data
    X = rng.normal(size=(50, n_dim))
    for x in X:
        qt.partial_fit(x)

    # Transform a value
    x_test = np.array([0.0, 0.0, 0.0])
    qt.transform(x_test.copy())

    # Reset
    qt.reset()

    # After reset, digests should be empty (no data seen)
    # TDigest with no data will still return values, but they'll be different
    assert all(digest.n_values == 0 for digest in qt.digests)


def test_quantile_normal_output(rng):
    """Test normal output distribution mode."""
    n_dim = 2
    qt = QuantileTransformer(n_dim=n_dim, output_distribution="normal")

    # Fit on uniform data
    X = rng.uniform(low=0, high=10, size=(200, n_dim))
    for x in X:
        qt.partial_fit(x)

    # Transform should map to approximately standard normal
    X_transformed = np.array([qt.transform(x.copy()) for x in X])

    # Check that distribution is roughly N(0,1)
    for i in range(X_transformed.shape[1]):
        assert np.abs(np.mean(X_transformed[:, i])) < 0.3
        assert np.abs(np.std(X_transformed[:, i]) - 1.0) < 0.3


def test_quantile_partial_fit_transform(rng):
    """Test partial_fit_transform method."""
    n_dim = 3
    qt = QuantileTransformer(n_dim=n_dim)

    x = rng.normal(size=n_dim)

    # First call partial_fit_transform
    y1 = qt.partial_fit_transform(x.copy())

    # Should be equivalent to partial_fit then transform
    qt2 = QuantileTransformer(n_dim=n_dim)
    qt2.partial_fit(x)
    y2 = qt2.transform(x.copy())

    assert np.allclose(y1, y2)


def test_quantile_invalid_distribution():
    """Test that invalid output_distribution raises error."""
    with pytest.raises(ValueError, match="output_distribution must be"):
        QuantileTransformer(n_dim=3, output_distribution="invalid")


def test_quantile_serialization_to_dict(fitted_transformer):
    """Test serialization to dictionary."""
    data = fitted_transformer.to_dict()

    # Check structure
    assert "version" in data
    assert "class" in data
    assert "config" in data
    assert "state" in data

    # Check values
    assert data["version"] == "1.0"
    assert data["class"] == "QuantileTransformer"
    assert data["config"]["n_dim"] == fitted_transformer.n_dim
    assert data["config"]["max_centroids"] == fitted_transformer.max_centroids
    assert data["config"]["output_distribution"] == fitted_transformer.output_distribution
    assert "digests" in data["state"]

    # Check that digests are serialized as dicts (not pickle)
    assert isinstance(data["state"]["digests"], list)
    assert len(data["state"]["digests"]) == fitted_transformer.n_dim
    assert isinstance(data["state"]["digests"][0], dict)
    assert "centroids" in data["state"]["digests"][0]


def test_quantile_serialization_roundtrip(rng):
    """Test that serialization and deserialization preserves state."""
    n_dim = 3
    qt = QuantileTransformer(n_dim=n_dim)

    # Fit with data
    X = rng.exponential(scale=1.5, size=(100, n_dim))
    for x in X:
        qt.partial_fit(x)

    # Transform with original
    x_test = rng.exponential(scale=1.5, size=n_dim)
    original_result = qt.transform(x_test.copy())

    # Serialize and deserialize
    data = qt.to_dict()
    restored_qt = QuantileTransformer.from_dict(data)

    # Transform with restored
    restored_result = restored_qt.transform(x_test.copy())

    # Results should be identical
    assert np.allclose(original_result, restored_result)

    # Config should be identical
    assert qt.n_dim == restored_qt.n_dim
    assert qt.max_centroids == restored_qt.max_centroids
    assert qt.output_distribution == restored_qt.output_distribution


def test_quantile_json_serialization(rng):
    """Test JSON string serialization."""
    n_dim = 3
    qt = QuantileTransformer(n_dim=n_dim, output_distribution="normal")

    # Fit with data
    X = rng.normal(size=(100, n_dim))
    for x in X:
        qt.partial_fit(x)

    # Transform with original
    x_test = rng.normal(size=n_dim)
    original_result = qt.transform(x_test.copy())

    # Serialize to JSON string
    json_str = qt.to_json()

    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert parsed["class"] == "QuantileTransformer"

    # Deserialize from JSON
    restored_qt = QuantileTransformer.from_json(json_str)

    # Transform with restored
    restored_result = restored_qt.transform(x_test.copy())

    # Results should be identical
    assert np.allclose(original_result, restored_result)


def test_quantile_serialization_empty():
    """Test serialization of unfitted transformer."""
    qt = QuantileTransformer(n_dim=3, max_centroids=200)

    # Should be able to serialize even if not fitted
    data = qt.to_dict()
    restored = QuantileTransformer.from_dict(data)

    # Config should be preserved
    assert restored.n_dim == 3
    assert restored.max_centroids == 200
    assert len(restored.digests) == 3


def test_quantile_deserialization_wrong_class():
    """Test that deserializing wrong class raises error."""
    data = {
        "version": "1.0",
        "class": "WrongClass",
        "config": {
            "n_dim": 3,
            "max_centroids": 1000,
            "output_distribution": "uniform",
        },
        "state": {"digests": []},
    }

    with pytest.raises(ValueError, match="Cannot deserialize"):
        QuantileTransformer.from_dict(data)


def test_quantile_serialization_preserves_cdfs(rng):
    """Test that serialization preserves CDF estimation."""
    n_dim = 2
    qt = QuantileTransformer(n_dim=n_dim)

    # Fit with known distribution
    X = rng.exponential(scale=1.0, size=(500, n_dim))
    for x in X:
        qt.partial_fit(x)

    # Get CDF values from original
    original_cdf_0 = qt.digests[0].cdf(1.0)
    original_cdf_1 = qt.digests[1].cdf(2.0)

    # Serialize and deserialize
    data = qt.to_dict()
    restored = QuantileTransformer.from_dict(data)

    # CDF values should be preserved
    restored_cdf_0 = restored.digests[0].cdf(1.0)
    restored_cdf_1 = restored.digests[1].cdf(2.0)

    assert np.allclose(original_cdf_0, restored_cdf_0)
    assert np.allclose(original_cdf_1, restored_cdf_1)


def test_quantile_serialization_no_pickle(rng):
    """Test that serialization does not use pickle."""
    n_dim = 2
    qt = QuantileTransformer(n_dim=n_dim)

    # Fit with data
    X = rng.normal(size=(50, n_dim))
    for x in X:
        qt.partial_fit(x)

    # Serialize to JSON
    json_str = qt.to_json()

    # Should be valid JSON (not base64-encoded pickle)
    parsed = json.loads(json_str)

    # Digests should be JSON objects, not base64 strings
    assert isinstance(parsed["state"]["digests"], list)
    assert isinstance(parsed["state"]["digests"][0], dict)
    assert "centroids" in parsed["state"]["digests"][0]

    # Centroids should be a list of dicts
    assert isinstance(parsed["state"]["digests"][0]["centroids"], list)
    if len(parsed["state"]["digests"][0]["centroids"]) > 0:
        assert isinstance(parsed["state"]["digests"][0]["centroids"][0], dict)


def test_quantile_different_output_modes(rng):
    """Test both output distribution modes."""
    n_dim = 2
    X = rng.exponential(scale=1.0, size=(100, n_dim))

    # Uniform output
    qt_uniform = QuantileTransformer(n_dim=n_dim, output_distribution="uniform")
    for x in X:
        qt_uniform.partial_fit(x)

    x_test = rng.exponential(scale=1.0, size=n_dim)
    y_uniform = qt_uniform.transform(x_test.copy())

    # Normal output
    qt_normal = QuantileTransformer(n_dim=n_dim, output_distribution="normal")
    for x in X:
        qt_normal.partial_fit(x)

    y_normal = qt_normal.transform(x_test.copy())

    # Uniform should be in [0, 1]
    assert np.all(y_uniform >= 0.0)
    assert np.all(y_uniform <= 1.0)

    # Normal should have a wider range
    assert y_normal.min() < y_uniform.min() or y_normal.max() > y_uniform.max()
