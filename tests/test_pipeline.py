import json

import numpy as np
import pytest
from numpy.random import default_rng
from onorm import MinMaxScaler, MultivariateNormalizer, Pipeline, StandardScaler, Winsorizer


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


def test_pipeline_basic(sample_data):
    """Test basic Pipeline functionality."""
    X, n_dim = sample_data

    # Create pipeline: standardize then min-max scale
    pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])

    for x in X:
        pipeline.partial_fit(x)

    # Transform should produce finite results
    x_norm = pipeline.transform(X[-1])
    assert np.all(np.isfinite(x_norm)), "Should produce finite results"


def test_pipeline_order_matters(rng):
    """Test that order of normalizers in pipeline matters."""
    n_dim = 3
    X = rng.normal(loc=10, scale=5, size=(100, n_dim))

    # Different pipeline orders
    pipeline1 = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])
    pipeline2 = Pipeline([MinMaxScaler(n_dim=n_dim), StandardScaler(n_dim=n_dim)])

    for x in X:
        pipeline1.partial_fit(x)
        pipeline2.partial_fit(x)

    # Transform same point
    x_test = X[-1].copy()
    x_norm1 = pipeline1.transform(x_test.copy())
    x_norm2 = pipeline2.transform(x_test.copy())

    # Results should be different (order matters)
    assert not np.allclose(x_norm1, x_norm2), "Order should matter in pipeline"


def test_pipeline_single_normalizer(rng):
    """Test pipeline with single normalizer."""
    n_dim = 3
    pipeline = Pipeline([StandardScaler(n_dim=n_dim)])

    X = rng.normal(size=(50, n_dim))
    for x in X:
        pipeline.partial_fit(x)

    x_norm = pipeline.transform(X[-1])
    assert np.all(np.isfinite(x_norm)), "Should produce finite results"


def test_pipeline_three_normalizers(rng):
    """Test pipeline with three normalizers."""
    n_dim = 3

    # Pipeline: Winsorize outliers -> Standardize -> MinMax scale
    pipeline = Pipeline(
        [
            Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9)),
            StandardScaler(n_dim=n_dim),
            MinMaxScaler(n_dim=n_dim),
        ]
    )

    X = rng.normal(size=(100, n_dim))
    X[0] = [100, 100, 100]  # Add outlier

    for x in X:
        pipeline.partial_fit(x)

    x_norm = pipeline.transform(X[-1])
    assert np.all(np.isfinite(x_norm)), "Should handle three normalizers"


def test_pipeline_reset(rng):
    """Test that reset() resets all normalizers in pipeline."""
    n_dim = 3

    scaler1 = StandardScaler(n_dim=n_dim)
    scaler2 = MinMaxScaler(n_dim=n_dim)
    pipeline = Pipeline([scaler1, scaler2])

    # Fit with data
    X = rng.normal(size=(50, n_dim))
    for x in X:
        pipeline.partial_fit(x)

    # Verify state changed
    assert scaler1.n > 0, "StandardScaler should have data"
    assert not np.all(np.isinf(scaler2.min)), "MinMaxScaler should have data"

    # Reset and verify
    pipeline.reset()

    assert scaler1.n == 0, "StandardScaler should be reset"
    assert np.all(np.isinf(scaler2.min)), "MinMaxScaler should be reset"


def test_pipeline_partial_fit_transform(rng):
    """Test that partial_fit_transform works correctly."""
    n_dim = 3
    x = rng.normal(size=n_dim)

    # Compare partial_fit_transform with separate calls
    pipeline1 = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])
    pipeline1.partial_fit(x)
    x_separate = pipeline1.transform(x)

    pipeline2 = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])
    x_combined = pipeline2.partial_fit_transform(x)

    assert np.allclose(
        x_combined, x_separate, atol=1e-10
    ), "partial_fit_transform should match separate calls"


def test_pipeline_maintains_shape(rng):
    """Test that Pipeline maintains array shape."""
    n_dim = 5
    pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])

    X = rng.normal(size=(50, n_dim))
    for x in X:
        pipeline.partial_fit(x)

    x_test = rng.normal(size=n_dim)
    x_transformed = pipeline.transform(x_test)

    assert x_transformed.shape == x_test.shape, "Shape should be preserved"
    assert len(x_transformed) == n_dim, "Dimension should be preserved"


def test_pipeline_empty():
    """Test pipeline with no normalizers."""
    pipeline = Pipeline([])
    x = np.array([1.0, 2.0, 3.0])

    pipeline.partial_fit(x)
    x_transformed = pipeline.transform(x)

    # With no normalizers, output should equal input
    assert np.allclose(x_transformed, x), "Empty pipeline should return unchanged data"


def test_pipeline_consistent_transforms(sample_data):
    """Test that pipeline produces consistent transforms."""
    X, n_dim = sample_data
    pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])

    for x in X:
        pipeline.partial_fit(x)

    # Transform same point twice
    x_test = X[-1].copy()
    x_norm1 = pipeline.transform(x_test.copy())
    x_norm2 = pipeline.transform(x_test.copy())

    # Should be identical
    assert np.allclose(x_norm1, x_norm2), "Transform should be deterministic"


def test_pipeline_different_dimensions(rng):
    """Test Pipeline with different dimensions."""
    for n_dim in [2, 5, 10]:
        pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])
        X = rng.normal(size=(50, n_dim))

        for x in X:
            pipeline.partial_fit(x)

        x_norm = pipeline.transform(X[-1])

        assert len(x_norm) == n_dim, f"Should work with {n_dim} dimensions"
        assert np.all(np.isfinite(x_norm)), f"Should produce finite results for {n_dim}D"


def test_pipeline_winsorize_before_standardize(rng):
    """Test that outliers are winsorized before being passed to standardization.

    This test verifies that the Pipeline correctly applies transformations
    sequentially: each normalizer receives the transformed output from the
    previous stage, not the raw input.
    """
    n_dim = 3

    # Create dataset with outlier AFTER some normal data
    X = rng.normal(loc=0, scale=1, size=(100, n_dim))
    X[50] = [100.0, 100.0, 100.0]  # Outlier appears after 50 normal points

    # Pipeline: Winsorize -> Standardize
    pipeline = Pipeline([Winsorizer(n_dim=n_dim, clip_q=(0.05, 0.95)), StandardScaler(n_dim=n_dim)])

    for x in X:
        pipeline.partial_fit(x)

    # Check StandardScaler's learned statistics
    scaler = pipeline.normalizers[1]

    # Mean should be close to 0 (proper sequential transformation)
    assert np.all(np.abs(scaler.mean) < 5), (
        f"StandardScaler mean {scaler.mean} is too large. "
        "This suggests the outlier was not winsorized before standardization."
    )

    # Standard deviation should be reasonable
    std = np.sqrt(scaler.variance)
    assert np.all(std < 10), (
        f"StandardScaler std {std} is too large. "
        "This suggests the outlier was not winsorized before standardization."
    )

    # Verify sequential transformation by tracking inputs
    X_test = rng.normal(loc=0, scale=1, size=(50, n_dim))
    X_test[25] = [100.0, 100.0, 100.0]  # Another outlier

    pipeline2 = Pipeline(
        [Winsorizer(n_dim=n_dim, clip_q=(0.05, 0.95)), StandardScaler(n_dim=n_dim)]
    )

    # Track what the second normalizer sees
    inputs_to_scaler = []
    original_partial_fit = pipeline2.normalizers[1].partial_fit

    def tracked_partial_fit(x):
        inputs_to_scaler.append(x.copy())
        return original_partial_fit(x)

    pipeline2.normalizers[1].partial_fit = tracked_partial_fit

    for x in X_test:
        pipeline2.partial_fit(x)

    # Check that the outlier was clipped before reaching StandardScaler
    data_seen_by_scaler = inputs_to_scaler[25]

    assert np.all(data_seen_by_scaler < 50), (
        f"StandardScaler saw {data_seen_by_scaler} for the outlier, "
        "but it should have been clipped by Winsorizer first."
    )

    # Test transformation: outlier should be handled gracefully
    x_outlier = np.array([100.0, 100.0, 100.0])
    x_normalized = pipeline.transform(x_outlier.copy())

    assert np.all(np.isfinite(x_normalized)), "Should produce finite results"


def test_pipeline_serialization_to_dict(rng):
    """Test serialization to dictionary."""
    n_dim = 3
    pipeline = Pipeline([Winsorizer(n_dim=n_dim, clip_q=(0.1, 0.9)), StandardScaler(n_dim=n_dim)])

    # Fit with data
    X = rng.normal(size=(50, n_dim))
    for x in X:
        pipeline.partial_fit(x)

    data = pipeline.to_dict()

    # Check structure
    assert "version" in data
    assert "class" in data
    assert "config" in data
    assert "state" in data

    # Check values
    assert data["version"] == "1.0"
    assert data["class"] == "Pipeline"
    assert "normalizers" in data["state"]

    # Check that normalizers are serialized
    assert isinstance(data["state"]["normalizers"], list)
    assert len(data["state"]["normalizers"]) == 2
    assert data["state"]["normalizers"][0]["class"] == "Winsorizer"
    assert data["state"]["normalizers"][1]["class"] == "StandardScaler"


def test_pipeline_serialization_roundtrip(rng):
    """Test that serialization and deserialization preserves state."""
    n_dim = 3
    pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])

    # Fit with data
    X = rng.normal(loc=5, scale=2, size=(100, n_dim))
    for x in X:
        pipeline.partial_fit(x)

    # Transform with original pipeline
    x_test = rng.normal(loc=5, scale=2, size=n_dim)
    original_result = pipeline.transform(x_test.copy())

    # Serialize and deserialize
    data = pipeline.to_dict()
    restored_pipeline = Pipeline.from_dict(data)

    # Transform with restored pipeline
    restored_result = restored_pipeline.transform(x_test.copy())

    # Results should be identical
    assert np.allclose(original_result, restored_result)

    # Pipeline structure should be preserved
    assert len(restored_pipeline.normalizers) == len(pipeline.normalizers)


def test_pipeline_json_serialization(rng):
    """Test JSON string serialization."""
    n_dim = 3
    pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])

    # Fit with data
    X = rng.normal(size=(50, n_dim))
    for x in X:
        pipeline.partial_fit(x)

    # Transform with original pipeline
    x_test = rng.normal(size=n_dim)
    original_result = pipeline.transform(x_test.copy())

    # Serialize to JSON string
    json_str = pipeline.to_json()

    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert parsed["class"] == "Pipeline"

    # Deserialize from JSON
    restored_pipeline = Pipeline.from_json(json_str)

    # Transform with restored pipeline
    restored_result = restored_pipeline.transform(x_test.copy())

    # Results should be identical
    assert np.allclose(original_result, restored_result)


def test_pipeline_serialization_empty():
    """Test serialization of unfitted pipeline."""
    n_dim = 3
    pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])

    # Should be able to serialize even if not fitted
    data = pipeline.to_dict()
    restored = Pipeline.from_dict(data)

    # Structure should be preserved
    assert len(restored.normalizers) == 2
    assert isinstance(restored.normalizers[0], StandardScaler)
    assert isinstance(restored.normalizers[1], MinMaxScaler)


def test_pipeline_deserialization_wrong_class():
    """Test that deserializing wrong class raises error."""
    data = {
        "version": "1.0",
        "class": "WrongClass",
        "config": {},
        "state": {"normalizers": []},
    }

    with pytest.raises(ValueError, match="Cannot deserialize"):
        Pipeline.from_dict(data)


def test_pipeline_serialization_nested(rng):
    """Test serialization with nested pipelines."""
    n_dim = 3

    # Create nested pipeline
    inner_pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])
    outer_pipeline = Pipeline([Winsorizer(n_dim=n_dim), inner_pipeline])

    # Fit with data
    X = rng.normal(size=(50, n_dim))
    for x in X:
        outer_pipeline.partial_fit(x)

    # Transform with original
    x_test = rng.normal(size=n_dim)
    original_result = outer_pipeline.transform(x_test.copy())

    # Serialize and deserialize
    data = outer_pipeline.to_dict()
    restored = Pipeline.from_dict(data)

    # Transform with restored
    restored_result = restored.transform(x_test.copy())

    # Results should be identical
    assert np.allclose(original_result, restored_result)

    # Check structure
    assert len(restored.normalizers) == 2
    assert isinstance(restored.normalizers[0], Winsorizer)
    assert isinstance(restored.normalizers[1], Pipeline)
    assert len(restored.normalizers[1].normalizers) == 2


def test_pipeline_serialization_all_normalizers(rng):
    """Test serialization with all normalizer types."""
    n_dim = 3
    pipeline = Pipeline(
        [
            Winsorizer(n_dim=n_dim, clip_q=(0.05, 0.95)),
            StandardScaler(n_dim=n_dim, with_mean=True, with_std=True),
            MultivariateNormalizer(n_dim=n_dim),
            MinMaxScaler(n_dim=n_dim),
        ]
    )

    # Fit with data
    X = rng.normal(size=(100, n_dim))
    for x in X:
        pipeline.partial_fit(x)

    # Transform with original
    x_test = rng.normal(size=n_dim)
    original_result = pipeline.transform(x_test.copy())

    # Serialize and deserialize
    data = pipeline.to_dict()
    restored = Pipeline.from_dict(data)

    # Transform with restored
    restored_result = restored.transform(x_test.copy())

    # Results should be identical
    assert np.allclose(original_result, restored_result)

    # Check all normalizers are preserved
    assert len(restored.normalizers) == 4
    assert isinstance(restored.normalizers[0], Winsorizer)
    assert isinstance(restored.normalizers[1], StandardScaler)
    assert isinstance(restored.normalizers[2], MultivariateNormalizer)
    assert isinstance(restored.normalizers[3], MinMaxScaler)
