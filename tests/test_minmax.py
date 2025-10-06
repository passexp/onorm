import json

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


def test_minmax_serialization_to_dict(fitted_scaler):
    """Test serialization to dictionary."""
    data = fitted_scaler.to_dict()

    # Check structure
    assert "version" in data
    assert "class" in data
    assert "config" in data
    assert "state" in data

    # Check values
    assert data["version"] == "1.0"
    assert data["class"] == "MinMaxScaler"
    assert data["config"]["n_dim"] == fitted_scaler.n_dim
    assert "min" in data["state"]
    assert "max" in data["state"]

    # Check that state arrays are base64 encoded strings
    assert isinstance(data["state"]["min"], str)
    assert isinstance(data["state"]["max"], str)


def test_minmax_serialization_roundtrip(fitted_scaler, sample_data):
    """Test that serialization and deserialization preserves state."""
    X, _ = sample_data

    # Transform with original scaler
    x_test = X[-1]
    original_result = fitted_scaler.transform(x_test.copy())

    # Serialize and deserialize
    data = fitted_scaler.to_dict()
    restored_scaler = MinMaxScaler.from_dict(data)

    # Transform with restored scaler
    restored_result = restored_scaler.transform(x_test.copy())

    # Results should be identical
    assert np.allclose(original_result, restored_result)

    # State should be identical
    assert np.allclose(fitted_scaler.min, restored_scaler.min)
    assert np.allclose(fitted_scaler.max, restored_scaler.max)
    assert fitted_scaler.n_dim == restored_scaler.n_dim


def test_minmax_json_serialization(fitted_scaler, sample_data):
    """Test JSON string serialization."""
    X, _ = sample_data

    # Transform with original scaler
    x_test = X[-1]
    original_result = fitted_scaler.transform(x_test.copy())

    # Serialize to JSON string
    json_str = fitted_scaler.to_json()

    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert parsed["class"] == "MinMaxScaler"

    # Deserialize from JSON
    restored_scaler = MinMaxScaler.from_json(json_str)

    # Transform with restored scaler
    restored_result = restored_scaler.transform(x_test.copy())

    # Results should be identical
    assert np.allclose(original_result, restored_result)


def test_minmax_serialization_empty_scaler():
    """Test serialization of unfitted scaler."""
    scaler = MinMaxScaler(n_dim=3)

    # Should be able to serialize even if not fitted
    data = scaler.to_dict()
    restored = MinMaxScaler.from_dict(data)

    # State should be preserved (inf/-inf)
    assert np.all(np.isinf(restored.min))
    assert np.all(np.isneginf(restored.max))
    assert restored.n_dim == 3


def test_minmax_deserialization_wrong_class():
    """Test that deserializing wrong class raises error."""
    data = {
        "version": "1.0",
        "class": "WrongClass",
        "config": {"n_dim": 3},
        "state": {"min": "", "max": ""},
    }

    with pytest.raises(ValueError, match="Cannot deserialize"):
        MinMaxScaler.from_dict(data)


def test_minmax_serialization_preserves_extremes(rng):
    """Test that serialization preserves edge cases."""
    n_dim = 3
    scaler = MinMaxScaler(n_dim=n_dim)

    # Fit with data including extreme values
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [100.0, -50.0, 0.0],
            [1.0, 2.0, 3.0],  # Back to normal
        ]
    )
    for x in X:
        scaler.partial_fit(x)

    # Serialize and deserialize
    data = scaler.to_dict()
    restored = MinMaxScaler.from_dict(data)

    # Test that extreme ranges are preserved
    x_test = np.array([50.0, 0.0, 1.5])
    original = scaler.transform(x_test.copy())
    restored_result = restored.transform(x_test.copy())

    assert np.allclose(original, restored_result)
