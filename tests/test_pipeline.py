import numpy as np
from numpy.random import default_rng

from onorm import MinMaxScaler, Pipeline, StandardScaler, Winsorizer


def test_pipeline_basic():
    """Test basic Pipeline functionality."""
    rng = default_rng(2022)
    n_dim = 3

    # Create pipeline: standardize then min-max scale
    pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])

    X = rng.normal(loc=5, scale=2, size=(100, n_dim))

    # Fit pipeline
    for x in X:
        pipeline.partial_fit(x)

    # Transform should work
    x_test = X[-1]
    x_norm = pipeline.transform(x_test)

    # Pipeline should produce finite results
    # Note: StandardScaler can produce values outside [0,1], then MinMax scales those
    # So final result depends on the standardized range seen during fitting
    assert np.all(np.isfinite(x_norm)), "Should produce finite results"


def test_pipeline_order_matters():
    """Test that order of normalizers in pipeline matters."""
    rng = default_rng(2022)
    n_dim = 3
    X = rng.normal(loc=10, scale=5, size=(100, n_dim))

    # Pipeline 1: Standard then MinMax
    pipeline1 = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])

    # Pipeline 2: MinMax then Standard
    pipeline2 = Pipeline([MinMaxScaler(n_dim=n_dim), StandardScaler(n_dim=n_dim)])

    # Fit both
    for x in X:
        pipeline1.partial_fit(x)
        pipeline2.partial_fit(x)

    # Transform same point
    x_test = X[-1].copy()
    x_norm1 = pipeline1.transform(x_test.copy())
    x_norm2 = pipeline2.transform(x_test.copy())

    # Results should be different (order matters)
    assert not np.allclose(x_norm1, x_norm2), "Order should matter in pipeline"


def test_pipeline_single_normalizer():
    """Test pipeline with single normalizer."""
    rng = default_rng(2022)
    n_dim = 3

    # Pipeline with single normalizer
    pipeline = Pipeline([StandardScaler(n_dim=n_dim)])

    X = rng.normal(size=(50, n_dim))

    for x in X:
        pipeline.partial_fit(x)

    x_test = X[-1]
    x_norm = pipeline.transform(x_test)

    # Should work like single normalizer
    assert np.all(np.isfinite(x_norm)), "Should produce finite results"


def test_pipeline_three_normalizers():
    """Test pipeline with three normalizers."""
    rng = default_rng(2022)
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

    x_test = X[-1]
    x_norm = pipeline.transform(x_test)

    # Should produce finite results
    assert np.all(np.isfinite(x_norm)), "Should handle three normalizers"


def test_pipeline_reset():
    """Test that reset() resets all normalizers in pipeline."""
    rng = default_rng(2022)
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

    # Reset
    pipeline.reset()

    # State should be reset
    assert scaler1.n == 0, "StandardScaler should be reset"
    assert np.all(np.isinf(scaler2.min)), "MinMaxScaler should be reset"


def test_pipeline_partial_fit_transform():
    """Test that partial_fit_transform works correctly."""
    rng = default_rng(2022)
    n_dim = 3

    x = rng.normal(size=n_dim)

    # partial_fit_transform should be equivalent to partial_fit then transform
    pipeline1 = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])
    pipeline1.partial_fit(x)
    x_norm_separate = pipeline1.transform(x)

    pipeline2 = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])
    x_norm_combined = pipeline2.partial_fit_transform(x)

    assert np.allclose(
        x_norm_combined, x_norm_separate, atol=1e-10
    ), "partial_fit_transform should match separate calls"


def test_pipeline_maintains_shape():
    """Test that Pipeline maintains array shape."""
    rng = default_rng(2022)
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

    # Should handle empty pipeline gracefully
    pipeline.partial_fit(x)
    x_transformed = pipeline.transform(x)

    # With no normalizers, output should equal input
    assert np.allclose(x_transformed, x), "Empty pipeline should return unchanged data"


def test_pipeline_consistent_transforms():
    """Test that pipeline produces consistent transforms."""
    rng = default_rng(2022)
    n_dim = 3

    pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])

    # Fit with data
    X = rng.normal(size=(100, n_dim))
    for x in X:
        pipeline.partial_fit(x)

    # Transform same point twice
    x_test = X[-1].copy()
    x_norm1 = pipeline.transform(x_test.copy())
    x_norm2 = pipeline.transform(x_test.copy())

    # Should be identical
    assert np.allclose(x_norm1, x_norm2), "Transform should be deterministic"


def test_pipeline_different_dimensions():
    """Test Pipeline with different dimensions."""
    rng = default_rng(2022)

    for n_dim in [2, 5, 10]:
        pipeline = Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)])
        X = rng.normal(size=(50, n_dim))

        for x in X:
            pipeline.partial_fit(x)

        x_test = X[-1]
        x_norm = pipeline.transform(x_test)

        assert len(x_norm) == n_dim, f"Should work with {n_dim} dimensions"
        assert np.all(np.isfinite(x_norm)), f"Should produce finite results for {n_dim}D"


def test_pipeline_winsorize_before_standardize():
    """Test that outliers are winsorized before being passed to standardization.

    This test verifies that the Pipeline correctly applies transformations
    sequentially: each normalizer receives the transformed output from the
    previous stage, not the raw input.
    """
    rng = default_rng(2022)
    n_dim = 3

    # Create dataset with outlier AFTER some normal data
    # This is important because winsorizer needs context to identify outliers
    X = rng.normal(loc=0, scale=1, size=(100, n_dim))
    X[50] = [100.0, 100.0, 100.0]  # Outlier appears after 50 normal points

    # Pipeline: Winsorize -> Standardize
    # Winsorizer should clip the outlier before StandardScaler sees it
    pipeline = Pipeline([
        Winsorizer(n_dim=n_dim, clip_q=(0.05, 0.95)),
        StandardScaler(n_dim=n_dim)
    ])

    # Fit the pipeline
    for x in X:
        pipeline.partial_fit(x)

    # Check StandardScaler's learned statistics
    # If pipeline works correctly, StandardScaler should have learned from
    # winsorized data, so its mean and std should be reasonable
    scaler = pipeline.normalizers[1]  # Get the StandardScaler

    # Mean should be close to 0 (the data mean)
    # With proper sequential transformation, the outlier gets clipped
    # before affecting StandardScaler too much
    assert np.all(np.abs(scaler.mean) < 5), (
        f"StandardScaler mean {scaler.mean} is too large. "
        "This suggests the outlier was not winsorized before standardization."
    )

    # Standard deviation should be reasonable (around 1-2 for normal data)
    std = np.sqrt(scaler.variance)
    assert np.all(std < 10), (
        f"StandardScaler std {std} is too large. "
        "This suggests the outlier was not winsorized before standardization."
    )

    # Now test that the pipeline applies transformations sequentially
    # We can verify this by checking that each normalizer processes transformed data
    X_test = rng.normal(loc=0, scale=1, size=(50, n_dim))
    X_test[25] = [100.0, 100.0, 100.0]  # Another outlier

    pipeline2 = Pipeline([
        Winsorizer(n_dim=n_dim, clip_q=(0.05, 0.95)),
        StandardScaler(n_dim=n_dim)
    ])

    # Track what the second normalizer sees by capturing its inputs
    inputs_to_scaler = []
    original_partial_fit = pipeline2.normalizers[1].partial_fit

    def tracked_partial_fit(x):
        inputs_to_scaler.append(x.copy())
        return original_partial_fit(x)

    pipeline2.normalizers[1].partial_fit = tracked_partial_fit

    for x in X_test:
        pipeline2.partial_fit(x)

    # The StandardScaler should see transformed data, not raw data
    # Check that the outlier was clipped before reaching StandardScaler
    outlier_idx = 25
    data_seen_by_scaler = inputs_to_scaler[outlier_idx]

    # The outlier (100, 100, 100) should have been clipped to something much smaller
    assert np.all(data_seen_by_scaler < 50), (
        f"StandardScaler saw {data_seen_by_scaler} for the outlier, "
        "but it should have been clipped by Winsorizer first. "
        "This indicates the pipeline is NOT applying transformations sequentially."
    )

    # Test transformation: outlier should be handled gracefully
    x_outlier = np.array([100.0, 100.0, 100.0])
    x_normalized = pipeline.transform(x_outlier.copy())

    # After winsorizing and standardizing, should produce reasonable values
    assert np.all(np.isfinite(x_normalized)), "Should produce finite results"
