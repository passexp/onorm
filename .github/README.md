# Online Normalization (onorm)

[![PyPI version](https://img.shields.io/pypi/v/onorm)](https://pypi.org/project/onorm/)
[![PyPI status](https://img.shields.io/pypi/status/onorm)](https://pypi.org/project/onorm/)
[![Python versions](https://img.shields.io/pypi/pyversions/onorm)](https://pypi.org/project/onorm/)
[![Downloads](https://static.pepy.tech/badge/onorm)](https://pepy.tech/project/onorm)
[![License](https://img.shields.io/pypi/l/onorm)](https://github.com/passexp/onorm/blob/main/LICENSE)
[![CI](https://github.com/ddimmery/onorm/actions/workflows/ci.yml/badge.svg)](https://github.com/ddimmery/onorm/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://passexp.github.io/onorm/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code-of-conduct.md)

`onorm` provides online (incremental) normalization algorithms for streaming data. These normalizers update their statistics incrementally without storing historical data, making them suitable for large-scale or real-time applications.

## Installation

```bash
pip install onorm
```

## Features

- **StandardScaler**: Online standardization (z-score normalization)
- **MinMaxScaler**: Online min-max scaling to [0, 1]
- **Winsorizer**: Online outlier clipping using quantiles
- **QuantileTransformer**: Transform features to uniform or normal distribution via marginal CDFs
- **MultivariateNormalizer**: Online decorrelation and standardization
- **Pipeline**: Chain multiple normalizers sequentially

All normalizers support:

- Incremental updates via `partial_fit()`
- Transformation via `transform()`
- Combined operation via `partial_fit_transform()`
- State reset via `reset()`
- Serialization via `to_dict()` / `from_dict()` and `to_json()` / `from_json()`

## Usage Example

Let's compare online normalization with and without outlier handling. We'll process a stream of data points and track how well each approach maintains normalized statistics.


```python
import numpy as np
import pandas as pd
from numpy.random import default_rng
from onorm import Pipeline, StandardScaler, Winsorizer
from plotnine import aes, geom_line, geom_vline, ggplot, labs, theme, theme_minimal

rng = default_rng(2024)
```


```python
# Generate streaming data with outliers
n_samples = 1000
n_dim = 5

X = rng.normal(loc=10, scale=1, size=(n_samples, n_dim))

# Add some outliers
outlier_indices = [100, 250, 500, 750]
for idx in outlier_indices:
    X[idx] = rng.uniform(-100, 100, size=n_dim)

print(f"Generated {n_samples} samples with {len(outlier_indices)} outliers")
```

    Generated 1000 samples with 4 outliers



```python
# Approach 1: StandardScaler only (sensitive to outliers)
scaler_only = StandardScaler(n_dim=n_dim)

# Approach 2: Pipeline with Winsorizer + StandardScaler (robust to outliers)
pipeline = Pipeline([Winsorizer(n_dim=n_dim, clip_q=(0.05, 0.95)), StandardScaler(n_dim=n_dim)])

# Track mean estimates over time
scaler_means = []
pipeline_means = []

for x in X:
    scaler_only.partial_fit(x)
    pipeline.partial_fit(x)

    scaler_means.append(scaler_only.mean[0])
    pipeline_means.append(pipeline.normalizers[1].mean[0])

print(f"StandardScaler final mean: {scaler_only.mean[0]:.2f}")
print(f"Pipeline final mean: {pipeline.normalizers[1].mean[0]:.2f}")
```

    StandardScaler final mean: 9.84
    Pipeline final mean: 10.02


### Visualization

The plot shows how the estimated mean evolves as data streams in. The pipeline with winsorization maintains stable estimates when outliers appear (red lines), while the standard scaler is more affected by extreme values.


```python
# Prepare data for plotting
true_mean = X[~np.isin(np.arange(len(X)), outlier_indices), 0].mean()

df = pd.DataFrame(
    {
        "Sample": range(n_samples),
        "StandardScaler": scaler_means,
        "Pipeline": pipeline_means,
        "True Mean": true_mean,
    }
)

df_long = pd.melt(df, id_vars=["Sample"], var_name="Method", value_name="Estimated Mean")

# Plot
(
    ggplot(df_long, aes(x="Sample", y="Estimated Mean", color="Method"))
    + geom_line()
    + geom_vline(xintercept=outlier_indices, color="red", alpha=0.3)
    + labs(title="Mean Estimation Over Time", x="Sample Index", y="Estimated Mean")
    + theme_minimal()
    + theme(legend_position="bottom")
)
```


    
![png](README_files/README_6_0.png)
    


### Key Takeaways

- **Online Learning**: All normalizers update incrementally without storing historical data
- **Robustness**: Use `Pipeline` with `Winsorizer` to handle outliers in streaming data
- **Efficiency**: Memory footprint remains constant regardless of stream length
- **Flexibility**: Mix and match normalizers to build custom preprocessing pipelines

For more details, see the [documentation](https://passexp.github.io/onorm/).

## QuantileTransformer: Handling Non-Normal Distributions

The `QuantileTransformer` uses incremental quantile estimation to transform features to a uniform [0,1] or normal distribution, making it ideal for skewed or heavy-tailed data streams.


```python
from onorm import QuantileTransformer

# Generate skewed data (log-normal distribution)
X_skewed = rng.lognormal(mean=0, sigma=1, size=(500, 3))

# Create transformer to map to uniform distribution
qt_uniform = QuantileTransformer(n_dim=3, output_distribution="uniform")

# Create transformer to map to normal distribution
qt_normal = QuantileTransformer(n_dim=3, output_distribution="normal")

# Process stream
X_uniform = []
X_normal = []

for x in X_skewed:
    qt_uniform.partial_fit(x)
    qt_normal.partial_fit(x)
    X_uniform.append(qt_uniform.transform(x.copy()))
    X_normal.append(qt_normal.transform(x.copy()))

X_uniform = np.array(X_uniform)
X_normal = np.array(X_normal)

print(f"Original data - Mean: {X_skewed[:, 0].mean():.2f}, Std: {X_skewed[:, 0].std():.2f}")
print(f"Uniform transform - Mean: {X_uniform[:, 0].mean():.2f}, Std: {X_uniform[:, 0].std():.2f}")
print(f"Normal transform - Mean: {X_normal[:, 0].mean():.2f}, Std: {X_normal[:, 0].std():.2f}")
```

    Original data - Mean: 1.70, Std: 2.14
    Uniform transform - Mean: 0.48, Std: 0.29
    Normal transform - Mean: -0.08, Std: 1.68


## Serialization: Save and Load Normalizer State

All normalizers support efficient serialization to JSON, making it easy to save trained models and deploy them in production systems.


```python
# Train a pipeline on streaming data
trained_pipeline = Pipeline([
    Winsorizer(n_dim=3, clip_q=(0.05, 0.95)),
    StandardScaler(n_dim=3)
])

training_data = rng.normal(loc=5, scale=2, size=(100, 3))
for x in training_data:
    trained_pipeline.partial_fit(x)

# Serialize to JSON
json_state = trained_pipeline.to_json()
print(f"Serialized size: {len(json_state)} bytes")

# Later: deserialize and use
restored_pipeline = Pipeline.from_json(json_state)

# Verify it produces the same results
test_point = rng.normal(loc=5, scale=2, size=3)
original_result = trained_pipeline.transform(test_point.copy())
restored_result = restored_pipeline.transform(test_point.copy())

print(f"Results match: {np.allclose(original_result, restored_result)}")
```

    Serialized size: 33191 bytes
    Results match: True

