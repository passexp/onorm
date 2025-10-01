# Online Normalization (onorm)

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code-of-conduct.md)
[![ci](https://github.com/ddimmery/onorm/actions/workflows/ci.yml/badge.svg)](https://github.com/ddimmery/onorm/actions/workflows/ci.yml)
![PyPI](https://img.shields.io/pypi/v/onorm)

A library for online (incremental/streaming) normalization of data, particularly useful for sequential experimentation and machine learning applications.

## Features

- **Online Normalization**: Update statistics incrementally without storing all data
- **Multiple Normalizers**:
  - `StandardScaler` - Z-score normalization (mean=0, std=1)
  - `MinMaxScaler` - Scale to [0, 1] range
  - `Winsorizer` - Clip outliers using quantiles
  - `MultivariateNormalizer` - Multivariate decorrelation
- **Pipeline Support**: Chain multiple normalizers together

## Installation

```bash
pip install onorm
```

## Quick Start

### StandardScaler - Z-score Normalization

```python
import numpy as np
from onorm import StandardScaler

# Create scaler for 3-dimensional data
scaler = StandardScaler(n_dim=3)

# Simulate streaming data
for i in range(100):
    x = np.random.normal(loc=5, scale=2, size=3)

    # Update statistics and transform
    x_normalized = scaler.partial_fit_transform(x)

    # x_normalized has mean ≈ 0 and std ≈ 1
```

### MinMaxScaler - Scale to [0, 1]

```python
from onorm import MinMaxScaler

scaler = MinMaxScaler(n_dim=3)

for i in range(100):
    x = np.random.normal(size=3)
    x_normalized = scaler.partial_fit_transform(x)

    # x_normalized is in range [0, 1]
```

### Pipeline - Combine Multiple Normalizers

```python
from onorm import Pipeline, StandardScaler, MinMaxScaler

# First standardize, then scale to [0, 1]
pipeline = Pipeline([
    StandardScaler(n_dim=3),
    MinMaxScaler(n_dim=3)
])

for i in range(100):
    x = np.random.normal(size=3)
    x_normalized = pipeline.partial_fit_transform(x)
```

## Documentation

For detailed documentation, visit [https://passexp.github.io/onorm/](https://passexp.github.io/onorm/)
