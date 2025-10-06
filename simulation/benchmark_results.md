# Performance Benchmark Report for onorm
**Generated:** 2025-10-05 21:40:13
**Random Seed:** 2022

---

## Executive Summary
This report benchmarks the computational performance of all normalizers in the `onorm` package to verify that:

1. Performance does not degrade substantially as sample size increases
2. All normalizers maintain O(1) or O(d²) per-observation complexity
3. Pipelines have expected overhead from component normalizers

### Sample Size Scaling (d=5, Normal)

Verifying that normalizers maintain constant per-observation complexity as sample size increases:

| Normalizer | n=100 (μs) | n=250 (μs) | n=500 (μs) | n=1,000 (μs) | n=2,500 (μs) | n=5,000 (μs) | n=10,000 (μs) | Ratio (max/min) | Constant? |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------------|----------|
| MinMaxScaler | 5.44 | 5.37 | 5.42 | 5.30 | 5.25 | 5.31 | 5.20 | 1.06 | ✓ Yes |
| MultivariateNormalizer | 24.05 | 22.54 | 22.04 | 21.81 | 21.75 | 21.62 | 21.65 | 1.11 | ✓ Yes |
| StandardScaler | 9.37 | 9.08 | 9.23 | 9.03 | 8.98 | 11.14 | 8.98 | 1.24 | ✓ Yes |
| Standard»MinMax | 24.48 | 23.89 | 24.14 | 23.99 | 23.78 | 23.72 | 24.02 | 1.03 | ✓ Yes |
| Standard»Multivariate | 55.15 | 54.02 | 54.38 | 54.17 | 53.86 | 53.53 | 53.93 | 1.04 | ✓ Yes |
| Winsorizer(5-95%) | 27.39 | 27.64 | 28.27 | 28.43 | 28.58 | 28.46 | 28.63 | 1.05 | ✓ Yes |

![Sample Size Scaling](scaling_sample_size.png)

### Dimensionality Scaling (n=500, Normal)

Performance across different dimensionalities:

| Normalizer | d=5 (μs) | d=10 (μs) | d=20 (μs) | d=50 (μs) |
|------------|----------|----------|----------|----------|
| MinMaxScaler | 5.41 | 5.31 | 5.43 | 5.39 |
| MultivariateNormalizer | 22.00 | 23.97 | 30.33 | 72.64 |
| StandardScaler | 9.12 | 9.10 | 9.90 | 9.31 |
| Standard»MinMax | 24.05 | 24.09 | 24.19 | 25.30 |
| Standard»Multivariate | 54.72 | 75.49 | 68.41 | 151.93 |
| Winsorizer(5-95%) | 28.25 | 55.10 | 107.33 | 265.17 |

![Dimensionality Scaling](scaling_dimensionality.png)

## Detailed Results

### Correlated Distribution (d=5)

| Normalizer | n | Mean (μs/obs) | Std (μs) | Median (μs/obs) | Time/1000 obs (ms) |
|------------|---|---------------|----------|-----------------|-------------------|
| MinMaxScaler | 500 | 6.35 | 0.27 | 6.28 | 6.35 |
| MultivariateNormalizer | 500 | 25.90 | 1.30 | 25.21 | 25.90 |
| StandardScaler | 500 | 11.01 | 0.41 | 11.10 | 11.01 |
| Standard»MinMax | 500 | 27.56 | 0.54 | 27.38 | 27.56 |
| Standard»Multivariate | 500 | 56.59 | 3.77 | 54.24 | 56.59 |
| Winsorizer(5-95%) | 500 | 32.45 | 0.49 | 32.33 | 32.45 |

### Mixed(95N+5C) Distribution (d=5)

| Normalizer | n | Mean (μs/obs) | Std (μs) | Median (μs/obs) | Time/1000 obs (ms) |
|------------|---|---------------|----------|-----------------|-------------------|
| MinMaxScaler | 500 | 5.30 | 0.14 | 5.20 | 5.30 |
| MultivariateNormalizer | 500 | 22.33 | 0.54 | 22.00 | 22.33 |
| StandardScaler | 500 | 9.10 | 0.15 | 9.10 | 9.10 |
| Standard»MinMax | 500 | 24.39 | 0.65 | 23.89 | 24.39 |
| Standard»Multivariate | 500 | 105.48 | 61.49 | 71.29 | 105.48 |
| Winsorizer(5-95%) | 500 | 28.38 | 0.33 | 28.27 | 28.38 |

### Normal Distribution (d=5)

| Normalizer | n | Mean (μs/obs) | Std (μs) | Median (μs/obs) | Time/1000 obs (ms) |
|------------|---|---------------|----------|-----------------|-------------------|
| MinMaxScaler | 100 | 5.44 | 0.18 | 5.40 | 5.44 |
| MinMaxScaler | 250 | 5.37 | 0.24 | 5.27 | 5.37 |
| MinMaxScaler | 500 | 5.31 | 0.24 | 5.21 | 5.31 |
| MinMaxScaler | 500 | 5.50 | 0.41 | 5.20 | 5.50 |
| MinMaxScaler | 500 | 5.42 | 0.12 | 5.44 | 5.42 |
| MinMaxScaler | 1,000 | 5.30 | 0.21 | 5.21 | 5.30 |
| MinMaxScaler | 2,500 | 5.25 | 0.06 | 5.22 | 5.25 |
| MinMaxScaler | 5,000 | 5.31 | 0.28 | 5.18 | 5.31 |
| MinMaxScaler | 10,000 | 5.20 | 0.07 | 5.18 | 5.20 |
| MultivariateNormalizer | 100 | 24.05 | 3.27 | 22.70 | 24.05 |
| MultivariateNormalizer | 250 | 22.54 | 0.87 | 22.20 | 22.54 |
| MultivariateNormalizer | 500 | 21.86 | 0.25 | 21.80 | 21.86 |
| MultivariateNormalizer | 500 | 22.09 | 0.41 | 22.22 | 22.09 |
| MultivariateNormalizer | 500 | 22.04 | 0.32 | 22.16 | 22.04 |
| MultivariateNormalizer | 1,000 | 21.81 | 0.24 | 21.74 | 21.81 |
| MultivariateNormalizer | 2,500 | 21.75 | 0.18 | 21.70 | 21.75 |
| MultivariateNormalizer | 5,000 | 21.62 | 0.17 | 21.60 | 21.62 |
| MultivariateNormalizer | 10,000 | 21.65 | 0.20 | 21.54 | 21.65 |
| StandardScaler | 100 | 9.37 | 0.45 | 9.23 | 9.37 |
| StandardScaler | 250 | 9.08 | 0.27 | 8.94 | 9.08 |
| StandardScaler | 500 | 9.04 | 0.16 | 8.96 | 9.04 |
| StandardScaler | 500 | 9.09 | 0.16 | 9.10 | 9.09 |
| StandardScaler | 500 | 9.23 | 0.20 | 9.18 | 9.23 |
| StandardScaler | 1,000 | 9.03 | 0.10 | 9.01 | 9.03 |
| StandardScaler | 2,500 | 8.98 | 0.06 | 8.97 | 8.98 |
| StandardScaler | 5,000 | 11.14 | 4.57 | 9.03 | 11.14 |
| StandardScaler | 10,000 | 8.98 | 0.08 | 8.97 | 8.98 |
| Standard»MinMax | 100 | 24.48 | 1.04 | 24.01 | 24.48 |
| Standard»MinMax | 250 | 23.89 | 0.49 | 23.75 | 23.89 |
| Standard»MinMax | 500 | 23.83 | 0.39 | 23.71 | 23.83 |
| Standard»MinMax | 500 | 24.18 | 0.39 | 23.94 | 24.18 |
| Standard»MinMax | 500 | 24.14 | 0.44 | 23.84 | 24.14 |
| Standard»MinMax | 1,000 | 23.99 | 0.24 | 23.90 | 23.99 |
| Standard»MinMax | 2,500 | 23.78 | 0.15 | 23.72 | 23.78 |
| Standard»MinMax | 5,000 | 23.72 | 0.20 | 23.65 | 23.72 |
| Standard»MinMax | 10,000 | 24.02 | 0.16 | 24.01 | 24.02 |
| Standard»Multivariate | 100 | 55.15 | 1.76 | 54.29 | 55.15 |
| Standard»Multivariate | 250 | 54.02 | 0.75 | 53.70 | 54.02 |
| Standard»Multivariate | 500 | 54.20 | 0.59 | 54.03 | 54.20 |
| Standard»Multivariate | 500 | 55.57 | 1.39 | 54.88 | 55.57 |
| Standard»Multivariate | 500 | 54.38 | 0.86 | 54.09 | 54.38 |
| Standard»Multivariate | 1,000 | 54.17 | 0.37 | 54.07 | 54.17 |
| Standard»Multivariate | 2,500 | 53.86 | 0.29 | 53.78 | 53.86 |
| Standard»Multivariate | 5,000 | 53.53 | 0.19 | 53.51 | 53.53 |
| Standard»Multivariate | 10,000 | 53.93 | 0.21 | 53.97 | 53.93 |
| Winsorizer(5-95%) | 100 | 27.39 | 1.50 | 26.96 | 27.39 |
| Winsorizer(5-95%) | 250 | 27.64 | 0.55 | 27.52 | 27.64 |
| Winsorizer(5-95%) | 500 | 28.12 | 0.25 | 28.12 | 28.12 |
| Winsorizer(5-95%) | 500 | 28.37 | 0.45 | 28.21 | 28.37 |
| Winsorizer(5-95%) | 500 | 28.27 | 0.37 | 28.15 | 28.27 |
| Winsorizer(5-95%) | 1,000 | 28.43 | 0.66 | 28.21 | 28.43 |
| Winsorizer(5-95%) | 2,500 | 28.58 | 0.13 | 28.55 | 28.58 |
| Winsorizer(5-95%) | 5,000 | 28.46 | 0.06 | 28.46 | 28.46 |
| Winsorizer(5-95%) | 10,000 | 28.63 | 0.06 | 28.62 | 28.63 |

### Normal Distribution (d=10)

| Normalizer | n | Mean (μs/obs) | Std (μs) | Median (μs/obs) | Time/1000 obs (ms) |
|------------|---|---------------|----------|-----------------|-------------------|
| MinMaxScaler | 500 | 5.31 | 0.18 | 5.19 | 5.31 |
| MultivariateNormalizer | 500 | 23.97 | 0.49 | 23.66 | 23.97 |
| StandardScaler | 500 | 9.10 | 0.15 | 9.09 | 9.10 |
| Standard»MinMax | 500 | 24.09 | 0.38 | 23.96 | 24.09 |
| Standard»Multivariate | 500 | 75.49 | 18.58 | 64.86 | 75.49 |
| Winsorizer(5-95%) | 500 | 55.10 | 0.38 | 55.08 | 55.10 |

### Normal Distribution (d=20)

| Normalizer | n | Mean (μs/obs) | Std (μs) | Median (μs/obs) | Time/1000 obs (ms) |
|------------|---|---------------|----------|-----------------|-------------------|
| MinMaxScaler | 500 | 5.43 | 0.25 | 5.31 | 5.43 |
| MultivariateNormalizer | 500 | 30.33 | 0.93 | 30.13 | 30.33 |
| StandardScaler | 500 | 9.90 | 0.50 | 9.77 | 9.90 |
| Standard»MinMax | 500 | 24.19 | 0.38 | 24.00 | 24.19 |
| Standard»Multivariate | 500 | 68.41 | 0.62 | 68.21 | 68.41 |
| Winsorizer(5-95%) | 500 | 107.33 | 0.74 | 107.05 | 107.33 |

### Normal Distribution (d=50)

| Normalizer | n | Mean (μs/obs) | Std (μs) | Median (μs/obs) | Time/1000 obs (ms) |
|------------|---|---------------|----------|-----------------|-------------------|
| MinMaxScaler | 500 | 5.39 | 0.15 | 5.35 | 5.39 |
| MultivariateNormalizer | 500 | 72.64 | 1.31 | 72.00 | 72.64 |
| StandardScaler | 500 | 9.31 | 0.20 | 9.21 | 9.31 |
| Standard»MinMax | 500 | 25.30 | 0.48 | 25.16 | 25.30 |
| Standard»Multivariate | 500 | 151.93 | 0.41 | 152.00 | 151.93 |
| Winsorizer(5-95%) | 500 | 265.17 | 1.14 | 265.31 | 265.17 |

### Uniform Distribution (d=5)

| Normalizer | n | Mean (μs/obs) | Std (μs) | Median (μs/obs) | Time/1000 obs (ms) |
|------------|---|---------------|----------|-----------------|-------------------|
| MinMaxScaler | 500 | 5.27 | 0.12 | 5.21 | 5.27 |
| MultivariateNormalizer | 500 | 22.18 | 0.41 | 22.04 | 22.18 |
| StandardScaler | 500 | 9.12 | 0.18 | 9.07 | 9.12 |
| Standard»MinMax | 500 | 24.05 | 0.53 | 23.73 | 24.05 |
| Standard»Multivariate | 500 | 54.09 | 0.62 | 53.82 | 54.09 |
| Winsorizer(5-95%) | 500 | 28.22 | 0.35 | 28.22 | 28.22 |

## Methodology

- **Timing:** `time.perf_counter()` for high-resolution measurements
- **Replications:** 5 runs with fresh data to reduce Monte Carlo error
- **Metrics:** Time per observation (microseconds), extrapolated time per 1000 observations
- **Constant Time Criterion:** Time ratio (max/min) < 1.5 across sample sizes

## Conclusions

✓ **All 6 normalizers passed the constant-time scaling test.**

All normalizers demonstrate efficient online performance suitable for streaming data applications.
