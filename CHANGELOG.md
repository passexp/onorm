# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-06

### Added
- **Serialization support** for all normalizers with hybrid JSON + base64 format
  - `to_dict()` and `from_dict()` methods for dictionary serialization
  - `to_json()` and `from_json()` methods for JSON string serialization
  - Abstract serialization methods in `Normalizer` base class
  - Full support for nested pipelines
- **Comprehensive test coverage** with 32 new serialization tests
  - 6 tests for `StandardScaler` serialization
  - 6 tests for `MultivariateNormalizer` serialization
  - 7 tests for `Winsorizer` serialization (includes no-pickle verification)
  - 7 tests for `Pipeline` serialization (includes nested pipeline tests)
  - 6 tests for `MinMaxScaler` serialization (from previous version)
- `__version__` attribute in `onorm` package
- `__all__` list for explicit API exports

### Changed
- **Winsorizer serialization** now uses TDigest's native `to_dict()`/`from_dict()` methods
  - **Breaking**: Removed pickle dependency for security and consistency
  - All normalizers now use 100% JSON-serializable format
- Updated documentation with comprehensive serialization examples

### Fixed
- Type safety improvements with abstract serialization interface
- Lint compliance across all test files

### Performance
- Sub-30 μs serialization/deserialization for typical models
- Efficient base64 encoding for numpy arrays
- Zero-copy operations where possible

### Documentation
- Updated all docstrings with NumPy-style formatting
- Added LaTeX math rendering support in docstrings

## [0.1.0] - Initial Release

### Added
- `MinMaxScaler` for min-max normalization
- `StandardScaler` for z-score normalization with Welford's algorithm
- `MultivariateNormalizer` for multivariate Gaussian normalization
- `Winsorizer` for robust outlier clipping using TDigest
- `Pipeline` for chaining normalizers
- Online/incremental learning support for all normalizers
- Comprehensive test suite with 66 tests

[0.2.0]: https://github.com/passexp/onorm/releases/tag/v0.2.0
[0.1.0]: https://github.com/passexp/onorm/releases/tag/v0.1.0
