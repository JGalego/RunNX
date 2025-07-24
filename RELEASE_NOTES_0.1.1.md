# RunNX v0.1.1 Release Notes

## 🎉 Release Highlights

RunNX v0.1.1 is a major enhancement release focusing on **ONNX binary format support**, **formal verification**, and **comprehensive testing**. This release establishes RunNX as a mathematically verifiable ONNX runtime with excellent test coverage and robust CI/CD.

## 📊 Release Metrics

- **✅ 242 Tests Passing** (All green!)
- **📈 96.48% Code Coverage** (Excellent quality)
- **🔧 Zero Clippy Warnings** (Clean codebase)
- **🚀 Enhanced CI/CD Pipeline** (Reliable automation)

## ✨ New Features

### 🔧 ONNX Binary Format Support
- **Full ONNX protobuf binary format support** with automatic format detection
- **Seamless conversion** between JSON and ONNX binary formats
- **~60% smaller file sizes** compared to JSON format
- **Faster loading performance** for binary format
- **Complete interoperability** with other ONNX runtime implementations

### 🧮 Formal Verification System
- **Mathematical verification** of operator correctness using Why3
- **Comprehensive specifications** for all ONNX operators (Add, Mul, MatMul, ReLU, Sigmoid, Transpose, Reshape)
- **Automated verification pipeline** with Alt-Ergo theorem prover
- **Runtime contract verification** with formal assertions
- **Property-based testing** for mathematical properties (commutativity, associativity, monotonicity)

## 🚀 Enhancements

### Format Auto-Detection
- `Model::from_file()` automatically detects JSON (.json) vs ONNX (.onnx) formats
- Dual format CLI runner supporting both JSON and ONNX binary model files
- Comprehensive examples demonstrating format compatibility and conversion

### Test Coverage Improvements
- **Enhanced test coverage to 96.48%** with 242 comprehensive tests
- **Comprehensive CLI testing** with 13 CLI runner tests
- **36 converter tests** covering all format conversion scenarios
- **Property-based formal verification tests** ensuring mathematical correctness

### CI/CD Pipeline Enhancements
- **Workflow consolidation** - unified coverage pipeline into main CI workflow
- **CLI binary availability fix** - ensures runner binary is built before coverage tests
- **Cross-platform compatibility** - fixed Windows CI issues with platform-independent paths
- **Enhanced formal verification pipeline** with proper dependency management

## 🐛 Bug Fixes

- **CLI Binary Availability**: Fixed "No such file or directory" errors in CI by ensuring runner binary is built before coverage tests
- **Windows CI Compatibility**: Fixed hardcoded Unix paths causing Windows test failures
- **Protocol Buffers Support**: Added missing `protoc` installation in CI jobs
- **Formal Verification Integration**: Fixed conditional test compilation for formal verification features
- **Dependency Updates**: Updated GitHub Actions and resolved security vulnerabilities

## 📚 Documentation

- Added comprehensive [Format Compatibility Guide](docs/FORMAT_COMPATIBILITY.md)
- Updated README with dual format examples and usage patterns
- Complete formal verification documentation and examples
- Improved code documentation with format-specific examples

## 🚀 Migration Guide

### From v0.1.0 to v0.1.1

#### Update Dependencies
```toml
[dependencies]
runnx = "0.1.1"  # Updated from 0.1.0
```

#### New API Methods
```rust
// New ONNX binary format methods
let model = Model::from_onnx_file("model.onnx")?;
model.to_onnx_file("output.onnx")?;

// Auto-detection (recommended)
let model = Model::from_file("model.onnx")?;  // Auto-detects format
model.to_file("output.onnx")?;              // Auto-detects format
```

#### CLI Usage
```bash
# Now supports both formats
cargo run --bin runnx-runner -- --model model.onnx --input input.json
cargo run --bin runnx-runner -- --model model.json --input input.json
```

## 🎯 Performance Improvements

- **Binary ONNX format**: ~60% smaller file sizes and faster loading
- **Optimized test suite**: 242 tests complete in ~40 seconds
- **Enhanced CI pipeline**: More reliable and faster builds

## 🔮 Looking Ahead

The next release (v0.2.0) will focus on:
- Additional ONNX operators (Softmax, BatchNorm, etc.)
- GPU acceleration support
- Quantization support
- Model optimization passes

## 🙏 Acknowledgments

Special thanks to the Rust and ONNX communities for their excellent tools and documentation that made this release possible.

---

**Full Changelog**: https://github.com/JGalego/runnx/compare/v0.1.0...v0.1.1
