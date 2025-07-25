# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-07-25

### ✨ Features

- **📊 Graph Visualization System**: Comprehensive model graph visualization capabilities
  - `Model::print_graph()` method for beautiful terminal ASCII art visualization
  - `Model::to_dot()` method for professional Graphviz export
  - Dynamic title box sizing that adapts to any graph name length
  - Rich information display: shapes, data types, attributes, and statistics
  - Topological sorting with correct execution order
  - Cycle detection for robust graph handling
  - CLI options: `--graph` for terminal visualization, `--dot` for file export

- **🎨 Professional Diagram Generation**: Publication-quality graphics support
  - Graphviz DOT format export for professional diagrams
  - Support for PNG, SVG, and PDF output formats
  - Color-coded visual elements:
    - Green ellipses for input tensors
    - Blue diamonds for initializers (weights/biases)
    - Rectangular boxes for operations
    - Red ellipses for output tensors
    - Directed arrows showing data flow

- **🛠️ YOLO Model Support**: Essential operators for YOLO object detection
  - `Concat`: Tensor concatenation for feature fusion
  - `Slice`: Tensor slicing operations (simplified)
  - `Upsample`: Feature map upsampling for FPN (simplified)
  - `MaxPool`: Max pooling operations (simplified)
  - `Softmax`: Classification probability computation
  - `NonMaxSuppression`: Object detection post-processing (simplified)
  - Formal verification contracts for all YOLO operators

### 🚀 Enhancements

- **Terminal Visualization**: Rich ASCII art output with Unicode symbols and structured layout
- **CLI Integration**: Seamless command-line graph visualization workflow
- **Documentation**: Comprehensive examples showing terminal output and DOT format
- **Code Organization**: Clean separation of concerns with dedicated graph module
- **Testing**: Complete test coverage for graph functionality and visualization

### 📚 Documentation

- Enhanced README with complete graph visualization guide
- Terminal output examples for complex neural networks
- DOT format syntax explanation and best practices
- Asset management documentation
- Professional diagram generation workflow
- Programmatic usage examples

## [0.1.1] - 2025-07-24

### ✨ Features

- **🔧 ONNX Binary Format Support**: Full support for standard ONNX protobuf binary format
  - `Model::from_onnx_file()` and `Model::to_onnx_file()` methods
  - Automatic format detection based on file extension
  - Seamless conversion between JSON and ONNX binary formats
  - Smaller file sizes (~60% compression vs JSON)
  - Faster loading performance for binary format
  - Full interoperability with other ONNX runtime implementations

- **🧮 Formal Verification System**: Mathematical verification of operator correctness using Why3
  - Comprehensive Why3 specifications for all ONNX operators (Add, Mul, MatMul, ReLU, Sigmoid, Transpose, Reshape)
  - Automated verification pipeline with Alt-Ergo theorem prover
  - Runtime contract verification with formal assertions
  - Property-based testing for mathematical properties (commutativity, associativity, monotonicity)
  - Complete verification automation with Python integration

### 🚀 Enhancements

- **Format Auto-Detection**: `Model::from_file()` automatically detects JSON (.json) vs ONNX (.onnx) formats
- **Dual Format CLI**: Command-line runner now supports both JSON and ONNX binary model files
- **Comprehensive Examples**: Added `onnx_demo.rs` demonstrating format compatibility and conversion
- **Enhanced Documentation**: New format compatibility guide with best practices
- **Cross-Platform Compatibility**: Fixed Windows CI issues with platform-independent file paths
- **CI/CD Improvements**: Enhanced formal verification pipeline with proper dependency management
- **Test Coverage Enhancement**: Improved overall test coverage to 96.48% with comprehensive CLI testing
- **Workflow Consolidation**: Unified coverage pipeline into main CI workflow for better maintainability

### 🐛 Bug Fixes

- **Windows CI Compatibility**: Fixed hardcoded Unix paths causing Windows test failures
- **Protocol Buffers Support**: Added missing `protoc` installation in CI jobs
- **Formal Verification Integration**: Fixed conditional test compilation for formal verification features
- **Dependency Updates**: Updated GitHub Actions and resolved security vulnerabilities
- **CLI Binary Availability**: Fixed "No such file or directory" errors in CI by ensuring runner binary is built before coverage tests

### 📚 Documentation

- Added comprehensive [Format Compatibility Guide](docs/FORMAT_COMPATIBILITY.md)
- Updated README with dual format examples and usage patterns
- Improved code documentation with format-specific examples
- Complete formal verification documentation and examples

## [0.1.0] - 2025-07-21

### ✨ Features

- **Core ONNX Runtime**: Minimal, verifiable ONNX runtime implementation in Rust
- **Tensor Operations**: Support for Add, Mul, MatMul, Conv, ReLU, Sigmoid, Reshape, Transpose
- **Model Loading**: Load and validate ONNX models from files
- **Inference Engine**: Execute ONNX models with comprehensive error handling
- **CLI Runner**: Command-line tool for running ONNX models (`runnx-runner`)
- **Async Support**: Optional async execution with Tokio
- **Benchmarking**: Performance benchmarks using Criterion

### 🏗️ Infrastructure

- **CI/CD Pipeline**: Comprehensive GitHub Actions workflow
- **Multi-platform Testing**: Ubuntu, Windows, macOS support
- **Code Coverage**: Integrated with Codecov (97% coverage)
- **Documentation**: Complete API documentation with examples
- **Security Auditing**: Automated dependency security checks
- **MSRV Support**: Minimum Supported Rust Version 1.81

### 📦 Dependencies

- **ndarray 0.16**: High-performance n-dimensional arrays
- **thiserror 2.0**: Error handling and propagation
- **serde 1.0**: Serialization support
- **criterion 0.6**: Benchmarking framework
- **env_logger 0.11**: Flexible logging

### 🧪 Testing

- **115 Unit Tests**: Comprehensive test coverage
- **14 CLI Tests**: Command-line interface testing
- **11 Integration Tests**: End-to-end functionality testing
- **16 Documentation Tests**: Ensure examples work correctly

<!-- generated by git-cliff -->
