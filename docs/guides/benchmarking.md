# Performance Benchmarking Guide

This document describes the comprehensive benchmark suite for RunNX, which measures performance across different operation categories and use cases.

## Benchmark Structure

### 1. `inference_benchmark.rs` - Core Tensor Operations
Tests fundamental tensor operations across different sizes:

- **Tensor Operations**: Add, multiply, matrix multiplication with sizes 10×10 to 500×500
- **Activation Functions**: ReLU and Sigmoid with 100 to 10K elements
- **Tensor Creation**: Zeros, ones, and from_shape_vec operations
- **Shape Operations**: Reshape and transpose across different sizes
- **Model Inference**: Simple linear models with batch sizes 1-16

### 2. `operator_benchmark.rs` - ONNX Operator Performance
Benchmarks ONNX operator implementations:

- **Basic Operators**: Add, Mul, MatMul with various tensor sizes
- **Activation Operators**: ReLU, Sigmoid, Softmax
- **Shape Operators**: Reshape, Transpose, Concat
- **Pooling Operators**: MaxPool with different kernel sizes

### 3. `cv_benchmark.rs` - Computer Vision Workloads
Specialized benchmarks for computer vision tasks, particularly YOLO-style networks:

- **Computer Vision Operators**: 
  - Conv layers with different channel counts (3, 32, 64, 128)
  - MaxPool operations across YOLO feature map sizes (416→13)
  - Upsample/Resize for Feature Pyramid Networks

- **YOLO Post-processing**:
  - Softmax for classification heads (80, 91, 1000 classes)
  - Sigmoid for objectness/bbox regression
  - Non-Maximum Suppression with varying box counts

- **Batch Operations**:
  - Batch normalization across different batch sizes
  - Feature map concatenation for FPN

- **Shape Manipulation**:
  - Detection head flattening (13×13, 26×26, 52×52)
  - Transpose for data layout changes

## Running Benchmarks

### Individual Benchmarks
```bash
# Core tensor operations
cargo bench --bench inference_benchmark

# ONNX operators
cargo bench --bench operator_benchmark

# Computer vision workloads
cargo bench --bench cv_benchmark
```

### Complete Benchmark Suite
```bash
# Run all benchmarks with automated reporting
./scripts/benchmark.sh
```

### Specific Benchmark Groups
```bash
# Only tensor operations
cargo bench --bench inference_benchmark tensor_operations

# Only activation functions
cargo bench --bench inference_benchmark activation_functions

# Only YOLO post-processing
cargo bench --bench cv_benchmark yolo_postprocessing
```

## Benchmark Results

Results are automatically saved to `target/criterion/` with:
- HTML reports for detailed analysis
- CSV data for programmatic processing
- Performance trend tracking over time

### Key Metrics Tracked

1. **Throughput**: Operations per second
2. **Latency**: Time per operation (nanoseconds to microseconds)
3. **Scalability**: Performance across different input sizes
4. **Memory Usage**: Peak memory consumption during operations

## Interpreting Results

### Good Performance Indicators
- Linear scaling with input size for element-wise operations
- Sub-linear scaling for more complex operations (e.g., MatMul)
- Consistent performance across different batch sizes
- Low variance in repeated measurements

### Performance Regression Detection
- Monitor for >5% performance degradation
- Pay attention to outliers in timing measurements
- Check for memory allocation patterns
- Validate against baseline measurements

## Integration with CI/CD

The benchmark suite integrates with continuous integration:

```yaml
# Example CI integration
- name: Performance Benchmarks
  run: |
    cargo bench --bench inference_benchmark --message-format=json | tee bench_results.json
    # Process results and fail if regression > 10%
```

## Benchmark Configuration

### Hardware Recommendations
- Dedicated benchmark runner without background processes
- Consistent CPU frequency (disable turbo boost)
- Sufficient RAM to avoid swapping
- SSD storage for consistent I/O

### Environment Variables
```bash
export CRITERION_HOME="target/criterion"
export CARGO_TARGET_DIR="target"
# Reduce noise from system processes
export CARGO_INCREMENTAL=0
```

## Formal Verification Performance

The benchmark suite also measures the performance impact of formal verification features:

```bash
# Run with formal verification enabled
cargo bench --features formal-verification

# Compare against baseline
cargo bench --no-default-features
```

This helps quantify the overhead of runtime contract checking and ensures that formal verification doesn't significantly impact production performance.

## Benchmark Data Analysis

Results can be analyzed using the criterion tooling:

```bash
# Generate comparison reports
criterion-compare baseline current

# Extract performance metrics
cargo bench -- --output-format json
```

For continuous monitoring, integrate with performance tracking tools like:
- Prometheus + Grafana for time-series analysis
- Custom dashboards for regression detection
- Automated alerts for performance degradation
