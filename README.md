# RunNX

A minimal, **mathematically verifiable** ONNX runtime implementation in Rust.

[![Crates.io](https://img.shields.io/crates/v/runnx.svg)](https://crates.io/crates/runnx)
[![Documentation](https://docs.rs/runnx/badge.svg)](https://docs.rs/runnx)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/jgalego/runnx/actions/workflows/ci.yml/badge.svg)](https://github.com/jgalego/runnx/actions/workflows/ci.yml)
[![Formal Verification](https://github.com/jgalego/runnx/actions/workflows/formal-verification.yml/badge.svg)](https://github.com/jgalego/runnx/actions/workflows/formal-verification.yml)
[![codecov](https://codecov.io/gh/jgalego/runnx/branch/main/graph/badge.svg)](https://codecov.io/gh/jgalego/runnx)

![RunNX](runnx.jpg)

## Overview

> Fast, fearless, and **formally verified** ONNX in Rust.

This project provides a minimal, educational ONNX runtime implementation focused on:
- **Simplicity**: Easy to understand and modify
- **Verifiability**: **Formal mathematical verification** using Why3 and property-based testing
- **Performance**: Efficient operations using ndarray
- **Safety**: Memory-safe Rust implementation with mathematical guarantees

## Features

- ✅ Basic tensor operations (`Add`, `Mul`, `MatMul`, `Conv`, &c.)
- ✅ **Formal mathematical specifications** with Why3
- ✅ **Property-based testing** for mathematical correctness
- ✅ **Runtime invariant verification**
- ✅ Model loading and validation  
- ✅ Inference execution
- ✅ Error handling and logging
- ✅ Benchmarking support
- ✅ Async support (optional)
- ✅ Command-line runner
- ✅ Comprehensive examples

## Quick Start

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
runnx = "0.1.0"
```

### Basic Usage

```rust
use runnx::{Model, Tensor};

// Load a model
let model = Model::from_file("model.onnx")?;

// Create input tensor
let input = Tensor::from_array(ndarray::array![[1.0, 2.0, 3.0]]);

// Run inference
let outputs = model.run(&[("input", input)])?;

// Get results
let result = outputs.get("output").unwrap();
println!("Result: {:?}", result.data());
```

### Command Line Usage

```bash
# Run inference on a model
cargo run --bin runnx-runner -- --model model.onnx --input input.json

# Run with async support
cargo run --features async --bin runnx-runner -- --model model.onnx --input input.json
```

## Architecture

The runtime is organized into several key components:

### Core Components

- **Model**: ONNX model representation and loading
- **Graph**: Computational graph with nodes and edges
- **Tensor**: N-dimensional array wrapper with type safety
- **Operators**: Implementation of ONNX operations
- **Runtime**: Execution engine with optimizations

### Supported Operators

| Operator      | Status   | Notes                       |
| ------------- | -------- | --------------------------- |
| `Add`         | ✅      | Element-wise addition        |
| `Mul`         | ✅      | Element-wise multiplication  |
| `MatMul`      | ✅      | Matrix multiplication        |
| `Conv`        | ✅      | 2D Convolution               |
| `Relu`        | ✅      | Rectified Linear Unit        |
| `Sigmoid`     | ✅      | Sigmoid activation           |
| `Reshape`     | ✅      | Tensor reshaping             |
| `Transpose`   | ✅      | Tensor transposition         |

## Examples

### Simple Linear Model

```rust
use runnx::{Model, Tensor};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    // Create a simple linear transformation: y = x * w + b
    let weights = Array2::from_shape_vec((3, 2), vec![0.5, 0.3, 0.2, 0.4, 0.1, 0.6])?;
    let bias = Array2::from_shape_vec((1, 2), vec![0.1, 0.2])?;
    
    let input = Tensor::from_array(Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0])?);
    let w_tensor = Tensor::from_array(weights);
    let b_tensor = Tensor::from_array(bias);
    
    // Manual computation for verification
    let result1 = input.matmul(&w_tensor)?;
    let result2 = result1.add(&b_tensor)?;
    
    println!("Linear transformation result: {:?}", result2.data());
    Ok(())
}
```

### Model Loading and Inference

```rust
use runnx::{Model, Tensor};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model from file
    let model = Model::from_file("path/to/model.onnx")?;
    
    // Print model information
    println!("Model: {}", model.name());
    println!("Inputs: {:?}", model.input_names());
    println!("Outputs: {:?}", model.output_names());
    
    // Prepare inputs
    let mut inputs = HashMap::new();
    inputs.insert("input", Tensor::zeros(&[1, 3, 224, 224]));
    
    // Run inference
    let outputs = model.run(&inputs)?;
    
    // Process outputs
    for (name, tensor) in outputs {
        println!("Output '{}': shape {:?}", name, tensor.shape());
    }
    
    Ok(())
}
```

## Performance

The runtime includes benchmarking capabilities:

```bash
# Run benchmarks
cargo bench

# Generate HTML reports
cargo bench -- --output-format html
```

Example benchmark results:
- Basic operations: ~10-50 µs
- Small model inference: ~100-500 µs
- Medium model inference: ~1-10 ms

## Formal Verification

RunNX includes comprehensive formal verification capabilities to ensure mathematical correctness:

### 🔬 Mathematical Specifications

The runtime includes formal specifications for all tensor operations using Why3:

```why3
(** Addition operation specification *)
function add_spec (a b: tensor) : tensor
  requires { valid_tensor a /\ valid_tensor b }
  requires { a.shape = b.shape }
  ensures  { valid_tensor result }
  ensures  { result.shape = a.shape }
  ensures  { forall i. 0 <= i < length result.data ->
             result.data[i] = a.data[i] + b.data[i] }
```

### 🧪 Property-Based Testing

Automatic verification of mathematical properties:

```rust
use runnx::formal::contracts::{AdditionContracts, ActivationContracts};

// Test addition commutativity: a + b = b + a
let result1 = tensor_a.add_with_contracts(&tensor_b)?;
let result2 = tensor_b.add_with_contracts(&tensor_a)?;
assert_eq!(result1.data(), result2.data());

// Test ReLU idempotency: ReLU(ReLU(x)) = ReLU(x)  
let relu_once = tensor.relu_with_contracts()?;
let relu_twice = relu_once.relu_with_contracts()?;
assert_eq!(relu_once.data(), relu_twice.data());
```

### 🔍 Runtime Verification

Dynamic checking of invariants during execution:

```rust
use runnx::formal::runtime_verification::InvariantMonitor;

let monitor = InvariantMonitor::new();
let result = tensor.add(&other)?;

// Verify numerical stability and bounds
assert!(monitor.verify_operation(&[&tensor, &other], &[&result]));
```

### 🎯 Verified Properties

The formal verification system proves:

- **Addition**: Commutativity, associativity, identity
- **Matrix Multiplication**: Associativity, distributivity  
- **ReLU**: Idempotency, monotonicity, non-negativity
- **Sigmoid**: Boundedness (0, 1), monotonicity, symmetry
- **Numerical Stability**: Overflow/underflow prevention

### 📊 Running Formal Verification

```bash
# Install Why3 (optional, for complete formal proofs)
make -C formal install-why3

# Run all verification (tests + proofs)
make -C formal all

# Run only property-based tests (no Why3 required)
cargo test formal --lib

# Run verification example
cargo run --example formal_verification

# Generate verification report  
make -C formal report
```

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with logging
RUST_LOG=debug cargo test

# Run specific test
cargo test test_tensor_operations
```

### Building Documentation

```bash
# Build and open documentation
cargo doc --open

# Build with private items
cargo doc --document-private-items
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Run `cargo test` and `cargo bench`
6. Submit a pull request

## License

This project is licensed under

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)


## Acknowledgments

- [ONNX](https://onnx.ai/) - Open Neural Network Exchange format
- [ndarray](https://github.com/rust-ndarray/ndarray) - Rust's `ndarray` library
- [Candle](https://github.com/huggingface/candle) - Inspiration for some design patterns

## Roadmap

- [ ] Add more operators (Softmax, BatchNorm, etc.)
- [ ] GPU acceleration support
- [ ] Quantization support
- [ ] Model optimization passes
- [ ] WASM compilation target
- [ ] Python bindings
