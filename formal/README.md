# Formal Verification for RunNX

This directory contains formal specifications and verification tools for the RunNX ONNX runtime, making it mathematically verifiable using formal methods.

## 🎯 Overview

The formal verification setup provides:

1. **Mathematical Specifications** - Precise mathematical definitions of tensor operations
2. **Property-Based Testing** - Automated testing of mathematical properties
3. **Runtime Verification** - Dynamic checking of invariants during execution
4. **Integration with Why3** - Formal proofs using state-of-the-art theorem provers

## 📁 Structure

```
formal/
├── operator_specs.mlw         # Why3 specifications for ONNX operators
├── verify_operators.py        # Python bridge for verification
├── Makefile                   # Automation scripts
├── test-verification.sh       # Shell script for testing verification
├── why3-wrapper.sh           # Why3 execution wrapper
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Rust 1.81+
- Python 3.8+
- Why3 (optional, for formal proofs)

### Install Why3 (Recommended)

```bash
# Using the provided makefile
make install-why3

# Or manually via opam
opam init
opam install why3 alt-ergo
```

### Run Verification

```bash
# Run all verification (tests + formal proofs)
make all

# Run only property-based tests (no Why3 required)
cargo test formal --release

# Run formal proofs (requires Why3)
make verify

# Run the comprehensive verification script
make verify-script
```

## 🔬 Formal Specifications

### Tensor Operations

Our specifications cover:

- **Element-wise operations**: Addition, multiplication with commutativity and associativity
- **Matrix multiplication**: Associativity and distributivity laws
- **Activation functions**: ReLU (idempotency, monotonicity) and Sigmoid (boundedness)
- **Shape operations**: Reshape and transpose with invariant preservation
- **Numerical stability**: Bounds checking and overflow prevention
- **YOLO operators**: Softmax, Concat, Slice, Upsample, MaxPool, NonMaxSuppression with probability and shape guarantees

### Mathematical Properties Verified

#### Addition
- **Commutativity**: `a + b = b + a`
- **Associativity**: `(a + b) + c = a + (b + c)`
- **Identity**: `a + 0 = a`

#### Matrix Multiplication  
- **Associativity**: `(A × B) × C = A × (B × C)`
- **Distributivity**: `A × (B + C) = A × B + A × C`

#### ReLU Activation
- **Idempotency**: `ReLU(ReLU(x)) = ReLU(x)`
- **Monotonicity**: `x ≤ y ⇒ ReLU(x) ≤ ReLU(y)`
- **Non-negativity**: `∀i: ReLU(x)[i] ≥ 0`

#### Sigmoid Activation
- **Boundedness**: `∀i: 0 < σ(x)[i] < 1`
- **Monotonicity**: `x < y ⇒ σ(x) < σ(y)`
- **Symmetry**: `σ(-x) = 1 - σ(x)`

#### Softmax Activation (YOLO)
- **Probability distribution**: `∑ softmax(x)[i] = 1.0`
- **Boundedness**: `∀i: 0 < softmax(x)[i] < 1`
- **Numerical stability**: `softmax(x) = softmax(x - max(x))`

#### YOLO Object Detection Operators
- **Concatenation**: Shape preservation and data integrity
- **Slicing**: Bounds checking and subset properties
- **Upsampling**: Scale factor validation and shape scaling
- **MaxPooling**: Monotonicity preservation
- **Non-Maximum Suppression**: Score ordering and IoU threshold guarantees

## 🧪 Property-Based Testing

The verification includes extensive property-based tests using the `proptest` crate:

```rust
// Example: Test addition commutativity
proptest! {
    #[test]
    fn test_add_commutativity(
        a in prop::array::uniform32(prop::num::f32::NORMAL, 2..10),
        b in prop::array::uniform32(prop::num::f32::NORMAL, 2..10)
    ) {
        let tensor_a = Tensor::from_array(Array2::from_shape_vec(shape, a.to_vec()).unwrap());
        let tensor_b = Tensor::from_array(Array2::from_shape_vec(shape, b.to_vec()).unwrap());
        
        let result1 = tensor_a.add(&tensor_b).unwrap();
        let result2 = tensor_b.add(&tensor_a).unwrap();
        
        assert_eq!(result1.data(), result2.data());
    }
}

// Example: Test YOLO softmax probability distribution
proptest! {
    #[test]
    fn test_softmax_probability_distribution(
        data in prop::collection::vec(prop::num::f32::NORMAL, 1..20)
    ) {
        if let Ok(tensor) = Tensor::from_shape_vec(&[data.len()], data) {
            if let Ok(softmax_result) = tensor.softmax() {
                // Test 1: Sum should equal 1.0
                let sum: f32 = softmax_result.data().iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-5);
                
                // Test 2: All values should be positive and < 1.0
                for &value in softmax_result.data().iter() {
                    prop_assert!(value > 0.0 && value < 1.0);
                }
            }
        }
    }
}

// Example: Test YOLO softmax numerical stability
proptest! {
    #[test]
    fn test_softmax_numerical_stability(
        data in prop::collection::vec(prop::num::f32::NORMAL, 1..10),
        shift in prop::num::f32::NORMAL
    ) {
        // Softmax should be invariant under constant shifts
        let softmax1 = tensor.softmax().unwrap();
        let shifted_tensor = apply_constant_shift(&tensor, shift);
        let softmax2 = shifted_tensor.softmax().unwrap();
        
        for (a, b) in softmax1.data().iter().zip(softmax2.data().iter()) {
            prop_assert!((a - b).abs() < 1e-5);
        }
    }
}
```

## 🔒 Runtime Verification

The formal module provides runtime invariant checking:

```rust
use runnx::formal::runtime_verification::InvariantMonitor;

let monitor = InvariantMonitor::new();
let result = tensor.add(&other).unwrap();

// Verify numerical stability
assert!(monitor.verify_operation(&[&tensor, &other], &[&result]));
```

## ⚙️ Integration with Rust Code

### Contract Annotations

Operations are annotated with formal contracts:

```rust
impl AdditionContracts for Tensor {
    // @requires: self.shape() == other.shape()
    // @ensures: result.shape() == self.shape()
    // @ensures: forall i: result[i] == self[i] + other[i]
    // Property: Commutativity - a + b == b + a
    fn add_with_contracts(&self, other: &Tensor) -> Result<Tensor> {
        // Implementation with pre/post condition checks
    }
}

impl YoloOperatorContracts for Tensor {
    // @requires: true (no preconditions)
    // @ensures: result.shape() == self.shape()
    // @ensures: sum(result.data()) == 1.0 (probability distribution)
    // @ensures: forall i: 0 < result[i] < 1
    fn softmax_with_contracts(&self) -> Result<Tensor> {
        // Implementation with probability distribution verification
    }
}
```

### Debug Assertions

In debug builds, mathematical properties are automatically verified:

```rust
#[cfg(debug_assertions)]
{
    let reverse_result = other.add(self)?;
    for (a, b) in result.data().iter().zip(reverse_result.data().iter()) {
        debug_assert!((a - b).abs() < f32::EPSILON, "Addition must be commutative");
    }
}
```

## 🔧 Why3 Integration

### Theorem Provers

The setup supports multiple theorem provers:
- **Alt-Ergo** - Primary SMT solver
- **CVC5** - Alternative SMT solver  
- **Z3** - Microsoft's SMT solver

### Proof Generation

Generate verification conditions:

```bash
why3 prove -P alt-ergo operator_specs.mlw
why3 prove -P z3 operator_specs.mlw
```

### Interactive Proofs

Use the Why3 IDE for interactive proof development:

```bash
make ide
```

## 📊 Verification Reports

Generate detailed HTML reports:

```bash
make report
# Open formal/_why3session/index.html
```

The reports show:
- Proof obligations
- Verification status
- Counterexamples (if any)
- Performance metrics

## 🎛️ Configuration

### Runtime Verification

Configure invariant checking:

```rust
let mut monitor = InvariantMonitor::new();
monitor.check_bounds = true;        // Enable bounds checking
monitor.check_stability = true;     // Enable numerical stability checks
monitor.epsilon = 1e-6;            // Set numerical tolerance
```

### Why3 Settings

Configure Why3 provers through command-line options:
- Theorem prover preferences (`-P alt-ergo`, `-P z3`)
- Proof strategies (`--strategy`)
- Timeout settings (`--timeout`)

## 🚨 Common Issues

### Alt-Ergo Configuration Error

If you get the error "No prover in /home/runner/.why3.conf corresponds to 'alt-ergo'":

```bash
# Method 1: Detect provers automatically
why3 config detect

# Method 2: Check available provers
why3 config list-provers

# Method 3: Manually configure Alt-Ergo (if installed)
why3 config add-prover alt-ergo /usr/local/bin/alt-ergo

# Method 4: Use any available prover
why3 prove operator_specs.mlw  # Without specifying -P alt-ergo
```

Our verification scripts are designed to gracefully handle missing provers and will automatically detect available ones.

### Why3 Installation

If Why3 installation fails:
```bash
# On Ubuntu/Debian
sudo apt update && sudo apt install opam
opam init --disable-sandboxing

# On macOS  
brew install opam
```

### Proof Failures

If formal proofs fail:
1. Check that mathematical properties hold in your implementation
2. Adjust numerical tolerances for floating-point operations
3. Use interactive Why3 IDE to debug proofs

### Test Failures

For property-based test failures:
1. Check the generated test cases
2. Verify edge cases in your implementation
3. Adjust test parameters if needed

## 📚 Further Reading

- [Why3 Manual](https://why3.lri.fr/manual.html)
- [Formal Methods in Software Engineering](https://link.springer.com/book/10.1007/978-3-319-57288-8)
- [Property-Based Testing with PropTest](https://docs.rs/proptest/)
- [SMT Solvers: Theory and Practice](https://smt-lib.org/)

## 🤝 Contributing

To add new formal specifications:

1. Add mathematical specifications to `.mlw` files
2. Implement runtime contracts in `src/formal.rs`  
3. Add property-based tests
4. Update documentation

## 📄 License

This formal verification setup is licensed under the same terms as RunNX (MIT OR Apache-2.0).
