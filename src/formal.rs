//! Formal specifications and contracts for RunNX
//!
//! This module provides formal specifications that can be verified
//! using external tools like Why3, Dafny, or property-based testing.

use crate::error::Result;
use crate::tensor::Tensor;

/// Mathematical contracts for tensor operations
pub mod contracts {
    use super::*;

    /// Addition operation contracts
    pub trait AdditionContracts {
        // @requires: self.shape() == other.shape()
        // @ensures: result.shape() == self.shape()
        // @ensures: forall i: result[i] == self[i] + other[i]
        // Property: Commutativity - a + b == b + a
        // Property: Associativity - (a + b) + c == a + (b + c)
        // Property: Identity - a + 0 == a
        fn add_with_contracts(&self, other: &Tensor) -> Result<Tensor>;
    }

    /// Matrix multiplication contracts
    pub trait MatMulContracts {
        // @requires: self.shape().len() == 2 && other.shape().len() == 2
        // @requires: self.shape()[1] == other.shape()[0]
        // @ensures: result.shape() == [self.shape()[0], other.shape()[1]]
        // Property: Associativity - (A * B) * C == A * (B * C)
        // Property: Distributivity - A * (B + C) == A * B + A * C
        fn matmul_with_contracts(&self, other: &Tensor) -> Result<Tensor>;
    }

    /// Activation function contracts
    pub trait ActivationContracts {
        // @requires: true (no preconditions)
        // @ensures: result.shape() == self.shape()
        // @ensures: forall i: result[i] == max(0, self[i])
        // Property: Idempotency - ReLU(ReLU(x)) == ReLU(x)
        // Property: Monotonicity - x <= y => ReLU(x) <= ReLU(y)
        // Property: Non-negativity - forall i: result[i] >= 0
        fn relu_with_contracts(&self) -> Result<Tensor>;

        // @requires: true (no preconditions)
        // @ensures: result.shape() == self.shape()
        // @ensures: forall i: 0 < result[i] < 1
        // Property: Bounded output - 0 < sigmoid(x) < 1
        // Property: Monotonicity - x < y => sigmoid(x) < sigmoid(y)
        // Property: Symmetry - sigmoid(-x) == 1 - sigmoid(x)
        fn sigmoid_with_contracts(&self) -> Result<Tensor>;
    }

    /// Numerical stability contracts
    pub trait StabilityContracts {
        // @requires: true
        // @ensures: forall i: result[i].is_finite()
        // Property: No overflow - operations don't produce infinity
        // Property: No underflow - operations don't produce NaN
        fn is_numerically_stable(&self) -> bool;

        // @requires: epsilon > 0
        // @ensures: forall i: abs(self[i]) < 1/epsilon
        // Property: Bounded values prevent numerical instability
        fn check_bounds(&self, epsilon: f32) -> bool;
    }
}

/// Implementation of formal contracts
impl contracts::AdditionContracts for Tensor {
    fn add_with_contracts(&self, other: &Tensor) -> Result<Tensor> {
        // Precondition check
        if self.shape() != other.shape() {
            return Err(crate::error::OnnxError::invalid_dimensions(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            )));
        }

        let result = self.add(other)?;

        // Postcondition checks
        debug_assert_eq!(
            result.shape(),
            self.shape(),
            "Result shape must match input shape"
        );

        // Mathematical property verification (in debug mode)
        #[cfg(debug_assertions)]
        {
            // Check commutativity with a small tolerance for floating point
            let reverse_result = other.add(self)?;
            for (a, b) in result.data().iter().zip(reverse_result.data().iter()) {
                debug_assert!((a - b).abs() < f32::EPSILON, "Addition must be commutative");
            }
        }

        Ok(result)
    }
}

impl contracts::MatMulContracts for Tensor {
    fn matmul_with_contracts(&self, other: &Tensor) -> Result<Tensor> {
        // Precondition checks
        if self.shape().len() != 2 || other.shape().len() != 2 {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        if self.shape()[1] != other.shape()[0] {
            return Err(crate::error::OnnxError::invalid_dimensions(format!(
                "Inner dimensions must match: {} vs {}",
                self.shape()[1],
                other.shape()[0]
            )));
        }

        let result = self.matmul(other)?;

        // Postcondition checks
        debug_assert_eq!(result.shape()[0], self.shape()[0]);
        debug_assert_eq!(result.shape()[1], other.shape()[1]);

        Ok(result)
    }
}

impl contracts::ActivationContracts for Tensor {
    fn relu_with_contracts(&self) -> Result<Tensor> {
        let result = self.relu();

        // Postcondition checks
        debug_assert_eq!(result.shape(), self.shape());

        #[cfg(debug_assertions)]
        {
            // Verify non-negativity
            for &value in result.data().iter() {
                debug_assert!(value >= 0.0, "ReLU output must be non-negative");
            }

            // Verify idempotency
            let double_relu = result.relu();
            for (a, b) in result.data().iter().zip(double_relu.data().iter()) {
                debug_assert_eq!(a, b, "ReLU must be idempotent");
            }
        }

        Ok(result)
    }

    fn sigmoid_with_contracts(&self) -> Result<Tensor> {
        let result = self.sigmoid();

        // Postcondition checks
        debug_assert_eq!(result.shape(), self.shape());

        #[cfg(debug_assertions)]
        {
            // Verify bounded output
            for &value in result.data().iter() {
                debug_assert!(
                    value > 0.0 && value < 1.0,
                    "Sigmoid output must be in (0, 1)"
                );
            }
        }

        Ok(result)
    }
}

impl contracts::StabilityContracts for Tensor {
    fn is_numerically_stable(&self) -> bool {
        self.data().iter().all(|&x| x.is_finite())
    }

    fn check_bounds(&self, epsilon: f32) -> bool {
        if epsilon <= 0.0 {
            return false;
        }

        let bound = 1.0 / epsilon;
        self.data().iter().all(|&x| x.abs() < bound)
    }
}

/// Property-based testing utilities
#[cfg(test)]
pub mod property_tests {
    use super::*;
    use crate::tensor::Tensor;
    use ndarray::Array2;

    /// Generate random tensor for property testing
    pub fn random_tensor(shape: &[usize], seed: u64) -> Tensor {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut rng_state = hasher.finish();

        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                ((rng_state as f32) / (u64::MAX as f32)) * 2.0 - 1.0 // [-1, 1]
            })
            .collect();

        match shape.len() {
            2 => Tensor::from_array(Array2::from_shape_vec((shape[0], shape[1]), data).unwrap()),
            _ => panic!("Only 2D tensors supported in this example"),
        }
    }

    /// Test mathematical properties
    pub fn test_associativity<F>(op: F, a: &Tensor, b: &Tensor, c: &Tensor) -> bool
    where
        F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        if let (Ok(ab), Ok(bc)) = (op(a, b), op(b, c)) {
            if let (Ok(ab_c), Ok(a_bc)) = (op(&ab, c), op(a, &bc)) {
                // Check if results are approximately equal
                return ab_c
                    .data()
                    .iter()
                    .zip(a_bc.data().iter())
                    .all(|(x, y)| (x - y).abs() < 1e-6);
            }
        }
        false
    }
}

/// Why3 integration helpers
pub mod why3_integration {
    use super::*;

    /// Generate Why3 proof obligations from Rust contracts
    pub fn generate_proof_obligations() -> Vec<String> {
        vec![
            "goal add_commutativity: forall a b. add_spec a b = add_spec b a".to_string(),
            "goal add_associativity: forall a b c. add_spec (add_spec a b) c = add_spec a (add_spec b c)".to_string(),
            "goal relu_idempotent: forall a. relu_spec (relu_spec a) = relu_spec a".to_string(),
            "goal sigmoid_bounds: forall a i. 0.0 < (sigmoid_spec a).data[i] < 1.0".to_string(),
            "goal matmul_associativity: forall a b c. matmul_spec (matmul_spec a b) c = matmul_spec a (matmul_spec b c)".to_string(),
        ]
    }

    /// Convert Rust tensor to Why3 representation
    pub fn to_why3_tensor(tensor: &Tensor) -> String {
        let shape_str = tensor
            .shape()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("; ");

        let data_str = tensor
            .data()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("; ");

        format!("{{ shape = [{shape_str}]; data = [{data_str}]; valid = true }}")
    }
}

/// Runtime verification helpers
pub mod runtime_verification {
    use super::contracts::StabilityContracts;
    use super::*;

    /// Monitor for checking invariants at runtime
    pub struct InvariantMonitor {
        check_bounds: bool,
        check_stability: bool,
        epsilon: f32,
    }

    impl InvariantMonitor {
        pub fn new() -> Self {
            Self {
                check_bounds: true,
                check_stability: true,
                epsilon: 1e-6,
            }
        }

        pub fn verify_operation(&self, input: &[&Tensor], output: &[&Tensor]) -> bool {
            let mut valid = true;

            if self.check_stability {
                for tensor in input.iter().chain(output.iter()) {
                    if !tensor.is_numerically_stable() {
                        eprintln!("Warning: Numerical instability detected");
                        valid = false;
                    }
                }
            }

            if self.check_bounds {
                for tensor in input.iter().chain(output.iter()) {
                    if !tensor.check_bounds(self.epsilon) {
                        eprintln!("Warning: Values exceed numerical bounds");
                        valid = false;
                    }
                }
            }

            valid
        }
    }

    impl Default for InvariantMonitor {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::contracts::{ActivationContracts, AdditionContracts, StabilityContracts};
    use super::property_tests::*;

    #[test]
    fn test_addition_contracts() {
        let a = random_tensor(&[2, 3], 12345);
        let b = random_tensor(&[2, 3], 54321);

        let result = a.add_with_contracts(&b);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.shape(), a.shape());
    }

    #[test]
    fn test_relu_properties() {
        let tensor = random_tensor(&[3, 4], 98765);
        let relu_result = tensor.relu_with_contracts().unwrap();

        // Test idempotency
        let double_relu = relu_result.relu_with_contracts().unwrap();
        for (a, b) in relu_result.data().iter().zip(double_relu.data().iter()) {
            assert_eq!(a, b);
        }

        // Test non-negativity
        for &value in relu_result.data().iter() {
            assert!(value >= 0.0);
        }
    }

    #[test]
    fn test_numerical_stability() {
        let tensor = random_tensor(&[2, 2], 11111);
        assert!(tensor.is_numerically_stable());
        assert!(tensor.check_bounds(1e-3));
    }
}
