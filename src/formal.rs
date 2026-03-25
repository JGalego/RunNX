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
        fn relu_with_contracts(&self) -> Result<Tensor>;

        // @requires: true (no preconditions)
        // @ensures: result.shape() == self.shape()
        // @ensures: forall i: 0 < result[i] < 1
        // Property: Bounded output - sigmoid maps R to (0, 1)
        fn sigmoid_with_contracts(&self) -> Result<Tensor>;
    }

    /// YOLO-specific operator contracts
    pub trait YoloOperatorContracts {
        // @requires: true (no preconditions)
        // @ensures: result.shape() == self.shape()
        // @ensures: sum(result.data()) == 1.0 (probability distribution)
        // @ensures: forall i: 0 < result[i] < 1
        // Property: Numerical stability - softmax(x) == softmax(x - max(x))
        // Property: Probability conservation - sum of outputs equals 1
        fn softmax_with_contracts(&self) -> Result<Tensor>;

        // @requires: self.shape().len() == other.shape().len()
        // @requires: axis < self.shape().len()
        // @requires: forall i != axis: self.shape()[i] == other.shape()[i]
        // @ensures: result.shape().len() == self.shape().len()
        // @ensures: result.shape()[axis] == self.shape()[axis] + other.shape()[axis]
        // @ensures: forall i != axis: result.shape()[i] == self.shape()[i]
        // Property: Data preservation - all input data appears in output
        fn concat_with_contracts(&self, other: &Tensor, axis: usize) -> Result<Tensor>;

        // @requires: starts.len() == ends.len()
        // @requires: forall i: 0 <= starts[i] < ends[i] <= self.shape()[i]
        // @ensures: result.shape()[i] == ends[i] - starts[i] for specified axes
        // Property: Subset - output contains subset of input data
        fn slice_with_contracts(
            &self,
            starts: &[i64],
            ends: &[i64],
            axes: Option<&[i64]>,
            steps: Option<&[i64]>,
        ) -> Result<Tensor>;

        // @requires: scale_factors.len() == self.shape().len()
        // @requires: forall i: scale_factors[i] > 0
        // @ensures: result.shape()[i] == (self.shape()[i] as f32 * scale_factors[i]) as usize
        // Property: Scale invariance - shape scales by exact factors
        fn upsample_with_contracts(&self, scale_factors: &[f32]) -> Result<Tensor>;

        // @requires: kernel_size.len() == spatial_dims
        // @requires: strides.len() == spatial_dims
        // @requires: padding.len() == spatial_dims
        // @ensures: result contains maximum values from each pooling window
        // Property: Monotonicity - if x <= y elementwise, then maxpool(x) <= maxpool(y)
        fn maxpool_with_contracts(
            &self,
            kernel_size: &[usize],
            strides: &[usize],
            padding: &[usize],
        ) -> Result<Tensor>;
    }

    /// Non-Maximum Suppression contracts
    pub trait NmsContracts {
        // @requires: boxes.shape() == [n, 4] && scores.shape() == [n]
        // @requires: 0.0 <= iou_threshold <= 1.0
        // @requires: score_threshold >= 0.0
        // @ensures: result indices are sorted by descending scores
        // @ensures: forall i, j in result: IoU(boxes[i], boxes[j]) <= iou_threshold
        // Property: Score ordering - output indices maintain score order
        fn nms_with_contracts(
            boxes: &Tensor,
            scores: &Tensor,
            iou_threshold: f32,
            score_threshold: f32,
            max_output_size: Option<usize>,
        ) -> Result<Vec<usize>>;
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
            // f32 addition is bitwise commutative for identical operands fed through
            // the same code path.  Use exact equality; if this ever fails it indicates
            // a deeper bug, not floating-point rounding.
            let reverse_result = other.add(self)?;
            for (a, b) in result.data().iter().zip(reverse_result.data().iter()) {
                debug_assert_eq!(a, b, "Addition must be commutative");
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

impl contracts::YoloOperatorContracts for Tensor {
    fn softmax_with_contracts(&self) -> Result<Tensor> {
        // Precondition checks
        if self.data().iter().any(|&x| !x.is_finite()) {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Input contains non-finite values".to_string(),
            ));
        }

        let result = self.softmax()?;

        // Postcondition checks
        debug_assert_eq!(result.shape(), self.shape());

        #[cfg(debug_assertions)]
        {
            // Softmax is applied per-slice along the last axis.
            // Verify that each such slice sums to 1.0 and contains values in (0, 1).
            let ndim = result.shape().len();
            let axis_size = result.shape()[ndim - 1];
            let outer_size: usize = result.shape()[..ndim - 1].iter().product();

            let flat: Vec<f32> = result.data().iter().cloned().collect();
            for outer in 0..outer_size {
                let slice = &flat[outer * axis_size..(outer + 1) * axis_size];
                let sum: f32 = slice.iter().sum();
                debug_assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "Softmax slice {outer} must sum to 1.0, got: {sum}"
                );
                for &value in slice {
                    debug_assert!(
                        value > 0.0 && value < 1.0,
                        "Softmax output must be in (0, 1), got: {value}"
                    );
                }
            }
        }

        Ok(result)
    }

    fn concat_with_contracts(&self, other: &Tensor, axis: usize) -> Result<Tensor> {
        // Precondition checks
        if self.shape().len() != other.shape().len() {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Tensors must have same number of dimensions for concatenation".to_string(),
            ));
        }

        if axis >= self.shape().len() {
            return Err(crate::error::OnnxError::invalid_dimensions(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                self.shape().len()
            )));
        }

        for (i, (&dim_a, &dim_b)) in self.shape().iter().zip(other.shape().iter()).enumerate() {
            if i != axis && dim_a != dim_b {
                return Err(crate::error::OnnxError::invalid_dimensions(format!(
                    "Dimension mismatch at axis {i}: {dim_a} vs {dim_b}"
                )));
            }
        }

        Tensor::concat(&[self, other], axis)
    }

    fn slice_with_contracts(
        &self,
        starts: &[i64],
        ends: &[i64],
        axes: Option<&[i64]>,
        steps: Option<&[i64]>,
    ) -> Result<Tensor> {
        // Precondition checks
        if starts.len() != ends.len() {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Starts and ends arrays must have same length".to_string(),
            ));
        }

        for (&start, &end) in starts.iter().zip(ends.iter()) {
            if start >= end {
                return Err(crate::error::OnnxError::invalid_dimensions(format!(
                    "Invalid slice range: start {start} >= end {end}"
                )));
            }
        }

        self.slice(starts, ends, axes, steps)
    }

    fn upsample_with_contracts(&self, scale_factors: &[f32]) -> Result<Tensor> {
        // Precondition checks
        if scale_factors.len() != self.shape().len() {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Scale factors must match tensor dimensions".to_string(),
            ));
        }

        for &scale in scale_factors {
            if scale <= 0.0 {
                return Err(crate::error::OnnxError::invalid_dimensions(
                    "Scale factors must be positive".to_string(),
                ));
            }
        }

        // Upsample via the contract interface requires a Tensor-level method;
        // use operators::upsample_op for the full implementation.
        Err(crate::error::OnnxError::unsupported_operation(
            "Upsample: use operators::execute_operator with OperatorType::Upsample".to_string(),
        ))
    }

    fn maxpool_with_contracts(
        &self,
        kernel_size: &[usize],
        strides: &[usize],
        _padding: &[usize],
    ) -> Result<Tensor> {
        // Precondition checks
        if kernel_size.is_empty() {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Kernel size cannot be empty".to_string(),
            ));
        }

        if kernel_size.len() != strides.len() {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Kernel size and strides must have same length".to_string(),
            ));
        }

        // MaxPool via the contract interface requires a Tensor-level method;
        // use operators::execute_operator with OperatorType::MaxPool for the full implementation.
        Err(crate::error::OnnxError::unsupported_operation(
            "MaxPool: use operators::execute_operator with OperatorType::MaxPool".to_string(),
        ))
    }
}

impl contracts::NmsContracts for Tensor {
    fn nms_with_contracts(
        boxes: &Tensor,
        scores: &Tensor,
        iou_threshold: f32,
        score_threshold: f32,
        _max_output_size: Option<usize>,
    ) -> Result<Vec<usize>> {
        // Precondition checks
        if boxes.shape().len() != 2 || boxes.shape()[1] != 4 {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Boxes tensor must have shape [N, 4]".to_string(),
            ));
        }

        if scores.shape().len() != 1 || scores.shape()[0] != boxes.shape()[0] {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Scores tensor must have shape [N] matching boxes".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&iou_threshold) {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "IoU threshold must be in [0, 1]".to_string(),
            ));
        }

        if score_threshold < 0.0 {
            return Err(crate::error::OnnxError::invalid_dimensions(
                "Score threshold must be non-negative".to_string(),
            ));
        }

        // For now, return error as NMS is not fully implemented
        Err(crate::error::OnnxError::unsupported_operation(
            "NMS operator not fully implemented".to_string(),
        ))
    }
}

impl contracts::ActivationContracts for Tensor {
    fn relu_with_contracts(&self) -> Result<Tensor> {
        let result = self.relu()?;

        // Postcondition checks
        debug_assert_eq!(result.shape(), self.shape());

        #[cfg(debug_assertions)]
        {
            // Verify non-negativity
            for &value in result.data().iter() {
                debug_assert!(value >= 0.0, "ReLU output must be non-negative");
            }

            // Verify idempotency
            let double_relu = result.relu()?;
            for (a, b) in result.data().iter().zip(double_relu.data().iter()) {
                debug_assert_eq!(a, b, "ReLU must be idempotent");
            }
        }

        Ok(result)
    }

    fn sigmoid_with_contracts(&self) -> Result<Tensor> {
        let result = self.sigmoid()?;

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

    /// Generate random tensor for property testing.
    ///
    /// Uses a simple, stable LCG whose initial state is derived directly from
    /// the caller-supplied `seed`, so the output is deterministic and reproducible
    /// across Rust versions (unlike `DefaultHasher`, which is not guaranteed to be stable).
    pub fn random_tensor(shape: &[usize], seed: u64) -> Tensor {
        // LCG parameters from Numerical Recipes (64-bit variant)
        let mut rng_state: u64 = seed.wrapping_add(1); // avoid state = 0

        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map high 32 bits to [-1, 1]
                let bits = (rng_state >> 32) as u32;
                (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        match shape.len() {
            1 => Tensor::from_shape_vec(&[shape[0]], data).expect("Failed to create 1D tensor"),
            2 => Tensor::from_array(Array2::from_shape_vec((shape[0], shape[1]), data).unwrap()),
            _ => {
                // For N-dimensional tensors, use the generic from_shape_vec
                Tensor::from_shape_vec(shape, data).expect("Failed to create N-D tensor")
            }
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

    /// Return the names of the Why3 goals defined in `formal/operators.mlw`.
    ///
    /// These correspond to the predicates and properties declared in the
    /// `RealOperators` theory and can be passed to Why3 tools (e.g.
    /// `why3 prove`) for automated verification.
    pub fn generate_proof_obligations() -> Vec<String> {
        vec![
            // Arithmetic properties (RealOperators theory, operators.mlw)
            "RealOperators.add_commutativity".to_string(),
            "RealOperators.add_associativity".to_string(),
            "RealOperators.mul_commutativity".to_string(),
            // Activation function properties
            "RealOperators.relu_idempotent".to_string(),
            "RealOperators.relu_monotonic".to_string(),
            "RealOperators.sigmoid_bounded".to_string(),
            "RealOperators.sigmoid_monotonic".to_string(),
            // Operator specifications
            "RealOperators.add_spec".to_string(),
            "RealOperators.mul_spec".to_string(),
            "RealOperators.sub_spec".to_string(),
            "RealOperators.div_spec".to_string(),
            "RealOperators.relu_spec".to_string(),
            "RealOperators.softmax_spec".to_string(),
            "RealOperators.reshape_spec".to_string(),
            "RealOperators.transpose_spec".to_string(),
            "RealOperators.matmul_spec".to_string(),
            "RealOperators.conv_spec".to_string(),
            "RealOperators.nms_spec".to_string(),
            "RealOperators.pad_spec".to_string(),
            "RealOperators.slice_spec".to_string(),
            "RealOperators.split_spec".to_string(),
        ]
    }

    /// Serialise a Rust tensor into a Why3 record literal compatible with the
    /// `TensorTypes.tensor` type defined in `formal/tensors.mlw`.
    ///
    /// The produced string can be embedded directly in Why3 goals or lemmas.
    pub fn to_why3_tensor(tensor: &Tensor) -> String {
        // Why3 array literals use the syntax: (make n default)[0 <- v0][1 <- v1]...
        let shape_len = tensor.shape().len();
        let shape_init = format!("(Array.make {} 0)", shape_len);
        let shape_str = tensor
            .shape()
            .iter()
            .enumerate()
            .fold(shape_init, |acc, (i, &d)| format!("{acc}[{i} <- {d}]"));

        let data_len = tensor.data().len();
        let data_init = format!("(Array.make {} 0.0)", data_len);
        let data_str = tensor
            .data()
            .iter()
            .enumerate()
            .fold(data_init, |acc, (i, &v)| format!("{acc}[{i} <- {}]", v));

        format!("{{ shape = {shape_str}; data = {data_str} }}")
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
    use super::contracts::{
        ActivationContracts, AdditionContracts, NmsContracts, StabilityContracts,
        YoloOperatorContracts,
    };
    use super::property_tests::*;
    use crate::tensor::Tensor;
    use ndarray::Array1;

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
    fn test_softmax_contracts() {
        let tensor = random_tensor(&[3, 4], 42424);
        let softmax_result = tensor.softmax_with_contracts();

        assert!(softmax_result.is_ok());
        let result = softmax_result.unwrap();

        // Test probability distribution: softmax is applied per-row along the
        // last axis, so each row (not the whole tensor) must sum to 1.0.
        let ndim = result.shape().len();
        let axis_size = result.shape()[ndim - 1];
        let outer_size: usize = result.shape()[..ndim - 1].iter().product();
        let flat = result.data();
        let flat = flat.as_slice().unwrap();
        for outer in 0..outer_size {
            let row_sum: f32 = flat[outer * axis_size..(outer + 1) * axis_size]
                .iter()
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "Softmax row {outer} should sum to 1.0, got: {row_sum}"
            );
        }

        // Test bounded output
        for &value in result.data().iter() {
            assert!(
                value > 0.0 && value < 1.0,
                "Softmax values should be in (0, 1), got: {value}"
            );
        }
    }

    #[test]
    fn test_softmax_stability() {
        let tensor = random_tensor(&[2, 3], 11111);
        let softmax1 = tensor.softmax_with_contracts().unwrap();

        // Add a constant to all elements (should not change softmax significantly)
        let shifted_data: Vec<f32> = tensor.data().iter().map(|&x| x + 10.0).collect(); // Use smaller shift
        let shifted_tensor = Tensor::from_shape_vec(tensor.shape(), shifted_data).unwrap();
        let softmax2 = shifted_tensor.softmax_with_contracts().unwrap();

        // Results should be approximately equal (allowing for small numerical differences)
        for (a, b) in softmax1.data().iter().zip(softmax2.data().iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Softmax should be stable under shifts: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_concat_contracts_validation() {
        let a = random_tensor(&[2, 3], 12345);
        let b = random_tensor(&[2, 4], 54321); // Different shape

        // Should fail due to incompatible shapes
        let result = a.concat_with_contracts(&b, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_contracts_validation() {
        let tensor = random_tensor(&[4, 5], 98765);

        // Invalid slice: start >= end
        let result = tensor.slice_with_contracts(&[1, 3], &[1, 2], None, None);
        assert!(result.is_err());

        // Invalid slice: mismatched array lengths
        let result = tensor.slice_with_contracts(&[1], &[2, 3], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_contracts() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
        let result = tensor.slice_with_contracts(&[1], &[3], None, None).unwrap();
        assert_eq!(result.shape(), &[2]);
        let data = result.data().as_slice().unwrap();
        assert_eq!(data, &[2.0, 3.0]);
    }

    #[test]
    fn test_upsample_contracts_validation() {
        let tensor = random_tensor(&[2, 3], 13579);

        // Invalid scale: wrong number of factors
        let result = tensor.upsample_with_contracts(&[2.0]); // Should be 2 factors
        assert!(result.is_err());

        // Invalid scale: negative factor
        let result = tensor.upsample_with_contracts(&[2.0, -1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_maxpool_contracts_validation() {
        let tensor = random_tensor(&[1, 1], 24680);

        // Invalid kernel: empty
        let result = tensor.maxpool_with_contracts(&[], &[], &[]);
        assert!(result.is_err());

        // Invalid kernel: mismatched lengths
        let result = tensor.maxpool_with_contracts(&[2], &[1, 1], &[0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_nms_contracts_validation() {
        // Invalid boxes shape
        let boxes_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let boxes = Tensor::from_shape_vec(&[3, 4], boxes_data).unwrap(); // Correct [N, 4] shape
        let scores_data = vec![0.9, 0.8, 0.7];
        let scores = Tensor::from_shape_vec(&[3], scores_data).unwrap();

        // This should work - both tensors have correct shapes
        let result = Tensor::nms_with_contracts(&boxes, &scores, 0.5, 0.1, Some(10));
        // We expect it to fail because NMS is not implemented yet
        assert!(result.is_err());

        // Test with invalid boxes shape [3, 3] instead of [N, 4]
        let invalid_boxes_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let invalid_boxes = Tensor::from_shape_vec(&[3, 3], invalid_boxes_data).unwrap();

        let result = Tensor::nms_with_contracts(&invalid_boxes, &scores, 0.5, 0.1, Some(10));
        assert!(result.is_err());

        // Test with mismatched scores length [4] instead of [3]
        let mismatched_scores_data = vec![0.9, 0.8, 0.7, 0.6];
        let mismatched_scores = Tensor::from_shape_vec(&[4], mismatched_scores_data).unwrap();

        let result = Tensor::nms_with_contracts(&boxes, &mismatched_scores, 0.5, 0.1, Some(10));
        assert!(result.is_err());

        // Test with invalid IoU threshold > 1.0
        let result = Tensor::nms_with_contracts(&boxes, &scores, 1.5, 0.1, Some(10));
        assert!(result.is_err());
    }

    #[test]
    fn test_numerical_stability() {
        let tensor = random_tensor(&[2, 2], 11111);
        assert!(tensor.is_numerically_stable());
        assert!(tensor.check_bounds(1e-3));
    }
}
