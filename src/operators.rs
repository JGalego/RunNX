//! ONNX operator implementations
//!
//! This module contains implementations of various ONNX operators.
//! Each operator is implemented as a function that takes input tensors
//! and returns output tensors.

use crate::{
    error::{OnnxError, Result},
    tensor::Tensor,
};

/// Supported ONNX operators
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorType {
    Add,
    Mul,
    MatMul,
    Conv,
    Relu,
    Sigmoid,
    Reshape,
    Transpose,
}

impl OperatorType {
    /// Parse operator type from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "Add" => Ok(OperatorType::Add),
            "Mul" => Ok(OperatorType::Mul),
            "MatMul" => Ok(OperatorType::MatMul),
            "Conv" => Ok(OperatorType::Conv),
            "Relu" => Ok(OperatorType::Relu),
            "Sigmoid" => Ok(OperatorType::Sigmoid),
            "Reshape" => Ok(OperatorType::Reshape),
            "Transpose" => Ok(OperatorType::Transpose),
            _ => Err(OnnxError::unsupported_operation(s)),
        }
    }
}

/// Execute an operator with given inputs
pub fn execute_operator(
    op_type: &OperatorType,
    inputs: &[Tensor],
    _attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    match op_type {
        OperatorType::Add => add_op(inputs),
        OperatorType::Mul => mul_op(inputs),
        OperatorType::MatMul => matmul_op(inputs),
        OperatorType::Conv => conv_op(inputs),
        OperatorType::Relu => relu_op(inputs),
        OperatorType::Sigmoid => sigmoid_op(inputs),
        OperatorType::Reshape => reshape_op(inputs),
        OperatorType::Transpose => transpose_op(inputs),
    }
}

/// Add operator implementation
///
/// Performs element-wise addition of two tensors.
///
/// # Arguments
/// * `inputs` - Array of exactly 2 tensors
///
/// # Returns
/// * Single output tensor with element-wise sum
fn add_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Add operator requires exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].add(&inputs[1])?;
    Ok(vec![result])
}

/// Multiply operator implementation
///
/// Performs element-wise multiplication of two tensors.
///
/// # Arguments
/// * `inputs` - Array of exactly 2 tensors
///
/// # Returns
/// * Single output tensor with element-wise product
fn mul_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Mul operator requires exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].mul(&inputs[1])?;
    Ok(vec![result])
}

/// Matrix multiplication operator implementation
///
/// Performs matrix multiplication of two 2D tensors.
///
/// # Arguments
/// * `inputs` - Array of exactly 2 tensors (both must be 2D)
///
/// # Returns
/// * Single output tensor with matrix product
fn matmul_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "MatMul operator requires exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].matmul(&inputs[1])?;
    Ok(vec![result])
}

/// 2D Convolution operator implementation (simplified)
///
/// Performs a basic 2D convolution operation.
/// This is a simplified implementation for educational purposes.
///
/// # Arguments
/// * `inputs` - Array of 2 tensors: [input, kernel]
///
/// # Returns
/// * Single output tensor with convolution result
fn conv_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() < 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Conv operator requires at least 2 inputs (input, kernel), got {}",
            inputs.len()
        )));
    }

    let input = &inputs[0];
    let kernel = &inputs[1];

    // Simplified 2D convolution for 4D tensors (NCHW format)
    if input.ndim() != 4 || kernel.ndim() != 4 {
        return Err(OnnxError::invalid_dimensions(
            "Conv operator requires 4D tensors (NCHW format)".to_string(),
        ));
    }

    // For simplicity, we'll just return the input tensor
    // A full implementation would perform the actual convolution
    log::warn!("Conv operator is not fully implemented, returning input tensor");
    Ok(vec![input.clone()])
}

/// ReLU operator implementation
///
/// Applies the Rectified Linear Unit function: f(x) = max(0, x)
///
/// # Arguments
/// * `inputs` - Array of exactly 1 tensor
///
/// # Returns
/// * Single output tensor with ReLU applied element-wise
fn relu_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Relu operator requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].relu();
    Ok(vec![result])
}

/// Sigmoid operator implementation
///
/// Applies the Sigmoid function: f(x) = 1 / (1 + exp(-x))
///
/// # Arguments
/// * `inputs` - Array of exactly 1 tensor
///
/// # Returns
/// * Single output tensor with Sigmoid applied element-wise
fn sigmoid_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Sigmoid operator requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].sigmoid();
    Ok(vec![result])
}

/// Reshape operator implementation
///
/// Reshapes a tensor to a new shape while preserving the total number of elements.
///
/// # Arguments
/// * `inputs` - Array of 2 tensors: [data, shape]
///
/// # Returns
/// * Single output tensor with new shape
fn reshape_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Reshape operator requires exactly 2 inputs (data, shape), got {}",
            inputs.len()
        )));
    }

    let data = &inputs[0];
    let shape_tensor = &inputs[1];

    // Extract shape from the second tensor
    let new_shape: Vec<usize> = shape_tensor.data().iter().map(|&x| x as usize).collect();

    let result = data.reshape(&new_shape)?;
    Ok(vec![result])
}

/// Transpose operator implementation
///
/// Transposes the input tensor by reversing or permuting the axes.
///
/// # Arguments
/// * `inputs` - Array of exactly 1 tensor
///
/// # Returns
/// * Single output tensor with transposed dimensions
fn transpose_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Transpose operator requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].transpose()?;
    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use ndarray::{Array1, Array2};
    use std::collections::HashMap;

    #[test]
    fn test_operator_type_from_str() {
        assert_eq!(OperatorType::from_str("Add").unwrap(), OperatorType::Add);
        assert_eq!(OperatorType::from_str("Relu").unwrap(), OperatorType::Relu);
        assert!(OperatorType::from_str("Unknown").is_err());
    }

    #[test]
    fn test_add_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let inputs = vec![a, b];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Add, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);

        let expected = vec![5.0, 7.0, 9.0];
        for (actual, &expected) in result[0].data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let inputs = vec![a]; // Only 1 input, should be 2
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Add, &inputs, &attrs);
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![2.0, 3.0, 4.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![5.0, 6.0, 7.0]));
        let inputs = vec![a, b];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Mul, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);

        let expected = vec![10.0, 18.0, 28.0];
        for (actual, &expected) in result[0].data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_op() {
        let a = Tensor::from_array(
            Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap(),
        );
        let b = Tensor::from_array(
            Array2::from_shape_vec((3, 2), vec![1., 2., 3., 4., 5., 6.]).unwrap(),
        );
        let inputs = vec![a, b];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::MatMul, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[2, 2]);
    }

    #[test]
    fn test_relu_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]));
        let inputs = vec![a];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Relu, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);

        let expected = vec![0.0, 0.0, 1.0, 2.0];
        for (actual, &expected) in result[0].data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![0.0]));
        let inputs = vec![a];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Sigmoid, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);

        // Sigmoid of 0 should be 0.5
        assert!((result[0].data()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_reshape_op() {
        let data = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let shape = Tensor::from_array(Array1::from_vec(vec![3.0, 2.0])); // New shape [3, 2]
        let inputs = vec![data, shape];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Reshape, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[3, 2]);
    }

    #[test]
    fn test_transpose_op() {
        let a = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let inputs = vec![a];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Transpose, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[3, 2]);
    }
}
