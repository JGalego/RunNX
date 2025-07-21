//! ONNX operator implementations
//!
//! This module contains implementations of various ONNX operators.
//! Each operator is implemented as a function that takes input tensors
//! and returns output tensors.

use crate::{
    error::{OnnxError, Result},
    tensor::Tensor,
};
use std::str::FromStr;

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

impl FromStr for OperatorType {
    type Err = OnnxError;

    fn from_str(s: &str) -> Result<Self> {
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
        assert_eq!("Add".parse::<OperatorType>().unwrap(), OperatorType::Add);
        assert_eq!("Mul".parse::<OperatorType>().unwrap(), OperatorType::Mul);
        assert_eq!("MatMul".parse::<OperatorType>().unwrap(), OperatorType::MatMul);
        assert_eq!("Conv".parse::<OperatorType>().unwrap(), OperatorType::Conv);
        assert_eq!("Relu".parse::<OperatorType>().unwrap(), OperatorType::Relu);
        assert_eq!("Sigmoid".parse::<OperatorType>().unwrap(), OperatorType::Sigmoid);
        assert_eq!("Reshape".parse::<OperatorType>().unwrap(), OperatorType::Reshape);
        assert_eq!("Transpose".parse::<OperatorType>().unwrap(), OperatorType::Transpose);
        
        // Test unknown operator
        let result = "Unknown".parse::<OperatorType>();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown"));
        
        // Test case sensitivity
        assert!("add".parse::<OperatorType>().is_err());
        assert!("ADD".parse::<OperatorType>().is_err());
    }

    #[test]
    fn test_operator_type_debug() {
        assert_eq!(format!("{:?}", OperatorType::Add), "Add");
        assert_eq!(format!("{:?}", OperatorType::Conv), "Conv");
    }

    #[test]
    fn test_operator_type_clone_eq() {
        let op1 = OperatorType::Add;
        let op2 = op1.clone();
        assert_eq!(op1, op2);
        
        assert_ne!(OperatorType::Add, OperatorType::Mul);
    }

    #[test]
    fn test_add_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let inputs = vec![a, b];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Add, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);

        let expected = [5.0, 7.0, 9.0];
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
        assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));

        // Test with too many inputs
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let c = Tensor::from_array(Array1::from_vec(vec![7.0, 8.0, 9.0]));
        let inputs = vec![a, b, c]; // 3 inputs, should be 2
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Add, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));
    }

    #[test]
    fn test_mul_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![2.0, 3.0, 4.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![5.0, 6.0, 7.0]));
        let inputs = vec![a, b];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Mul, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);

        let expected = [10.0, 18.0, 28.0];
        for (actual, &expected) in result[0].data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mul_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let inputs = vec![a]; // Only 1 input, should be 2
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Mul, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));
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
    fn test_matmul_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let inputs = vec![a]; // Only 1 input, should be 2
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::MatMul, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));
    }

    #[test]
    fn test_conv_op() {
        // Test valid 4D tensors (NCHW format)
        let input = Tensor::zeros(&[1, 1, 3, 3]); // Batch=1, Channel=1, H=3, W=3
        let kernel = Tensor::zeros(&[1, 1, 2, 2]); // OutChannels=1, InChannels=1, KH=2, KW=2
        let inputs = vec![input, kernel];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Conv, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);
        // Note: This is a simplified implementation that just returns the input
        assert_eq!(result[0].shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv_op_insufficient_inputs() {
        let input = Tensor::zeros(&[1, 1, 3, 3]);
        let inputs = vec![input]; // Only 1 input, should be at least 2
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Conv, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 2 inputs"));
    }

    #[test]
    fn test_conv_op_wrong_dimensions() {
        // Test with 3D input (invalid for Conv)
        let input = Tensor::zeros(&[1, 3, 3]); // 3D tensor
        let kernel = Tensor::zeros(&[1, 1, 2, 2]); // 4D tensor
        let inputs = vec![input, kernel];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Conv, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("4D tensors"));

        // Test with 3D kernel (invalid for Conv)
        let input = Tensor::zeros(&[1, 1, 3, 3]); // 4D tensor
        let kernel = Tensor::zeros(&[1, 2, 2]); // 3D tensor
        let inputs = vec![input, kernel];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Conv, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("4D tensors"));
    }

    #[test]
    fn test_relu_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]));
        let inputs = vec![a];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Relu, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);

        let expected = [0.0, 0.0, 1.0, 2.0];
        for (actual, &expected) in result[0].data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_relu_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let inputs = vec![a, b]; // 2 inputs, should be 1
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Relu, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 1 input"));
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
    fn test_sigmoid_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let inputs = vec![a, b]; // 2 inputs, should be 1
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Sigmoid, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 1 input"));
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
    fn test_reshape_op_wrong_inputs() {
        let data = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let inputs = vec![data]; // Only 1 input, should be 2
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Reshape, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));

        // Test with too many inputs
        let data = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let shape = Tensor::from_array(Array1::from_vec(vec![3.0, 2.0]));
        let extra = Tensor::from_array(Array1::from_vec(vec![1.0]));
        let inputs = vec![data, shape, extra]; // 3 inputs, should be 2
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Reshape, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));
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

    #[test]
    fn test_transpose_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let inputs = vec![a, b]; // 2 inputs, should be 1
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Transpose, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 1 input"));
    }

    #[test] 
    fn test_execute_operator_with_attributes() {
        // Test that attributes parameter is accepted (though currently unused)
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let inputs = vec![a];
        let mut attrs = HashMap::new();
        attrs.insert("test_attr".to_string(), "test_value".to_string());

        let result = execute_operator(&OperatorType::Relu, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_operator_types_execute() {
        // Ensure all operator types can be executed without panicking
        let tensor_1d = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let tensor_2d = Tensor::from_array(Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap());
        let tensor_4d = Tensor::zeros(&[1, 1, 2, 2]);
        let shape_tensor = Tensor::from_array(Array1::from_vec(vec![3.0, 2.0]));
        let attrs = HashMap::new();

        // Test all single-input operators
        assert!(execute_operator(&OperatorType::Relu, &[tensor_1d.clone()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::Sigmoid, &[tensor_1d.clone()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::Transpose, &[tensor_2d.clone()], &attrs).is_ok());

        // Test dual-input operators
        assert!(execute_operator(&OperatorType::Add, &[tensor_1d.clone(), tensor_1d.clone()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::Mul, &[tensor_1d.clone(), tensor_1d.clone()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::MatMul, &[tensor_2d.clone(), tensor_2d.transpose().unwrap()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::Reshape, &[tensor_2d.clone(), shape_tensor], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::Conv, &[tensor_4d.clone(), tensor_4d.clone()], &attrs).is_ok());
    }
}
