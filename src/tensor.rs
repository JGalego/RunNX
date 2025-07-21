//! Tensor implementation for ONNX runtime
//!
//! This module provides the [`Tensor`] type, which is a wrapper around ndarray
//! with additional functionality for ONNX operations.

use crate::error::{OnnxError, Result};
use ndarray::{ArrayBase, ArrayD, Data, Dimension, IxDyn};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A multi-dimensional tensor for neural network computations
///
/// `Tensor` is a wrapper around ndarray's `ArrayD<f32>` that provides
/// additional functionality specific to ONNX operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    data: ArrayD<f32>,
}

impl Tensor {
    /// Create a new tensor from an ndarray
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    /// use ndarray::Array2;
    ///
    /// let array = Array2::from_elem((2, 3), 1.0);
    /// let tensor = Tensor::from_array(array);
    /// assert_eq!(tensor.shape(), &[2, 3]);
    /// ```
    pub fn from_array<S, D>(array: ArrayBase<S, D>) -> Self
    where
        S: Data<Elem = f32>,
        D: Dimension,
    {
        Self {
            data: array.to_owned().into_dyn(),
        }
    }

    /// Create a tensor filled with zeros
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[2, 3, 4]);
    /// assert_eq!(tensor.shape(), &[2, 3, 4]);
    /// assert!(tensor.data().iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    /// Create a tensor filled with ones
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    ///
    /// let tensor = Tensor::ones(&[2, 2]);
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// assert!(tensor.data().iter().all(|&x| x == 1.0));
    /// ```
    pub fn ones(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::ones(IxDyn(shape)),
        }
    }

    /// Create a tensor from raw data and shape
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_shape_vec(&[2, 3], data).unwrap();
    /// assert_eq!(tensor.shape(), &[2, 3]);
    /// ```
    pub fn from_shape_vec(shape: &[usize], data: Vec<f32>) -> Result<Self> {
        let array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| OnnxError::invalid_dimensions(e.to_string()))?;
        Ok(Self { data: array })
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &ArrayD<f32> {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.data
    }

    /// Element-wise addition
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    /// use ndarray::Array1;
    ///
    /// let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
    /// let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
    /// let result = a.add(&b).unwrap();
    ///
    /// let expected = vec![5.0, 7.0, 9.0];
    /// for (actual, &expected) in result.data().iter().zip(expected.iter()) {
    ///     assert!((actual - expected).abs() < 1e-6);
    /// }
    /// ```
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape() != other.shape() {
            return Err(OnnxError::shape_mismatch(self.shape(), other.shape()));
        }

        Ok(Tensor {
            data: &self.data + &other.data,
        })
    }

    /// Element-wise multiplication
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    /// use ndarray::Array1;
    ///
    /// let a = Tensor::from_array(Array1::from_vec(vec![2.0, 3.0, 4.0]));
    /// let b = Tensor::from_array(Array1::from_vec(vec![5.0, 6.0, 7.0]));
    /// let result = a.mul(&b).unwrap();
    ///
    /// let expected = vec![10.0, 18.0, 28.0];
    /// for (actual, &expected) in result.data().iter().zip(expected.iter()) {
    ///     assert!((actual - expected).abs() < 1e-6);
    /// }
    /// ```
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape() != other.shape() {
            return Err(OnnxError::shape_mismatch(self.shape(), other.shape()));
        }

        Ok(Tensor {
            data: &self.data * &other.data,
        })
    }

    /// Matrix multiplication
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    /// use ndarray::Array2;
    ///
    /// let a = Tensor::from_array(Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap());
    /// let b = Tensor::from_array(Array2::from_shape_vec((3, 2), vec![1., 2., 3., 4., 5., 6.]).unwrap());
    /// let result = a.matmul(&b).unwrap();
    ///
    /// assert_eq!(result.shape(), &[2, 2]);
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(OnnxError::invalid_dimensions(
                "Matrix multiplication requires 2D tensors",
            ));
        }

        let self_shape = self.shape();
        let other_shape = other.shape();

        if self_shape[1] != other_shape[0] {
            return Err(OnnxError::shape_mismatch(
                &[self_shape[0], other_shape[1]],
                &[self_shape[0], self_shape[1]],
            ));
        }

        let self_2d = self
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| OnnxError::invalid_dimensions(e.to_string()))?;
        let other_2d = other
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| OnnxError::invalid_dimensions(e.to_string()))?;

        let result = self_2d.dot(&other_2d);
        Ok(Tensor::from_array(result))
    }

    /// Reshape the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    ///
    /// let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    /// let reshaped = tensor.reshape(&[3, 2]).unwrap();
    /// assert_eq!(reshaped.shape(), &[3, 2]);
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len() {
            return Err(OnnxError::invalid_dimensions(format!(
                "Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                self.len(),
                new_shape,
                new_len
            )));
        }

        let reshaped = self
            .data
            .view()
            .to_shape(IxDyn(new_shape))
            .map_err(|e| OnnxError::invalid_dimensions(e.to_string()))?
            .to_owned();

        Ok(Tensor { data: reshaped })
    }

    /// Transpose the tensor (swap axes)
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    ///
    /// let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    /// let transposed = tensor.transpose().unwrap();
    /// assert_eq!(transposed.shape(), &[3, 2]);
    /// ```
    pub fn transpose(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(OnnxError::invalid_dimensions(
                "Transpose currently only supports 2D tensors",
            ));
        }

        let transposed = self.data.t().to_owned();
        Ok(Tensor { data: transposed })
    }

    /// Apply ReLU activation (max(0, x))
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    /// use ndarray::Array1;
    ///
    /// let tensor = Tensor::from_array(Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]));
    /// let result = tensor.relu();
    ///
    /// let expected = vec![0.0, 0.0, 1.0, 2.0];
    /// for (actual, &expected) in result.data().iter().zip(expected.iter()) {
    ///     assert!((actual - expected).abs() < 1e-6);
    /// }
    /// ```
    pub fn relu(&self) -> Tensor {
        let data = self.data.mapv(|x| x.max(0.0));
        Tensor { data }
    }

    /// Apply Sigmoid activation (1 / (1 + exp(-x)))
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    /// use ndarray::Array1;
    ///
    /// let tensor = Tensor::from_array(Array1::from_vec(vec![0.0]));
    /// let result = tensor.sigmoid();
    ///
    /// // Sigmoid of 0 should be 0.5
    /// assert!((result.data()[0] - 0.5).abs() < 1e-6);
    /// ```
    pub fn sigmoid(&self) -> Tensor {
        let data = self.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        Tensor { data }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{:?}\n{}", self.shape(), self.data)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    #[test]
    fn test_tensor_creation() {
        let array = Array2::from_elem((2, 3), 1.0);
        let tensor = Tensor::from_array(array);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_tensor_from_different_array_types() {
        // Test 1D array
        let array1d = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let tensor1d = Tensor::from_array(array1d);
        assert_eq!(tensor1d.shape(), &[3]);
        assert_eq!(tensor1d.ndim(), 1);

        // Test 3D array
        let array3d = Array3::zeros((2, 3, 4));
        let tensor3d = Tensor::from_array(array3d);
        assert_eq!(tensor3d.shape(), &[2, 3, 4]);
        assert_eq!(tensor3d.ndim(), 3);
        assert_eq!(tensor3d.len(), 24);
    }

    #[test]
    fn test_empty_tensor() {
        let tensor = Tensor::zeros(&[0]);
        assert!(tensor.is_empty());
        assert_eq!(tensor.len(), 0);
    }

    #[test]
    fn test_zeros_and_ones() {
        let zeros = Tensor::zeros(&[2, 3]);
        assert!(zeros.data().iter().all(|&x| x == 0.0));
        assert_eq!(zeros.shape(), &[2, 3]);
        assert_eq!(zeros.len(), 6);

        let ones = Tensor::ones(&[2, 3]);
        assert!(ones.data().iter().all(|&x| x == 1.0));
        assert_eq!(ones.shape(), &[2, 3]);
        assert_eq!(ones.len(), 6);
    }

    #[test]
    fn test_zeros_ones_different_shapes() {
        // Test 1D
        let zeros_1d = Tensor::zeros(&[5]);
        assert_eq!(zeros_1d.shape(), &[5]);
        assert!(zeros_1d.data().iter().all(|&x| x == 0.0));

        // Test 3D
        let ones_3d = Tensor::ones(&[2, 2, 2]);
        assert_eq!(ones_3d.shape(), &[2, 2, 2]);
        assert!(ones_3d.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_shape_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_shape_vec(&[2, 2], data).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);

        // Test accessing data
        let tensor_data = tensor.data();
        assert_eq!(tensor_data[[0, 0]], 1.0);
        assert_eq!(tensor_data[[0, 1]], 2.0);
        assert_eq!(tensor_data[[1, 0]], 3.0);
        assert_eq!(tensor_data[[1, 1]], 4.0);
    }

    #[test]
    fn test_from_shape_vec_invalid() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Tensor::from_shape_vec(&[2, 2], data); // 3 elements for 4 slots
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid dimensions"));

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = Tensor::from_shape_vec(&[2, 2], data); // 5 elements for 4 slots
        assert!(result.is_err());
    }

    #[test]
    fn test_data_accessors() {
        let mut tensor = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Test immutable access
        let data = tensor.data();
        assert_eq!(data[[0, 0]], 1.0);

        // Test mutable access
        let data_mut = tensor.data_mut();
        data_mut[[0, 0]] = 10.0;
        assert_eq!(tensor.data()[[0, 0]], 10.0);
    }

    #[test]
    fn test_add() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let result = a.add(&b).unwrap();

        let expected = [5.0, 7.0, 9.0];
        for (actual, &expected) in result.data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add_2d() {
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = a.add(&b).unwrap();

        let expected = [6.0, 8.0, 10.0, 12.0];
        for (actual, &expected) in result.data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add_shape_mismatch() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let result = a.add(&b);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_add_different_shapes_same_elements() {
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_shape_vec(&[4, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = a.add(&b);
        assert!(result.is_err()); // Should fail even with same number of elements
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_array(Array1::from_vec(vec![2.0, 3.0, 4.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![5.0, 6.0, 7.0]));
        let result = a.mul(&b).unwrap();

        let expected = [10.0, 18.0, 28.0];
        for (actual, &expected) in result.data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mul_shape_mismatch() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let result = a.mul(&b);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_array(
            Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap(),
        );
        let b = Tensor::from_array(
            Array2::from_shape_vec((3, 2), vec![1., 2., 3., 4., 5., 6.]).unwrap(),
        );
        let result = a.matmul(&b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);

        // Expected: [[22, 28], [49, 64]]
        let data = result.data();
        let expected = [22.0, 28.0, 49.0, 64.0];
        for (actual, &expected) in data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_invalid_shapes() {
        let a = Tensor::from_array(Array2::from_elem((2, 3), 1.0));
        let b = Tensor::from_array(Array2::from_elem((4, 2), 1.0)); // Incompatible dimensions
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_matmul_non_2d() {
        // Test with 1D tensor
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array2::from_elem((3, 2), 1.0));
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2D tensors"));

        // Test with 3D tensor
        let a = Tensor::from_array(Array3::from_elem((2, 3, 4), 1.0));
        let b = Tensor::from_array(Array2::from_elem((4, 2), 1.0));
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2D tensors"));
    }

    #[test]
    fn test_matmul_identity() {
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let identity = Tensor::from_shape_vec(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let result = a.matmul(&identity).unwrap();

        // Should get back the original matrix
        for (actual, expected) in result.data().iter().zip(a.data().iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.len(), 6);

        // Test that data is preserved
        let original_data: Vec<f32> = tensor.data().iter().cloned().collect();
        let reshaped_data: Vec<f32> = reshaped.data().iter().cloned().collect();
        for (orig, reshaped) in original_data.iter().zip(reshaped_data.iter()) {
            assert!((orig - reshaped).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reshape_to_1d() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let reshaped = tensor.reshape(&[6]).unwrap();
        assert_eq!(reshaped.shape(), &[6]);
        assert_eq!(reshaped.ndim(), 1);
    }

    #[test]
    fn test_reshape_from_1d() {
        let tensor = Tensor::from_shape_vec(&[6], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let reshaped = tensor.reshape(&[2, 3]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 3]);
        assert_eq!(reshaped.ndim(), 2);
    }

    #[test]
    fn test_reshape_invalid() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let result = tensor.reshape(&[2, 2]); // Different number of elements
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Cannot reshape"));
        assert!(error_msg.contains("6 elements"));
        assert!(error_msg.contains("4 elements"));
    }

    #[test]
    fn test_transpose() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let transposed = tensor.transpose().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);

        // Check specific values
        let orig_data = tensor.data();
        let trans_data = transposed.data();
        assert_eq!(orig_data[[0, 1]], trans_data[[1, 0]]);
        assert_eq!(orig_data[[1, 2]], trans_data[[2, 1]]);
    }

    #[test]
    fn test_transpose_non_2d() {
        // Test with 1D tensor
        let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let result = tensor.transpose();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2D tensors"));

        // Test with 3D tensor
        let tensor = Tensor::from_array(Array3::from_elem((2, 3, 4), 1.0));
        let result = tensor.transpose();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2D tensors"));
    }

    #[test]
    fn test_transpose_square_matrix() {
        let tensor = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape(), &[2, 2]);
        let data = transposed.data();
        assert_eq!(data[[0, 0]], 1.0);
        assert_eq!(data[[0, 1]], 3.0);
        assert_eq!(data[[1, 0]], 2.0);
        assert_eq!(data[[1, 1]], 4.0);
    }

    #[test]
    fn test_relu() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]));
        let result = tensor.relu();

        let expected = [0.0, 0.0, 0.0, 1.0, 2.0];
        for (actual, &expected) in result.data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_relu_all_positive() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let result = tensor.relu();

        // Should be unchanged for all positive values
        for (actual, expected) in result.data().iter().zip(tensor.data().iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_relu_all_negative() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![-1.0, -2.0, -3.0]));
        let result = tensor.relu();

        // Should be all zeros for all negative values
        assert!(result.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_sigmoid() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![0.0]));
        let result = tensor.sigmoid();

        // Sigmoid of 0 should be 0.5
        assert!((result.data()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_extreme_values() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![-10.0, 10.0]));
        let result = tensor.sigmoid();

        let data = result.data();
        // Sigmoid of large negative should be close to 0
        assert!(data[0] < 0.01);
        // Sigmoid of large positive should be close to 1
        assert!(data[1] > 0.99);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![-1.0, 1.0]));
        let result = tensor.sigmoid();

        let data = result.data();
        // Sigmoid is symmetric around 0.5: sigmoid(-x) + sigmoid(x) = 1
        assert!((data[0] + data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_equality() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let c = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 4.0]));
        let d = Tensor::from_array(Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap());

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d); // Different shapes
    }

    #[test]
    fn test_tensor_equality_with_tolerance() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![1.0000001, 2.0000001, 3.0000001]));

        // Should be equal within tolerance
        assert_eq!(a, b);
    }

    #[test]
    fn test_tensor_display() {
        let tensor = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let display_string = format!("{tensor}");

        assert!(display_string.contains("Tensor[2, 2]"));
        assert!(display_string.contains("1"));
        assert!(display_string.contains("2"));
        assert!(display_string.contains("3"));
        assert!(display_string.contains("4"));
    }

    #[test]
    fn test_tensor_clone() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let cloned = tensor.clone();

        assert_eq!(tensor, cloned);
        assert_eq!(tensor.shape(), cloned.shape());
        assert_eq!(tensor.len(), cloned.len());
    }

    #[test]
    fn test_complex_operations_chain() {
        // Test chaining operations
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 2], vec![2.0, 2.0, 2.0, 2.0]).unwrap();

        let added = a.add(&b).unwrap();
        let multiplied = added.mul(&b).unwrap();
        let relu_result = multiplied.relu();
        let sigmoid_result = relu_result.sigmoid();

        assert_eq!(sigmoid_result.shape(), &[2, 2]);
        // All values should be positive after ReLU and between 0 and 1 after sigmoid
        assert!(sigmoid_result.data().iter().all(|&x| x > 0.0 && x < 1.0));
    }
}
