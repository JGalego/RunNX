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
            .into_shape(IxDyn(new_shape))
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
    use ndarray::{Array1, Array2};

    #[test]
    fn test_tensor_creation() {
        let array = Array2::from_elem((2, 3), 1.0);
        let tensor = Tensor::from_array(array);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.ndim(), 2);
    }

    #[test]
    fn test_zeros_and_ones() {
        let zeros = Tensor::zeros(&[2, 3]);
        assert!(zeros.data().iter().all(|&x| x == 0.0));

        let ones = Tensor::ones(&[2, 3]);
        assert!(ones.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_shape_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_shape_vec(&[2, 2], data).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn test_from_shape_vec_invalid() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Tensor::from_shape_vec(&[2, 2], data);
        assert!(result.is_err());
    }

    #[test]
    fn test_add() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let result = a.add(&b).unwrap();

        let expected = vec![5.0, 7.0, 9.0];
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
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_array(Array1::from_vec(vec![2.0, 3.0, 4.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![5.0, 6.0, 7.0]));
        let result = a.mul(&b).unwrap();

        let expected = vec![10.0, 18.0, 28.0];
        for (actual, &expected) in result.data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
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
        let expected = vec![22.0, 28.0, 49.0, 64.0];
        for (actual, &expected) in data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_invalid_shapes() {
        let a = Tensor::from_array(Array2::from_elem((2, 3), 1.0));
        let b = Tensor::from_array(Array2::from_elem((4, 2), 1.0)); // Incompatible
        let result = a.matmul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.len(), 6);
    }

    #[test]
    fn test_reshape_invalid() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let result = tensor.reshape(&[2, 2]); // Different number of elements
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let transposed = tensor.transpose().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
    }

    #[test]
    fn test_relu() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]));
        let result = tensor.relu();

        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        for (actual, &expected) in result.data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![0.0]));
        let result = tensor.sigmoid();

        // Sigmoid of 0 should be 0.5
        assert!((result.data()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_equality() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let c = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 4.0]));

        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
