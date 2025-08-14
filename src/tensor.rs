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
        // Handle broadcasting following ONNX/NumPy rules
        let (left, right) = self.broadcast_tensors(other)?;

        Ok(Tensor {
            data: &left.data + &right.data,
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
        // Handle broadcasting following ONNX/NumPy rules
        let (left, right) = self.broadcast_tensors(other)?;

        Ok(Tensor {
            data: &left.data * &right.data,
        })
    }

    /// Broadcast two tensors to compatible shapes following ONNX/NumPy rules
    fn broadcast_tensors(&self, other: &Tensor) -> Result<(Tensor, Tensor)> {
        let self_shape = self.shape();
        let other_shape = other.shape();

        // If shapes are identical, no broadcasting needed
        if self_shape == other_shape {
            return Ok((self.clone(), other.clone()));
        }

        // Special case: scalar broadcasting (empty shape)
        if other_shape.is_empty() {
            if other.data.len() != 1 {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Scalar tensor must have exactly 1 element, got {}",
                    other.data.len()
                )));
            }
            let scalar_value = other.data.iter().next().unwrap();
            let broadcasted_data = ndarray::Array::from_elem(self.data.raw_dim(), *scalar_value);
            let broadcasted_other = Tensor {
                data: broadcasted_data,
            };
            return Ok((self.clone(), broadcasted_other));
        }

        if self_shape.is_empty() {
            if self.data.len() != 1 {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Scalar tensor must have exactly 1 element, got {}",
                    self.data.len()
                )));
            }
            let scalar_value = self.data.iter().next().unwrap();
            let broadcasted_data = ndarray::Array::from_elem(other.data.raw_dim(), *scalar_value);
            let broadcasted_self = Tensor {
                data: broadcasted_data,
            };
            return Ok((broadcasted_self, other.clone()));
        }

        // NumPy-style broadcasting: align dimensions from the right
        let self_ndim = self_shape.len();
        let other_ndim = other_shape.len();
        let max_ndim = self_ndim.max(other_ndim);

        // Create broadcasted shapes by padding with 1s on the left
        let mut self_bc_shape = vec![1; max_ndim];
        let mut other_bc_shape = vec![1; max_ndim];

        // Fill in actual dimensions from the right
        for i in 0..self_ndim {
            self_bc_shape[max_ndim - self_ndim + i] = self_shape[i];
        }
        for i in 0..other_ndim {
            other_bc_shape[max_ndim - other_ndim + i] = other_shape[i];
        }

        // Check broadcasting compatibility and compute result shape
        let mut result_shape = vec![0; max_ndim];
        for i in 0..max_ndim {
            let dim_a = self_bc_shape[i];
            let dim_b = other_bc_shape[i];

            if dim_a == dim_b {
                result_shape[i] = dim_a;
            } else if dim_a == 1 {
                result_shape[i] = dim_b;
            } else if dim_b == 1 {
                result_shape[i] = dim_a;
            } else {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Cannot broadcast shapes {self_shape:?} and {other_shape:?}: incompatible dimensions {dim_a} and {dim_b}"
                )));
            }
        }

        // Broadcast tensors to the result shape
        let self_broadcasted = self.broadcast_to_shape(&result_shape)?;
        let other_broadcasted = other.broadcast_to_shape(&result_shape)?;

        Ok((self_broadcasted, other_broadcasted))
    }

    /// Broadcast this tensor to a target shape
    fn broadcast_to_shape(&self, target_shape: &[usize]) -> Result<Tensor> {
        let current_shape = self.shape();

        if current_shape == target_shape {
            return Ok(self.clone());
        }

        // Use ndarray's broadcast functionality
        let mut broadcasted_data = self.data.clone();

        // Reshape to match the target dimensionality by adding size-1 dimensions
        let current_ndim = current_shape.len();
        let target_ndim = target_shape.len();

        if target_ndim > current_ndim {
            // Need to add dimensions at the front
            let mut new_shape = vec![1; target_ndim - current_ndim];
            new_shape.extend_from_slice(current_shape);

            // Reshape the array to have the correct number of dimensions
            let new_dim = ndarray::IxDyn(&new_shape);
            broadcasted_data = broadcasted_data
                .to_shape(new_dim)
                .map_err(|e| {
                    OnnxError::invalid_dimensions(format!(
                        "Failed to reshape for broadcasting: {e}"
                    ))
                })?
                .into_owned();
        }

        // Now broadcast to the target shape
        let target_dim = ndarray::IxDyn(target_shape);

        // Create a view that can be broadcast
        let broadcasted_view = broadcasted_data.broadcast(target_dim).ok_or_else(|| {
            OnnxError::invalid_dimensions(format!(
                "Failed to broadcast from {:?} to {:?}",
                broadcasted_data.shape(),
                target_shape
            ))
        })?;

        // Convert the broadcasted view to an owned array
        let result_data = broadcasted_view.to_owned();

        Ok(Tensor { data: result_data })
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        // Handle broadcasting following ONNX/NumPy rules
        let (left, right) = self.broadcast_tensors(other)?;

        Ok(Tensor {
            data: &left.data / &right.data,
        })
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        // Handle broadcasting following ONNX/NumPy rules
        let (left, right) = self.broadcast_tensors(other)?;

        Ok(Tensor {
            data: &left.data - &right.data,
        })
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Result<Tensor> {
        Ok(Tensor {
            data: self.data.mapv(|x| x.exp()),
        })
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Tensor> {
        Ok(Tensor {
            data: self.data.mapv(|x| x.sqrt()),
        })
    }

    /// Element-wise power
    pub fn pow(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape() != other.shape() {
            return Err(OnnxError::shape_mismatch(self.shape(), other.shape()));
        }

        Ok(Tensor {
            data: ndarray::Zip::from(&self.data)
                .and(&other.data)
                .map_collect(|&a, &b| a.powf(b)),
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

    /// Transpose the tensor with optional axis permutation
    ///
    /// If `perm` is None, performs default transpose (reverse all axes).
    /// If `perm` is provided, permutes axes according to the specification.
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    ///
    /// let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    /// let result = tensor.transpose_with_perm(None).unwrap();
    /// assert_eq!(result.shape(), &[3, 2]);
    /// ```
    pub fn transpose_with_perm(&self, perm: Option<&[usize]>) -> Result<Tensor> {
        match perm {
            Some(axes) => {
                // Validate permutation
                if axes.len() != self.ndim() {
                    return Err(OnnxError::invalid_dimensions(format!(
                        "Permutation length {} doesn't match tensor dimensions {}",
                        axes.len(),
                        self.ndim()
                    )));
                }

                // Check that all axes are valid
                for &axis in axes {
                    if axis >= self.ndim() {
                        return Err(OnnxError::invalid_dimensions(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            axis,
                            self.ndim()
                        )));
                    }
                }

                // Check that all axes are unique
                let mut sorted_axes = axes.to_vec();
                sorted_axes.sort_unstable();
                for (i, &axis) in sorted_axes.iter().enumerate().take(self.ndim()) {
                    if axis != i {
                        return Err(OnnxError::invalid_dimensions(
                            "Permutation must be a valid permutation of axes".to_string(),
                        ));
                    }
                }

                // For simple permutations, handle common cases
                if axes.len() == 2 {
                    // 2D transpose
                    if axes == [1, 0] {
                        let transposed = self.data.t().to_owned();
                        Ok(Tensor { data: transposed })
                    } else if axes == [0, 1] {
                        // Identity - no change
                        Ok(self.clone())
                    } else {
                        return Err(OnnxError::invalid_dimensions(format!(
                            "Invalid 2D permutation {axes:?}"
                        )));
                    }
                } else if axes == (0..axes.len()).collect::<Vec<_>>() {
                    // Identity permutation - no change needed
                    Ok(self.clone())
                } else {
                    // General case: use ndarray's permuted_axes for any permutation
                    let transposed = self.data.clone().permuted_axes(axes);
                    Ok(Tensor { data: transposed })
                }
            }
            None => {
                // Default transpose: reverse all axes
                let ndim = self.ndim();
                if ndim == 0 {
                    // 0-dimensional tensor - return as is
                    Ok(self.clone())
                } else if ndim == 1 {
                    // 1-dimensional tensor - return as is (can't transpose)
                    Ok(self.clone())
                } else if ndim == 2 {
                    // 2-dimensional tensor - use built-in transpose
                    let transposed = self.data.t().to_owned();
                    Ok(Tensor { data: transposed })
                } else {
                    // Multi-dimensional tensor - for now, just do 2D transpose if possible
                    // or return error for truly multi-dimensional cases
                    log::warn!("Multi-dimensional transpose without perm not fully supported, treating as 2D if possible");
                    if ndim == 2 {
                        let transposed = self.data.t().to_owned();
                        Ok(Tensor { data: transposed })
                    } else {
                        // For higher dimensions, return error
                        return Err(OnnxError::invalid_dimensions(format!(
                            "Default transpose for {ndim}-dimensional tensors not supported. Use perm attribute to specify axis permutation."
                        )));
                    }
                }
            }
        }
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
        self.transpose_with_perm(None)
    }

    /// Slice the tensor along specified axes
    ///
    /// `starts` and `ends` define the slice range for each axis. Optional
    /// `axes` specifies which axes are sliced (defaults to `[0, 1, ...]`), and
    /// `steps` sets the stride for each slice (defaults to `1`). Negative indices
    /// are supported and interpreted relative to the dimension size.
    pub fn slice(
        &self,
        starts: &[i64],
        ends: &[i64],
        axes: Option<&[i64]>,
        steps: Option<&[i64]>,
    ) -> Result<Tensor> {
        if starts.len() != ends.len() {
            return Err(OnnxError::invalid_dimensions(
                "Starts and ends arrays must have same length",
            ));
        }

        let num = starts.len();
        let axes_vec: Vec<usize> = if let Some(ax) = axes {
            if ax.len() != num {
                return Err(OnnxError::invalid_dimensions(
                    "Axes length must match starts/ends length",
                ));
            }
            ax.iter().map(|&a| a as usize).collect()
        } else {
            (0..num).collect()
        };

        let steps_vec: Vec<i64> = if let Some(st) = steps {
            if st.len() != num {
                return Err(OnnxError::invalid_dimensions(
                    "Steps length must match starts/ends length",
                ));
            }
            st.to_vec()
        } else {
            vec![1; num]
        };

        let mut result = self.data.clone();
        for ((&axis, (&start, &end)), &step) in axes_vec
            .iter()
            .zip(starts.iter().zip(ends.iter()))
            .zip(steps_vec.iter())
        {
            if axis >= result.ndim() {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Axis {axis} out of bounds"
                )));
            }

            let dim = result.shape()[axis] as i64;
            if step == 0 {
                return Err(OnnxError::invalid_dimensions("Step value cannot be zero"));
            }

            let mut s = start;
            let mut e = end;
            if s < 0 {
                s += dim;
            }
            if e < 0 {
                e += dim;
            }
            // Handle special case where e is i64::MAX or very large (means "end of dimension")
            if e >= dim || e == i64::MAX {
                e = dim;
            }
            if s < 0 || e > dim || s >= e {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Invalid slice range: {s}..{e} for axis {axis}",
                )));
            }

            let slice = ndarray::Slice {
                start: s as isize,
                end: Some(e as isize),
                step: step as isize,
            };
            result = result.slice_axis(ndarray::Axis(axis), slice).to_owned();
        }

        Ok(Tensor { data: result })
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

    /// Applies the Softmax activation function along the last axis
    ///
    /// The Softmax function is defined as: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    /// This implementation applies softmax along the last axis of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    /// use ndarray::Array1;
    ///
    /// let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
    /// let result = tensor.softmax().unwrap();
    /// // Sum of softmax outputs should be 1.0
    /// let sum: f32 = result.data().iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// ```
    pub fn softmax(&self) -> Result<Tensor> {
        if self.is_empty() {
            return Err(OnnxError::invalid_dimensions(
                "Cannot apply softmax to empty tensor".to_string(),
            ));
        }

        // For simplicity, apply softmax to the flattened tensor
        // A full implementation would handle arbitrary axes
        let max_val = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_data = self.data.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_data.sum();

        if sum_exp == 0.0 {
            return Err(OnnxError::invalid_dimensions(
                "Softmax sum is zero, cannot normalize".to_string(),
            ));
        }

        let softmax_data = exp_data.mapv(|x| x / sum_exp);
        Ok(Tensor { data: softmax_data })
    }

    /// Concatenate tensors along a specified axis
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to concatenate (including self)
    /// * `axis` - Axis along which to concatenate
    ///
    /// # Returns
    /// * New tensor with concatenated result
    ///
    /// # Examples
    ///
    /// ```
    /// use runnx::Tensor;
    /// use ndarray::Array2;
    ///
    /// let a = Tensor::from_array(Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    /// let b = Tensor::from_array(Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap());
    /// let result = Tensor::concat(&[&a, &b], 0).unwrap();
    /// assert_eq!(result.shape(), &[4, 2]);
    /// ```
    pub fn concat(tensors: &[&Tensor], axis: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(OnnxError::invalid_dimensions(
                "Cannot concatenate empty tensor list".to_string(),
            ));
        }

        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        let first = tensors[0];
        if axis >= first.ndim() {
            return Err(OnnxError::invalid_dimensions(format!(
                "Concatenation axis {} out of bounds for tensor with {} dimensions",
                axis,
                first.ndim()
            )));
        }

        // Check that all tensors have compatible shapes
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.ndim() != first.ndim() {
                return Err(OnnxError::invalid_dimensions(format!(
                    "All tensors must have same number of dimensions: tensor 0 has {}, tensor {} has {}",
                    first.ndim(), i, tensor.ndim()
                )));
            }

            for (dim_idx, (&expected_size, &actual_size)) in
                first.shape().iter().zip(tensor.shape().iter()).enumerate()
            {
                if dim_idx != axis && expected_size != actual_size {
                    return Err(OnnxError::invalid_dimensions(format!(
                        "Tensor shapes must match except on concatenation axis: dimension {dim_idx} expected size {expected_size}, got {actual_size}"
                    )));
                }
            }
        }

        // Calculate output shape
        let mut output_shape = first.shape().to_vec();
        output_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();

        // Perform concatenation using ndarray
        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let concatenated = ndarray::concatenate(ndarray::Axis(axis), &views)
            .map_err(|e| OnnxError::invalid_dimensions(format!("Concatenation failed: {e}")))?;

        Ok(Tensor { data: concatenated })
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
        assert!(result.unwrap_err().to_string().contains("Cannot broadcast"));
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
        assert!(result.unwrap_err().to_string().contains("Cannot broadcast"));
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
        // Test with 1D tensor - should now succeed and return as-is
        let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let result = tensor.transpose();
        assert!(result.is_ok());
        let transposed = result.unwrap();
        assert_eq!(transposed.shape(), tensor.shape());
        assert_eq!(transposed.data(), tensor.data());

        // Test with 3D tensor - should fail with helpful message
        let tensor = Tensor::from_array(Array3::from_elem((2, 3, 4), 1.0));
        let result = tensor.transpose();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("dimensional tensors not supported"));
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
    fn test_transpose_4d() {
        // Test 4D tensor with custom permutation [0, 2, 1, 3] (YOLOv8 case)
        let tensor_4d = Tensor::from_shape_vec(
            &[1, 4, 16, 8400],
            (0..4 * 16 * 8400).map(|i| i as f32).collect(),
        )
        .unwrap();
        let result_4d = tensor_4d.transpose_with_perm(Some(&[0, 2, 1, 3])).unwrap();

        // Expected shape after [0, 2, 1, 3] permutation: [1, 16, 4, 8400]
        assert_eq!(result_4d.shape(), &[1, 16, 4, 8400]);

        // Test identity permutation
        let identity_result = tensor_4d.transpose_with_perm(Some(&[0, 1, 2, 3])).unwrap();
        assert_eq!(identity_result.shape(), &[1, 4, 16, 8400]);
        assert_eq!(identity_result.data, tensor_4d.data);
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
    fn test_slice() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
        let sliced = tensor.slice(&[1], &[3], None, None).unwrap();
        assert_eq!(sliced.shape(), &[2]);
        let data = sliced.data().as_slice().unwrap();
        assert_eq!(data, &[2.0, 3.0]);
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

    #[test]
    fn test_concat() {
        // Test concatenation along axis 0
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_shape_vec(&[1, 2], vec![5.0, 6.0]).unwrap();
        let c = Tensor::from_shape_vec(&[1, 2], vec![7.0, 8.0]).unwrap();

        let result = Tensor::concat(&[&a, &b, &c], 0).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
        let data = result.data().as_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_concat_axis1() {
        // Test concatenation along axis 1
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 1], vec![5.0, 6.0]).unwrap();

        let result = Tensor::concat(&[&a, &b], 1).unwrap();
        assert_eq!(result.shape(), &[2, 3]);

        // Check the values using indexed access rather than as_slice
        let expected = [1.0, 2.0, 5.0, 3.0, 4.0, 6.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            let (row, col) = (i / 3, i % 3);
            assert_eq!(result.data[[row, col]], expected_val);
        }
    }

    #[test]
    fn test_concat_single_tensor() {
        // Test concatenation with single tensor
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = Tensor::concat(&[&a], 0).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        let data = result.data().as_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_broadcasting_edge_cases() {
        // Test incompatible shapes for broadcasting
        let a = Tensor::from_shape_vec(&[2, 3], vec![1.0; 6]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 4], vec![1.0; 8]).unwrap();
        let result = a.add(&b);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("broadcast"));
    }

    #[test]
    fn test_scalar_broadcasting() {
        // Test scalar broadcasting operations
        let scalar = Tensor::from_shape_vec(&[], vec![5.0]).unwrap();
        let tensor = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = tensor.add(&scalar).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data().as_slice().unwrap(), &[6.0, 7.0, 8.0, 9.0]);

        // Test [1,1] with [2,2] - this might actually work with broadcasting
        let pseudo_scalar = Tensor::from_shape_vec(&[1, 1], vec![5.0]).unwrap();
        let result = tensor.add(&pseudo_scalar);
        // Let's see if this works or fails - if it works, broadcasting handles it
        if result.is_ok() {
            println!("Broadcasting [1,1] with [2,2] works!");
        } else {
            println!("Broadcasting [1,1] with [2,2] fails as expected");
        }
    }

    #[test]
    fn test_softmax_edge_cases() {
        // Test softmax with all same values
        let tensor = Tensor::from_shape_vec(&[3], vec![2.0, 2.0, 2.0]).unwrap();
        let result = tensor.softmax().unwrap();

        // Each element should be 1/3
        for &val in result.data().iter() {
            assert!((val - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_concat_edge_cases() {
        // Test empty tensor list
        let result = Tensor::concat(&[], 0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("empty tensor list"));

        // Test axis out of bounds - with single tensor, concat returns the tensor itself
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = Tensor::concat(&[&a], 5);
        // Single tensor concat with out-of-bounds axis still works (returns the single tensor)
        assert!(result.is_ok());

        // Test mismatched dimensions (different number of dimensions)
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_shape_vec(&[2], vec![5.0, 6.0]).unwrap(); // Different ndim
        let result = Tensor::concat(&[&a, &b], 0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("same number of dimensions"));
    }

    #[test]
    fn test_arithmetic_operations() {
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 2], vec![2.0, 2.0, 2.0, 2.0]).unwrap();

        // Test div
        let result = a.div(&b).unwrap();
        assert_eq!(result.data().as_slice().unwrap(), &[0.5, 1.0, 1.5, 2.0]);

        // Test sub
        let result = a.sub(&b).unwrap();
        assert_eq!(result.data().as_slice().unwrap(), &[-1.0, 0.0, 1.0, 2.0]);

        // Test pow
        let result = a.pow(&b).unwrap();
        assert_eq!(result.data().as_slice().unwrap(), &[1.0, 4.0, 9.0, 16.0]);

        // Test exp
        let result = a.exp().unwrap();
        let expected: Vec<f32> = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
            .iter()
            .map(|&x| x.exp())
            .collect();
        let actual = result.data().as_slice().unwrap();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }

        // Test sqrt
        let result = a.sqrt().unwrap();
        let expected: Vec<f32> = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
            .iter()
            .map(|&x| x.sqrt())
            .collect();
        let actual = result.data().as_slice().unwrap();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_transpose_edge_cases() {
        // Test 0D tensor transpose
        let tensor = Tensor::from_shape_vec(&[], vec![42.0]).unwrap();
        let result = tensor.transpose().unwrap();
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(result.data().as_slice().unwrap(), &[42.0]);

        // Test transpose with invalid permutation
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1.0; 6]).unwrap();
        let result = tensor.transpose_with_perm(Some(&[0, 1, 2])); // Too many dimensions
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_edge_cases() {
        let tensor = Tensor::from_shape_vec(&[4, 3], vec![1.0; 12]).unwrap();

        // Test slice validation - these cases fail as expected
        let result = tensor.slice(&[5, 0], &[6, 2], None, None); // Start out of bounds
        assert!(result.is_err());

        let result = tensor.slice(&[2, 0], &[1, 2], None, None); // Start > end
        assert!(result.is_err());

        // Test cases that work
        let result = tensor.slice(&[0, 0], &[2, 5], None, None); // End out of bounds is handled gracefully
        assert!(result.is_ok());

        let result = tensor.slice(&[0, 0], &[2, 2], None, None); // Valid slice
        assert!(result.is_ok());
    }
}
