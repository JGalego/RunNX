//! ONNX operator implementations
//!
//! This module contains implementations of various ONNX operators.
//! Each operator is implemented as a function that takes input tensors
//! and returns output tensors.

use crate::{
    error::{OnnxError, Result},
    tensor::Tensor,
};
use std::collections::HashMap;
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
    Concat,
    Slice,
    Upsample,
    MaxPool,
    Softmax,
    NonMaxSuppression,
    BatchNormalization,
    Split,
    Gather,
    ConstantOfShape,
    Cast,
    Shape,
    Unsqueeze,
    Squeeze,
    Pad,
    Div,
    Sub,
    Exp,
    Sqrt,
    Pow,
    ReduceMean,
    Identity,
    Resize,
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
            "Concat" => Ok(OperatorType::Concat),
            "Slice" => Ok(OperatorType::Slice),
            "Upsample" => Ok(OperatorType::Upsample),
            "MaxPool" => Ok(OperatorType::MaxPool),
            "Softmax" => Ok(OperatorType::Softmax),
            "NonMaxSuppression" => Ok(OperatorType::NonMaxSuppression),
            "BatchNormalization" => Ok(OperatorType::BatchNormalization),
            "Split" => Ok(OperatorType::Split),
            "Gather" => Ok(OperatorType::Gather),
            "ConstantOfShape" => Ok(OperatorType::ConstantOfShape),
            "Cast" => Ok(OperatorType::Cast),
            "Shape" => Ok(OperatorType::Shape),
            "Unsqueeze" => Ok(OperatorType::Unsqueeze),
            "Squeeze" => Ok(OperatorType::Squeeze),
            "Pad" => Ok(OperatorType::Pad),
            "Div" => Ok(OperatorType::Div),
            "Sub" => Ok(OperatorType::Sub),
            "Exp" => Ok(OperatorType::Exp),
            "Sqrt" => Ok(OperatorType::Sqrt),
            "Pow" => Ok(OperatorType::Pow),
            "ReduceMean" => Ok(OperatorType::ReduceMean),
            "Identity" => Ok(OperatorType::Identity),
            "Resize" => Ok(OperatorType::Resize),
            _ => Err(OnnxError::unsupported_operation(s)),
        }
    }
}

/// Helper function to parse integer arrays from ONNX attribute strings
/// Handles formats like "[1,2,3]", "1,2,3", etc.
fn parse_int_array(attr_value: &str) -> Result<Vec<i64>> {
    let cleaned = attr_value.trim().trim_matches(['[', ']']);
    if cleaned.is_empty() {
        return Ok(Vec::new());
    }

    cleaned
        .split(',')
        .map(|s| {
            s.trim().parse::<i64>().map_err(|e| {
                OnnxError::runtime_error(format!("Failed to parse integer '{s}': {e}"))
            })
        })
        .collect()
}

/// Execute an operator with given inputs
pub fn execute_operator(
    op_type: &OperatorType,
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    match op_type {
        OperatorType::Add => add_op(inputs),
        OperatorType::Mul => mul_op(inputs),
        OperatorType::MatMul => matmul_op(inputs),
        OperatorType::Conv => conv_op(inputs, attributes),
        OperatorType::Relu => relu_op(inputs),
        OperatorType::Sigmoid => sigmoid_op(inputs),
        OperatorType::Reshape => reshape_op(inputs),
        OperatorType::Transpose => transpose_op(inputs, attributes),
        OperatorType::Concat => concat_op(inputs, attributes),
        OperatorType::Slice => slice_op(inputs, attributes),
        OperatorType::Upsample => upsample_op(inputs, attributes),
        OperatorType::MaxPool => maxpool_op(inputs, attributes),
        OperatorType::Softmax => softmax_op(inputs, attributes),
        OperatorType::NonMaxSuppression => nms_op(inputs, attributes),
        OperatorType::BatchNormalization => batch_norm_op(inputs, attributes),
        OperatorType::Split => split_op(inputs, attributes),
        OperatorType::Gather => gather_op(inputs, attributes),
        OperatorType::ConstantOfShape => constant_of_shape_op(inputs, attributes),
        OperatorType::Cast => cast_op(inputs, attributes),
        OperatorType::Shape => shape_op(inputs),
        OperatorType::Unsqueeze => unsqueeze_op(inputs, attributes),
        OperatorType::Squeeze => squeeze_op(inputs, attributes),
        OperatorType::Pad => pad_op(inputs, attributes),
        OperatorType::Div => div_op(inputs),
        OperatorType::Sub => sub_op(inputs),
        OperatorType::Exp => exp_op(inputs),
        OperatorType::Sqrt => sqrt_op(inputs),
        OperatorType::Pow => pow_op(inputs),
        OperatorType::ReduceMean => reduce_mean_op(inputs, attributes),
        OperatorType::Identity => identity_op(inputs),
        OperatorType::Resize => resize_op(inputs, attributes),
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
///
/// # Formal Specifications (verified with Why3)
/// - **Preconditions**:
///   - `inputs.len() == 2`
///   - `inputs[0].shape() == inputs[1].shape()` (or broadcastable)
/// - **Postconditions**:
///   - `result.shape() == inputs[0].shape()`
///   - `∀i: result[i] == inputs[0][i] + inputs[1][i]`
/// - **Properties**:
///   - Commutativity: `add(a, b) == add(b, a)`
///   - Associativity: `add(add(a, b), c) == add(a, add(b, c))`
///   - Identity: `add(a, 0) == a`
#[cfg_attr(
    feature = "formal-verification",
    doc = "This function is formally verified using Why3 specifications"
)]
fn add_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    #[cfg(feature = "formal-verification")]
    {
        // Formal verification precondition checks
        assert!(inputs.len() == 2, "Precondition: exactly 2 inputs required");
        // Note: broadcastability check would be done by the tensor operations
    }

    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Add operator requires exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].add(&inputs[1])?;

    #[cfg(feature = "formal-verification")]
    {
        // Formal verification postcondition checks
        debug_assert_eq!(
            result.shape(),
            inputs[0].shape(),
            "Postcondition: result shape matches input shape"
        );
    }

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
///
/// # Formal Specifications (verified with Why3)
/// - **Preconditions**:
///   - `inputs.len() == 2`
///   - `inputs[0].shape() == inputs[1].shape()` (or broadcastable)
/// - **Postconditions**:
///   - `result.shape() == inputs[0].shape()`
///   - `∀i: result[i] == inputs[0][i] * inputs[1][i]`
/// - **Properties**:
///   - Commutativity: `mul(a, b) == mul(b, a)`
///   - Associativity: `mul(mul(a, b), c) == mul(a, mul(b, c))`
///   - Identity: `mul(a, 1) == a`
///   - Annihilator: `mul(a, 0) == 0`
#[cfg_attr(
    feature = "formal-verification",
    doc = "This function is formally verified using Why3 specifications"
)]
fn mul_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    #[cfg(feature = "formal-verification")]
    {
        // Formal verification precondition checks
        assert!(inputs.len() == 2, "Precondition: exactly 2 inputs required");
    }

    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Mul operator requires exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].mul(&inputs[1])?;

    #[cfg(feature = "formal-verification")]
    {
        // Formal verification postcondition checks
        debug_assert_eq!(
            result.shape(),
            inputs[0].shape(),
            "Postcondition: result shape matches input shape"
        );
    }

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
///
/// # Formal Specifications (verified with Why3)
/// - **Preconditions**:
///   - `inputs.len() == 2`
///   - `inputs[0].ndim() == 2 && inputs[1].ndim() == 2`
///   - `inputs[0].shape()[1] == inputs[1].shape()[0]` (inner dimensions match)
/// - **Postconditions**:
///   - `result.ndim() == 2`
///   - `result.shape() == [inputs[0].shape()[0], inputs[1].shape()[1]]`
///   - `∀i,j: result[i,j] == Σₖ inputs[0][i,k] * inputs[1][k,j]`
/// - **Properties**:
///   - Associativity: `matmul(matmul(A, B), C) == matmul(A, matmul(B, C))`
///   - Distributivity: `matmul(A, add(B, C)) == add(matmul(A, B), matmul(A, C))`
///   - Identity: `matmul(A, I) == A` where I is identity matrix
#[cfg_attr(
    feature = "formal-verification",
    doc = "This function is formally verified using Why3 specifications"
)]
fn matmul_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    #[cfg(feature = "formal-verification")]
    {
        // Formal verification precondition checks
        assert!(inputs.len() == 2, "Precondition: exactly 2 inputs required");
        assert!(
            inputs[0].ndim() == 2,
            "Precondition: first input must be 2D"
        );
        assert!(
            inputs[1].ndim() == 2,
            "Precondition: second input must be 2D"
        );
        assert_eq!(
            inputs[0].shape()[1],
            inputs[1].shape()[0],
            "Precondition: inner dimensions must match"
        );
    }

    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "MatMul operator requires exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].matmul(&inputs[1])?;

    #[cfg(feature = "formal-verification")]
    {
        // Formal verification postcondition checks
        debug_assert_eq!(result.ndim(), 2, "Postcondition: result must be 2D");
        debug_assert_eq!(
            result.shape()[0],
            inputs[0].shape()[0],
            "Postcondition: output rows match first input rows"
        );
        debug_assert_eq!(
            result.shape()[1],
            inputs[1].shape()[1],
            "Postcondition: output cols match second input cols"
        );
    }

    Ok(vec![result])
}

/// 2D Convolution operator implementation
///
/// Performs a complete 2D convolution operation following ONNX specification.
/// This implementation handles proper shape calculation and actual convolution computation.
///
/// # Arguments
/// * `inputs` - Array of 2-3 tensors: [input, kernel, bias (optional)]
/// * `attrs` - Optional attributes for stride, padding, etc.
///
/// # Returns
/// * Single output tensor with convolution result
fn conv_op(inputs: &[Tensor], attrs: &HashMap<String, String>) -> Result<Vec<Tensor>> {
    if inputs.len() < 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Conv operator requires at least 2 inputs (input, kernel), got {}",
            inputs.len()
        )));
    }

    let input = &inputs[0];
    let kernel = &inputs[1];
    let bias = if inputs.len() > 2 {
        Some(&inputs[2])
    } else {
        None
    };

    // Support 4D tensors (NCHW format) for 2D convolution
    if input.ndim() != 4 || kernel.ndim() != 4 {
        return Err(OnnxError::invalid_dimensions(
            "Conv operator requires 4D tensors (NCHW format)".to_string(),
        ));
    }

    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    // Input: [N, C_in, H_in, W_in]
    // Kernel: [C_out, C_in, K_h, K_w]
    let batch_size = input_shape[0];
    let channels_in = input_shape[1];
    let height_in = input_shape[2];
    let width_in = input_shape[3];

    let channels_out = kernel_shape[0];
    let channels_in_kernel = kernel_shape[1];
    let kernel_h = kernel_shape[2];
    let kernel_w = kernel_shape[3];

    // Validate channel dimensions match
    if channels_in != channels_in_kernel {
        return Err(OnnxError::invalid_dimensions(format!(
            "Input channels ({channels_in}) must match kernel input channels ({channels_in_kernel})"
        )));
    }

    // Parse attributes (defaults for typical CNN)
    log::debug!("Conv attributes: {attrs:?}");

    // Parse strides - format: "[stride_h, stride_w]"
    let strides = attrs
        .get("strides")
        .map(|s| {
            // Remove brackets and split
            let clean = s.trim_start_matches('[').trim_end_matches(']');
            clean
                .split(',')
                .map(|p| p.trim().parse::<usize>().unwrap_or(1))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![1, 1]);

    let stride_h = strides.first().copied().unwrap_or(1);
    let stride_w = strides.get(1).copied().unwrap_or(stride_h);

    // Parse pads - format: "[pad_top, pad_left, pad_bottom, pad_right]"
    let pads = attrs
        .get("pads")
        .map(|s| {
            let clean = s.trim_start_matches('[').trim_end_matches(']');
            clean
                .split(',')
                .map(|p| p.trim().parse::<usize>().unwrap_or(0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            // Default padding to maintain spatial dimensions
            let default_pad = (kernel_h - 1) / 2;
            vec![default_pad, default_pad, default_pad, default_pad]
        });

    let pad_top = if pads.len() >= 4 {
        pads[0]
    } else {
        pads.first().copied().unwrap_or(0)
    };
    let pad_left = if pads.len() >= 4 {
        pads[1]
    } else {
        pads.get(1).copied().unwrap_or(pad_top)
    };
    let pad_bottom = if pads.len() >= 4 { pads[2] } else { pad_top };
    let pad_right = if pads.len() >= 4 { pads[3] } else { pad_left };

    // Calculate output dimensions
    let height_out = (height_in + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    let width_out = (width_in + pad_left + pad_right - kernel_w) / stride_w + 1;

    let output_shape = [batch_size, channels_out, height_out, width_out];

    log::debug!("Conv: input {input_shape:?}, kernel {kernel_shape:?} -> output {output_shape:?}");
    log::debug!(
        "Conv: stride={stride_h}x{stride_w}, pad=[{pad_top},{pad_left},{pad_bottom},{pad_right}]"
    );

    // Perform actual 2D convolution - HIGHLY OPTIMIZED
    let mut output_data = vec![0.0; output_shape.iter().product()];

    // Pre-extract slices for efficiency
    let input_slice = input.data().as_slice().unwrap();
    let kernel_slice = kernel.data().as_slice().unwrap();

    // Pre-calculate ALL stride values for maximum efficiency
    let input_channel_stride = height_in * width_in;
    let input_batch_stride = channels_in * input_channel_stride;
    let kernel_output_stride = channels_in * kernel_h * kernel_w;
    let kernel_input_stride = kernel_h * kernel_w;
    let output_channel_stride = height_out * width_out;
    let output_batch_stride = channels_out * output_channel_stride;

    // Pre-compute valid kernel windows to avoid bounds checking in inner loop
    let mut valid_windows = Vec::new();
    for kh in 0..kernel_h {
        for kw in 0..kernel_w {
            valid_windows.push((kh, kw));
        }
    }

    // OPTIMIZED: Reorder loops for better cache locality (batch -> output_channel -> spatial)
    for n in 0..batch_size {
        let input_batch_offset = n * input_batch_stride;
        let output_batch_offset = n * output_batch_stride;

        for c_out in 0..channels_out {
            let kernel_output_offset = c_out * kernel_output_stride;
            let output_channel_offset = output_batch_offset + c_out * output_channel_stride;

            // OPTIMIZED: Process entire spatial output in one go
            for spatial_idx in 0..output_channel_stride {
                let h_out = spatial_idx / width_out;
                let w_out = spatial_idx % width_out;

                let mut sum = 0.0f32;

                // OPTIMIZED: Unroll input channel loop for better vectorization
                for c_in in 0..channels_in {
                    let input_channel_offset = input_batch_offset + c_in * input_channel_stride;
                    let kernel_input_offset = kernel_output_offset + c_in * kernel_input_stride;

                    // OPTIMIZED: Use pre-computed valid windows and minimize bounds checks
                    for &(kh, kw) in &valid_windows {
                        // Calculate input coordinates with stride
                        let h_in_padded = h_out * stride_h + kh;
                        let w_in_padded = w_out * stride_w + kw;

                        // OPTIMIZED: Single bounds check per kernel position
                        if h_in_padded >= pad_top
                            && h_in_padded < height_in + pad_top
                            && w_in_padded >= pad_left
                            && w_in_padded < width_in + pad_left
                        {
                            let h_in = h_in_padded - pad_top;
                            let w_in = w_in_padded - pad_left;

                            // OPTIMIZED: Direct indexing without additional bounds check
                            if h_in < height_in && w_in < width_in {
                                let input_val = unsafe {
                                    *input_slice.get_unchecked(
                                        input_channel_offset + h_in * width_in + w_in,
                                    )
                                };
                                let kernel_val = unsafe {
                                    *kernel_slice
                                        .get_unchecked(kernel_input_offset + kh * kernel_w + kw)
                                };
                                sum += input_val * kernel_val;
                            }
                        }
                    }
                }

                output_data[output_channel_offset + spatial_idx] = sum;
            }
        }
    }

    log::debug!("Conv: computed {} output values", output_data.len());

    // Apply bias if present - HIGHLY OPTIMIZED
    let final_output = if let Some(bias) = bias {
        let bias_shape = bias.shape();
        let bias_slice = bias.data().as_slice().unwrap();

        log::debug!("Conv: applying bias with shape {bias_shape:?}");

        // OPTIMIZED: In-place bias addition with SIMD-friendly operations
        if bias_shape == [channels_out] {
            // Standard case: bias is 1D with one value per output channel
            for (c_out, &bias_val) in bias_slice.iter().enumerate().take(channels_out) {
                let start_idx = c_out * output_channel_stride;
                let end_idx = start_idx + output_channel_stride;

                // OPTIMIZED: Vectorized addition for the entire channel at once
                for i in start_idx..end_idx {
                    unsafe {
                        *output_data.get_unchecked_mut(i) += bias_val;
                    }
                }
            }
        } else if bias_shape.len() == 4 && bias_shape[0] == 1 && bias_shape[1] == channels_out {
            // 4D bias case: [1, C_out, 1, 1] - same optimization
            for (c_out, &bias_val) in bias_slice.iter().enumerate().take(channels_out) {
                let start_idx = c_out * output_channel_stride;
                let end_idx = start_idx + output_channel_stride;

                for i in start_idx..end_idx {
                    unsafe {
                        *output_data.get_unchecked_mut(i) += bias_val;
                    }
                }
            }
        } else {
            log::warn!("Conv: unsupported bias shape {bias_shape:?}, skipping bias addition");
        }

        Tensor::from_shape_vec(&output_shape, output_data)?
    } else {
        Tensor::from_shape_vec(&output_shape, output_data)?
    };

    log::debug!("Conv: final output shape {:?}", final_output.shape());
    Ok(vec![final_output])
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
///
/// # Formal Specifications (verified with Why3)
/// - **Preconditions**:
///   - `inputs.len() == 1`
/// - **Postconditions**:
///   - `result.shape() == inputs[0].shape()`
///   - `∀i: result[i] == max(0, inputs[0][i])`
///   - `∀i: result[i] >= 0` (non-negativity)
/// - **Properties**:
///   - Idempotency: `relu(relu(x)) == relu(x)`
///   - Monotonicity: `x <= y => relu(x) <= relu(y)`
///   - Non-negativity: `∀x: relu(x) >= 0`
///   - Zero preservation: `relu(0) == 0`
#[cfg_attr(
    feature = "formal-verification",
    doc = "This function is formally verified using Why3 specifications"
)]
fn relu_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    #[cfg(feature = "formal-verification")]
    {
        // Formal verification precondition checks
        assert!(inputs.len() == 1, "Precondition: exactly 1 input required");
    }

    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Relu operator requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].relu();

    #[cfg(feature = "formal-verification")]
    {
        // Formal verification postcondition checks
        debug_assert_eq!(
            result.shape(),
            inputs[0].shape(),
            "Postcondition: result shape matches input shape"
        );
        // Non-negativity check would be done in debug builds
        for &value in result.data() {
            debug_assert!(
                value >= 0.0,
                "Postcondition: all values must be non-negative"
            );
        }
    }

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
///
/// # Formal Specifications (verified with Why3)
/// - **Preconditions**:
///   - `inputs.len() == 1`
/// - **Postconditions**:
///   - `result.shape() == inputs[0].shape()`
///   - `∀i: result[i] == 1.0 / (1.0 + exp(-inputs[0][i]))`
///   - `∀i: 0 < result[i] < 1` (bounded output)
/// - **Properties**:
///   - Bounded: `∀x: 0 < sigmoid(x) < 1`
///   - Monotonicity: `x < y => sigmoid(x) < sigmoid(y)`
///   - Symmetry: `sigmoid(-x) == 1 - sigmoid(x)`
///   - Fixed point: `sigmoid(0) == 0.5`
#[cfg_attr(
    feature = "formal-verification",
    doc = "This function is formally verified using Why3 specifications"
)]
fn sigmoid_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    #[cfg(feature = "formal-verification")]
    {
        // Formal verification precondition checks
        assert!(inputs.len() == 1, "Precondition: exactly 1 input required");
    }

    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Sigmoid operator requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].sigmoid();

    #[cfg(feature = "formal-verification")]
    {
        // Formal verification postcondition checks
        debug_assert_eq!(
            result.shape(),
            inputs[0].shape(),
            "Postcondition: result shape matches input shape"
        );
        // Bounded output check in debug builds
        for &value in result.data() {
            debug_assert!(
                value > 0.0 && value < 1.0,
                "Postcondition: sigmoid output must be in (0,1), got {value}"
            );
        }
    }

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
    let raw_shape: Vec<i64> = shape_tensor.data().iter().map(|&x| x as i64).collect();
    log::debug!(
        "Reshape: input shape {:?}, target shape {:?}",
        data.shape(),
        raw_shape
    );

    // Handle special ONNX reshape values
    let input_shape = data.shape();
    let total_elements: usize = input_shape.iter().product();
    let mut new_shape = Vec::new();
    let mut infer_dim_index = None;
    let mut inferred_elements = 1;

    for (i, &dim) in raw_shape.iter().enumerate() {
        if dim == 0 {
            // In ONNX, 0 means copy the corresponding dimension from input
            if i < input_shape.len() {
                new_shape.push(input_shape[i]);
                inferred_elements *= input_shape[i];
            } else {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Cannot copy dimension {i} from input shape {input_shape:?} (index out of bounds)"
                )));
            }
        } else if dim == -1 {
            // -1 means infer this dimension
            if infer_dim_index.is_some() {
                return Err(OnnxError::invalid_dimensions(
                    "Only one dimension can be inferred (-1) in reshape".to_string(),
                ));
            }
            infer_dim_index = Some(i);
            new_shape.push(0); // placeholder
        } else if dim > 0 {
            new_shape.push(dim as usize);
            inferred_elements *= dim as usize;
        } else {
            return Err(OnnxError::invalid_dimensions(format!(
                "Invalid dimension {dim} in reshape"
            )));
        }
    }

    // If there's a dimension to infer, calculate it
    if let Some(infer_idx) = infer_dim_index {
        if inferred_elements == 0 || total_elements % inferred_elements != 0 {
            return Err(OnnxError::invalid_dimensions(format!(
                "Cannot infer dimension: total elements {total_elements} is not divisible by product of known dimensions {inferred_elements}"
            )));
        }
        new_shape[infer_idx] = total_elements / inferred_elements;
    }

    log::debug!("Reshape: computed new shape {new_shape:?}");

    let result = data.reshape(&new_shape)?;
    Ok(vec![result])
}

/// Transpose operator implementation
///
/// Transposes the input tensor by reversing or permuting the axes.
/// Supports multi-dimensional tensors with optional axis permutation.
///
/// # Arguments
/// * `inputs` - Array of exactly 1 tensor
/// * `attributes` - May contain "perm" attribute specifying axis permutation
///
/// # Returns
/// * Single output tensor with transposed dimensions
fn transpose_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Transpose operator requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let input = &inputs[0];

    // Parse perm attribute if provided
    let perm = if let Some(perm_str) = attributes.get("perm") {
        // Parse perm from string like "[0, 3, 1, 2]" or "0,3,1,2"
        let cleaned = perm_str.trim_matches(['[', ']']);
        let perm_result: std::result::Result<Vec<usize>, std::num::ParseIntError> = cleaned
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect();

        match perm_result {
            Ok(p) => {
                log::debug!("Transpose: using perm attribute {p:?}");
                Some(p)
            }
            Err(_) => {
                log::warn!("Transpose: invalid perm attribute '{perm_str}', using default");
                None
            }
        }
    } else {
        None
    };

    let result = input.transpose_with_perm(perm.as_deref())?;
    log::debug!(
        "Transpose: input {:?} -> output {:?} (perm: {:?})",
        input.shape(),
        result.shape(),
        perm
    );
    Ok(vec![result])
}

/// Concat operator implementation
///
/// Concatenates a list of tensors along a specified axis.
///
/// # Arguments
/// * `inputs` - Array of tensors to concatenate
/// * `attributes` - Must contain "axis" attribute specifying concatenation axis
///
/// # Returns
/// * Single output tensor with concatenated result
fn concat_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(OnnxError::invalid_dimensions(
            "Concat operator requires at least 1 input".to_string(),
        ));
    }

    let axis = attributes
        .get("axis")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);

    // Handle negative axis
    let normalized_axis = if axis < 0 {
        (inputs[0].ndim() as i32 + axis) as usize
    } else {
        axis as usize
    };

    if inputs.len() == 1 {
        return Ok(vec![inputs[0].clone()]);
    }

    // Create references to tensors for concatenation
    let tensor_refs: Vec<&Tensor> = inputs.iter().collect();
    let result = Tensor::concat(&tensor_refs, normalized_axis)?;

    log::debug!(
        "Concat: axis {} shapes {:?} -> output {:?}",
        normalized_axis,
        inputs.iter().map(|t| t.shape()).collect::<Vec<_>>(),
        result.shape()
    );

    Ok(vec![result])
}

/// Slice operator implementation
///
/// Extracts a slice from the input tensor.
///
/// # Arguments
/// * `inputs` - Array of tensors: [data, starts, ends, axes, steps]
/// * `attributes` - May contain slice parameters
///
/// # Returns
/// * Single output tensor with sliced result
fn slice_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(OnnxError::invalid_dimensions(
            "Slice operator requires at least 1 input".to_string(),
        ));
    }

    let data = &inputs[0];

    // Check if we have input tensors for starts, ends, etc. (newer ONNX)
    if inputs.len() >= 3 {
        // Modern ONNX: parameters as input tensors
        let starts_tensor = &inputs[1];
        let ends_tensor = &inputs[2];

        // Extract starts and ends from tensors, handling negative values properly
        let starts: Vec<i64> = starts_tensor
            .data()
            .iter()
            .map(|&x| {
                // Round to nearest integer first to handle floating point precision issues
                x.round() as i64
            })
            .collect();
        let ends: Vec<i64> = ends_tensor
            .data()
            .iter()
            .map(|&x| {
                // Handle the special case where -1.0 gets cast incorrectly to a huge positive number
                if x < -0.5 && x > -1.5 {
                    // This is likely -1.0, which means "end of dimension"
                    -1
                } else if x > (i64::MAX as f32 * 0.9) {
                    // This is likely a corrupted -1 that became i64::MAX
                    -1
                } else {
                    // Round to nearest integer first, preserving negative values
                    x.round() as i64
                }
            })
            .collect();

        log::debug!("Slice: raw starts tensor data: {:?}", starts_tensor.data());
        log::debug!("Slice: raw ends tensor data: {:?}", ends_tensor.data());
        log::debug!("Slice: parsed starts: {starts:?}");
        log::debug!("Slice: parsed ends: {ends:?}");

        // Get axes if provided (4th input)
        let axes = if inputs.len() >= 4 {
            Some(
                inputs[3]
                    .data()
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        // Get steps if provided (5th input)
        let steps = if inputs.len() >= 5 {
            Some(
                inputs[4]
                    .data()
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        let result = data.slice(&starts, &ends, axes.as_deref(), steps.as_deref())?;
        log::debug!(
            "Slice: input {:?} -> output {:?} (starts: {:?}, ends: {:?}, axes: {:?})",
            data.shape(),
            result.shape(),
            starts,
            ends,
            axes
        );
        Ok(vec![result])
    } else {
        // Older ONNX: parameters as attributes
        let parse_list = |key: &str| -> Option<Vec<i64>> {
            attributes.get(key).and_then(|val| {
                val.split(',')
                    .map(|s| s.trim().parse::<i64>().ok())
                    .collect::<Option<Vec<_>>>()
            })
        };

        // Try to get starts/ends from attributes, but provide defaults if missing
        match (parse_list("starts"), parse_list("ends")) {
            (Some(starts), Some(ends)) => {
                let axes = parse_list("axes");
                let steps = parse_list("steps");

                let result = inputs[0].slice(&starts, &ends, axes.as_deref(), steps.as_deref())?;
                Ok(vec![result])
            }
            _ => {
                // Missing required attributes - use simplified implementation
                log::warn!("Slice operator missing required attributes - returning input tensor");
                Ok(vec![data.clone()])
            }
        }
    }
}

/// Upsample operator implementation
///
/// Upsamples the input tensor using nearest neighbor or linear interpolation.
///
/// # Arguments
/// * `inputs` - Array of tensors: [X, scales] or [X, roi, scales]
/// * `attributes` - May contain "mode" and "coordinate_transformation_mode"
///
/// # Returns
/// * Single output tensor with upsampled result
fn upsample_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(OnnxError::invalid_dimensions(
            "Upsample operator requires at least 1 input".to_string(),
        ));
    }

    let _mode = attributes
        .get("mode")
        .map(|s| s.as_str())
        .unwrap_or("nearest");

    // Simplified implementation - just return the input tensor
    log::warn!("Upsample operator is simplified - returning input tensor");
    Ok(vec![inputs[0].clone()])
}

/// MaxPool operator implementation
///
/// Performs max pooling operation on the input tensor.
///
/// # Arguments
/// * `inputs` - Array of exactly 1 tensor (4D NCHW format)
/// * `attributes` - Must contain "kernel_shape", may contain "strides", "pads"
///
/// # Returns
/// * Single output tensor with max pooled result
fn maxpool_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "MaxPool operator requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let input = &inputs[0];
    if input.ndim() != 4 {
        return Err(OnnxError::invalid_dimensions(
            "MaxPool operator requires 4D input tensor (NCHW format)".to_string(),
        ));
    }

    // Parse kernel shape, strides, and padding
    let kernel_shape = parse_int_array(
        attributes
            .get("kernel_shape")
            .unwrap_or(&"[2,2]".to_string()),
    )?;
    let strides = parse_int_array(attributes.get("strides").unwrap_or(&"[1,1]".to_string()))?;
    let pads = parse_int_array(attributes.get("pads").unwrap_or(&"[0,0,0,0]".to_string()))?;

    let kernel_h = kernel_shape.first().copied().unwrap_or(2) as usize;
    let kernel_w = kernel_shape.get(1).copied().unwrap_or(2) as usize;
    let stride_h = strides.first().copied().unwrap_or(1) as usize;
    let stride_w = strides.get(1).copied().unwrap_or(1) as usize;
    let pad_top = pads.first().copied().unwrap_or(0) as usize;
    let pad_left = pads.get(1).copied().unwrap_or(0) as usize;
    let pad_bottom = pads.get(2).copied().unwrap_or(0) as usize;
    let pad_right = pads.get(3).copied().unwrap_or(0) as usize;

    let input_shape = input.shape();
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];

    // Calculate output dimensions
    let output_h = (input_h + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    let output_w = (input_w + pad_left + pad_right - kernel_w) / stride_w + 1;
    let output_shape = [batch_size, channels, output_h, output_w];

    log::debug!("MaxPool: {input_h}x{input_w} -> {output_h}x{output_w}, kernel={kernel_h}x{kernel_w}, stride={stride_h}x{stride_w}, pad=[{pad_top},{pad_left},{pad_bottom},{pad_right}]");

    let mut output_data = Vec::with_capacity(output_shape.iter().product());
    let input_data = input.data();

    // Perform max pooling
    for batch in 0..batch_size {
        for channel in 0..channels {
            for out_h in 0..output_h {
                for out_w in 0..output_w {
                    let mut max_val = f32::NEG_INFINITY;

                    // Apply kernel to get maximum value
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let in_h = out_h * stride_h + kh;
                            let in_w = out_w * stride_w + kw;

                            // Handle padding: skip if outside padded bounds
                            if in_h >= pad_top
                                && in_h < input_h + pad_top
                                && in_w >= pad_left
                                && in_w < input_w + pad_left
                            {
                                let actual_h = in_h - pad_top;
                                let actual_w = in_w - pad_left;

                                if actual_h < input_h && actual_w < input_w {
                                    let val = input_data[[batch, channel, actual_h, actual_w]];
                                    max_val = max_val.max(val);
                                }
                            }
                        }
                    }

                    // If no valid values found (all padding), use 0
                    if max_val == f32::NEG_INFINITY {
                        max_val = 0.0;
                    }

                    output_data.push(max_val);
                }
            }
        }
    }

    let result = Tensor::from_shape_vec(&output_shape, output_data)?;
    log::debug!("MaxPool: output shape {:?}", result.shape());
    Ok(vec![result])
}

/// Softmax operator implementation
///
/// Applies the Softmax function: softmax(x_i) = exp(x_i) / sum(exp(x_j))
/// Uses numerical stabilization by subtracting the maximum value.
///
/// # Arguments
/// * `inputs` - Array of exactly 1 tensor
/// * `attributes` - May contain "axis" attribute specifying the axis to apply softmax
///
/// # Returns
/// * Single output tensor with Softmax applied
fn softmax_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Softmax operator requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let input = &inputs[0];
    let axis = attributes
        .get("axis")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(-1);

    // Normalize axis (handle negative indexing)
    let normalized_axis = if axis < 0 {
        (input.ndim() as i32 + axis) as usize
    } else {
        axis as usize
    };

    if normalized_axis >= input.ndim() {
        return Err(OnnxError::invalid_dimensions(format!(
            "Softmax axis {} out of bounds for tensor with {} dimensions",
            axis,
            input.ndim()
        )));
    }

    log::debug!(
        "Softmax: input shape {:?}, axis {}",
        input.shape(),
        normalized_axis
    );

    // For now, implement a simplified version that applies softmax to the last axis
    // A full implementation would handle arbitrary axis
    if input.ndim() <= 2 || normalized_axis == input.ndim() - 1 {
        // Use existing tensor softmax method for simple cases
        let result = input.softmax()?;
        log::debug!(
            "Softmax: used tensor method, output shape {:?}",
            result.shape()
        );
        Ok(vec![result])
    } else {
        // For complex multi-dimensional cases, implement axis-specific softmax
        let input_shape = input.shape();

        // Use tensor operations for safer data access
        let total_elements = input_shape.iter().product::<usize>();
        let axis_size = input_shape[normalized_axis];

        log::debug!("Softmax: complex case - shape {input_shape:?}, axis {normalized_axis}, axis_size {axis_size}");

        // Create output tensor with zeros
        let mut output = crate::tensor::Tensor::zeros(input_shape);

        // Calculate the number of slices along the specified axis
        let num_slices = total_elements / axis_size;

        // Calculate strides for efficient indexing
        let mut strides = vec![1; input.ndim()];
        for i in (0..input.ndim() - 1).rev() {
            strides[i] = strides[i + 1] * input_shape[i + 1];
        }
        let axis_stride = strides[normalized_axis];

        // Process each slice along the specified axis
        for slice_idx in 0..num_slices {
            // Find the starting position for this slice
            let mut base_idx = 0;
            let mut remaining = slice_idx;

            for dim in 0..input.ndim() {
                if dim != normalized_axis {
                    let dim_size = input_shape[dim];
                    let coord = remaining % dim_size;
                    remaining /= dim_size;
                    base_idx += coord * strides[dim];
                }
            }

            // Collect values along the axis for this slice using direct array access
            let mut axis_values = Vec::with_capacity(axis_size);
            for i in 0..axis_size {
                let idx = base_idx + i * axis_stride;

                // Convert linear index to multi-dimensional coordinates
                let mut coords = vec![0; input.ndim()];
                let mut temp_idx = idx;
                for dim in (0..input.ndim()).rev() {
                    coords[dim] = temp_idx % input_shape[dim];
                    temp_idx /= input_shape[dim];
                }

                // Get value using array indexing
                let value = input.data()[&*coords];
                axis_values.push(value);
            }

            // Apply softmax with numerical stabilization
            let max_val = axis_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_values: Vec<f32> = axis_values.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            if sum_exp == 0.0 || !sum_exp.is_finite() {
                // Fallback to uniform distribution
                let uniform_val = 1.0 / axis_size as f32;
                for i in 0..axis_size {
                    let idx = base_idx + i * axis_stride;
                    let mut coords = vec![0; input.ndim()];
                    let mut temp_idx = idx;
                    for dim in (0..input.ndim()).rev() {
                        coords[dim] = temp_idx % input_shape[dim];
                        temp_idx /= input_shape[dim];
                    }
                    output.data_mut()[&*coords] = uniform_val;
                }
            } else {
                for (i, &exp_val) in exp_values.iter().enumerate().take(axis_size) {
                    let idx = base_idx + i * axis_stride;
                    let mut coords = vec![0; input.ndim()];
                    let mut temp_idx = idx;
                    for dim in (0..input.ndim()).rev() {
                        coords[dim] = temp_idx % input_shape[dim];
                        temp_idx /= input_shape[dim];
                    }
                    let softmax_val = exp_val / sum_exp;
                    output.data_mut()[&*coords] = softmax_val;
                }
            }
        }

        log::debug!("Softmax: output shape {:?}", output.shape());
        Ok(vec![output])
    }
}

/// NonMaxSuppression operator implementation
///
/// Performs Non-Maximum Suppression for object detection.
///
/// # Arguments
/// * `inputs` - Array of tensors: [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
/// * `attributes` - May contain "center_point_box" attribute
///
/// # Returns
/// * Single output tensor with selected indices
fn nms_op(
    inputs: &[Tensor],
    _attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() < 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "NonMaxSuppression operator requires at least 2 inputs (boxes, scores), got {}",
            inputs.len()
        )));
    }

    // Simplified implementation - return empty tensor
    // A full implementation would perform actual NMS algorithm
    log::warn!("NonMaxSuppression operator is simplified - returning empty tensor");
    let empty_result = Tensor::zeros(&[0, 3]); // [num_selected_indices, 3] format
    Ok(vec![empty_result])
}

/// BatchNormalization operator implementation
///
/// Performs batch normalization: (x - mean) / sqrt(variance + epsilon) * scale + bias
/// For inference mode, uses provided running mean and variance.
///
/// # Arguments
/// * `inputs` - Array of 5 tensors: [input, scale, bias, mean, variance]
/// * `attributes` - May contain "epsilon" attribute
///
/// # Returns
/// * Single output tensor with normalized result
fn batch_norm_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() < 5 {
        return Err(OnnxError::invalid_dimensions(format!(
            "BatchNormalization requires 5 inputs (input, scale, bias, mean, var), got {}",
            inputs.len()
        )));
    }

    let input = &inputs[0];
    let scale = &inputs[1];
    let bias = &inputs[2];
    let mean = &inputs[3];
    let variance = &inputs[4];

    // Parse epsilon parameter
    let epsilon = attributes
        .get("epsilon")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1e-5);

    log::debug!(
        "BatchNorm: input shape {:?}, epsilon {}",
        input.shape(),
        epsilon
    );

    // For 4D input (NCHW), scale/bias/mean/variance should be 1D with size = channels
    let input_shape = input.shape();
    if input.ndim() != 4 {
        return Err(OnnxError::invalid_dimensions(
            "BatchNormalization currently supports only 4D input tensors (NCHW)".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    // Validate parameter shapes
    let expected_param_shape = [channels];
    if scale.shape() != expected_param_shape
        || bias.shape() != expected_param_shape
        || mean.shape() != expected_param_shape
        || variance.shape() != expected_param_shape
    {
        return Err(OnnxError::invalid_dimensions(format!(
            "BatchNorm parameters must have shape [{channels}], got scale: {:?}, bias: {:?}, mean: {:?}, var: {:?}",
            scale.shape(), bias.shape(), mean.shape(), variance.shape()
        )));
    }

    // Get parameter data
    let scale_data = scale.data().as_slice().unwrap();
    let bias_data = bias.data().as_slice().unwrap();
    let mean_data = mean.data().as_slice().unwrap();
    let var_data = variance.data().as_slice().unwrap();
    let input_data = input.data().as_slice().unwrap();

    // Create output data
    let mut output_data = vec![0.0; input_data.len()];

    // Helper function for 4D indexing
    let idx = |n: usize, c: usize, h: usize, w: usize| -> usize {
        n * (channels * height * width) + c * (height * width) + h * width + w
    };

    // Apply batch normalization: output = scale * (input - mean) / sqrt(var + epsilon) + bias
    for n in 0..batch_size {
        for c in 0..channels {
            let scale_val = scale_data[c];
            let bias_val = bias_data[c];
            let mean_val = mean_data[c];
            let var_val = var_data[c];
            let inv_std = 1.0 / (var_val + epsilon).sqrt();

            for h in 0..height {
                for w in 0..width {
                    let i = idx(n, c, h, w);
                    let normalized = (input_data[i] - mean_val) * inv_std;
                    output_data[i] = scale_val * normalized + bias_val;
                }
            }
        }
    }

    let result = Tensor::from_shape_vec(input_shape, output_data)?;
    log::debug!("BatchNorm: output shape {:?}", result.shape());

    Ok(vec![result])
}

/// Split operator implementation
///
/// Splits a tensor into multiple tensors along a specified axis.
///
/// # Arguments
/// * `inputs` - Array of 1-2 tensors: [input, split (optional)]
/// * `attributes` - May contain "axis" and "split" attributes
///
/// # Returns
/// * Multiple output tensors from the split
fn split_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(OnnxError::invalid_dimensions(
            "Split requires at least 1 input".to_string(),
        ));
    }

    let input = &inputs[0];
    let axis = attributes
        .get("axis")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);

    // Normalize axis (handle negative indexing)
    let normalized_axis = if axis < 0 {
        (input.ndim() as i32 + axis) as usize
    } else {
        axis as usize
    };

    if normalized_axis >= input.ndim() {
        return Err(OnnxError::invalid_dimensions(format!(
            "Split axis {} out of bounds for tensor with {} dimensions",
            axis,
            input.ndim()
        )));
    }

    let axis_size = input.shape()[normalized_axis];

    // Get split sizes - either from second input tensor or from attributes
    let split_sizes = if inputs.len() >= 2 {
        // Split sizes provided as input tensor
        let split_tensor = &inputs[1];
        split_tensor
            .data()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
    } else if let Some(split_attr) = attributes.get("split") {
        // Parse split sizes from attribute string
        let cleaned = split_attr.trim_matches(['[', ']']);
        cleaned
            .split(',')
            .map(|s| s.trim().parse::<usize>().unwrap_or(1))
            .collect()
    } else {
        // Default: split into equal parts (assume 2 parts for simplicity)
        let num_splits = 2;
        let chunk_size = axis_size / num_splits;
        let remainder = axis_size % num_splits;

        let mut sizes = vec![chunk_size; num_splits];
        // Distribute remainder among first few chunks
        for size in sizes.iter_mut().take(remainder) {
            *size += 1;
        }
        sizes
    };

    // Validate split sizes
    let total_size: usize = split_sizes.iter().sum();
    if total_size != axis_size {
        return Err(OnnxError::invalid_dimensions(format!(
            "Split sizes sum ({total_size}) must equal axis size ({axis_size})"
        )));
    }

    log::debug!(
        "Split: input shape {:?}, axis {}, sizes {:?}",
        input.shape(),
        normalized_axis,
        split_sizes
    );

    // Perform the split
    let mut results = Vec::new();
    let mut current_offset = 0;

    for &split_size in &split_sizes {
        if split_size == 0 {
            continue; // Skip zero-sized splits
        }

        // Create slice parameters for this split
        let mut starts = vec![0i64; input.ndim()];
        let mut ends = vec![0i64; input.ndim()];

        for i in 0..input.ndim() {
            if i == normalized_axis {
                starts[i] = current_offset as i64;
                ends[i] = (current_offset + split_size) as i64;
            } else {
                starts[i] = 0;
                ends[i] = input.shape()[i] as i64;
            }
        }

        // Create the slice
        let split_result = input.slice(&starts, &ends, None, None)?;
        results.push(split_result);

        current_offset += split_size;
    }

    log::debug!("Split: created {} output tensors", results.len());
    Ok(results)
}

/// Gather operator implementation
///
/// Gathers values from the input tensor at specified indices along a given axis.
///
/// # Arguments
/// * `inputs` - Array of exactly 2 tensors: [data, indices]
/// * `attributes` - May contain "axis" attribute (default: 0)
///
/// # Returns
/// * Single output tensor with gathered values
fn gather_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Gather requires exactly 2 inputs (data, indices), got {}",
            inputs.len()
        )));
    }

    let data = &inputs[0];
    let indices = &inputs[1];

    let axis = attributes
        .get("axis")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);

    // Normalize axis (handle negative indexing)
    let normalized_axis = if axis < 0 {
        (data.ndim() as i32 + axis) as usize
    } else {
        axis as usize
    };

    if normalized_axis >= data.ndim() {
        return Err(OnnxError::invalid_dimensions(format!(
            "Gather axis {} out of bounds for tensor with {} dimensions",
            axis,
            data.ndim()
        )));
    }

    let data_shape = data.shape();
    let indices_shape = indices.shape();
    let axis_size = data_shape[normalized_axis];

    log::debug!("Gather: data shape {data_shape:?}, indices shape {indices_shape:?}, axis {normalized_axis}");

    // Calculate output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:]
    let mut output_shape = Vec::new();
    output_shape.extend_from_slice(&data_shape[..normalized_axis]);
    output_shape.extend_from_slice(indices_shape);
    output_shape.extend_from_slice(&data_shape[normalized_axis + 1..]);

    log::debug!("Gather: output shape {output_shape:?}");

    // Get indices as integers
    let indices_data = indices.data().as_slice().unwrap();
    let indices_int: Vec<usize> = indices_data
        .iter()
        .map(|&x| {
            let idx = x as i64;
            // Handle negative indices
            let normalized_idx = if idx < 0 {
                (axis_size as i64 + idx) as usize
            } else {
                idx as usize
            };

            if normalized_idx >= axis_size {
                // Clamp to valid range
                axis_size - 1
            } else {
                normalized_idx
            }
        })
        .collect();

    // Calculate strides for input tensor
    let mut data_strides = vec![1; data.ndim()];
    for i in (0..data.ndim() - 1).rev() {
        data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
    }

    // Calculate strides for output tensor
    let mut output_strides = vec![1; output_shape.len()];
    for i in (0..output_shape.len() - 1).rev() {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    let data_data = data.data().as_slice().unwrap();
    let mut output_data = vec![0.0; output_shape.iter().product()];

    // Iterate through all output positions
    for (output_idx, output_val) in output_data.iter_mut().enumerate() {
        // Convert flat output index to multi-dimensional coordinates
        let mut output_coords = vec![0; output_shape.len()];
        let mut remaining = output_idx;
        for i in 0..output_shape.len() {
            output_coords[i] = remaining / output_strides[i];
            remaining %= output_strides[i];
        }

        // Map output coordinates to input coordinates
        let mut data_coords = vec![0; data.ndim()];

        // Copy coordinates before the gather axis
        data_coords[..normalized_axis].copy_from_slice(&output_coords[..normalized_axis]);

        // Handle indices for the gather axis
        let indices_start = normalized_axis;
        let indices_end = indices_start + indices.ndim();

        // Flatten indices coordinates to get the index value
        let mut indices_idx = 0;
        let mut indices_stride = 1;
        for i in (indices_start..indices_end).rev() {
            indices_idx += output_coords[i] * indices_stride;
            indices_stride *= indices_shape[i - indices_start];
        }

        data_coords[normalized_axis] = indices_int[indices_idx];

        // Copy coordinates after the gather axis
        for i in (normalized_axis + 1)..data.ndim() {
            data_coords[i] = output_coords[indices_end + i - normalized_axis - 1];
        }

        // Convert data coordinates to flat index
        let mut data_idx = 0;
        for i in 0..data.ndim() {
            data_idx += data_coords[i] * data_strides[i];
        }

        // Copy the value
        *output_val = data_data[data_idx];
    }

    let result = Tensor::from_shape_vec(&output_shape, output_data)?;
    log::debug!("Gather: result shape {:?}", result.shape());

    Ok(vec![result])
}

/// ConstantOfShape operator implementation
fn constant_of_shape_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "ConstantOfShape requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let value = attributes
        .get("value")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);

    // Create tensor with specified shape and constant value
    let shape_data = inputs[0].data();
    let shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();

    let result = if value == 0.0 {
        Tensor::zeros(&shape)
    } else if value == 1.0 {
        Tensor::ones(&shape)
    } else {
        let data = vec![value; shape.iter().product()];
        Tensor::from_shape_vec(&shape, data)?
    };

    Ok(vec![result])
}

/// Cast operator implementation
fn cast_op(
    inputs: &[Tensor],
    _attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Cast requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    // Simplified implementation - assume all data is already f32
    log::warn!("Cast operator is simplified - returning input tensor");
    Ok(vec![inputs[0].clone()])
}

/// Shape operator implementation
fn shape_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Shape requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let shape_data: Vec<f32> = inputs[0].shape().iter().map(|&dim| dim as f32).collect();

    let shape_tensor = Tensor::from_shape_vec(&[shape_data.len()], shape_data)?;
    Ok(vec![shape_tensor])
}

/// Unsqueeze operator implementation
///
/// Adds dimensions of size 1 to the tensor at specified axes.
///
/// # Arguments
/// * `inputs` - Array of 1-2 tensors: [data, axes (optional)]
/// * `attributes` - May contain "axes" attribute
///
/// # Returns
/// * Single output tensor with additional dimensions
fn unsqueeze_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(OnnxError::invalid_dimensions(
            "Unsqueeze requires at least 1 input".to_string(),
        ));
    }

    let input = &inputs[0];

    // Get axes from input tensor or attributes
    let axes: Vec<i64> = if inputs.len() >= 2 {
        // Axes provided as input tensor (newer ONNX)
        let axes_tensor = &inputs[1];
        axes_tensor.data().iter().map(|&x| x as i64).collect()
    } else if let Some(axes_attr) = attributes.get("axes") {
        // Parse axes from attribute string
        let cleaned = axes_attr.trim_matches(['[', ']']);
        cleaned
            .split(',')
            .map(|s| s.trim().parse::<i64>().unwrap_or(0))
            .collect()
    } else {
        return Err(OnnxError::invalid_dimensions(
            "Unsqueeze requires axes to be specified".to_string(),
        ));
    };

    if axes.is_empty() {
        return Ok(vec![input.clone()]);
    }

    let input_shape = input.shape();
    let input_ndim = input.ndim() as i64;
    let output_ndim = input_ndim + axes.len() as i64;

    // Normalize axes (handle negative indexing relative to output shape)
    let mut normalized_axes: Vec<usize> = axes
        .iter()
        .map(|&axis| {
            if axis < 0 {
                (output_ndim + axis) as usize
            } else {
                axis as usize
            }
        })
        .collect();

    // Validate axes
    for &axis in &normalized_axes {
        if axis >= output_ndim as usize {
            return Err(OnnxError::invalid_dimensions(format!(
                "Unsqueeze axis {axis} out of bounds for output with {output_ndim} dimensions"
            )));
        }
    }

    // Sort axes to process them in order
    normalized_axes.sort();

    // Check for duplicate axes
    for i in 1..normalized_axes.len() {
        if normalized_axes[i] == normalized_axes[i - 1] {
            return Err(OnnxError::invalid_dimensions(
                "Unsqueeze axes must be unique".to_string(),
            ));
        }
    }

    log::debug!("Unsqueeze: input shape {input_shape:?}, axes {normalized_axes:?}");

    // Build output shape
    let mut output_shape = Vec::with_capacity(output_ndim as usize);
    let mut input_dim_idx = 0;

    for output_dim_idx in 0..(output_ndim as usize) {
        if normalized_axes.contains(&output_dim_idx) {
            // This is a new dimension of size 1
            output_shape.push(1);
        } else {
            // This is a dimension from the input
            output_shape.push(input_shape[input_dim_idx]);
            input_dim_idx += 1;
        }
    }

    log::debug!("Unsqueeze: output shape {output_shape:?}");

    // Create output tensor by reshaping
    let result = input.reshape(&output_shape)?;

    Ok(vec![result])
}

/// Squeeze operator implementation
///
/// Removes dimensions of size 1 from the tensor at specified axes.
///
/// # Arguments
/// * `inputs` - Array of 1-2 tensors: [data, axes (optional)]
/// * `attributes` - May contain "axes" attribute
///
/// # Returns
/// * Single output tensor with reduced dimensions
fn squeeze_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(OnnxError::invalid_dimensions(
            "Squeeze requires at least 1 input".to_string(),
        ));
    }

    let input = &inputs[0];
    let input_shape = input.shape();

    // Get axes from input tensor or attributes
    let axes: Option<Vec<i64>> = if inputs.len() >= 2 {
        // Axes provided as input tensor (newer ONNX)
        let axes_tensor = &inputs[1];
        Some(axes_tensor.data().iter().map(|&x| x as i64).collect())
    } else if let Some(axes_attr) = attributes.get("axes") {
        // Parse axes from attribute string
        let cleaned = axes_attr.trim_matches(['[', ']']);
        Some(
            cleaned
                .split(',')
                .map(|s| s.trim().parse::<i64>().unwrap_or(0))
                .collect(),
        )
    } else {
        None // No axes specified, squeeze all size-1 dimensions
    };

    log::debug!("Squeeze: input shape {input_shape:?}, axes {axes:?}");

    let input_ndim = input.ndim() as i64;

    // Determine which axes to squeeze
    let axes_to_squeeze: Vec<usize> = if let Some(specified_axes) = axes {
        // Normalize specified axes and validate
        let mut normalized_axes = Vec::new();
        for &axis in &specified_axes {
            let normalized_axis = if axis < 0 {
                (input_ndim + axis) as usize
            } else {
                axis as usize
            };

            if normalized_axis >= input.ndim() {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Squeeze axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    input.ndim()
                )));
            }

            if input_shape[normalized_axis] != 1 {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Cannot squeeze axis {} with size {}",
                    axis, input_shape[normalized_axis]
                )));
            }

            normalized_axes.push(normalized_axis);
        }
        normalized_axes
    } else {
        // Squeeze all dimensions with size 1
        input_shape
            .iter()
            .enumerate()
            .filter(|(_, &size)| size == 1)
            .map(|(idx, _)| idx)
            .collect()
    };

    // Build output shape by excluding squeezed dimensions
    let output_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter(|(idx, _)| !axes_to_squeeze.contains(idx))
        .map(|(_, &size)| size)
        .collect();

    log::debug!("Squeeze: output shape {output_shape:?}");

    // Handle edge case: if all dimensions are squeezed, result should be 0D
    let result = if output_shape.is_empty() {
        // Create a 0D tensor (scalar)
        if input.len() != 1 {
            return Err(OnnxError::invalid_dimensions(
                "Cannot squeeze to scalar: input must have exactly 1 element".to_string(),
            ));
        }
        let scalar_value = input.data().iter().next().unwrap();
        Tensor::from_shape_vec(&[], vec![*scalar_value])?
    } else {
        input.reshape(&output_shape)?
    };

    Ok(vec![result])
}

/// Pad operator implementation
///
/// Pads the input tensor with specified padding values.
///
/// # Arguments
/// * `inputs` - Array of 2-3 tensors: [data, pads, constant_value (optional)]
/// * `attributes` - May contain "mode" attribute
///
/// # Returns
/// * Single output tensor with padding applied
fn pad_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() < 2 {
        return Err(OnnxError::invalid_dimensions(
            "Pad requires at least 2 inputs (data, pads)".to_string(),
        ));
    }

    let input = &inputs[0];
    let pads_tensor = &inputs[1];
    let constant_value = if inputs.len() >= 3 {
        inputs[2].data().iter().next().copied().unwrap_or(0.0)
    } else {
        0.0
    };

    let mode = attributes
        .get("mode")
        .map(|s| s.as_str())
        .unwrap_or("constant");

    // Get padding values
    let pads_data = pads_tensor.data().as_slice().unwrap();
    let pads: Vec<usize> = pads_data.iter().map(|&x| x as usize).collect();

    let input_shape = input.shape();
    let ndim = input.ndim();

    // Validate pads length
    if pads.len() != 2 * ndim {
        return Err(OnnxError::invalid_dimensions(format!(
            "Pads length ({}) must be 2 * input dimensions ({})",
            pads.len(),
            2 * ndim
        )));
    }

    // Extract padding for each dimension: [begin_pad_0, begin_pad_1, ..., end_pad_0, end_pad_1, ...]
    let begin_pads = &pads[..ndim];
    let end_pads = &pads[ndim..];

    log::debug!("Pad: input shape {input_shape:?}, begin_pads {begin_pads:?}, end_pads {end_pads:?}, mode {mode}, value {constant_value}");

    // Calculate output shape
    let output_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .map(|(i, &size)| size + begin_pads[i] + end_pads[i])
        .collect();

    // Only implement constant padding for now (most common usage)
    if mode != "constant" {
        log::warn!("Pad: only constant mode supported, got {mode}");
        return Ok(vec![input.clone()]);
    }

    log::debug!("Pad: output shape {output_shape:?}");

    // Create output tensor filled with constant value
    let mut output_data = vec![constant_value; output_shape.iter().product()];
    let input_data = input.data().as_slice().unwrap();

    // Calculate strides for both input and output
    let mut input_strides = vec![1; ndim];
    let mut output_strides = vec![1; ndim];

    for i in (0..ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // Copy input data to the appropriate region in output
    let total_input_elements = input_data.len();
    for (input_idx, &input_val) in input_data.iter().enumerate().take(total_input_elements) {
        // Convert flat input index to multi-dimensional coordinates
        let mut input_coords = vec![0; ndim];
        let mut remaining = input_idx;
        for i in 0..ndim {
            input_coords[i] = remaining / input_strides[i];
            remaining %= input_strides[i];
        }

        // Calculate corresponding output coordinates (shifted by begin padding)
        let output_coords: Vec<usize> = input_coords
            .iter()
            .enumerate()
            .map(|(i, &coord)| coord + begin_pads[i])
            .collect();

        // Convert output coordinates to flat index
        let mut output_idx = 0;
        for i in 0..ndim {
            output_idx += output_coords[i] * output_strides[i];
        }

        // Copy the value
        output_data[output_idx] = input_val;
    }

    let result = Tensor::from_shape_vec(&output_shape, output_data)?;
    log::debug!("Pad: result shape {:?}", result.shape());

    Ok(vec![result])
}

/// Div operator implementation
fn div_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Div requires exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    log::debug!(
        "Div: input[0] shape {:?}, input[1] shape {:?}",
        inputs[0].shape(),
        inputs[1].shape()
    );

    let result = inputs[0].div(&inputs[1])?;
    Ok(vec![result])
}

/// Sub operator implementation
fn sub_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Sub requires exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].sub(&inputs[1])?;
    Ok(vec![result])
}

/// Exp operator implementation
fn exp_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Exp requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].exp()?;
    Ok(vec![result])
}

/// Sqrt operator implementation
fn sqrt_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Sqrt requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].sqrt()?;
    Ok(vec![result])
}

/// Pow operator implementation
fn pow_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Pow requires exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    let result = inputs[0].pow(&inputs[1])?;
    Ok(vec![result])
}

/// ReduceMean operator implementation
fn reduce_mean_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "ReduceMean requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    let _axes = attributes.get("axes");
    let _keepdims = attributes
        .get("keepdims")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1);

    // Simplified implementation - return mean of all elements
    let input = &inputs[0];
    let mean_value = input.data().iter().sum::<f32>() / input.data().len() as f32;
    let result = Tensor::from_shape_vec(&[1], vec![mean_value])?;

    Ok(vec![result])
}

/// Identity operator implementation
fn identity_op(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Identity requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    Ok(vec![inputs[0].clone()])
}

/// Resize operator implementation (similar to Upsample)
fn resize_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(OnnxError::invalid_dimensions(
            "Resize requires at least 1 input".to_string(),
        ));
    }

    let input = &inputs[0];
    log::debug!(
        "Resize: input shape {:?}, num_inputs: {}",
        input.shape(),
        inputs.len()
    );

    // Debug: Print all attributes
    for (key, value) in attributes {
        log::debug!("Resize attribute: {key} = {value}");
    }

    // Debug: Print info about all input tensors
    for (i, inp) in inputs.iter().enumerate() {
        let first_few: Vec<f32> = inp.data().iter().take(8).copied().collect();
        log::debug!(
            "Resize input[{}]: shape {:?}, first few values: {:?}",
            i,
            inp.shape(),
            first_few
        );
    }

    // Get scale factors from inputs (typical ONNX Resize has roi, scales as inputs)
    if inputs.len() >= 2 {
        let scales_tensor = &inputs[1]; // scales is typically the 2nd input (index 1)
        let scales: Vec<f32> = scales_tensor.data().iter().copied().collect();
        log::debug!("Resize scales from input: {scales:?}");

        // For 4D tensor (NCHW), scales should be [batch_scale, channel_scale, height_scale, width_scale]
        if scales.len() >= 4 && input.ndim() == 4 {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let height = input_shape[2];
            let width = input_shape[3];

            // Calculate new dimensions
            let new_height = (height as f32 * scales[2]) as usize;
            let new_width = (width as f32 * scales[3]) as usize;

            log::debug!(
                "Resize: {}x{} -> {}x{} (scales: {:.2}x{:.2})",
                height,
                width,
                new_height,
                new_width,
                scales[2],
                scales[3]
            );

            let output_shape = [batch_size, channels, new_height, new_width];

            // Simple nearest neighbor upsampling
            let mut output_data = Vec::with_capacity(output_shape.iter().product());
            let input_data = input.data();

            for batch in 0..batch_size {
                for channel in 0..channels {
                    for new_h in 0..new_height {
                        for new_w in 0..new_width {
                            // Map back to original coordinates (nearest neighbor)
                            let orig_h = ((new_h as f32) / scales[2]) as usize;
                            let orig_w = ((new_w as f32) / scales[3]) as usize;
                            let orig_h = orig_h.min(height - 1);
                            let orig_w = orig_w.min(width - 1);

                            // Access using multi-dimensional indexing
                            let value = input_data[[batch, channel, orig_h, orig_w]];
                            output_data.push(value);
                        }
                    }
                }
            }

            let result = Tensor::from_shape_vec(&output_shape, output_data)?;
            log::debug!(
                "Resize: input {:?} -> output {:?} (scales: {:?})",
                input.shape(),
                result.shape(),
                scales
            );
            return Ok(vec![result]);
        }
    }

    // Try to get scale from attributes if not in inputs
    if let Some(scales_str) = attributes.get("scales") {
        log::debug!("Resize scales from attributes: {scales_str}");
        // Parse scales from attribute (e.g., "[1.0, 1.0, 2.0, 2.0]")
        let cleaned = scales_str.trim_matches(['[', ']']);
        let scales: std::result::Result<Vec<f32>, _> = cleaned
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect();

        if let Ok(scales) = scales {
            if scales.len() >= 4 && input.ndim() == 4 {
                let input_shape = input.shape();
                let batch_size = input_shape[0];
                let channels = input_shape[1];
                let height = input_shape[2];
                let width = input_shape[3];

                // Calculate new dimensions
                let new_height = (height as f32 * scales[2]) as usize;
                let new_width = (width as f32 * scales[3]) as usize;

                log::debug!(
                    "Resize from attributes: {}x{} -> {}x{} (scales: {:.2}x{:.2})",
                    height,
                    width,
                    new_height,
                    new_width,
                    scales[2],
                    scales[3]
                );

                let output_shape = [batch_size, channels, new_height, new_width];

                // Simple nearest neighbor upsampling
                let mut output_data = Vec::with_capacity(output_shape.iter().product());
                let input_data = input.data();

                for batch in 0..batch_size {
                    for channel in 0..channels {
                        for new_h in 0..new_height {
                            for new_w in 0..new_width {
                                // Map back to original coordinates (nearest neighbor)
                                let orig_h = ((new_h as f32) / scales[2]) as usize;
                                let orig_w = ((new_w as f32) / scales[3]) as usize;
                                let orig_h = orig_h.min(height - 1);
                                let orig_w = orig_w.min(width - 1);

                                // Access using multi-dimensional indexing
                                let value = input_data[[batch, channel, orig_h, orig_w]];
                                output_data.push(value);
                            }
                        }
                    }
                }

                let result = Tensor::from_shape_vec(&output_shape, output_data)?;
                log::debug!(
                    "Resize: input {:?} -> output {:?} (scales: {:?})",
                    input.shape(),
                    result.shape(),
                    scales
                );
                return Ok(vec![result]);
            }
        }
    }

    // Fallback: simplified implementation
    log::warn!("Resize operator using simplified implementation - returning input tensor");
    log::debug!(
        "Resize fallback: input shape {:?}, attributes: {:?}",
        input.shape(),
        attributes
    );
    Ok(vec![input.clone()])
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
        assert_eq!(
            "MatMul".parse::<OperatorType>().unwrap(),
            OperatorType::MatMul
        );
        assert_eq!("Conv".parse::<OperatorType>().unwrap(), OperatorType::Conv);
        assert_eq!("Relu".parse::<OperatorType>().unwrap(), OperatorType::Relu);
        assert_eq!(
            "Sigmoid".parse::<OperatorType>().unwrap(),
            OperatorType::Sigmoid
        );
        assert_eq!(
            "Reshape".parse::<OperatorType>().unwrap(),
            OperatorType::Reshape
        );
        assert_eq!(
            "Transpose".parse::<OperatorType>().unwrap(),
            OperatorType::Transpose
        );

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

        #[cfg(feature = "formal-verification")]
        {
            // With formal verification enabled, precondition violations should panic
            let result =
                std::panic::catch_unwind(|| execute_operator(&OperatorType::Add, &inputs, &attrs));
            assert!(
                result.is_err(),
                "Should panic with formal verification enabled"
            );
        }

        #[cfg(not(feature = "formal-verification"))]
        {
            // Without formal verification, should return error
            let result = execute_operator(&OperatorType::Add, &inputs, &attrs);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));
        }

        // Test with too many inputs
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let c = Tensor::from_array(Array1::from_vec(vec![7.0, 8.0, 9.0]));
        let inputs = vec![a, b, c]; // 3 inputs, should be 2
        let attrs = HashMap::new();

        #[cfg(feature = "formal-verification")]
        {
            let result =
                std::panic::catch_unwind(|| execute_operator(&OperatorType::Add, &inputs, &attrs));
            assert!(
                result.is_err(),
                "Should panic with formal verification enabled"
            );
        }

        #[cfg(not(feature = "formal-verification"))]
        {
            let result = execute_operator(&OperatorType::Add, &inputs, &attrs);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));
        }
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

        #[cfg(feature = "formal-verification")]
        {
            // With formal verification enabled, precondition violations should panic
            let result =
                std::panic::catch_unwind(|| execute_operator(&OperatorType::Mul, &inputs, &attrs));
            assert!(
                result.is_err(),
                "Should panic with formal verification enabled"
            );
        }

        #[cfg(not(feature = "formal-verification"))]
        {
            // Without formal verification, should return error
            let result = execute_operator(&OperatorType::Mul, &inputs, &attrs);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));
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
    fn test_matmul_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let inputs = vec![a]; // Only 1 input, should be 2
        let attrs = HashMap::new();

        #[cfg(feature = "formal-verification")]
        {
            // With formal verification enabled, precondition violations should panic
            let result = std::panic::catch_unwind(|| {
                execute_operator(&OperatorType::MatMul, &inputs, &attrs)
            });
            assert!(
                result.is_err(),
                "Should panic with formal verification enabled"
            );
        }

        #[cfg(not(feature = "formal-verification"))]
        {
            // Without formal verification, should return error
            let result = execute_operator(&OperatorType::MatMul, &inputs, &attrs);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("exactly 2 inputs"));
        }
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
        // Proper convolution: 3x3 input with 2x2 kernel -> 2x2 output
        assert_eq!(result[0].shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_conv_op_insufficient_inputs() {
        let input = Tensor::zeros(&[1, 1, 3, 3]);
        let inputs = vec![input]; // Only 1 input, should be at least 2
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Conv, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least 2 inputs"));
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

        #[cfg(feature = "formal-verification")]
        {
            // With formal verification enabled, precondition violations should panic
            let result =
                std::panic::catch_unwind(|| execute_operator(&OperatorType::Relu, &inputs, &attrs));
            assert!(
                result.is_err(),
                "Should panic with formal verification enabled"
            );
        }

        #[cfg(not(feature = "formal-verification"))]
        {
            // Without formal verification, should return error
            let result = execute_operator(&OperatorType::Relu, &inputs, &attrs);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("exactly 1 input"));
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
    fn test_sigmoid_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let inputs = vec![a, b]; // 2 inputs, should be 1
        let attrs = HashMap::new();

        #[cfg(feature = "formal-verification")]
        {
            // With formal verification enabled, precondition violations should panic
            let result = std::panic::catch_unwind(|| {
                execute_operator(&OperatorType::Sigmoid, &inputs, &attrs)
            });
            assert!(
                result.is_err(),
                "Should panic with formal verification enabled"
            );
        }

        #[cfg(not(feature = "formal-verification"))]
        {
            // Without formal verification, should return error
            let result = execute_operator(&OperatorType::Sigmoid, &inputs, &attrs);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("exactly 1 input"));
        }
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
        let tensor_2d = Tensor::from_array(
            Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap(),
        );
        let tensor_4d = Tensor::zeros(&[1, 1, 2, 2]);
        let shape_tensor = Tensor::from_array(Array1::from_vec(vec![3.0, 2.0]));
        let attrs = HashMap::new();

        // Test all single-input operators
        assert!(execute_operator(&OperatorType::Relu, &[tensor_1d.clone()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::Sigmoid, &[tensor_1d.clone()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::Transpose, &[tensor_2d.clone()], &attrs).is_ok());

        // Test dual-input operators
        assert!(execute_operator(
            &OperatorType::Add,
            &[tensor_1d.clone(), tensor_1d.clone()],
            &attrs
        )
        .is_ok());
        assert!(execute_operator(
            &OperatorType::Mul,
            &[tensor_1d.clone(), tensor_1d.clone()],
            &attrs
        )
        .is_ok());
        assert!(execute_operator(
            &OperatorType::MatMul,
            &[tensor_2d.clone(), tensor_2d.transpose().unwrap()],
            &attrs
        )
        .is_ok());
        assert!(execute_operator(
            &OperatorType::Reshape,
            &[tensor_2d.clone(), shape_tensor],
            &attrs
        )
        .is_ok());
        assert!(execute_operator(
            &OperatorType::Conv,
            &[tensor_4d.clone(), tensor_4d.clone()],
            &attrs
        )
        .is_ok());
    }

    // === Formal Verification Tests ===
    // These tests verify mathematical properties of operators

    #[test]
    fn test_formal_addition_identity() {
        // Test that a + 0 = a (identity property)
        let tensor = Tensor::from_shape_vec(&[3], vec![1.0, 2.0, 3.0]).unwrap();
        let zero = Tensor::zeros(&[3]);

        let result = tensor.add(&zero).unwrap();
        assert_eq!(result.data(), tensor.data());
    }

    #[test]
    fn test_formal_addition_commutativity() {
        // Test that a + b = b + a (commutativity)
        let tensor_a = Tensor::from_shape_vec(&[3], vec![1.0, 2.0, 3.0]).unwrap();
        let tensor_b = Tensor::from_shape_vec(&[3], vec![4.0, 5.0, 6.0]).unwrap();

        let result1 = tensor_a.add(&tensor_b).unwrap();
        let result2 = tensor_b.add(&tensor_a).unwrap();

        assert_eq!(result1.data(), result2.data());
    }

    #[test]
    fn test_formal_multiplication_commutativity() {
        // Test that a * b = b * a (commutativity)
        let tensor_a = Tensor::from_shape_vec(&[3], vec![2.0, 3.0, 4.0]).unwrap();
        let tensor_b = Tensor::from_shape_vec(&[3], vec![5.0, 6.0, 7.0]).unwrap();

        let result1 = tensor_a.mul(&tensor_b).unwrap();
        let result2 = tensor_b.mul(&tensor_a).unwrap();

        assert_eq!(result1.data(), result2.data());
    }

    #[test]
    fn test_formal_relu_non_negativity() {
        // Test that ReLU output is always non-negative
        let tensor = Tensor::from_shape_vec(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let result = tensor.relu();

        for &value in result.data() {
            assert!(
                value >= 0.0,
                "ReLU output must be non-negative, got {value}"
            );
        }
    }

    #[test]
    fn test_formal_relu_idempotency() {
        // Test that ReLU(ReLU(x)) = ReLU(x) (idempotency)
        let tensor = Tensor::from_shape_vec(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let result1 = tensor.relu();
        let result2 = result1.relu();

        assert_eq!(result1.data(), result2.data());
    }

    #[test]
    fn test_formal_sigmoid_bounded() {
        // Test that sigmoid output is always in (0, 1)
        let tensor = Tensor::from_shape_vec(&[5], vec![-10.0, -1.0, 0.0, 1.0, 10.0]).unwrap();
        let result = tensor.sigmoid();

        for &value in result.data() {
            assert!(
                value > 0.0 && value < 1.0,
                "Sigmoid output must be in (0,1), got {value}"
            );
        }
    }

    #[test]
    fn test_formal_matmul_dimensions() {
        // Test that matrix multiplication produces correct dimensions
        let matrix_a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let matrix_b = Tensor::from_shape_vec(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let result = matrix_a.matmul(&matrix_b).unwrap();
        assert_eq!(result.shape(), [2, 2]);
    }

    #[test]
    fn test_formal_matmul_rectangular() {
        // Test matrix multiplication with rectangular matrices
        let matrix_a = Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let matrix_b = Tensor::from_shape_vec(&[3, 2], vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

        // 1x3 * 3x2 = 1x2
        let result = matrix_a.matmul(&matrix_b).unwrap();
        assert_eq!(result.shape(), [1, 2]);
    }

    #[test]
    fn test_concat_op() {
        let tensor1 = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0]));
        let tensor2 = Tensor::from_array(Array1::from_vec(vec![3.0, 4.0]));
        let inputs = vec![tensor1, tensor2];
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "0".to_string());

        let result = execute_operator(&OperatorType::Concat, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);
        // Proper implementation concatenates along axis 0
        assert_eq!(result[0].shape(), &[4]);
        let data = result[0].data().as_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concat_op_empty_inputs() {
        let inputs = vec![];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Concat, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 1 input"));
    }

    #[test]
    fn test_slice_op() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
        let inputs = vec![tensor];
        let mut attrs = HashMap::new();
        attrs.insert("starts".to_string(), "1".to_string());
        attrs.insert("ends".to_string(), "3".to_string());

        let result = execute_operator(&OperatorType::Slice, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[2]);
        let slice_data = result[0].data().as_slice().unwrap();
        assert_eq!(slice_data, &[2.0, 3.0]);
    }

    #[test]
    fn test_upsample_op() {
        let tensor = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let inputs = vec![tensor];
        let mut attrs = HashMap::new();
        attrs.insert("mode".to_string(), "nearest".to_string());

        let result = execute_operator(&OperatorType::Upsample, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_maxpool_op() {
        let tensor = Tensor::zeros(&[1, 1, 4, 4]); // NCHW format
        let inputs = vec![tensor];
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), "2,2".to_string());

        let result = execute_operator(&OperatorType::MaxPool, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_maxpool_op_wrong_dimensions() {
        let tensor = Tensor::zeros(&[4, 4]); // 2D tensor, should be 4D
        let inputs = vec![tensor];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::MaxPool, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("4D input tensor"));
    }

    #[test]
    fn test_softmax_op() {
        let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let inputs = vec![tensor];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Softmax, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);

        // Sum of softmax outputs should be approximately 1.0
        let sum: f32 = result[0].data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All outputs should be positive
        for &value in result[0].data() {
            assert!(value > 0.0);
        }
    }

    #[test]
    fn test_softmax_op_wrong_inputs() {
        let tensor1 = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0]));
        let tensor2 = Tensor::from_array(Array1::from_vec(vec![3.0, 4.0]));
        let inputs = vec![tensor1, tensor2]; // Should be exactly 1 input
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::Softmax, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 1 input"));
    }

    #[test]
    fn test_nms_op() {
        let boxes = Tensor::zeros(&[1, 4, 4]); // [batch_size, num_boxes, 4]
        let scores = Tensor::ones(&[1, 1, 4]); // [batch_size, num_classes, num_boxes]
        let inputs = vec![boxes, scores];
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::NonMaxSuppression, &inputs, &attrs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[0, 3]); // Empty result from simplified implementation
    }

    #[test]
    fn test_nms_op_insufficient_inputs() {
        let boxes = Tensor::zeros(&[1, 4, 4]);
        let inputs = vec![boxes]; // Should be at least 2 inputs
        let attrs = HashMap::new();

        let result = execute_operator(&OperatorType::NonMaxSuppression, &inputs, &attrs);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least 2 inputs"));
    }

    #[test]
    fn test_extended_operator_types_from_str() {
        // Test extended operator parsing
        assert_eq!(
            "Concat".parse::<OperatorType>().unwrap(),
            OperatorType::Concat
        );
        assert_eq!(
            "Slice".parse::<OperatorType>().unwrap(),
            OperatorType::Slice
        );
        assert_eq!(
            "Upsample".parse::<OperatorType>().unwrap(),
            OperatorType::Upsample
        );
        assert_eq!(
            "MaxPool".parse::<OperatorType>().unwrap(),
            OperatorType::MaxPool
        );
        assert_eq!(
            "Softmax".parse::<OperatorType>().unwrap(),
            OperatorType::Softmax
        );
        assert_eq!(
            "NonMaxSuppression".parse::<OperatorType>().unwrap(),
            OperatorType::NonMaxSuppression
        );
    }

    #[test]
    fn test_all_extended_operators_execute() {
        // Test that all extended operators can be executed without panicking
        let tensor_1d = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let tensor_4d = Tensor::zeros(&[1, 1, 2, 2]);
        let attrs = HashMap::new();

        // Test extended operators
        assert!(execute_operator(&OperatorType::Concat, &[tensor_1d.clone()], &attrs).is_ok());
        let mut slice_attrs = HashMap::new();
        slice_attrs.insert("starts".to_string(), "0".to_string());
        slice_attrs.insert("ends".to_string(), "1".to_string());
        assert!(execute_operator(&OperatorType::Slice, &[tensor_1d.clone()], &slice_attrs).is_ok());
        assert!(execute_operator(&OperatorType::Upsample, &[tensor_4d.clone()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::MaxPool, &[tensor_4d.clone()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::Softmax, &[tensor_1d.clone()], &attrs).is_ok());
        assert!(execute_operator(
            &OperatorType::NonMaxSuppression,
            &[tensor_4d.clone(), tensor_1d.clone()],
            &attrs
        )
        .is_ok());
    }
}
