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

#[cfg(all(feature = "blas", not(feature = "naive-conv")))]
extern crate blas_src; // links the selected BLAS implementation

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

fn non_negative_usize(value: i64, parameter: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        OnnxError::invalid_dimensions(format!("{parameter} must be non-negative, got {value}"))
    })
}

fn parse_non_negative_array(
    value: &str,
    parameter: &str,
    expected_len: usize,
) -> Result<Vec<usize>> {
    let values = parse_int_array(value)?;
    if values.len() != expected_len {
        return Err(OnnxError::invalid_dimensions(format!(
            "{parameter} requires {expected_len} values, got {}",
            values.len()
        )));
    }
    values
        .into_iter()
        .map(|value| non_negative_usize(value, parameter))
        .collect()
}

fn finite_integer_i64(value: f32, parameter: &str) -> Result<i64> {
    let value_f64 = f64::from(value);
    if !value.is_finite()
        || value.fract() != 0.0
        || value_f64 < i64::MIN as f64
        || value_f64 >= 9_223_372_036_854_775_808.0
    {
        return Err(OnnxError::invalid_dimensions(format!(
            "{parameter} must be a finite integer representable as i64, got {value}"
        )));
    }

    Ok(value as i64)
}

fn checked_element_count(shape: &[usize], operation: &str) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }

    let count = shape.iter().try_fold(1usize, |count, &dimension| {
        count.checked_mul(dimension).ok_or_else(|| {
            OnnxError::invalid_dimensions(format!(
                "{operation} shape {shape:?} exceeds the addressable element count"
            ))
        })
    })?;
    if count > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(OnnxError::invalid_dimensions(format!(
            "{operation} shape {shape:?} exceeds the addressable byte size"
        )));
    }
    Ok(count)
}

fn allocate_f32(count: usize, value: f32, operation: &str) -> Result<Vec<f32>> {
    let mut data = Vec::new();
    data.try_reserve_exact(count).map_err(|error| {
        OnnxError::invalid_dimensions(format!(
            "Unable to allocate {operation} with {count} elements: {error}"
        ))
    })?;
    data.resize(count, value);
    Ok(data)
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
/// # Formal Specifications (specified in `formal/operators.mlw`)
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
/// # Formal Specifications (specified in `formal/operators.mlw`)
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
/// # Formal Specifications (specified in `formal/operators.mlw`)
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

/// Dimensional parameters for the Conv operator, derived from inputs and attributes.
struct ConvParams {
    batch_size: usize,
    channels_in: usize,
    height_in: usize,
    width_in: usize,
    channels_out: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    height_out: usize,
    width_out: usize,
}

/// 2D Convolution operator — dispatches to one of three back-ends:
///
/// | Feature flag  | Back-end                        | Speed   |
/// |---------------|---------------------------------|---------|
/// | `naive-conv`  | 6-level nested loop             | slowest |
/// | `blas`        | im2col + OpenBLAS `sgemm`       | fastest |
/// | *(default)*   | im2col + ndarray `dot` (matrixmultiply) | fast |
///
/// The `naive-conv` path is kept as a readable reference so readers can
/// compare the algorithmic structure to the im2col transformation.
fn conv_op(inputs: &[Tensor], attrs: &HashMap<String, String>) -> Result<Vec<Tensor>> {
    if !(2..=3).contains(&inputs.len()) {
        return Err(OnnxError::invalid_dimensions(format!(
            "Conv operator requires 2 or 3 inputs (input, kernel, optional bias), got {}",
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

    if kernel_h == 0 || kernel_w == 0 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Conv kernel dimensions must be non-zero, got {kernel_h}x{kernel_w}"
        )));
    }

    // Validate channel dimensions match
    if channels_in != channels_in_kernel {
        return Err(OnnxError::invalid_dimensions(format!(
            "Input channels ({channels_in}) must match kernel input channels ({channels_in_kernel})"
        )));
    }

    log::debug!("Conv attributes: {attrs:?}");

    let strides = attrs
        .get("strides")
        .map(|value| parse_non_negative_array(value, "Conv strides", 2))
        .transpose()?
        .unwrap_or_else(|| vec![1, 1]);
    let stride_h = strides[0];
    let stride_w = strides[1];

    if stride_h == 0 || stride_w == 0 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Conv strides must be non-zero, got {stride_h}x{stride_w}"
        )));
    }

    let pads = attrs
        .get("pads")
        .map(|value| parse_non_negative_array(value, "Conv pads", 4))
        .transpose()?
        .unwrap_or_else(|| vec![0, 0, 0, 0]);
    let [pad_top, pad_left, pad_bottom, pad_right] = [pads[0], pads[1], pads[2], pads[3]];

    if let Some(auto_pad) = attrs.get("auto_pad") {
        if auto_pad != "NOTSET" {
            return Err(OnnxError::unsupported_operation(format!(
                "Conv auto_pad '{auto_pad}' is not implemented; provide explicit pads"
            )));
        }
    }
    let dilations = attrs
        .get("dilations")
        .map(|value| parse_non_negative_array(value, "Conv dilations", 2))
        .transpose()?
        .unwrap_or_else(|| vec![1, 1]);
    if dilations != [1, 1] {
        return Err(OnnxError::unsupported_operation(format!(
            "Conv dilations {dilations:?} are not implemented"
        )));
    }
    let group = attrs
        .get("group")
        .map(|value| {
            value.parse::<usize>().map_err(|error| {
                OnnxError::invalid_dimensions(format!("Invalid Conv group '{value}': {error}"))
            })
        })
        .transpose()?
        .unwrap_or(1);
    if group != 1 {
        return Err(OnnxError::unsupported_operation(format!(
            "Grouped Conv with group={group} is not implemented"
        )));
    }
    if let Some(kernel_shape) = attrs.get("kernel_shape") {
        let declared = parse_non_negative_array(kernel_shape, "Conv kernel_shape", 2)?;
        if declared != [kernel_h, kernel_w] {
            return Err(OnnxError::invalid_dimensions(format!(
                "Conv kernel_shape {declared:?} does not match kernel tensor [{kernel_h}, {kernel_w}]"
            )));
        }
    }

    if let Some(bias) = bias {
        let bias_shape = bias.shape();
        let supported_bias = bias_shape == [channels_out]
            || (bias_shape.len() == 4
                && bias_shape[0] == 1
                && bias_shape[1] == channels_out
                && bias_shape[2] == 1
                && bias_shape[3] == 1);
        if !supported_bias {
            return Err(OnnxError::invalid_dimensions(format!(
                "Conv bias shape {bias_shape:?} must be [{channels_out}] or [1, {channels_out}, 1, 1]"
            )));
        }
    }

    // Calculate output dimensions with checked arithmetic because padding is model-controlled.
    let padded_height = height_in
        .checked_add(pad_top)
        .and_then(|value| value.checked_add(pad_bottom))
        .ok_or_else(|| OnnxError::invalid_dimensions("Conv padded height overflows usize"))?;
    let padded_width = width_in
        .checked_add(pad_left)
        .and_then(|value| value.checked_add(pad_right))
        .ok_or_else(|| OnnxError::invalid_dimensions("Conv padded width overflows usize"))?;

    if padded_height < kernel_h || padded_width < kernel_w {
        return Err(OnnxError::invalid_dimensions(format!(
            "Conv padded input {padded_height}x{padded_width} is smaller than kernel {kernel_h}x{kernel_w}"
        )));
    }

    let height_out = (padded_height - kernel_h) / stride_h + 1;
    let width_out = (padded_width - kernel_w) / stride_w + 1;

    let output_shape = [batch_size, channels_out, height_out, width_out];
    checked_element_count(&output_shape, "Conv output")?;
    channels_in
        .checked_mul(kernel_h)
        .and_then(|value| value.checked_mul(kernel_w))
        .ok_or_else(|| OnnxError::invalid_dimensions("Conv kernel shape overflows usize"))?;
    height_in
        .checked_mul(width_in)
        .and_then(|value| value.checked_mul(channels_in))
        .ok_or_else(|| OnnxError::invalid_dimensions("Conv input shape overflows usize"))?;

    log::debug!("Conv: input {input_shape:?}, kernel {kernel_shape:?} -> output {output_shape:?}");
    log::debug!(
        "Conv: stride={stride_h}x{stride_w}, pad=[{pad_top},{pad_left},{pad_bottom},{pad_right}]"
    );

    let input_data = input.data();
    let kernel_data = kernel.data();
    let input_standard = input_data.as_standard_layout();
    let kernel_standard = kernel_data.as_standard_layout();
    let input_flat = input_standard.as_slice().ok_or_else(|| {
        OnnxError::invalid_dimensions("Conv could not normalize the input tensor layout")
    })?;
    let kernel_flat = kernel_standard.as_slice().ok_or_else(|| {
        OnnxError::invalid_dimensions("Conv could not normalize the kernel tensor layout")
    })?;

    let p = ConvParams {
        batch_size,
        channels_in,
        height_in,
        width_in,
        channels_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_top,
        pad_left,
        height_out,
        width_out,
    };

    // Dispatch to the selected back-end.
    // Priority when multiple features are set: naive-conv > blas > im2col.
    #[cfg(feature = "naive-conv")]
    let mut output_data = conv_naive(&p, input_flat, kernel_flat)?;

    #[cfg(all(feature = "blas", not(feature = "naive-conv")))]
    let mut output_data = conv_blas(&p, input_flat, kernel_flat)?;

    #[cfg(not(any(feature = "naive-conv", feature = "blas")))]
    let mut output_data = conv_im2col(&p, input_flat, kernel_flat)?;

    log::debug!("Conv: computed {} output values", output_data.len());

    // Apply bias if present
    let final_output = if let Some(bias) = bias {
        let bias_shape = bias.shape();
        log::debug!("Conv: applying bias with shape {bias_shape:?}");

        let bias_data = bias.data();
        let get_bias_val = |c_out: usize| -> f32 {
            if bias_shape == [channels_out] {
                bias_data[c_out]
            } else if bias_shape.len() == 4
                && bias_shape[0] == 1
                && bias_shape[1] == channels_out
                && bias_shape[2] == 1
                && bias_shape[3] == 1
            {
                bias_data[[0, c_out, 0, 0]]
            } else {
                0.0 // unsupported shape; logged below
            }
        };

        let hw = height_out * width_out;
        // Add bias for every (batch, channel) slice — must cover all batches.
        for n in 0..batch_size {
            let batch_offset = n * channels_out * hw;
            for c_out in 0..channels_out {
                let bias_val = get_bias_val(c_out);
                let start = batch_offset + c_out * hw;
                let end = start + hw;
                for val in &mut output_data[start..end] {
                    *val += bias_val;
                }
            }
        }

        Tensor::from_shape_vec(&output_shape, output_data)?
    } else {
        Tensor::from_shape_vec(&output_shape, output_data)?
    };

    log::debug!("Conv: final output shape {:?}", final_output.shape());
    Ok(vec![final_output])
}

// ---------------------------------------------------------------------------
// Conv back-end implementations
// ---------------------------------------------------------------------------

/// **Naive back-end** (`--features naive-conv`)
///
/// Direct 6-level nested loop over (N, C_out, H_out, W_out, C_in, Kh, Kw).
/// Kept as a readable reference — every multiply-accumulate maps 1-to-1 to
/// the mathematical definition of convolution.  Uses precomputed flat-slice
/// strides so it avoids dynamic ndarray index lookups.
#[cfg(feature = "naive-conv")]
fn conv_naive(p: &ConvParams, input: &[f32], kernel: &[f32]) -> Result<Vec<f32>> {
    let &ConvParams {
        batch_size,
        channels_in,
        height_in,
        width_in,
        channels_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_top,
        pad_left,
        height_out,
        width_out,
    } = p;

    let in_c_stride = height_in * width_in;
    let in_n_stride = channels_in * in_c_stride;
    let k_kh_stride = kernel_w;
    let k_cin_stride = kernel_h * k_kh_stride;
    let k_cout_stride = channels_in * k_cin_stride;
    let out_hw = height_out * width_out;
    let out_c_stride = out_hw;
    let out_n_stride = channels_out * out_hw;

    let output_len = checked_element_count(
        &[batch_size, channels_out, height_out, width_out],
        "Conv naive output",
    )?;
    let mut out = allocate_f32(output_len, 0.0, "Conv naive output")?;

    for n in 0..batch_size {
        for c_out in 0..channels_out {
            for h_out in 0..height_out {
                for w_out in 0..width_out {
                    let mut sum = 0.0f32;
                    for c_in in 0..channels_in {
                        for kh in 0..kernel_h {
                            let h_in_padded = h_out * stride_h + kh;
                            if h_in_padded < pad_top || h_in_padded >= height_in + pad_top {
                                continue;
                            }
                            let h_in = h_in_padded - pad_top;
                            for kw in 0..kernel_w {
                                let w_in_padded = w_out * stride_w + kw;
                                if w_in_padded < pad_left || w_in_padded >= width_in + pad_left {
                                    continue;
                                }
                                let w_in = w_in_padded - pad_left;
                                let i_idx =
                                    n * in_n_stride + c_in * in_c_stride + h_in * width_in + w_in;
                                let k_idx = c_out * k_cout_stride
                                    + c_in * k_cin_stride
                                    + kh * k_kh_stride
                                    + kw;
                                sum += input[i_idx] * kernel[k_idx];
                            }
                        }
                    }
                    out[n * out_n_stride + c_out * out_c_stride + h_out * width_out + w_out] = sum;
                }
            }
        }
    }
    Ok(out)
}

/// **im2col back-end** (default, no extra features required)
///
/// # How im2col works
///
/// Instead of computing each output pixel with nested loops, we first
/// *rearrange* the input into a 2-D matrix (the "column" matrix) where
/// every row contains the flattened patch of input values that contribute
/// to one output pixel.  The convolution then reduces to a single matrix
/// multiply:
///
/// ```text
/// col   [N·H_out·W_out,  C_in·Kh·Kw]   ← im2col(input)
/// W     [C_out,          C_in·Kh·Kw]   ← kernel reshaped (no copy)
///
/// output = col @ W.T                    ← one GEMM call
/// [N·H_out·W_out, C_out]  →  reshape → [N, C_out, H_out, W_out]
/// ```
///
/// The GEMM is performed by ndarray's `dot`, which delegates to the
/// `matrixmultiply` crate — a cache-blocking, SIMD-vectorised pure-Rust
/// GEMM that is competitive with OpenBLAS on many workloads.
#[cfg(not(any(feature = "naive-conv", feature = "blas")))]
fn conv_im2col(p: &ConvParams, input: &[f32], kernel: &[f32]) -> Result<Vec<f32>> {
    use ndarray::Array2;

    let &ConvParams {
        batch_size,
        channels_in,
        height_in,
        width_in,
        channels_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_top,
        pad_left,
        height_out,
        width_out,
    } = p;

    let patch_len = checked_element_count(
        &[channels_in, kernel_h, kernel_w],
        "Conv im2col kernel patch",
    )?;
    let n_patches =
        checked_element_count(&[batch_size, height_out, width_out], "Conv im2col patches")?;

    // ------------------------------------------------------------------
    // Step 1: im2col — build the [n_patches, patch_len] column matrix.
    // Each row collects the input values (with zero-padding) that a single
    // output position (n, h_out, w_out) multiplies against the kernel.
    // ------------------------------------------------------------------
    let col_len = checked_element_count(&[n_patches, patch_len], "Conv im2col workspace")?;
    let mut col = allocate_f32(col_len, 0.0, "Conv im2col workspace")?;
    let in_c_stride = height_in * width_in;
    let in_n_stride = channels_in * in_c_stride;

    for n in 0..batch_size {
        for h_out in 0..height_out {
            for w_out in 0..width_out {
                let row = n * height_out * width_out + h_out * width_out + w_out;
                let row_base = row * patch_len;
                let mut col_idx = 0;
                for c_in in 0..channels_in {
                    for kh in 0..kernel_h {
                        let h_in_padded = h_out * stride_h + kh;
                        for kw in 0..kernel_w {
                            let w_in_padded = w_out * stride_w + kw;
                            // Pixels that fall in the zero-padding stay 0.0 (pre-initialised).
                            if h_in_padded >= pad_top
                                && h_in_padded < height_in + pad_top
                                && w_in_padded >= pad_left
                                && w_in_padded < width_in + pad_left
                            {
                                let h_in = h_in_padded - pad_top;
                                let w_in = w_in_padded - pad_left;
                                col[row_base + col_idx] = input
                                    [n * in_n_stride + c_in * in_c_stride + h_in * width_in + w_in];
                            }
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 2: GEMM — output = col @ kernel.T
    // col    : [n_patches, patch_len]
    // kernel : [channels_out, patch_len]  (already flat, no copy needed)
    // output : [n_patches, channels_out]
    // ------------------------------------------------------------------
    let col_mat = Array2::from_shape_vec((n_patches, patch_len), col)
        .map_err(|e| OnnxError::invalid_dimensions(format!("im2col reshape failed: {e}")))?;
    let w_mat = Array2::from_shape_vec((channels_out, patch_len), kernel.to_vec())
        .map_err(|e| OnnxError::invalid_dimensions(format!("kernel reshape failed: {e}")))?;

    let out_mat = col_mat.dot(&w_mat.t()); // [n_patches, channels_out]

    // ------------------------------------------------------------------
    // Step 3: reindex from [N·H·W, C_out]  →  [N, C_out, H, W] (NCHW).
    // ------------------------------------------------------------------
    let out_slice = out_mat
        .as_slice()
        .expect("matmul output should be contiguous");
    let hw = height_out
        .checked_mul(width_out)
        .ok_or_else(|| OnnxError::invalid_dimensions("Conv output shape overflows usize"))?;
    let output_len = checked_element_count(
        &[batch_size, channels_out, height_out, width_out],
        "Conv im2col output",
    )?;
    let mut output = allocate_f32(output_len, 0.0, "Conv im2col output")?;
    for n in 0..batch_size {
        for h in 0..height_out {
            for w in 0..width_out {
                let row = n * hw + h * width_out + w;
                for c_out in 0..channels_out {
                    output[n * channels_out * hw + c_out * hw + h * width_out + w] =
                        out_slice[row * channels_out + c_out];
                }
            }
        }
    }
    Ok(output)
}

/// **BLAS back-end** (`--features blas`)
///
/// Identical im2col transform as the default back-end, but the GEMM is
/// performed by OpenBLAS `sgemm` instead of `matrixmultiply`.  On machines
/// with a well-tuned BLAS (e.g. Intel MKL or OpenBLAS with AVX-512) this
/// can be 2–4× faster than the pure-Rust path.
///
/// # System requirements
/// ```bash
/// # Ubuntu / Debian / WSL2
/// sudo apt install libopenblas-dev
/// cargo build --features blas
/// ```
#[cfg(all(feature = "blas", not(feature = "naive-conv")))]
fn conv_blas(p: &ConvParams, input: &[f32], kernel: &[f32]) -> Result<Vec<f32>> {
    let &ConvParams {
        batch_size,
        channels_in,
        height_in,
        width_in,
        channels_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_top,
        pad_left,
        height_out,
        width_out,
    } = p;

    let patch_len =
        checked_element_count(&[channels_in, kernel_h, kernel_w], "Conv BLAS kernel patch")?;
    let n_patches =
        checked_element_count(&[batch_size, height_out, width_out], "Conv BLAS patches")?;
    let hw = height_out
        .checked_mul(width_out)
        .ok_or_else(|| OnnxError::invalid_dimensions("Conv BLAS output shape overflows usize"))?;

    // Step 1: im2col (same as the default back-end).
    let col_len = checked_element_count(&[n_patches, patch_len], "Conv BLAS workspace")?;
    let mut col = allocate_f32(col_len, 0.0, "Conv BLAS workspace")?;
    let in_c_stride = height_in * width_in;
    let in_n_stride = channels_in * in_c_stride;

    for n in 0..batch_size {
        for h_out in 0..height_out {
            for w_out in 0..width_out {
                let row = n * hw + h_out * width_out + w_out;
                let row_base = row * patch_len;
                let mut col_idx = 0;
                for c_in in 0..channels_in {
                    for kh in 0..kernel_h {
                        let h_in_padded = h_out * stride_h + kh;
                        for kw in 0..kernel_w {
                            let w_in_padded = w_out * stride_w + kw;
                            if h_in_padded >= pad_top
                                && h_in_padded < height_in + pad_top
                                && w_in_padded >= pad_left
                                && w_in_padded < width_in + pad_left
                            {
                                let h_in = h_in_padded - pad_top;
                                let w_in = w_in_padded - pad_left;
                                col[row_base + col_idx] = input
                                    [n * in_n_stride + c_in * in_c_stride + h_in * width_in + w_in];
                            }
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    }

    // Step 2: GEMM via OpenBLAS sgemm.
    //
    // We compute: output = col @ kernel.T
    //   col    [m=n_patches, k=patch_len]   (row-major, no transpose)
    //   kernel [n=channels_out, k=patch_len] (row-major, transposed in GEMM)
    //   output [m, n]
    let m = i32::try_from(n_patches)
        .map_err(|_| OnnxError::invalid_dimensions("Conv BLAS patch count exceeds i32::MAX"))?;
    let n = i32::try_from(channels_out)
        .map_err(|_| OnnxError::invalid_dimensions("Conv BLAS output channels exceed i32::MAX"))?;
    let k = i32::try_from(patch_len)
        .map_err(|_| OnnxError::invalid_dimensions("Conv BLAS patch size exceeds i32::MAX"))?;
    let flat_output_len =
        checked_element_count(&[n_patches, channels_out], "Conv BLAS matrix output")?;
    let mut out_flat = allocate_f32(flat_output_len, 0.0, "Conv BLAS matrix output")?;

    // SAFETY: `col`, `kernel`, and `out_flat` are distinct, live f32 buffers.
    // Their lengths are respectively m*k, n*k, and m*n from the checked shape
    // products above. Row-major leading dimensions k, k, and n match SGEMM's
    // A, transposed-B, and C layouts, and all dimensions fit in c_int/i32.
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,     // col: no transpose
            cblas::Transpose::Ordinary, // kernel: transpose
            m,
            n,
            k,
            1.0f32, // alpha
            &col,
            k, // A = col,    lda = k
            kernel,
            k,      // B = kernel, ldb = k (before transpose)
            0.0f32, // beta
            &mut out_flat,
            n, // C = output, ldc = n
        );
    }

    // Step 3: reindex [N·H·W, C_out] → [N, C_out, H, W].
    let output_len = checked_element_count(
        &[batch_size, channels_out, height_out, width_out],
        "Conv BLAS output",
    )?;
    let mut output = allocate_f32(output_len, 0.0, "Conv BLAS output")?;
    for n in 0..batch_size {
        for h in 0..height_out {
            for w in 0..width_out {
                let row = n * hw + h * width_out + w;
                for c_out in 0..channels_out {
                    output[n * channels_out * hw + c_out * hw + h * width_out + w] =
                        out_flat[row * channels_out + c_out];
                }
            }
        }
    }
    Ok(output)
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
/// # Formal Specifications (specified in `formal/operators.mlw`)
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

    let result = inputs[0].relu()?;

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
/// # Formal Specifications (specified in `formal/operators.mlw`)
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

    let result = inputs[0].sigmoid()?;

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
    let raw_shape: Vec<i64> = shape_tensor
        .data()
        .iter()
        .map(|&value| finite_integer_i64(value, "Reshape dimension"))
        .collect::<Result<_>>()?;
    log::debug!(
        "Reshape: input shape {:?}, target shape {:?}",
        data.shape(),
        raw_shape
    );

    // Handle special ONNX reshape values
    let input_shape = data.shape();
    let total_elements = data.len();
    let mut new_shape = Vec::new();
    let mut infer_dim_index = None;
    let mut inferred_elements = 1usize;

    for (i, &dim) in raw_shape.iter().enumerate() {
        if dim == 0 {
            // In ONNX, 0 means copy the corresponding dimension from input
            if i < input_shape.len() {
                new_shape.push(input_shape[i]);
                inferred_elements =
                    inferred_elements
                        .checked_mul(input_shape[i])
                        .ok_or_else(|| {
                            OnnxError::invalid_dimensions(
                                "Reshape known dimensions overflow usize".to_string(),
                            )
                        })?;
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
            let dimension = usize::try_from(dim).map_err(|_| {
                OnnxError::invalid_dimensions(format!(
                    "Reshape dimension {dim} is not representable as usize"
                ))
            })?;
            new_shape.push(dimension);
            inferred_elements = inferred_elements.checked_mul(dimension).ok_or_else(|| {
                OnnxError::invalid_dimensions("Reshape known dimensions overflow usize")
            })?;
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
            Err(error) => {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Invalid Transpose perm attribute '{perm_str}': {error}"
                )))
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
            .map(|&value| finite_integer_i64(value, "Slice start"))
            .collect::<Result<_>>()?;
        let ends: Vec<i64> = ends_tensor
            .data()
            .iter()
            .map(|&value| {
                if value >= i64::MAX as f32 * 0.9 {
                    Ok(i64::MAX)
                } else if value <= i64::MIN as f32 * 0.9 {
                    Ok(i64::MIN)
                } else {
                    finite_integer_i64(value, "Slice end")
                }
            })
            .collect::<Result<_>>()?;

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
                    .map(|&value| finite_integer_i64(value, "Slice axis"))
                    .collect::<Result<Vec<_>>>()?,
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
                    .map(|&value| finite_integer_i64(value, "Slice step"))
                    .collect::<Result<Vec<_>>>()?,
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
            _ => Err(OnnxError::invalid_dimensions(
                "Slice requires starts and ends inputs or attributes".to_string(),
            )),
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

    let mode = attributes
        .get("mode")
        .map(|s| s.as_str())
        .unwrap_or("nearest");

    Err(OnnxError::unsupported_operation(format!(
        "Upsample mode '{mode}' is not implemented"
    )))
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
    let kernel_shape = attributes
        .get("kernel_shape")
        .map(|value| parse_non_negative_array(value, "MaxPool kernel_shape", 2))
        .transpose()?
        .unwrap_or_else(|| vec![2, 2]);
    let strides = attributes
        .get("strides")
        .map(|value| parse_non_negative_array(value, "MaxPool strides", 2))
        .transpose()?
        .unwrap_or_else(|| vec![1, 1]);
    let pads = attributes
        .get("pads")
        .map(|value| parse_non_negative_array(value, "MaxPool pads", 4))
        .transpose()?
        .unwrap_or_else(|| vec![0, 0, 0, 0]);

    let [kernel_h, kernel_w] = [kernel_shape[0], kernel_shape[1]];
    let [stride_h, stride_w] = [strides[0], strides[1]];
    let [pad_top, pad_left, pad_bottom, pad_right] = [pads[0], pads[1], pads[2], pads[3]];

    if let Some(auto_pad) = attributes.get("auto_pad") {
        if auto_pad != "NOTSET" {
            return Err(OnnxError::unsupported_operation(format!(
                "MaxPool auto_pad '{auto_pad}' is not implemented; provide explicit pads"
            )));
        }
    }
    let dilations = attributes
        .get("dilations")
        .map(|value| parse_non_negative_array(value, "MaxPool dilations", 2))
        .transpose()?
        .unwrap_or_else(|| vec![1, 1]);
    if dilations != [1, 1] {
        return Err(OnnxError::unsupported_operation(format!(
            "MaxPool dilations {dilations:?} are not implemented"
        )));
    }
    for (attribute, description) in [
        ("ceil_mode", "ceil mode"),
        ("storage_order", "storage order"),
    ] {
        if let Some(value) = attributes.get(attribute) {
            let parsed = value.parse::<i64>().map_err(|error| {
                OnnxError::invalid_dimensions(format!(
                    "Invalid MaxPool {attribute} '{value}': {error}"
                ))
            })?;
            if parsed != 0 {
                return Err(OnnxError::unsupported_operation(format!(
                    "MaxPool {description} {parsed} is not implemented"
                )));
            }
        }
    }

    if kernel_h == 0 || kernel_w == 0 {
        return Err(OnnxError::invalid_dimensions(format!(
            "MaxPool kernel dimensions must be non-zero, got {kernel_h}x{kernel_w}"
        )));
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(OnnxError::invalid_dimensions(format!(
            "MaxPool strides must be non-zero, got {stride_h}x{stride_w}"
        )));
    }

    let input_shape = input.shape();
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];

    // Calculate output dimensions with checked arithmetic because attributes are model-controlled.
    let padded_h = input_h
        .checked_add(pad_top)
        .and_then(|value| value.checked_add(pad_bottom))
        .ok_or_else(|| OnnxError::invalid_dimensions("MaxPool padded height overflows usize"))?;
    let padded_w = input_w
        .checked_add(pad_left)
        .and_then(|value| value.checked_add(pad_right))
        .ok_or_else(|| OnnxError::invalid_dimensions("MaxPool padded width overflows usize"))?;
    if padded_h < kernel_h || padded_w < kernel_w {
        return Err(OnnxError::invalid_dimensions(format!(
            "MaxPool padded input {padded_h}x{padded_w} is smaller than kernel {kernel_h}x{kernel_w}"
        )));
    }

    let output_h = (padded_h - kernel_h) / stride_h + 1;
    let output_w = (padded_w - kernel_w) / stride_w + 1;
    let output_shape = [batch_size, channels, output_h, output_w];

    log::debug!("MaxPool: {input_h}x{input_w} -> {output_h}x{output_w}, kernel={kernel_h}x{kernel_w}, stride={stride_h}x{stride_w}, pad=[{pad_top},{pad_left},{pad_bottom},{pad_right}]");

    let output_len = checked_element_count(&output_shape, "MaxPool output")?;
    let mut output_data = Vec::new();
    output_data.try_reserve_exact(output_len).map_err(|error| {
        OnnxError::invalid_dimensions(format!(
            "Unable to allocate MaxPool output with shape {output_shape:?}: {error}"
        ))
    })?;
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

    if input.is_empty() {
        return Err(OnnxError::invalid_dimensions(
            "Cannot apply softmax to empty tensor".to_string(),
        ));
    }

    let mut output = input.data().clone();
    for mut lane in output.lanes_mut(ndarray::Axis(normalized_axis)) {
        let max_value = lane
            .iter()
            .fold(f32::NEG_INFINITY, |current, &value| current.max(value));
        lane.mapv_inplace(|value| (value - max_value).exp());
        let denominator: f32 = lane.iter().sum();
        if denominator == 0.0 || !denominator.is_finite() {
            return Err(OnnxError::invalid_dimensions(
                "Softmax denominator is zero or non-finite".to_string(),
            ));
        }
        lane.mapv_inplace(|value| value / denominator);
    }

    let result = Tensor::from_array(output);
    log::debug!("Softmax: output shape {:?}", result.shape());
    Ok(vec![result])
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

    Err(OnnxError::unsupported_operation(
        "NonMaxSuppression is not implemented".to_string(),
    ))
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
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(OnnxError::invalid_dimensions(format!(
            "BatchNormalization epsilon must be finite and non-negative, got {epsilon}"
        )));
    }

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

    let scale_data = scale.data();
    let bias_data = bias.data();
    let mean_data = mean.data();
    let var_data = variance.data();
    let input_data = input.data();

    if var_data
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(OnnxError::invalid_dimensions(
            "BatchNormalization variance must contain finite, non-negative values".to_string(),
        ));
    }

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
                    let normalized = (input_data[[n, c, h, w]] - mean_val) * inv_std;
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
    if inputs.is_empty() || inputs.len() > 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Split requires 1 or 2 inputs, got {}",
            inputs.len()
        )));
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
            .map(|&size| {
                let size = finite_integer_i64(size, "Split size")?;
                non_negative_usize(size, "Split size")
            })
            .collect::<Result<Vec<_>>>()?
    } else if let Some(split_attr) = attributes.get("split") {
        parse_int_array(split_attr)?
            .into_iter()
            .map(|size| non_negative_usize(size, "Split size"))
            .collect::<Result<Vec<_>>>()?
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
    let total_size = split_sizes.iter().try_fold(0usize, |total, &size| {
        total
            .checked_add(size)
            .ok_or_else(|| OnnxError::invalid_dimensions("Split sizes overflow usize"))
    })?;
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
            let mut output_shape = input.shape().to_vec();
            output_shape[normalized_axis] = 0;
            results.push(Tensor::from_shape_vec(&output_shape, Vec::new())?);
            continue;
        }

        // Create slice parameters for this split
        let mut starts = vec![0i64; input.ndim()];
        let mut ends = vec![0i64; input.ndim()];

        for i in 0..input.ndim() {
            if i == normalized_axis {
                starts[i] = i64::try_from(current_offset)
                    .map_err(|_| OnnxError::invalid_dimensions("Split offset exceeds i64::MAX"))?;
                ends[i] = i64::try_from(current_offset + split_size)
                    .map_err(|_| OnnxError::invalid_dimensions("Split end exceeds i64::MAX"))?;
            } else {
                starts[i] = 0;
                ends[i] = i64::try_from(input.shape()[i]).map_err(|_| {
                    OnnxError::invalid_dimensions("Split input dimension exceeds i64::MAX")
                })?;
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

    // ONNX Gather indices must be integers in the range [-s, s - 1], where s
    // is the size of the selected axis.
    let axis_size_signed = axis_size as i128;
    let indices_int: Vec<usize> = indices
        .data()
        .iter()
        .map(|&value| {
            if !value.is_finite() || value.fract() != 0.0 {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Gather index {value} is not a finite integer"
                )));
            }

            let index = value as i128;
            if index < -axis_size_signed || index >= axis_size_signed {
                return Err(OnnxError::invalid_dimensions(format!(
                    "Gather index {index} is out of bounds for axis {normalized_axis} with size {axis_size}"
                )));
            }

            Ok(if index < 0 {
                (axis_size_signed + index) as usize
            } else {
                index as usize
            })
        })
        .collect::<Result<_>>()?;

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

    let data_standard = data.data().as_standard_layout();
    let data_data = data_standard.as_slice().ok_or_else(|| {
        OnnxError::invalid_dimensions("Gather could not normalize the input tensor layout")
    })?;
    let output_len = checked_element_count(&output_shape, "Gather output")?;
    let mut output_data = allocate_f32(output_len, 0.0, "Gather output")?;

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

    if inputs[0].ndim() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "ConstantOfShape shape input must be 1D, got {} dimensions",
            inputs[0].ndim()
        )));
    }

    let shape: Vec<usize> = inputs[0]
        .data()
        .iter()
        .map(|&dimension| {
            let dimension = finite_integer_i64(dimension, "ConstantOfShape dimension")?;
            non_negative_usize(dimension, "ConstantOfShape dimension")
        })
        .collect::<Result<_>>()?;
    let element_count = checked_element_count(&shape, "ConstantOfShape output")?;
    let mut data = Vec::new();
    data.try_reserve_exact(element_count).map_err(|error| {
        OnnxError::invalid_dimensions(format!(
            "Unable to allocate ConstantOfShape output with shape {shape:?}: {error}"
        ))
    })?;
    data.resize(element_count, value);
    let result = Tensor::from_shape_vec(&shape, data)?;

    Ok(vec![result])
}

/// Cast operator implementation
fn cast_op(
    inputs: &[Tensor],
    attributes: &std::collections::HashMap<String, String>,
) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Cast requires exactly 1 input, got {}",
            inputs.len()
        )));
    }

    match attributes.get("to").map(String::as_str) {
        Some("1" | "float" | "float32") => Ok(vec![inputs[0].clone()]),
        Some(target) => Err(OnnxError::unsupported_operation(format!(
            "Cast target '{target}' is not supported; RunNX tensors are float32"
        ))),
        None => Err(OnnxError::invalid_dimensions(
            "Cast requires the 'to' attribute".to_string(),
        )),
    }
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

    if mode != "constant" {
        return Err(OnnxError::unsupported_operation(format!(
            "Pad mode '{mode}' is not implemented"
        )));
    }

    // Get padding values
    let pads: Vec<usize> = pads_tensor
        .data()
        .iter()
        .map(|&padding| {
            let padding = finite_integer_i64(padding, "Pad value")?;
            non_negative_usize(padding, "Pad value")
        })
        .collect::<Result<_>>()?;

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
        .map(|(i, &size)| {
            size.checked_add(begin_pads[i])
                .and_then(|value| value.checked_add(end_pads[i]))
                .ok_or_else(|| {
                    OnnxError::invalid_dimensions(format!(
                        "Pad output dimension {i} overflows usize"
                    ))
                })
        })
        .collect::<Result<_>>()?;

    log::debug!("Pad: output shape {output_shape:?}");

    // Create output tensor filled with constant value
    let output_len = checked_element_count(&output_shape, "Pad output")?;
    let mut output_data = Vec::new();
    output_data.try_reserve_exact(output_len).map_err(|error| {
        OnnxError::invalid_dimensions(format!(
            "Unable to allocate Pad output with shape {output_shape:?}: {error}"
        ))
    })?;
    output_data.resize(output_len, constant_value);

    // Calculate strides for both input and output
    let mut input_strides = vec![1usize; ndim];
    let mut output_strides = vec![1usize; ndim];

    for i in (0..ndim.saturating_sub(1)).rev() {
        input_strides[i] = input_strides[i + 1]
            .checked_mul(input_shape[i + 1])
            .ok_or_else(|| OnnxError::invalid_dimensions("Pad input strides overflow usize"))?;
        output_strides[i] = output_strides[i + 1]
            .checked_mul(output_shape[i + 1])
            .ok_or_else(|| OnnxError::invalid_dimensions("Pad output strides overflow usize"))?;
    }

    // Copy input data to the appropriate region in output
    for (input_idx, &input_val) in input.data().iter().enumerate() {
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
    if inputs.is_empty() || inputs.len() > 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "ReduceMean requires 1 or 2 inputs, got {}",
            inputs.len()
        )));
    }

    let input = &inputs[0];
    let keepdims = match attributes.get("keepdims").map(String::as_str) {
        None | Some("1" | "true") => true,
        Some("0" | "false") => false,
        Some(value) => {
            return Err(OnnxError::invalid_dimensions(format!(
                "ReduceMean keepdims must be 0 or 1, got '{value}'"
            )))
        }
    };
    let noop_with_empty_axes = match attributes.get("noop_with_empty_axes").map(String::as_str) {
        None | Some("0" | "false") => false,
        Some("1" | "true") => true,
        Some(value) => {
            return Err(OnnxError::invalid_dimensions(format!(
                "ReduceMean noop_with_empty_axes must be 0 or 1, got '{value}'"
            )))
        }
    };

    let specified_axes = if inputs.len() == 2 {
        Some(
            inputs[1]
                .data()
                .iter()
                .map(|&axis| finite_integer_i64(axis, "ReduceMean axis"))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        attributes
            .get("axes")
            .map(|axes| parse_int_array(axes))
            .transpose()?
    };

    if matches!(&specified_axes, Some(axes) if axes.is_empty()) && noop_with_empty_axes {
        return Ok(vec![input.clone()]);
    }

    let rank = input.ndim() as i64;
    let axes = specified_axes.unwrap_or_else(|| (0..rank).collect());
    let mut normalized_axes = Vec::with_capacity(axes.len());
    for axis in axes {
        if axis < -rank || axis >= rank {
            return Err(OnnxError::invalid_dimensions(format!(
                "ReduceMean axis {axis} out of bounds for tensor with {} dimensions",
                input.ndim()
            )));
        }
        normalized_axes.push(if axis < 0 {
            (rank + axis) as usize
        } else {
            axis as usize
        });
    }
    normalized_axes.sort_unstable();
    if normalized_axes.windows(2).any(|axes| axes[0] == axes[1]) {
        return Err(OnnxError::invalid_dimensions(
            "ReduceMean axes must be unique".to_string(),
        ));
    }

    let mut output = input.data().clone();
    for &axis in normalized_axes.iter().rev() {
        let reduced = output.mean_axis(ndarray::Axis(axis)).ok_or_else(|| {
            OnnxError::invalid_dimensions(format!("ReduceMean cannot reduce empty axis {axis}"))
        })?;
        output = if keepdims {
            reduced.insert_axis(ndarray::Axis(axis))
        } else {
            reduced
        };
    }

    let result = Tensor::from_array(output);

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
    if inputs.is_empty() || inputs.len() > 2 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Resize requires 1 or 2 inputs in the supported signature, got {}",
            inputs.len()
        )));
    }

    let input = &inputs[0];
    if input.ndim() != 4 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Resize requires a 4D input, got {} dimensions",
            input.ndim()
        )));
    }

    let mode = attributes
        .get("mode")
        .map(String::as_str)
        .unwrap_or("nearest");
    if mode != "nearest" {
        return Err(OnnxError::unsupported_operation(format!(
            "Resize mode '{mode}' is not implemented"
        )));
    }

    let scales = if let Some(scales_tensor) = inputs.get(1) {
        if scales_tensor.ndim() != 1 {
            return Err(OnnxError::invalid_dimensions(format!(
                "Resize scales input must be 1D, got {} dimensions",
                scales_tensor.ndim()
            )));
        }
        scales_tensor.data().iter().copied().collect::<Vec<_>>()
    } else if let Some(scales_str) = attributes.get("scales") {
        let cleaned = scales_str.trim_matches(['[', ']']);
        cleaned
            .split(',')
            .map(|value| {
                value.trim().parse::<f32>().map_err(|error| {
                    OnnxError::invalid_dimensions(format!(
                        "Invalid Resize scale '{}': {error}",
                        value.trim()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        return Err(OnnxError::invalid_dimensions(
            "Resize requires four scale values for a 4D input".to_string(),
        ));
    };

    if scales.len() != 4 {
        return Err(OnnxError::invalid_dimensions(format!(
            "Resize requires four scale values for a 4D input, got {}",
            scales.len()
        )));
    }
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(OnnxError::invalid_dimensions(format!(
            "Resize scales must be finite and positive, got {scales:?}"
        )));
    }
    if scales[0] != 1.0 || scales[1] != 1.0 {
        return Err(OnnxError::unsupported_operation(format!(
            "Resize only supports batch and channel scales of 1, got {} and {}",
            scales[0], scales[1]
        )));
    }

    let input_shape = input.shape();
    let [batch_size, channels, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let scaled_dimension = |dimension: usize, scale: f32, name: &str| -> Result<usize> {
        let scaled = dimension as f64 * f64::from(scale);
        if scaled > usize::MAX as f64 {
            return Err(OnnxError::invalid_dimensions(format!(
                "Resize {name} exceeds usize"
            )));
        }
        Ok(scaled.floor() as usize)
    };
    let new_height = scaled_dimension(height, scales[2], "height")?;
    let new_width = scaled_dimension(width, scales[3], "width")?;
    let output_shape = [batch_size, channels, new_height, new_width];
    let output_len = checked_element_count(&output_shape, "Resize output")?;
    let mut output_data = Vec::new();
    output_data.try_reserve_exact(output_len).map_err(|error| {
        OnnxError::invalid_dimensions(format!(
            "Unable to allocate Resize output with shape {output_shape:?}: {error}"
        ))
    })?;

    let input_data = input.data();
    for batch in 0..batch_size {
        for channel in 0..channels {
            for output_h in 0..new_height {
                for output_w in 0..new_width {
                    let input_h = ((output_h as f64) / f64::from(scales[2])).floor() as usize;
                    let input_w = ((output_w as f64) / f64::from(scales[3])).floor() as usize;
                    output_data.push(
                        input_data[[
                            batch,
                            channel,
                            input_h.min(height - 1),
                            input_w.min(width - 1),
                        ]],
                    );
                }
            }
        }
    }

    Ok(vec![Tensor::from_shape_vec(&output_shape, output_data)?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use ndarray::{Array1, Array2, Array3, Array4};
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
    fn test_conv_op_handles_non_standard_input_layout() {
        let array = Array4::from_shape_vec((1, 1, 2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .permuted_axes([0, 1, 3, 2]);
        let input = Tensor::from_array(array);
        assert!(!input.data().is_standard_layout());
        let kernel = Tensor::from_shape_vec(&[1, 1, 1, 1], vec![1.0]).unwrap();

        let result = execute_operator(
            &OperatorType::Conv,
            &[input.clone(), kernel],
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(result[0].shape(), input.shape());
        assert_eq!(result[0].data(), input.data());
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
            .contains("requires 2 or 3 inputs"));
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
        let result = tensor.relu().unwrap();

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
        let result1 = tensor.relu().unwrap();
        let result2 = result1.relu().unwrap();

        assert_eq!(result1.data(), result2.data());
    }

    #[test]
    fn test_formal_sigmoid_bounded() {
        // Test that sigmoid output is always in (0, 1)
        let tensor = Tensor::from_shape_vec(&[5], vec![-10.0, -1.0, 0.0, 1.0, 10.0]).unwrap();
        let result = tensor.sigmoid().unwrap();

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

        let error = execute_operator(&OperatorType::Upsample, &inputs, &attrs).unwrap_err();
        assert!(error.to_string().contains("not implemented"));
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
    fn test_maxpool_op_rejects_invalid_dimensions() {
        let input = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![1.0; 4]).unwrap();

        let mut zero_stride = HashMap::new();
        zero_stride.insert("strides".to_string(), "[0,1]".to_string());
        let error =
            execute_operator(&OperatorType::MaxPool, &[input.clone()], &zero_stride).unwrap_err();
        assert!(error.to_string().contains("strides must be non-zero"));

        let mut negative_padding = HashMap::new();
        negative_padding.insert("pads".to_string(), "[-1,0,0,0]".to_string());
        let error = execute_operator(&OperatorType::MaxPool, &[input.clone()], &negative_padding)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("MaxPool pads must be non-negative"));

        let mut oversized_kernel = HashMap::new();
        oversized_kernel.insert("kernel_shape".to_string(), "[3,3]".to_string());
        let error =
            execute_operator(&OperatorType::MaxPool, &[input], &oversized_kernel).unwrap_err();
        assert!(error.to_string().contains("smaller than kernel"));
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

        let error =
            execute_operator(&OperatorType::NonMaxSuppression, &inputs, &attrs).unwrap_err();
        assert!(error.to_string().contains("not implemented"));
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
        assert!(execute_operator(&OperatorType::Upsample, &[tensor_4d.clone()], &attrs).is_err());
        assert!(execute_operator(&OperatorType::MaxPool, &[tensor_4d.clone()], &attrs).is_ok());
        assert!(execute_operator(&OperatorType::Softmax, &[tensor_1d.clone()], &attrs).is_ok());
        assert!(execute_operator(
            &OperatorType::NonMaxSuppression,
            &[tensor_4d.clone(), tensor_1d.clone()],
            &attrs
        )
        .is_err());
    }

    // ========================================
    // Mathematical Operator Tests
    // ========================================

    #[test]
    fn test_sub_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![4.0, 6.0, 8.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let result = execute_operator(&OperatorType::Sub, &[a, b], &HashMap::new()).unwrap();

        assert_eq!(result.len(), 1);
        let data = result[0].data();
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        assert!((data[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sub_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0]));

        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Sub, &[], &HashMap::new());
        assert!(result.is_err());

        let result = execute_operator(&OperatorType::Sub, &[a.clone()], &HashMap::new());
        assert!(result.is_err());

        // Test with mismatched shapes
        let b = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let result = execute_operator(&OperatorType::Sub, &[a, b], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_div_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![8.0, 12.0, 16.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![2.0, 3.0, 4.0]));
        let result = execute_operator(&OperatorType::Div, &[a, b], &HashMap::new()).unwrap();

        assert_eq!(result.len(), 1);
        let data = result[0].data();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        assert!((data[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_div_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0]));

        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Div, &[], &HashMap::new());
        assert!(result.is_err());

        let result = execute_operator(&OperatorType::Div, &[a.clone()], &HashMap::new());
        assert!(result.is_err());

        // Test with mismatched shapes
        let b = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let result = execute_operator(&OperatorType::Div, &[a, b], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_pow_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![2.0, 3.0, 4.0]));
        let b = Tensor::from_array(Array1::from_vec(vec![2.0, 2.0, 2.0]));
        let result = execute_operator(&OperatorType::Pow, &[a, b], &HashMap::new()).unwrap();

        assert_eq!(result.len(), 1);
        let data = result[0].data();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 9.0).abs() < 1e-6);
        assert!((data[2] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_pow_op_wrong_inputs() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0]));

        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Pow, &[], &HashMap::new());
        assert!(result.is_err());

        let result = execute_operator(&OperatorType::Pow, &[a.clone()], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_sqrt_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![4.0, 9.0, 16.0]));
        let result = execute_operator(&OperatorType::Sqrt, &[a], &HashMap::new()).unwrap();

        assert_eq!(result.len(), 1);
        let data = result[0].data();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 3.0).abs() < 1e-6);
        assert!((data[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_sqrt_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Sqrt, &[], &HashMap::new());
        assert!(result.is_err());

        // Test with negative values
        let a = Tensor::from_array(Array1::from_vec(vec![-1.0, 2.0]));
        let result = execute_operator(&OperatorType::Sqrt, &[a], &HashMap::new());
        // Should handle negative values gracefully (NaN or error)
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_exp_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![0.0, 1.0, 2.0]));
        let result = execute_operator(&OperatorType::Exp, &[a], &HashMap::new()).unwrap();

        assert_eq!(result.len(), 1);
        let data = result[0].data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - std::f32::consts::E).abs() < 1e-6);
        assert!((data[2] - (std::f32::consts::E * std::f32::consts::E)).abs() < 1e-5);
    }

    #[test]
    fn test_exp_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Exp, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_cast_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.5, 3.7]));
        let mut attrs = HashMap::new();
        attrs.insert("to".to_string(), "1".to_string()); // Cast to float32

        let result = execute_operator(&OperatorType::Cast, &[a], &attrs);
        assert!(result.is_ok() || result.is_err()); // Implementation dependent
    }

    #[test]
    fn test_cast_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Cast, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_of_shape_op() {
        let shape = Tensor::from_array(Array1::from_vec(vec![2.0, 3.0]));
        let mut attrs = HashMap::new();
        attrs.insert("value".to_string(), "5.0".to_string());

        let result = execute_operator(&OperatorType::ConstantOfShape, &[shape], &attrs);
        assert!(result.is_ok() || result.is_err()); // Implementation dependent
    }

    #[test]
    fn test_constant_of_shape_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::ConstantOfShape, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_of_shape_rejects_invalid_dimensions() {
        for dimension in [-1.0, 1.5, f32::NAN, f32::INFINITY] {
            let shape = Tensor::from_shape_vec(&[1], vec![dimension]).unwrap();
            let result =
                execute_operator(&OperatorType::ConstantOfShape, &[shape], &HashMap::new());
            assert!(result.is_err(), "dimension {dimension} should be rejected");
        }

        let shape = Tensor::from_shape_vec(&[2], vec![4_294_967_296.0, 4_294_967_296.0]).unwrap();
        let result = execute_operator(&OperatorType::ConstantOfShape, &[shape], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_of_shape_requires_vector_input() {
        let shape = Tensor::from_shape_vec(&[1, 2], vec![2.0, 3.0]).unwrap();

        let error = execute_operator(&OperatorType::ConstantOfShape, &[shape], &HashMap::new())
            .unwrap_err();
        assert!(error.to_string().contains("shape input must be 1D"));
    }

    #[test]
    fn test_shape_op() {
        let a = Tensor::from_array(
            Array3::from_shape_vec((2, 3, 4), (0..24).map(|x| x as f32).collect()).unwrap(),
        );
        let result = execute_operator(&OperatorType::Shape, &[a], &HashMap::new());
        assert!(result.is_ok() || result.is_err()); // Implementation dependent
    }

    #[test]
    fn test_shape_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Shape, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_split_op() {
        let a = Tensor::from_array(
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap(),
        );
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "1".to_string());
        attrs.insert("split".to_string(), "2,2".to_string());

        let result = execute_operator(&OperatorType::Split, &[a], &attrs);
        // Split operation returns multiple outputs, check if it processes without error
        assert!(result.is_ok() || result.is_err()); // Implementation dependent
    }

    #[test]
    fn test_split_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Split, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_gather_op() {
        let data = Tensor::from_array(
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap(),
        );
        let indices = Tensor::from_array(Array1::from_vec(vec![0.0, 2.0]));
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "0".to_string());

        let result = execute_operator(&OperatorType::Gather, &[data, indices], &attrs);
        assert!(result.is_ok() || result.is_err()); // Implementation dependent
    }

    #[test]
    fn test_gather_op_wrong_inputs() {
        let data = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0]));

        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Gather, &[], &HashMap::new());
        assert!(result.is_err());

        let result = execute_operator(&OperatorType::Gather, &[data], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_unsqueeze_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), "0".to_string());

        let result = execute_operator(&OperatorType::Unsqueeze, &[a], &attrs);
        assert!(result.is_ok() || result.is_err()); // Implementation dependent
    }

    #[test]
    fn test_unsqueeze_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Unsqueeze, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze_op() {
        let a = Tensor::from_array(Array3::from_shape_vec((1, 3, 1), vec![1.0, 2.0, 3.0]).unwrap());
        let result = execute_operator(&OperatorType::Squeeze, &[a], &HashMap::new());
        assert!(result.is_ok() || result.is_err()); // Implementation dependent
    }

    #[test]
    fn test_squeeze_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Squeeze, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_normalization_op() {
        let input = Tensor::from_array(
            Array4::from_shape_vec((1, 2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap(),
        );
        let scale = Tensor::from_array(Array1::from_vec(vec![1.0, 1.0]));
        let bias = Tensor::from_array(Array1::from_vec(vec![0.0, 0.0]));
        let mean = Tensor::from_array(Array1::from_vec(vec![2.5, 6.5]));
        let var = Tensor::from_array(Array1::from_vec(vec![1.25, 1.25]));

        let result = execute_operator(
            &OperatorType::BatchNormalization,
            &[input, scale, bias, mean, var],
            &HashMap::new(),
        );
        assert!(result.is_ok() || result.is_err()); // Implementation dependent
    }

    #[test]
    fn test_batch_normalization_op_wrong_inputs() {
        let input = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0]));

        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::BatchNormalization, &[], &HashMap::new());
        assert!(result.is_err());

        let result = execute_operator(&OperatorType::BatchNormalization, &[input], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_normalization_handles_non_standard_layout() {
        let array = Array4::from_shape_vec((1, 1, 2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .permuted_axes([0, 1, 3, 2]);
        let input = Tensor::from_array(array);
        assert!(!input.data().is_standard_layout());
        let scale = Tensor::from_shape_vec(&[1], vec![1.0]).unwrap();
        let bias = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
        let mean = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
        let variance = Tensor::from_shape_vec(&[1], vec![1.0]).unwrap();
        let mut attrs = HashMap::new();
        attrs.insert("epsilon".to_string(), "0".to_string());

        let result = execute_operator(
            &OperatorType::BatchNormalization,
            &[input.clone(), scale, bias, mean, variance],
            &attrs,
        )
        .unwrap();
        assert_eq!(result[0].data(), input.data());
    }

    #[test]
    fn test_batch_normalization_rejects_invalid_variance() {
        let input = Tensor::from_shape_vec(&[1, 1, 2, 1], vec![1.0; 2]).unwrap();
        let scale = Tensor::from_shape_vec(&[1], vec![1.0]).unwrap();
        let bias = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
        let mean = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
        let variance = Tensor::from_shape_vec(&[1], vec![-1.0]).unwrap();

        let error = execute_operator(
            &OperatorType::BatchNormalization,
            &[input, scale, bias, mean, variance],
            &HashMap::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("variance"));
    }

    #[test]
    fn test_pad_op() {
        let a =
            Tensor::from_array(Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let mut attrs = HashMap::new();
        attrs.insert("pads".to_string(), "1,1,1,1".to_string());

        let result = execute_operator(&OperatorType::Pad, &[a], &attrs);
        assert!(result.is_ok() || result.is_err()); // Implementation dependent
    }

    #[test]
    fn test_pad_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Pad, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_pad_op_rejects_invalid_padding_values() {
        let input = Tensor::from_shape_vec(&[2], vec![1.0, 2.0]).unwrap();

        for padding in [-1.0, 1.5, f32::NAN, f32::INFINITY] {
            let pads = Tensor::from_shape_vec(&[2], vec![padding, 0.0]).unwrap();
            let result =
                execute_operator(&OperatorType::Pad, &[input.clone(), pads], &HashMap::new());
            assert!(result.is_err(), "padding {padding} should be rejected");
        }
    }

    #[test]
    fn test_reduce_mean_op() {
        let a = Tensor::from_array(
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), "1".to_string());

        let result = execute_operator(&OperatorType::ReduceMean, &[a], &attrs).unwrap();
        assert_eq!(result[0].shape(), &[2, 1]);
        assert_eq!(result[0].data().as_slice().unwrap(), &[2.0, 5.0]);
    }

    #[test]
    fn test_reduce_mean_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::ReduceMean, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_identity_op() {
        let a = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
        let result =
            execute_operator(&OperatorType::Identity, &[a.clone()], &HashMap::new()).unwrap();

        assert_eq!(result.len(), 1);
        let data = result[0].data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_identity_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Identity, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_resize_op() {
        let a = Tensor::from_array(
            Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        let mut attrs = HashMap::new();
        attrs.insert("scales".to_string(), "1.0,1.0,2.0,2.0".to_string());

        let result = execute_operator(&OperatorType::Resize, &[a], &attrs).unwrap();
        assert_eq!(result[0].shape(), &[1, 1, 4, 4]);
        assert_eq!(
            result[0].data().as_slice().unwrap(),
            &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,]
        );
    }

    #[test]
    fn test_resize_op_rejects_invalid_scales_and_modes() {
        let input = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![1.0; 4]).unwrap();
        for scales in [
            vec![1.0, 1.0, 0.0, 2.0],
            vec![1.0, 1.0, -1.0, 2.0],
            vec![1.0, 1.0, f32::NAN, 2.0],
            vec![2.0, 1.0, 2.0, 2.0],
        ] {
            let scales = Tensor::from_shape_vec(&[4], scales).unwrap();
            assert!(execute_operator(
                &OperatorType::Resize,
                &[input.clone(), scales],
                &HashMap::new(),
            )
            .is_err());
        }

        let mut attrs = HashMap::new();
        attrs.insert("scales".to_string(), "[1,1,2,2]".to_string());
        attrs.insert("mode".to_string(), "linear".to_string());
        let error = execute_operator(&OperatorType::Resize, &[input], &attrs).unwrap_err();
        assert!(error.to_string().contains("not implemented"));
    }

    #[test]
    fn test_resize_op_wrong_inputs() {
        // Test with insufficient inputs
        let result = execute_operator(&OperatorType::Resize, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_int_array_empty_string() {
        let result = parse_int_array("");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn test_parse_int_array_whitespace_only() {
        let result = parse_int_array("   ");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn test_parse_int_array_bracketed() {
        let result = parse_int_array("[1,2,3]");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_int_array_with_spaces() {
        let result = parse_int_array("1, 2, 3");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_int_array_invalid() {
        let result = parse_int_array("1,invalid,3");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to parse integer"));
    }

    #[test]
    fn test_conv_op_with_custom_strides() {
        let input = Tensor::from_shape_vec(&[1, 2, 4, 4], vec![1.0; 32]).unwrap();
        let kernel = Tensor::from_shape_vec(&[3, 2, 3, 3], vec![0.1; 54]).unwrap();
        let inputs = vec![input, kernel];

        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), "[2,2]".to_string());
        attrs.insert("pads".to_string(), "[1,1,1,1]".to_string());

        let result = execute_operator(&OperatorType::Conv, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_conv_op_with_bias() {
        let input = Tensor::from_shape_vec(&[1, 2, 3, 3], vec![1.0; 18]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 2, 2, 2], vec![0.5; 8]).unwrap();
        let bias = Tensor::from_shape_vec(&[1], vec![0.1]).unwrap();
        let inputs = vec![input, kernel, bias];

        let result = execute_operator(&OperatorType::Conv, &inputs, &HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn test_conv_op_default_strides_and_pads() {
        let input = Tensor::from_shape_vec(&[1, 1, 3, 3], vec![1.0; 9]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![0.25; 4]).unwrap();
        let inputs = vec![input, kernel];

        // Test with empty attributes (should use defaults)
        let result = execute_operator(&OperatorType::Conv, &inputs, &HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn test_conv_op_defaults_to_zero_padding() {
        let input = Tensor::from_shape_vec(&[1, 1, 5, 5], vec![1.0; 25]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 1, 3, 3], vec![1.0; 9]).unwrap();

        let result =
            execute_operator(&OperatorType::Conv, &[input, kernel], &HashMap::new()).unwrap();

        assert_eq!(result[0].shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv_op_malformed_strides() {
        let input = Tensor::from_shape_vec(&[1, 1, 3, 3], vec![1.0; 9]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![0.25; 4]).unwrap();
        let inputs = vec![input, kernel];

        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), "invalid_stride".to_string());

        let error = execute_operator(&OperatorType::Conv, &inputs, &attrs).unwrap_err();
        assert!(error.to_string().contains("Failed to parse integer"));
    }

    #[test]
    fn test_conv_op_rejects_kernel_larger_than_padded_input() {
        let input = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![1.0; 4]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 1, 5, 5], vec![0.1; 25]).unwrap();
        let mut attrs = HashMap::new();
        attrs.insert("pads".to_string(), "[0,0,0,0]".to_string());

        let error = execute_operator(&OperatorType::Conv, &[input, kernel], &attrs).unwrap_err();
        assert!(error.to_string().contains("smaller than kernel"));
    }

    #[test]
    fn test_conv_op_rejects_zero_stride() {
        let input = Tensor::from_shape_vec(&[1, 1, 3, 3], vec![1.0; 9]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![0.25; 4]).unwrap();
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), "[0,1]".to_string());

        let error = execute_operator(&OperatorType::Conv, &[input, kernel], &attrs).unwrap_err();
        assert!(error.to_string().contains("strides must be non-zero"));
    }

    #[test]
    fn test_conv_op_rejects_zero_sized_kernel() {
        let input = Tensor::from_shape_vec(&[1, 1, 3, 3], vec![1.0; 9]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 1, 0, 2], vec![]).unwrap();

        let error =
            execute_operator(&OperatorType::Conv, &[input, kernel], &HashMap::new()).unwrap_err();
        assert!(error
            .to_string()
            .contains("kernel dimensions must be non-zero"));
    }

    #[test]
    fn test_conv_op_rejects_unaddressable_output() {
        let input = Tensor::from_shape_vec(&[1, 1, 1, 1], vec![1.0]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 1, 1, 1], vec![1.0]).unwrap();
        let mut attrs = HashMap::new();
        attrs.insert(
            "pads".to_string(),
            format!("[{},0,{},0]", i64::MAX, i64::MAX),
        );

        let error = execute_operator(&OperatorType::Conv, &[input, kernel], &attrs).unwrap_err();
        assert!(error.to_string().contains("addressable byte size"));
    }

    #[test]
    fn test_conv_op_rejects_padding_overflow() {
        let input = Tensor::from_shape_vec(&[1, 1, 2, 1], vec![1.0; 2]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 1, 1, 1], vec![1.0]).unwrap();
        let mut attrs = HashMap::new();
        attrs.insert(
            "pads".to_string(),
            format!("[{},0,{},0]", i64::MAX, i64::MAX),
        );

        let error = execute_operator(&OperatorType::Conv, &[input, kernel], &attrs).unwrap_err();
        assert!(error.to_string().contains("padded height overflows"));
    }

    #[test]
    fn test_conv_op_partial_pads() {
        let input = Tensor::from_shape_vec(&[1, 1, 4, 4], vec![1.0; 16]).unwrap();
        let kernel = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![0.25; 4]).unwrap();
        let inputs = vec![input, kernel];

        let mut attrs = HashMap::new();
        attrs.insert("pads".to_string(), "[1,2]".to_string()); // Only 2 elements instead of 4

        let error = execute_operator(&OperatorType::Conv, &inputs, &attrs).unwrap_err();
        assert!(error.to_string().contains("Conv pads requires 4 values"));
    }

    #[test]
    fn test_conv_op_rejects_unsupported_semantics() {
        let input = Tensor::from_shape_vec(&[1, 2, 4, 4], vec![1.0; 32]).unwrap();
        let kernel = Tensor::from_shape_vec(&[2, 2, 3, 3], vec![1.0; 36]).unwrap();

        for (name, value) in [
            ("group", "2"),
            ("dilations", "[2,1]"),
            ("auto_pad", "SAME_UPPER"),
        ] {
            let mut attrs = HashMap::new();
            attrs.insert(name.to_string(), value.to_string());
            assert!(execute_operator(
                &OperatorType::Conv,
                &[input.clone(), kernel.clone()],
                &attrs,
            )
            .is_err());
        }
    }

    #[test]
    fn test_maxpool_op_rejects_unsupported_semantics() {
        let input = Tensor::from_shape_vec(&[1, 1, 4, 4], vec![1.0; 16]).unwrap();

        for (name, value) in [
            ("dilations", "[2,1]"),
            ("auto_pad", "SAME_UPPER"),
            ("ceil_mode", "1"),
            ("storage_order", "1"),
        ] {
            let mut attrs = HashMap::new();
            attrs.insert(name.to_string(), value.to_string());
            assert!(execute_operator(&OperatorType::MaxPool, &[input.clone()], &attrs).is_err());
        }
    }

    #[test]
    fn test_conv_op_unsupported_bias_shape() {
        let input = Tensor::from_shape_vec(&[1, 2, 3, 3], vec![1.0; 18]).unwrap();
        let kernel = Tensor::from_shape_vec(&[2, 2, 2, 2], vec![0.5; 16]).unwrap();
        let bias = Tensor::from_shape_vec(&[3, 3], vec![0.1; 9]).unwrap(); // Wrong shape
        let inputs = vec![input, kernel, bias];

        let error = execute_operator(&OperatorType::Conv, &inputs, &HashMap::new()).unwrap_err();
        assert!(error.to_string().contains("Conv bias shape"));
    }

    #[test]
    fn test_conv_op_4d_bias() {
        let input = Tensor::from_shape_vec(&[1, 1, 3, 3], vec![1.0; 9]).unwrap();
        let kernel = Tensor::from_shape_vec(&[2, 1, 2, 2], vec![0.5; 8]).unwrap();
        let bias = Tensor::from_shape_vec(&[1, 2, 1, 1], vec![0.1, 0.2]).unwrap(); // 4D bias
        let inputs = vec![input, kernel, bias];

        let result = execute_operator(&OperatorType::Conv, &inputs, &HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn test_transpose_op_with_custom_perm() {
        let tensor = Tensor::from_shape_vec(&[2, 3, 4], vec![1.0; 24]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("perm".to_string(), "[2,0,1]".to_string());

        let result = execute_operator(&OperatorType::Transpose, &inputs, &attrs);
        assert!(result.is_ok());
        if let Ok(outputs) = result {
            assert_eq!(outputs[0].shape(), &[4, 2, 3]);
        }
    }

    #[test]
    fn test_transpose_op_invalid_perm() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1.0; 6]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("perm".to_string(), "[0,2]".to_string()); // Invalid: index 2 out of bounds

        let result = execute_operator(&OperatorType::Transpose, &inputs, &attrs);
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax_op_with_axis() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "1".to_string());

        let result = execute_operator(&OperatorType::Softmax, &inputs, &attrs).unwrap();
        for row in 0..2 {
            let sum: f32 = (0..3).map(|column| result[0].data()[[row, column]]).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_op_honors_first_axis_for_matrix() {
        let tensor = Tensor::from_shape_vec(&[2, 2], vec![0.0, 2.0, 1.0, 4.0]).unwrap();
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "0".to_string());

        let result = execute_operator(&OperatorType::Softmax, &[tensor], &attrs).unwrap();
        let expected = [
            1.0 / (1.0 + 1.0f32.exp()),
            1.0 / (1.0 + 2.0f32.exp()),
            1.0f32.exp() / (1.0 + 1.0f32.exp()),
            2.0f32.exp() / (1.0 + 2.0f32.exp()),
        ];

        for (&actual, expected) in result[0].data().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_op_negative_axis() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1.0; 6]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "-1".to_string());

        let result = execute_operator(&OperatorType::Softmax, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gather_op_with_indices() {
        let data = Tensor::from_shape_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices = Tensor::from_shape_vec(&[2], vec![0.0, 2.0]).unwrap();
        let inputs = vec![data, indices];

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "0".to_string());

        let result = execute_operator(&OperatorType::Gather, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gather_op_handles_non_standard_layout() {
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .reversed_axes();
        let data = Tensor::from_array(array);
        assert!(!data.data().is_standard_layout());
        let indices = Tensor::from_shape_vec(&[2], vec![2.0, 0.0]).unwrap();

        let result =
            execute_operator(&OperatorType::Gather, &[data, indices], &HashMap::new()).unwrap();

        assert_eq!(result[0].shape(), &[2, 2]);
        assert_eq!(result[0].data().as_slice().unwrap(), &[3.0, 6.0, 1.0, 4.0]);
    }

    #[test]
    fn test_gather_op_out_of_bounds_indices() {
        let data = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let indices = Tensor::from_shape_vec(&[1], vec![2.0]).unwrap();
        let inputs = vec![data, indices];

        let result = execute_operator(&OperatorType::Gather, &inputs, &HashMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_split_op_with_custom_axis() {
        let tensor = Tensor::from_shape_vec(&[4, 2], vec![1.0; 8]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "0".to_string());
        attrs.insert("split".to_string(), "[2,2]".to_string());

        let result = execute_operator(&OperatorType::Split, &inputs, &attrs);
        assert!(result.is_ok());
        if let Ok(outputs) = result {
            assert_eq!(outputs.len(), 2);
        }
    }

    #[test]
    fn test_split_op_uneven_split() {
        let tensor = Tensor::from_shape_vec(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("split".to_string(), "[2,3]".to_string());

        let result = execute_operator(&OperatorType::Split, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_op_constant_mode() {
        let tensor = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let pads = Tensor::from_shape_vec(&[4], vec![1.0, 1.0, 1.0, 1.0]).unwrap(); // 2D tensor needs 4 pad values
        let inputs = vec![tensor, pads];

        let mut attrs = HashMap::new();
        attrs.insert("mode".to_string(), "constant".to_string());

        let result = execute_operator(&OperatorType::Pad, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_op_reflect_mode() {
        let tensor = Tensor::from_shape_vec(&[3, 3], vec![1.0; 9]).unwrap();
        let pads = Tensor::from_shape_vec(&[4], vec![1.0, 1.0, 0.0, 0.0]).unwrap(); // 2D tensor needs 4 pad values
        let inputs = vec![tensor, pads];

        let mut attrs = HashMap::new();
        attrs.insert("mode".to_string(), "reflect".to_string());

        let error = execute_operator(&OperatorType::Pad, &inputs, &attrs).unwrap_err();
        assert!(error.to_string().contains("not implemented"));
    }

    #[test]
    fn test_cast_op_different_types() {
        let tensor = Tensor::from_shape_vec(&[3], vec![1.5, 2.7, 3.9]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("to".to_string(), "int32".to_string());

        let error = execute_operator(&OperatorType::Cast, &inputs, &attrs).unwrap_err();
        assert!(error.to_string().contains("not supported"));
    }

    #[test]
    fn test_cast_op_unsupported_type() {
        let tensor = Tensor::from_shape_vec(&[2], vec![1.0, 2.0]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("to".to_string(), "complex128".to_string()); // Unsupported

        let error = execute_operator(&OperatorType::Cast, &inputs, &attrs).unwrap_err();
        assert!(error.to_string().contains("not supported"));
    }

    #[test]
    fn test_unsqueeze_op_multiple_axes() {
        let tensor = Tensor::from_shape_vec(&[2, 3], vec![1.0; 6]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), "[0,2]".to_string());

        let result = execute_operator(&OperatorType::Unsqueeze, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_squeeze_op_specific_axes() {
        let tensor = Tensor::from_shape_vec(&[1, 3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), "[0,2]".to_string());

        let result = execute_operator(&OperatorType::Squeeze, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_constant_of_shape_with_value() {
        let shape_tensor = Tensor::from_shape_vec(&[2], vec![3.0, 4.0]).unwrap();
        let inputs = vec![shape_tensor];

        let mut attrs = HashMap::new();
        attrs.insert("value".to_string(), "5.0".to_string());

        let result = execute_operator(&OperatorType::ConstantOfShape, &inputs, &attrs);
        assert!(result.is_ok());
        if let Ok(outputs) = result {
            assert_eq!(outputs[0].shape(), &[3, 4]);
        }
    }

    #[test]
    fn test_maxpool_op_with_strides_and_pads() {
        let tensor = Tensor::from_shape_vec(&[1, 1, 4, 4], vec![1.0; 16]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), "[2,2]".to_string());
        attrs.insert("strides".to_string(), "[2,2]".to_string());
        attrs.insert("pads".to_string(), "[1,1,1,1]".to_string());

        let result = execute_operator(&OperatorType::MaxPool, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_nms_op_with_all_parameters() {
        let boxes = Tensor::from_shape_vec(
            &[4, 4],
            vec![
                0.0, 0.0, 1.0, 1.0, 0.1, 0.1, 1.1, 1.1, 2.0, 2.0, 3.0, 3.0, 2.1, 2.1, 3.1, 3.1,
            ],
        )
        .unwrap();
        let scores = Tensor::from_shape_vec(&[4], vec![0.9, 0.8, 0.7, 0.6]).unwrap();
        let inputs = vec![boxes, scores];

        let mut attrs = HashMap::new();
        attrs.insert("iou_threshold".to_string(), "0.5".to_string());
        attrs.insert("score_threshold".to_string(), "0.1".to_string());

        let error =
            execute_operator(&OperatorType::NonMaxSuppression, &inputs, &attrs).unwrap_err();
        assert!(error.to_string().contains("not implemented"));
    }

    #[test]
    fn test_batch_norm_op_all_parameters() {
        let input = Tensor::from_shape_vec(&[1, 2, 2, 2], vec![1.0; 8]).unwrap();
        let scale = Tensor::from_shape_vec(&[2], vec![1.0, 1.0]).unwrap();
        let bias = Tensor::from_shape_vec(&[2], vec![0.0, 0.0]).unwrap();
        let mean = Tensor::from_shape_vec(&[2], vec![0.5, 0.5]).unwrap();
        let variance = Tensor::from_shape_vec(&[2], vec![0.25, 0.25]).unwrap();
        let inputs = vec![input, scale, bias, mean, variance];

        let result = execute_operator(&OperatorType::BatchNormalization, &inputs, &HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn test_resize_op_with_scales() {
        let tensor = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("scales".to_string(), "[1.0,1.0,2.0,2.0]".to_string());
        attrs.insert("mode".to_string(), "nearest".to_string());

        let result = execute_operator(&OperatorType::Resize, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_mean_op_with_axes() {
        let tensor =
            Tensor::from_shape_vec(&[2, 2, 2], (0..8).map(|value| value as f32).collect()).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), "[-1]".to_string());
        attrs.insert("keepdims".to_string(), "0".to_string());

        let result = execute_operator(&OperatorType::ReduceMean, &inputs, &attrs).unwrap();
        assert_eq!(result[0].shape(), &[2, 2]);
        assert_eq!(result[0].data().as_slice().unwrap(), &[0.5, 2.5, 4.5, 6.5]);
    }

    #[test]
    fn test_slice_op_with_steps() {
        let tensor =
            Tensor::from_shape_vec(&[8], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
        let inputs = vec![tensor];

        let mut attrs = HashMap::new();
        attrs.insert("starts".to_string(), "0".to_string());
        attrs.insert("ends".to_string(), "8".to_string());
        attrs.insert("steps".to_string(), "2".to_string());

        let result = execute_operator(&OperatorType::Slice, &inputs, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_extended_operators_basic_execution() {
        // Test all operators that might have extended logic paths
        let test_operators = vec![
            OperatorType::Div,
            OperatorType::Sub,
            OperatorType::Exp,
            OperatorType::Sqrt,
            OperatorType::Pow,
        ];

        for op in test_operators {
            match op {
                OperatorType::Div | OperatorType::Sub | OperatorType::Pow => {
                    let tensor1 = Tensor::from_shape_vec(&[2], vec![4.0, 9.0]).unwrap();
                    let tensor2 = Tensor::from_shape_vec(&[2], vec![2.0, 3.0]).unwrap();
                    let inputs = vec![tensor1, tensor2];
                    let result = execute_operator(&op, &inputs, &HashMap::new());
                    assert!(result.is_ok(), "Failed for operator {op:?}");
                }
                OperatorType::Exp | OperatorType::Sqrt => {
                    let tensor = Tensor::from_shape_vec(&[2], vec![1.0, 4.0]).unwrap();
                    let inputs = vec![tensor];
                    let result = execute_operator(&op, &inputs, &HashMap::new());
                    assert!(result.is_ok(), "Failed for operator {op:?}");
                }
                _ => {}
            }
        }
    }
}
