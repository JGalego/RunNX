//! Targeted tests to improve coverage of operators.rs (~78% → 90%+)
//!
//! Each test section targets specific uncovered branches identified in the
//! coverage analysis.

use runnx::operators::{execute_operator, OperatorType};
use runnx::Tensor;
use std::collections::HashMap;

// ============================================================
// reshape_op — uncovered error branches
// ============================================================

#[test]
fn test_reshape_invalid_negative_dim() {
    // dim < -1 is invalid (only -1 is allowed to mean "infer")
    let data = Tensor::from_shape_vec(&[6], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    // Shape tensor with -2 (invalid)
    let shape = Tensor::from_shape_vec(&[2], vec![-2.0, 3.0]).unwrap();
    let result = execute_operator(&OperatorType::Reshape, &[data, shape], &HashMap::new());
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Invalid dimension"));
}

#[test]
fn test_reshape_dim_zero_out_of_bounds() {
    // dim == 0 means "copy from input", but index exceeds input ndim
    let data = Tensor::from_shape_vec(&[6], vec![1., 2., 3., 4., 5., 6.]).unwrap(); // 1D
                                                                                    // Target shape [0, 0, 6] — indices 0 and 1 are both out of range for 1D input
    let shape = Tensor::from_shape_vec(&[3], vec![0.0, 0.0, 6.0]).unwrap();
    let result = execute_operator(&OperatorType::Reshape, &[data, shape], &HashMap::new());
    // Index 1 is out of bounds for 1D input → error
    assert!(result.is_err());
}

#[test]
fn test_reshape_two_infer_dims() {
    // Two -1 dimensions → only one is allowed
    let data = Tensor::from_shape_vec(&[6], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let shape = Tensor::from_shape_vec(&[2], vec![-1.0, -1.0]).unwrap();
    let result = execute_operator(&OperatorType::Reshape, &[data, shape], &HashMap::new());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("inferred"));
}

#[test]
fn test_reshape_inferred_dim_not_divisible() {
    // total_elements=6, known dims product=4 → 6 % 4 != 0
    let data = Tensor::from_shape_vec(&[6], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let shape = Tensor::from_shape_vec(&[2], vec![-1.0, 4.0]).unwrap();
    let result = execute_operator(&OperatorType::Reshape, &[data, shape], &HashMap::new());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("divisible"));
}

// ============================================================
// concat_op — negative axis
// ============================================================

#[test]
fn test_concat_op_negative_axis() {
    let a = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let b = Tensor::from_shape_vec(&[2, 3], vec![7., 8., 9., 10., 11., 12.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "-1".to_string()); // last axis = 1 for 2D
    let result = execute_operator(&OperatorType::Concat, &[a, b], &attrs).unwrap();
    assert_eq!(result[0].shape(), &[2, 6]);
}

// ============================================================
// slice_op — missing attributes (fallback path)
// ============================================================

#[test]
fn test_slice_op_missing_attributes() {
    // Only 1 input and no starts/ends attrs → returns input unchanged
    let tensor = Tensor::from_shape_vec(&[4], vec![1., 2., 3., 4.]).unwrap();
    let result =
        execute_operator(&OperatorType::Slice, &[tensor.clone()], &HashMap::new()).unwrap();
    assert_eq!(result[0].shape(), tensor.shape());
}

#[test]
fn test_slice_op_with_axes_input() {
    // 4 inputs: data, starts, ends, axes
    let data = Tensor::from_shape_vec(&[3, 4], (0..12).map(|x| x as f32).collect()).unwrap();
    let starts = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
    let ends = Tensor::from_shape_vec(&[1], vec![2.0]).unwrap();
    let axes = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
    let result = execute_operator(
        &OperatorType::Slice,
        &[data, starts, ends, axes],
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[2, 4]);
}

#[test]
fn test_slice_op_with_steps_input() {
    // 5 inputs: data, starts, ends, axes, steps
    let data = Tensor::from_shape_vec(&[6], vec![0., 1., 2., 3., 4., 5.]).unwrap();
    let starts = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
    let ends = Tensor::from_shape_vec(&[1], vec![6.0]).unwrap();
    let axes = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
    let steps = Tensor::from_shape_vec(&[1], vec![2.0]).unwrap();
    let result = execute_operator(
        &OperatorType::Slice,
        &[data, starts, ends, axes, steps],
        &HashMap::new(),
    )
    .unwrap();
    // step=2 → every other element → 3 elements
    assert_eq!(result[0].shape(), &[3]);
}

// ============================================================
// split_op — additional branches
// ============================================================

#[test]
fn test_split_op_negative_axis() {
    let tensor = Tensor::from_shape_vec(&[4, 2], vec![1.0; 8]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "-1".to_string()); // last axis = 1
    attrs.insert("split".to_string(), "1,1".to_string());
    let result = execute_operator(&OperatorType::Split, &[tensor], &attrs).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].shape(), &[4, 1]);
}

#[test]
fn test_split_op_out_of_bounds_axis() {
    let tensor = Tensor::from_shape_vec(&[4, 2], vec![1.0; 8]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "5".to_string()); // Way out of range
    let result = execute_operator(&OperatorType::Split, &[tensor], &attrs);
    assert!(result.is_err());
}

#[test]
fn test_split_op_sizes_from_input_tensor() {
    // Split sizes provided as 2nd input tensor
    let tensor = Tensor::from_shape_vec(&[6], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let split_sizes = Tensor::from_shape_vec(&[3], vec![2.0, 2.0, 2.0]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());
    let result = execute_operator(&OperatorType::Split, &[tensor, split_sizes], &attrs).unwrap();
    assert_eq!(result.len(), 3);
    for chunk in &result {
        assert_eq!(chunk.shape(), &[2]);
    }
}

#[test]
fn test_split_op_default_equal_split() {
    // No split attr → equal split into 2 parts
    let tensor = Tensor::from_shape_vec(&[4], vec![1., 2., 3., 4.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());
    let result = execute_operator(&OperatorType::Split, &[tensor], &attrs).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].shape(), &[2]);
    assert_eq!(result[1].shape(), &[2]);
}

// ============================================================
// gather_op — negative axis, out-of-bounds axis, negative indices
// ============================================================

#[test]
fn test_gather_op_negative_axis() {
    let data = Tensor::from_shape_vec(&[3, 4], (0..12).map(|x| x as f32).collect()).unwrap();
    let indices = Tensor::from_shape_vec(&[2], vec![0.0, 1.0]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "-1".to_string()); // last axis = 1
    let result = execute_operator(&OperatorType::Gather, &[data, indices], &attrs).unwrap();
    assert_eq!(result[0].shape(), &[3, 2]);
}

#[test]
fn test_gather_op_out_of_bounds_axis() {
    let data = Tensor::from_shape_vec(&[3, 4], (0..12).map(|x| x as f32).collect()).unwrap();
    let indices = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "5".to_string());
    let result = execute_operator(&OperatorType::Gather, &[data, indices], &attrs);
    assert!(result.is_err());
}

#[test]
fn test_gather_op_negative_indices() {
    // Negative index -1 should map to last element
    let data = Tensor::from_shape_vec(&[4], vec![10., 20., 30., 40.]).unwrap();
    let indices = Tensor::from_shape_vec(&[1], vec![-1.0]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());
    let result = execute_operator(&OperatorType::Gather, &[data, indices], &attrs).unwrap();
    // -1 → index 3 → value 40
    assert!((result[0].data()[0] - 40.0).abs() < 1e-6);
}

// ============================================================
// unsqueeze_op — more error paths
// ============================================================

#[test]
fn test_unsqueeze_op_axis_out_of_bounds() {
    let tensor = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), "5".to_string()); // Out of range for 1D→2D output
    let result = execute_operator(&OperatorType::Unsqueeze, &[tensor], &attrs);
    assert!(result.is_err());
}

#[test]
fn test_unsqueeze_op_duplicate_axes() {
    let tensor = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), "0,0".to_string()); // Duplicate
    let result = execute_operator(&OperatorType::Unsqueeze, &[tensor], &attrs);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unique"));
}

#[test]
fn test_unsqueeze_op_empty_axes_attr() {
    // Empty axes string: split(',') on "" gives [""] → parse as i64 with unwrap_or(0) → [0]
    // so this unsqueezes at axis 0: [3] → [1, 3]
    let tensor = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), "".to_string());
    let result = execute_operator(&OperatorType::Unsqueeze, &[tensor], &attrs).unwrap();
    assert_eq!(result[0].shape(), &[1, 3]);
}

#[test]
fn test_unsqueeze_op_axes_from_input_tensor() {
    // Axes provided as 2nd input tensor (newer ONNX)
    let tensor = Tensor::from_shape_vec(&[2, 3], vec![1.; 6]).unwrap();
    let axes_tensor = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
    let result = execute_operator(
        &OperatorType::Unsqueeze,
        &[tensor, axes_tensor],
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[1, 2, 3]);
}

#[test]
fn test_unsqueeze_op_no_axes_specified() {
    // Neither input tensor nor attribute → error
    let tensor = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).unwrap();
    let result = execute_operator(&OperatorType::Unsqueeze, &[tensor], &HashMap::new());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("axes"));
}

// ============================================================
// squeeze_op — error paths and edge cases
// ============================================================

#[test]
fn test_squeeze_op_out_of_bounds_axis() {
    let tensor = Tensor::from_shape_vec(&[1, 3], vec![1., 2., 3.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), "5".to_string());
    let result = execute_operator(&OperatorType::Squeeze, &[tensor], &attrs);
    assert!(result.is_err());
}

#[test]
fn test_squeeze_op_non_unit_dimension() {
    // Trying to squeeze a dim that is not size 1 → error
    let tensor = Tensor::from_shape_vec(&[2, 3], vec![1.; 6]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), "0".to_string()); // dim 0 has size 2, not 1
    let result = execute_operator(&OperatorType::Squeeze, &[tensor], &attrs);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("squeeze"));
}

#[test]
fn test_squeeze_op_to_scalar() {
    // A [1,1,1] tensor squeezed to a 0D scalar
    let tensor = Tensor::from_shape_vec(&[1, 1, 1], vec![42.0]).unwrap();
    let result = execute_operator(&OperatorType::Squeeze, &[tensor], &HashMap::new()).unwrap();
    // Should produce a scalar — shape is [] and it has exactly 1 element
    assert_eq!(result[0].shape(), &[] as &[usize]);
    let val = *result[0].data().iter().next().unwrap();
    assert!((val - 42.0).abs() < 1e-6);
}

#[test]
fn test_squeeze_op_axes_from_input_tensor() {
    // Axes provided as 2nd input tensor
    let tensor = Tensor::from_shape_vec(&[1, 3, 1], vec![1., 2., 3.]).unwrap();
    let axes_tensor = Tensor::from_shape_vec(&[2], vec![0.0, 2.0]).unwrap();
    let result = execute_operator(
        &OperatorType::Squeeze,
        &[tensor, axes_tensor],
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[3]);
}

// ============================================================
// pad_op — constant value from 3rd input, non-constant mode
// ============================================================

#[test]
fn test_pad_op_constant_value_from_input() {
    let tensor = Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap();
    let pads = Tensor::from_shape_vec(&[4], vec![1., 0., 1., 0.]).unwrap();
    let const_val = Tensor::from_shape_vec(&[1], vec![9.0]).unwrap(); // Fill with 9.0
    let mut attrs = HashMap::new();
    attrs.insert("mode".to_string(), "constant".to_string());
    let result = execute_operator(&OperatorType::Pad, &[tensor, pads, const_val], &attrs).unwrap();
    // Output shape: [2+1+1, 2+0+0] = [4, 2]
    assert_eq!(result[0].shape(), &[4, 2]);
    // Padding values should be 9.0
    let data = result[0].data().as_slice().unwrap();
    assert!((data[0] - 9.0).abs() < 1e-6); // top row is padding
}

#[test]
fn test_pad_op_non_constant_mode_returns_input() {
    // Reflect mode → simplified to returning input unchanged
    let tensor = Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap();
    let pads = Tensor::from_shape_vec(&[4], vec![1., 1., 1., 1.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("mode".to_string(), "reflect".to_string());
    let result = execute_operator(&OperatorType::Pad, &[tensor.clone(), pads], &attrs).unwrap();
    assert_eq!(result[0].shape(), tensor.shape());
}

// ============================================================
// batch_norm_op — non-4D input and mismatched param shapes
// ============================================================

#[test]
fn test_batch_norm_op_non_4d_input() {
    let input = Tensor::from_shape_vec(&[3, 4], vec![1.0; 12]).unwrap(); // 2D, not 4D
    let scale = Tensor::from_shape_vec(&[4], vec![1.0; 4]).unwrap();
    let bias = Tensor::from_shape_vec(&[4], vec![0.0; 4]).unwrap();
    let mean = Tensor::from_shape_vec(&[4], vec![0.0; 4]).unwrap();
    let variance = Tensor::from_shape_vec(&[4], vec![1.0; 4]).unwrap();
    let result = execute_operator(
        &OperatorType::BatchNormalization,
        &[input, scale, bias, mean, variance],
        &HashMap::new(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("4D"));
}

#[test]
fn test_batch_norm_op_mismatched_param_shapes() {
    let input = Tensor::from_shape_vec(&[1, 3, 2, 2], vec![1.0; 12]).unwrap();
    let scale = Tensor::from_shape_vec(&[2], vec![1.0, 1.0]).unwrap(); // Should be [3]
    let bias = Tensor::from_shape_vec(&[3], vec![0.0; 3]).unwrap();
    let mean = Tensor::from_shape_vec(&[3], vec![0.0; 3]).unwrap();
    let variance = Tensor::from_shape_vec(&[3], vec![1.0; 3]).unwrap();
    let result = execute_operator(
        &OperatorType::BatchNormalization,
        &[input, scale, bias, mean, variance],
        &HashMap::new(),
    );
    assert!(result.is_err());
}

#[test]
fn test_batch_norm_op_custom_epsilon() {
    // input=1, mean=1, var=0.1, epsilon=1e-3 → (1-1)/sqrt(0.1+1e-3)*2 + 0.5 = 0.5
    let input = Tensor::from_shape_vec(&[1, 2, 2, 2], vec![1.0; 8]).unwrap();
    let scale = Tensor::from_shape_vec(&[2], vec![2.0, 2.0]).unwrap();
    let bias = Tensor::from_shape_vec(&[2], vec![0.5, 0.5]).unwrap();
    let mean = Tensor::from_shape_vec(&[2], vec![1.0, 1.0]).unwrap();
    let variance = Tensor::from_shape_vec(&[2], vec![0.1, 0.1]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("epsilon".to_string(), "1e-3".to_string());
    let result = execute_operator(
        &OperatorType::BatchNormalization,
        &[input, scale, bias, mean, variance],
        &attrs,
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[1, 2, 2, 2]);
    // normalized = (1.0 - 1.0) / sqrt(0.1 + 0.001) = 0 → output = 2.0 * 0 + 0.5 = 0.5
    assert!(result[0].data().iter().all(|&v| (v - 0.5).abs() < 1e-5));
}

// ============================================================
// softmax_op — axis out-of-bounds, multi-dim complex path
// ============================================================

#[test]
fn test_softmax_op_axis_out_of_bounds() {
    let tensor = Tensor::from_shape_vec(&[2, 3], vec![1.0; 6]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "5".to_string()); // Way out of range
    let result = execute_operator(&OperatorType::Softmax, &[tensor], &attrs);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of bounds"));
}

#[test]
fn test_softmax_op_multi_dim_non_last_axis() {
    // 3D tensor with axis=0 → triggers the complex multi-dimensional path
    let tensor = Tensor::from_shape_vec(&[2, 3, 4], (0..24).map(|x| x as f32).collect()).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());
    let result = execute_operator(&OperatorType::Softmax, &[tensor], &attrs).unwrap();
    assert_eq!(result[0].shape(), &[2, 3, 4]);
    // Each "slice" along axis 0 should sum to 1
    let data = result[0].data().as_slice().unwrap();
    // Check a few positions: elements at indices (0,0,0) and (1,0,0) should sum to 1
    let sum_at_00 = data[0] + data[12]; // [0,0,0] + [1,0,0]
    assert!((sum_at_00 - 1.0).abs() < 1e-5, "sum was {sum_at_00}");
}

// ============================================================
// resize_op — scales from 2nd input tensor
// ============================================================

#[test]
fn test_resize_op_scales_from_input_tensor() {
    // Pass scales as a 2nd input tensor (common ONNX pattern)
    let input = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap();
    let scales = Tensor::from_shape_vec(&[4], vec![1.0, 1.0, 2.0, 2.0]).unwrap();
    let result =
        execute_operator(&OperatorType::Resize, &[input, scales], &HashMap::new()).unwrap();
    // 2x2 scaled by 2 → 4x4
    assert_eq!(result[0].shape(), &[1, 1, 4, 4]);
}

// ============================================================
// conv_op — mismatched channels
// ============================================================

#[test]
fn test_conv_op_mismatched_channels() {
    // Input has 2 channels, kernel expects 3 input channels
    let input = Tensor::from_shape_vec(&[1, 2, 4, 4], vec![1.0; 32]).unwrap();
    let kernel = Tensor::from_shape_vec(&[1, 3, 2, 2], vec![0.1; 12]).unwrap(); // C_in=3 ≠ 2
    let result = execute_operator(&OperatorType::Conv, &[input, kernel], &HashMap::new());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("channels"));
}

// ============================================================
// maxpool_op — edge case: all padding (max_val stays NEG_INFINITY → 0)
// ============================================================

#[test]
fn test_maxpool_op_with_large_padding() {
    // Large padding means some windows may have no valid input pixels
    let tensor = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("kernel_shape".to_string(), "[2,2]".to_string());
    attrs.insert("strides".to_string(), "[1,1]".to_string());
    attrs.insert("pads".to_string(), "[1,1,1,1]".to_string());
    let result = execute_operator(&OperatorType::MaxPool, &[tensor], &attrs).unwrap();
    // With padding [1,1,1,1] and kernel [2,2], output shape is (2+1+1-2)/1+1 = 3
    assert_eq!(result[0].shape(), &[1, 1, 3, 3]);
}

// ============================================================
// Additional operator type coverage
// ============================================================

#[test]
fn test_operator_type_from_str_all_new_ops() {
    // Ensure all newer operators parse correctly
    let new_ops = vec![
        ("BatchNormalization", OperatorType::BatchNormalization),
        ("Split", OperatorType::Split),
        ("Gather", OperatorType::Gather),
        ("ConstantOfShape", OperatorType::ConstantOfShape),
        ("Cast", OperatorType::Cast),
        ("Shape", OperatorType::Shape),
        ("Unsqueeze", OperatorType::Unsqueeze),
        ("Squeeze", OperatorType::Squeeze),
        ("Pad", OperatorType::Pad),
        ("Div", OperatorType::Div),
        ("Sub", OperatorType::Sub),
        ("Exp", OperatorType::Exp),
        ("Sqrt", OperatorType::Sqrt),
        ("Pow", OperatorType::Pow),
        ("ReduceMean", OperatorType::ReduceMean),
        ("Identity", OperatorType::Identity),
        ("Resize", OperatorType::Resize),
    ];
    for (s, expected) in new_ops {
        let parsed = s.parse::<OperatorType>().unwrap();
        assert_eq!(parsed, expected, "Failed to parse '{s}'");
    }
}

#[test]
fn test_constant_of_shape_with_ones_value() {
    let shape_tensor = Tensor::from_shape_vec(&[2], vec![2.0, 3.0]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("value".to_string(), "1.0".to_string());
    let result = execute_operator(&OperatorType::ConstantOfShape, &[shape_tensor], &attrs).unwrap();
    assert_eq!(result[0].shape(), &[2, 3]);
    assert!(result[0].data().iter().all(|&x| (x - 1.0).abs() < 1e-6));
}

#[test]
fn test_shape_op_returns_correct_dims() {
    let tensor = Tensor::from_shape_vec(&[2, 3, 4], vec![0.0; 24]).unwrap();
    let result = execute_operator(&OperatorType::Shape, &[tensor], &HashMap::new()).unwrap();
    let dims: Vec<f32> = result[0].data().iter().copied().collect();
    assert_eq!(dims, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_reduce_mean_op_returns_mean() {
    let tensor = Tensor::from_shape_vec(&[4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = execute_operator(&OperatorType::ReduceMean, &[tensor], &HashMap::new()).unwrap();
    assert!((result[0].data()[0] - 2.5).abs() < 1e-6);
}

#[test]
fn test_upsample_op_empty_inputs_error() {
    let result = execute_operator(&OperatorType::Upsample, &[], &HashMap::new());
    assert!(result.is_err());
}

#[test]
fn test_resize_op_fallback_no_scales() {
    // Single input, no scales → fallback returns input unchanged
    let input = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap();
    let result =
        execute_operator(&OperatorType::Resize, &[input.clone()], &HashMap::new()).unwrap();
    assert_eq!(result[0].shape(), input.shape());
}

#[test]
fn test_maxpool_op_wrong_input_count() {
    let result = execute_operator(&OperatorType::MaxPool, &[], &HashMap::new());
    assert!(result.is_err());
}

// ============================================================
// transpose_op — perm string parse error (falls back to default)
// ============================================================

#[test]
fn test_transpose_op_unparseable_perm() {
    // "abc,def" can't parse as usize → warning logged, uses default reverse transpose
    let tensor = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("perm".to_string(), "abc,def".to_string());
    let result = execute_operator(&OperatorType::Transpose, &[tensor], &attrs).unwrap();
    // Default transpose reverses axes: [2,3] → [3,2]
    assert_eq!(result[0].shape(), &[3, 2]);
}

// ============================================================
// pad_op — wrong pads length error
// ============================================================

#[test]
fn test_pad_op_wrong_pads_length() {
    let tensor = Tensor::from_shape_vec(&[2, 3], vec![1.; 6]).unwrap(); // 2D → need 4 pads
    let pads = Tensor::from_shape_vec(&[3], vec![1., 0., 1.]).unwrap(); // only 3, should be 4
    let result = execute_operator(&OperatorType::Pad, &[tensor, pads], &HashMap::new());
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("2 * input dimensions"));
}

// ============================================================
// split_op — sizes don't sum to axis size, zero-sized chunk
// ============================================================

#[test]
fn test_split_op_sizes_mismatch() {
    let tensor = Tensor::from_shape_vec(&[6], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());
    attrs.insert("split".to_string(), "2,2,1".to_string()); // sum=5 ≠ 6
    let result = execute_operator(&OperatorType::Split, &[tensor], &attrs);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("must equal axis size"));
}

#[test]
fn test_split_op_zero_sized_chunk() {
    // A zero-sized split in the middle is silently skipped
    let tensor = Tensor::from_shape_vec(&[5], vec![1., 2., 3., 4., 5.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());
    attrs.insert("split".to_string(), "2,0,3".to_string());
    let result = execute_operator(&OperatorType::Split, &[tensor], &attrs).unwrap();
    // Zero-sized middle split is skipped → 2 outputs
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].shape(), &[2]);
    assert_eq!(result[1].shape(), &[3]);
}

// ============================================================
// gather_op — out-of-bounds positive index clamping
// ============================================================

#[test]
fn test_gather_op_index_clamped_to_last() {
    let data = Tensor::from_shape_vec(&[4], vec![10., 20., 30., 40.]).unwrap();
    let indices = Tensor::from_shape_vec(&[2], vec![2.0, 5.0]).unwrap(); // 5 >= size(4)
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());
    let result = execute_operator(&OperatorType::Gather, &[data, indices], &attrs).unwrap();
    assert!((result[0].data()[0] - 30.0).abs() < 1e-6); // index 2 → 30
    assert!((result[0].data()[1] - 40.0).abs() < 1e-6); // index 5 clamped to 3 → 40
}

// ============================================================
// resize_op — 2nd input tensor with insufficient scales or non-4D input
// ============================================================

#[test]
fn test_resize_op_two_inputs_insufficient_scales() {
    // scales tensor has < 4 elements → falls through to attribute/fallback path
    let input = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap();
    let scales = Tensor::from_shape_vec(&[2], vec![2.0, 2.0]).unwrap();
    let result = execute_operator(
        &OperatorType::Resize,
        &[input.clone(), scales],
        &HashMap::new(),
    )
    .unwrap();
    // Falls through to fallback; returns input unchanged
    assert_eq!(result[0].shape(), input.shape());
}

#[test]
fn test_resize_op_two_inputs_non_4d() {
    // input is 2D, not 4D → condition fails, falls through to fallback
    let input = Tensor::from_shape_vec(&[2, 3], vec![1.; 6]).unwrap();
    let scales = Tensor::from_shape_vec(&[4], vec![1.0, 1.0, 2.0, 2.0]).unwrap();
    let result = execute_operator(
        &OperatorType::Resize,
        &[input.clone(), scales],
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(result[0].shape(), input.shape());
}

// ============================================================
// batch_norm_op — multi-batch (batch_size > 1)
// ============================================================

#[test]
fn test_batch_norm_op_multi_batch() {
    let input = Tensor::from_shape_vec(&[3, 2, 2, 2], vec![1.0; 24]).unwrap(); // batch=3
    let scale = Tensor::from_shape_vec(&[2], vec![2.0, 2.0]).unwrap();
    let bias = Tensor::from_shape_vec(&[2], vec![0.5, 0.5]).unwrap();
    let mean = Tensor::from_shape_vec(&[2], vec![0.0, 0.0]).unwrap();
    let variance = Tensor::from_shape_vec(&[2], vec![1.0, 1.0]).unwrap();
    let result = execute_operator(
        &OperatorType::BatchNormalization,
        &[input, scale, bias, mean, variance],
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[3, 2, 2, 2]);
    // BN: (1 - 0) / sqrt(1 + 1e-5) * 2 + 0.5 ≈ 2.5
    let val = result[0].data().iter().next().unwrap();
    assert!((val - 2.5).abs() < 1e-3);
}

// ============================================================
// slice_op — special float handling for -1.0 end value
// ============================================================

#[test]
fn test_slice_op_negative_one_end_value() {
    // -1.0 in ends triggers the special `x < -0.5 && x > -1.5` branch → treated as -1
    let data = Tensor::from_shape_vec(&[4], vec![10., 20., 30., 40.]).unwrap();
    let starts = Tensor::from_shape_vec(&[1], vec![0.0]).unwrap();
    let ends = Tensor::from_shape_vec(&[1], vec![-1.0]).unwrap();
    let result =
        execute_operator(&OperatorType::Slice, &[data, starts, ends], &HashMap::new()).unwrap();
    // -1 means "up to but not including last" → first 3 elements
    assert_eq!(result[0].shape(), &[3]);
}

// ============================================================
// concat_op — single input early return
// ============================================================

#[test]
fn test_concat_op_single_input_returns_clone() {
    let a = Tensor::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());
    let result = execute_operator(&OperatorType::Concat, &[a.clone()], &attrs).unwrap();
    assert_eq!(result[0].shape(), a.shape());
    assert_eq!(
        result[0].data().as_slice().unwrap(),
        a.data().as_slice().unwrap()
    );
}
