//! Integration tests for the ONNX runtime
//!
//! These tests verify the complete functionality of the ONNX runtime
//! by testing end-to-end workflows.

use runnx::{Model, Tensor, Graph};
use std::collections::HashMap;
use ndarray::{Array1, Array, Array2};
use tempfile::NamedTempFile;

#[test]
fn test_complete_workflow() {
    // Create a model
    let model = Model::create_simple_linear();
    
    // Prepare input
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), Tensor::from_array(Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap().into_dyn()));
    
    // Run inference
    let outputs = model.run(&inputs).unwrap();
    
    // Verify outputs
    assert!(outputs.contains_key("output"));
    let output = outputs.get("output").unwrap();
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn test_model_serialization_roundtrip() {
    let original_model = Model::create_simple_linear();
    
    // Create temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path();
    
    // Save model
    original_model.to_file(file_path).unwrap();
    
    // Load model
    let loaded_model = Model::from_file(file_path).unwrap();
    
    // Compare models
    assert_eq!(original_model.name(), loaded_model.name());
    assert_eq!(original_model.input_names(), loaded_model.input_names());
    assert_eq!(original_model.output_names(), loaded_model.output_names());
    
    // Run inference on both and compare results
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), Tensor::from_array(Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap().into_dyn()));
    
    let original_outputs = original_model.run(&inputs).unwrap();
    let loaded_outputs = loaded_model.run(&inputs).unwrap();
    
    assert_eq!(original_outputs.len(), loaded_outputs.len());
    for (name, original_tensor) in &original_outputs {
        let loaded_tensor = loaded_outputs.get(name).unwrap();
        assert_eq!(*original_tensor, *loaded_tensor);
    }
}

#[test]
fn test_large_tensor_operations() {
    // Test with larger tensors to ensure scalability
    let size = 100;
    let a = Tensor::ones(&[size, size]);
    let b = Tensor::ones(&[size, size]);
    
    // Test addition
    let sum = a.add(&b).unwrap();
    assert_eq!(sum.shape(), &[size, size]);
    assert!(sum.data().iter().all(|&x| (x - 2.0).abs() < 1e-6));
    
    // Test matrix multiplication
    let product = a.matmul(&b).unwrap();
    assert_eq!(product.shape(), &[size, size]);
    assert!(product.data().iter().all(|&x| (x - (size as f32)).abs() < 1e-6));
}

#[test]
fn test_error_handling() {
    let model = Model::create_simple_linear();
    
    // Test with wrong input shape
    let mut wrong_inputs = HashMap::new();
    wrong_inputs.insert("input".to_string(), Tensor::ones(&[2, 2])); // Wrong shape
    
    let result = model.run(&wrong_inputs);
    assert!(result.is_err());
    
    // Test with missing input
    let empty_inputs = HashMap::new();
    let result = model.run(&empty_inputs);
    assert!(result.is_err());
}

#[test]
fn test_graph_validation() {
    let mut graph = Graph::new("test_graph".to_string());
    
    // Add an invalid node (references non-existent input)
    let invalid_node = onnx_rs_min::graph::Node::new(
        "invalid".to_string(),
        "Relu".to_string(),
        vec!["nonexistent_input".to_string()],
        vec!["output".to_string()],
    );
    graph.add_node(invalid_node);
    
    // Validation should fail
    assert!(graph.validate().is_err());
}

#[test]
fn test_activation_functions_edge_cases() {
    use ndarray::Array1;
    
    // Test ReLU with extreme values
    let extreme_values = Tensor::from_array(Array1::from_vec(vec![
        f32::NEG_INFINITY, -1000.0, -1.0, 0.0, 1.0, 1000.0, f32::INFINITY
    ]));
    let relu_result = extreme_values.relu();
    let relu_data = relu_result.data().as_slice().unwrap();
    
    assert_eq!(relu_data[0], 0.0); // -infinity -> 0
    assert_eq!(relu_data[1], 0.0); // -1000 -> 0
    assert_eq!(relu_data[2], 0.0); // -1 -> 0
    assert_eq!(relu_data[3], 0.0); // 0 -> 0
    assert_eq!(relu_data[4], 1.0); // 1 -> 1
    assert_eq!(relu_data[5], 1000.0); // 1000 -> 1000
    assert_eq!(relu_data[6], f32::INFINITY); // infinity -> infinity
    
    // Test Sigmoid with extreme values
    let sigmoid_result = extreme_values.sigmoid();
    let sigmoid_data = sigmoid_result.data().as_slice().unwrap();
    
    assert_eq!(sigmoid_data[0], 0.0); // sigmoid(-infinity) = 0
    assert!(sigmoid_data[1] < 1e-6); // sigmoid(-1000) ≈ 0
    assert!(sigmoid_data[2] < 0.5); // sigmoid(-1) < 0.5
    assert!((sigmoid_data[3] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
    assert!(sigmoid_data[4] > 0.5); // sigmoid(1) > 0.5
    assert!(sigmoid_data[5] > 1.0 - 1e-6); // sigmoid(1000) ≈ 1
    assert_eq!(sigmoid_data[6], 1.0); // sigmoid(infinity) = 1
}

#[test]
#[cfg(feature = "async")]
async fn test_async_inference() {
    let model = Model::create_simple_linear();
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0])));
    
    let outputs = model.run_async(&inputs).await.unwrap();
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_tensor_memory_layout() {
    // Test that tensors maintain correct memory layout
    let data = vec![1., 2., 3., 4., 5., 6.];
    let tensor = Tensor::from_shape_vec(&[2, 3], data.clone()).unwrap();
    
    let tensor_data = tensor.data().as_slice().unwrap();
    assert_eq!(tensor_data, data.as_slice());
    
    // Test reshape preserves data order
    let reshaped = tensor.reshape(&[3, 2]).unwrap();
    let reshaped_data = reshaped.data().as_slice().unwrap();
    assert_eq!(reshaped_data, data.as_slice());
}

#[test]
fn test_numerical_stability() {
    use ndarray::Array1;
    
    // Test with very small numbers
    let small_numbers = Tensor::from_array(Array1::from_vec(vec![1e-20, 1e-10, 1e-5]));
    let small_doubled = small_numbers.add(&small_numbers).unwrap();
    let expected = vec![2e-20, 2e-10, 2e-5];
    
    let result_data = small_doubled.data().as_slice().unwrap();
    for (actual, expected) in result_data.iter().zip(expected.iter()) {
        assert!((actual / expected - 1.0).abs() < 1e-10);
    }
    
    // Test with very large numbers
    let large_numbers = Tensor::from_array(Array1::from_vec(vec![1e10, 1e15, 1e20]));
    let large_doubled = large_numbers.add(&large_numbers).unwrap();
    let expected_large = vec![2e10, 2e15, 2e20];
    
    let result_large = large_doubled.data().as_slice().unwrap();
    for (actual, expected) in result_large.iter().zip(expected_large.iter()) {
        assert!((actual / expected - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_performance_regression() {
    // Ensure that basic operations complete within reasonable time
    let size = 500;
    let a = Tensor::ones(&[size, size]);
    let b = Tensor::ones(&[size, size]);
    
    let start = std::time::Instant::now();
    let _result = a.matmul(&b).unwrap();
    let elapsed = start.elapsed();
    
    // Matrix multiplication of 500x500 should complete within a few seconds
    assert!(elapsed.as_secs() < 5, "Matrix multiplication took too long: {:?}", elapsed);
    
    let start = std::time::Instant::now();
    let _result = a.add(&b).unwrap();
    let elapsed = start.elapsed();
    
    // Addition should be very fast
    assert!(elapsed.as_millis() < 100, "Addition took too long: {:?}", elapsed);
}

#[test]
fn test_concurrent_access() {
    use std::sync::Arc;
    use std::thread;
    
    let model = Arc::new(Model::create_simple_linear());
    let mut handles = vec![];
    
    // Run inference from multiple threads
    for i in 0..4 {
        let model_clone = Arc::clone(&model);
        let handle = thread::spawn(move || {
            let mut inputs = HashMap::new();
            inputs.insert("input".to_string(), Tensor::from_array(Array2::from_shape_vec((1, 3), vec![i as f32, (i+1) as f32, (i+2) as f32]).unwrap().into_dyn()));
            
            let outputs = model_clone.run(&inputs).unwrap();
            outputs.get("output").unwrap().clone()
        });
        handles.push(handle);
    }
    
    // Collect results
    let results: Vec<Tensor> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    
    // Verify all results have correct shape
    for result in results {
        assert_eq!(result.shape(), &[1, 2]);
    }
}
