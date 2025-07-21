//! Simple model example
//!
//! This example demonstrates how to create, save, and run a simple model
//! using the RunNX library.

use ndarray::{Array1, Array2};
use runnx::{Model, Tensor};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("=== Simple Model Example ===\n");

    // Create a simple linear model
    let model = Model::create_simple_linear();

    // Print model summary
    println!("Model Summary:");
    println!("{}", model.summary());

    // Create input data
    let input_data = vec![1.0, 2.0, 3.0];
    let input_tensor = Tensor::from_array(
        Array2::from_shape_vec((1, 3), input_data)
            .unwrap()
            .into_dyn(),
    );

    println!("Input tensor:");
    println!("  Shape: {:?}", input_tensor.shape());
    println!("  Data: {:?}", input_tensor.data().as_slice().unwrap());

    // Prepare inputs
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input_tensor);

    // Run inference
    println!("\nRunning inference...");
    let start_time = std::time::Instant::now();
    let outputs = model.run(&inputs)?;
    let inference_time = start_time.elapsed();

    // Display results
    println!("Inference completed in {:.2}µs", inference_time.as_micros());

    for (name, tensor) in &outputs {
        println!("\nOutput '{name}':");
        println!("  Shape: {:?}", tensor.shape());
        println!("  Data: {:?}", tensor.data().as_slice().unwrap());
    }

    // Manual verification
    println!("\n=== Manual Verification ===");

    // Expected calculation:
    // input: [1, 2, 3] (shape: [1, 3])
    // weights: [[0.5, 0.3], [0.2, 0.4], [0.1, 0.6]] (shape: [3, 2])
    // bias: [0.1, 0.2] (shape: [1, 2])
    //
    // matmul: [1, 2, 3] × [[0.5, 0.3], [0.2, 0.4], [0.1, 0.6]]
    //       = [1×0.5 + 2×0.2 + 3×0.1, 1×0.3 + 2×0.4 + 3×0.6]
    //       = [0.5 + 0.4 + 0.3, 0.3 + 0.8 + 1.8]
    //       = [1.2, 2.9]
    //
    // add bias: [1.2, 2.9] + [0.1, 0.2] = [1.3, 3.1]

    let expected_result = vec![1.3, 3.1];
    let actual_result = outputs.get("output").unwrap();
    let actual_data = actual_result.data().as_slice().unwrap();

    println!("Expected result: {expected_result:?}");
    println!("Actual result:   {actual_data:?}");

    // Verify the results
    let mut all_match = true;
    for (i, (&expected, &actual)) in expected_result.iter().zip(actual_data.iter()).enumerate() {
        let diff = (expected - actual).abs();
        if diff > 1e-6 {
            println!(
                "❌ Mismatch at index {i}: expected {expected:.6}, got {actual:.6} (diff: {diff:.6})"
            );
            all_match = false;
        }
    }

    if all_match {
        println!("✅ All results match expected values!");
    }

    // Save the model for later use
    println!("\n=== Saving Model ===");
    let model_path = "simple_model.json";
    model.to_file(model_path)?;
    println!("Model saved to: {model_path}");

    // Demonstrate loading the model back
    println!("\n=== Loading Model ===");
    let loaded_model = Model::from_file(model_path)?;
    println!("Model loaded successfully!");
    println!("Name: {}", loaded_model.name());
    println!("Inputs: {:?}", loaded_model.input_names());
    println!("Outputs: {:?}", loaded_model.output_names());

    // Run inference with loaded model to verify it works
    let loaded_outputs = loaded_model.run(&inputs)?;
    let loaded_result = loaded_outputs.get("output").unwrap();
    let loaded_data = loaded_result.data().as_slice().unwrap();

    println!("\nLoaded model output: {loaded_data:?}");

    // Verify loaded model produces same results
    let results_match = actual_data
        .iter()
        .zip(loaded_data.iter())
        .all(|(&a, &b)| (a - b).abs() < 1e-6);

    if results_match {
        println!("✅ Loaded model produces identical results!");
    } else {
        println!("❌ Loaded model results don't match!");
    }

    // Demonstrate tensor operations directly
    println!("\n=== Direct Tensor Operations ===");

    let a = Tensor::from_array(Array2::from_shape_vec(
        (2, 3),
        vec![1., 2., 3., 4., 5., 6.],
    )?);
    let b = Tensor::from_array(Array2::from_shape_vec(
        (2, 3),
        vec![1., 1., 1., 1., 1., 1.],
    )?);

    println!("Tensor A:");
    println!("{a}");

    println!("Tensor B:");
    println!("{b}");

    // Addition
    let c = a.add(&b)?;
    println!("A + B:");
    println!("{c}");

    // Multiplication
    let d = a.mul(&b)?;
    println!("A * B:");
    println!("{d}");

    // ReLU
    let e = Tensor::from_array(Array1::from_vec(vec![-2., -1., 0., 1., 2.]));
    let f = e.relu();
    println!("ReLU([-2, -1, 0, 1, 2]):");
    println!("{f}");

    // Sigmoid
    let g = Tensor::from_array(Array1::from_vec(vec![-1., 0., 1.]));
    let h = g.sigmoid();
    println!("Sigmoid([-1, 0, 1]):");
    println!("{h}");

    println!("\n=== Example Complete ===");
    Ok(())
}
