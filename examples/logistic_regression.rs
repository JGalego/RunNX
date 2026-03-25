//! Logistic regression example
//!
//! This example demonstrates how to build and run a binary logistic regression
//! model using the RunNX library.
//!
//! The model computes:
//!   probability = sigmoid(X @ W + b)
//!
//! Graph: input -> MatMul -> Add -> Sigmoid -> output

use ndarray::Array2;
use runnx::{
    graph::{Graph, Node, TensorSpec},
    model::{Model, ModelMetadata},
    Tensor,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Logistic Regression Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Build the model
    // -----------------------------------------------------------------------
    println!("=== Building Model ===");

    // Weights: [2, 1]  (2 input features -> 1 output)
    // Bias:    [1, 1]
    let weights = Tensor::from_shape_vec(&[2, 1], vec![0.5_f32, -0.3])?;
    let bias = Tensor::from_shape_vec(&[1, 1], vec![0.1_f32])?;

    let mut graph = Graph::new("logistic_regression".to_string());

    // Input: batch of samples, shape [N, 2]
    graph.add_input(TensorSpec::new("input".to_string(), vec![None, Some(2)]));

    // Output: predicted probabilities, shape [N, 1]
    graph.add_output(TensorSpec::new("output".to_string(), vec![None, Some(1)]));

    graph.add_initializer("weights".to_string(), weights);
    graph.add_initializer("bias".to_string(), bias);

    // input @ weights -> linear
    graph.add_node(Node::new(
        "matmul".to_string(),
        "MatMul".to_string(),
        vec!["input".to_string(), "weights".to_string()],
        vec!["linear".to_string()],
    ));

    // linear + bias -> logit
    graph.add_node(Node::new(
        "add_bias".to_string(),
        "Add".to_string(),
        vec!["linear".to_string(), "bias".to_string()],
        vec!["logit".to_string()],
    ));

    // sigmoid(logit) -> output
    graph.add_node(Node::new(
        "sigmoid".to_string(),
        "Sigmoid".to_string(),
        vec!["logit".to_string()],
        vec!["output".to_string()],
    ));

    let model = Model::with_metadata(
        ModelMetadata {
            name: "logistic_regression".to_string(),
            version: "1.0".to_string(),
            description: "Binary logistic regression: sigmoid(X @ W + b)".to_string(),
            producer: "RunNX Example".to_string(),
            onnx_version: "1.9.0".to_string(),
            domain: "".to_string(),
        },
        graph,
    );

    println!("Model Summary:");
    println!("{}", model.summary());

    // -----------------------------------------------------------------------
    // 2. Prepare input data (4 samples, 2 features each)
    // -----------------------------------------------------------------------
    println!("=== Running Inference ===");

    // Samples: [[1, 2], [3, 4], [-1, -2], [0.5, -0.5]]
    let input_data = vec![1.0_f32, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, -0.5];
    let input_tensor = Tensor::from_array(Array2::from_shape_vec((4, 2), input_data)?.into_dyn());

    println!("Input shape: {:?}", input_tensor.shape());

    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input_tensor);

    let start = std::time::Instant::now();
    let outputs = model.run(&inputs)?;
    println!(
        "Inference completed in {:.2}µs",
        start.elapsed().as_micros()
    );

    // -----------------------------------------------------------------------
    // 3. Display and verify results
    // -----------------------------------------------------------------------
    println!("\n=== Results ===");

    let output = outputs.get("output").unwrap();
    let probs = output.data();
    let probs_slice = probs.as_slice().unwrap();

    println!("Output shape: {:?}", output.shape());
    println!();

    for (i, p) in probs_slice.iter().enumerate() {
        println!(
            "  Sample {}: probability = {:.6}  (class = {})",
            i,
            p,
            if *p >= 0.5 { 1 } else { 0 }
        );
    }

    // -----------------------------------------------------------------------
    // 4. Manual verification
    // -----------------------------------------------------------------------
    println!("\n=== Manual Verification ===");

    // W = [0.5, -0.3]^T,  b = 0.1
    //
    // Sample 0: [1, 2]    -> 1*0.5 + 2*(-0.3) + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
    // Sample 1: [3, 4]    -> 3*0.5 + 4*(-0.3) + 0.1 = 1.5 - 1.2 + 0.1 = 0.4
    // Sample 2: [-1, -2]  -> -0.5 + 0.6 + 0.1 = 0.2
    // Sample 3: [0.5,-0.5]-> 0.25 + 0.15 + 0.1 = 0.5
    //
    // sigmoid(z) = 1 / (1 + exp(-z))
    let logits = [0.0_f32, 0.4, 0.2, 0.5];
    let expected: Vec<f32> = logits.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();

    println!("Expected: {:?}", expected);
    println!("Actual:   {:?}", probs_slice);

    let mut all_match = true;
    for (i, (&exp, &act)) in expected.iter().zip(probs_slice.iter()).enumerate() {
        let diff = (exp - act).abs();
        if diff > 1e-6 {
            println!("  Mismatch at sample {i}: expected {exp:.6}, got {act:.6} (diff: {diff:.6})");
            all_match = false;
        }
    }
    if all_match {
        println!("All results match expected values!");
    }

    // -----------------------------------------------------------------------
    // 5. Save and reload
    // -----------------------------------------------------------------------
    println!("\n=== Saving and Loading Model ===");

    let model_path = "logistic_regression.json";
    model.to_file(model_path)?;
    println!("Saved to: {model_path}");

    let loaded = Model::from_file(model_path)?;
    let loaded_outputs = loaded.run(&inputs)?;
    let loaded_probs = loaded_outputs.get("output").unwrap().data();

    let roundtrip_ok = probs_slice
        .iter()
        .zip(loaded_probs.as_slice().unwrap().iter())
        .all(|(&a, &b)| (a - b).abs() < 1e-6);

    if roundtrip_ok {
        println!("Loaded model produces identical results!");
    } else {
        println!("Loaded model results differ!");
    }

    std::fs::remove_file(model_path)?;

    println!("\n=== Example Complete ===");
    Ok(())
}
