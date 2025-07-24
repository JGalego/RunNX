use runnx::graph::TensorSpec;
use runnx::{error::Result, Graph, Model, Node};
use std::fs;

fn main() -> Result<()> {
    println!("Testing ONNX binary format support...");

    // Create a simple model
    let mut graph = Graph::new("test_graph".to_string());

    // Add input specification
    let input_spec = TensorSpec::new("input".to_string(), vec![Some(1), Some(3)]);
    graph.add_input(input_spec);

    // Add output specification
    let output_spec = TensorSpec::new("output".to_string(), vec![Some(1), Some(3)]);
    graph.add_output(output_spec);

    // Add a simple node (Relu - supported operator)
    let node = Node::new(
        "relu".to_string(),
        "Relu".to_string(),
        vec!["input".to_string()],
        vec!["output".to_string()],
    );
    graph.add_node(node);

    // Create model
    let model = Model::new(graph);

    println!("Model created with {} nodes", model.graph.nodes.len());

    // Test 1: Save and load as JSON (original format)
    println!("\n1. Testing JSON format:");
    let json_path = std::env::temp_dir().join("test_model.json");

    match model.to_json_file(&json_path) {
        Ok(_) => println!("✓ Successfully saved model as JSON"),
        Err(e) => println!("✗ Failed to save as JSON: {e}"),
    }

    match Model::from_json_file(&json_path) {
        Ok(loaded_model) => {
            println!("✓ Successfully loaded model from JSON");
            println!("  - Loaded {} nodes", loaded_model.graph.nodes.len());
            println!("  - Loaded {} inputs", loaded_model.graph.inputs.len());
            println!("  - Loaded {} outputs", loaded_model.graph.outputs.len());
        }
        Err(e) => println!("✗ Failed to load from JSON: {e}"),
    }

    // Test 2: Save and load as ONNX binary
    println!("\n2. Testing ONNX binary format:");
    let onnx_path = std::env::temp_dir().join("test_model.onnx");

    match model.to_onnx_file(&onnx_path) {
        Ok(_) => println!("✓ Successfully saved model as ONNX binary"),
        Err(e) => println!("✗ Failed to save as ONNX binary: {e}"),
    }

    match Model::from_onnx_file(&onnx_path) {
        Ok(loaded_model) => {
            println!("✓ Successfully loaded model from ONNX binary");
            println!("  - Loaded {} nodes", loaded_model.graph.nodes.len());
            println!("  - Loaded {} inputs", loaded_model.graph.inputs.len());
            println!("  - Loaded {} outputs", loaded_model.graph.outputs.len());
        }
        Err(e) => println!("✗ Failed to load from ONNX binary: {e}"),
    }

    // Test 3: Format auto-detection
    println!("\n3. Testing format auto-detection:");

    match Model::from_file(&json_path) {
        Ok(_) => println!("✓ Auto-detected JSON format"),
        Err(e) => println!("✗ Failed to auto-detect JSON: {e}"),
    }

    match Model::from_file(&onnx_path) {
        Ok(_) => println!("✓ Auto-detected ONNX binary format"),
        Err(e) => println!("✗ Failed to auto-detect ONNX binary: {e}"),
    }

    // Cleanup
    let _ = fs::remove_file(&json_path);
    let _ = fs::remove_file(&onnx_path);

    println!("\n✅ ONNX support testing completed!");

    Ok(())
}
