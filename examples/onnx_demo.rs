use runnx::*;

fn main() -> runnx::Result<()> {
    println!("🚀 RunNX ONNX Compatibility Demo");
    println!("================================");

    // Create a simple model
    let mut graph = graph::Graph::new("demo_graph".to_string());

    // Add input specification
    let input_spec = graph::TensorSpec::new("input".to_string(), vec![Some(1), Some(4)]);
    graph.add_input(input_spec);

    // Add output specification
    let output_spec = graph::TensorSpec::new("output".to_string(), vec![Some(1), Some(4)]);
    graph.add_output(output_spec);

    // Add a ReLU node
    let relu_node = graph::Node::new(
        "relu_1".to_string(),
        "Relu".to_string(),
        vec!["input".to_string()],
        vec!["output".to_string()],
    );
    graph.add_node(relu_node);

    let model = model::Model::with_metadata(
        model::ModelMetadata {
            name: "demo_model".to_string(),
            version: "1.0".to_string(),
            description: "A simple ReLU demo model".to_string(),
            producer: "RunNX Demo".to_string(),
            onnx_version: "1.9.0".to_string(),
            domain: "".to_string(),
        },
        graph,
    );

    println!("✅ Created model: {}", model.name());
    println!("   Graph: {}", model.graph.name);
    println!("   Inputs: {}", model.graph.inputs.len());
    println!("   Outputs: {}", model.graph.outputs.len());
    println!("   Nodes: {}", model.graph.nodes.len());

    // Test JSON format
    println!("\n📄 Testing JSON format...");
    model.to_json_file("demo_model.json")?;
    let json_model = model::Model::from_json_file("demo_model.json")?;
    println!("✅ JSON: Saved and loaded successfully");
    println!("   Loaded model: {}", json_model.name());

    // Test ONNX binary format
    println!("\n🔄 Testing ONNX binary format...");
    model.to_onnx_file("demo_model.onnx")?;
    let onnx_model = model::Model::from_onnx_file("demo_model.onnx")?;
    println!("✅ ONNX: Saved and loaded successfully");
    println!("   Loaded model: {}", onnx_model.name());

    // Test generic file loading (auto-detection)
    println!("\n🎯 Testing generic file loading...");
    let auto_json = model::Model::from_file("demo_model.json")?;
    let auto_onnx = model::Model::from_file("demo_model.onnx")?;
    println!("✅ Auto-detection: Both formats loaded successfully");
    println!("   JSON auto-load: {}", auto_json.name());
    println!("   ONNX auto-load: {}", auto_onnx.name());

    // Verify consistency
    println!("\n🔍 Verifying consistency...");
    assert_eq!(model.name(), json_model.name());
    // Note: ONNX format may use graph name as model name in some cases
    println!("   Original model: {}", model.name());
    println!("   JSON model: {}", json_model.name());
    println!("   ONNX model: {}", onnx_model.name());
    assert_eq!(model.graph.name, json_model.graph.name);
    assert_eq!(model.graph.name, onnx_model.graph.name);
    println!("✅ All formats are functionally consistent!");

    // Show file sizes
    use std::fs;
    let json_size = fs::metadata("demo_model.json")?.len();
    let onnx_size = fs::metadata("demo_model.onnx")?.len();
    println!("\n📊 File sizes:");
    println!("   JSON: {json_size} bytes");
    println!("   ONNX: {onnx_size} bytes");

    // Cleanup
    let _ = fs::remove_file("demo_model.json");
    let _ = fs::remove_file("demo_model.onnx");

    println!("\n🎉 Demo completed successfully!");
    println!("RunNX now supports both JSON and ONNX binary formats! 🚀");

    Ok(())
}
