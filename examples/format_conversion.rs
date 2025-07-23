//! Format conversion example
//!
//! This example demonstrates how to convert between JSON and ONNX binary formats.

use runnx::*;
use std::fs;

fn main() -> runnx::Result<()> {
    println!("🔄 RunNX Format Conversion Demo");
    println!("==============================");

    // Create a simple model with multiple operations
    let mut graph = graph::Graph::new("conversion_demo".to_string());

    // Add input specification
    let input_spec = graph::TensorSpec::new("input".to_string(), vec![Some(1), Some(3)]);
    graph.add_input(input_spec);

    // Add intermediate and output specifications
    let _relu_output_spec =
        graph::TensorSpec::new("relu_output".to_string(), vec![Some(1), Some(3)]);
    let final_output_spec = graph::TensorSpec::new("output".to_string(), vec![Some(1), Some(3)]);
    graph.add_output(final_output_spec);

    // Add ReLU node
    let relu_node = graph::Node::new(
        "relu_node".to_string(),
        "Relu".to_string(),
        vec!["input".to_string()],
        vec!["relu_output".to_string()],
    );
    graph.add_node(relu_node);

    // Add Sigmoid node
    let sigmoid_node = graph::Node::new(
        "sigmoid_node".to_string(),
        "Sigmoid".to_string(),
        vec!["relu_output".to_string()],
        vec!["output".to_string()],
    );
    graph.add_node(sigmoid_node);

    let model = model::Model::with_metadata(
        model::ModelMetadata {
            name: "conversion_demo".to_string(),
            version: "2.0".to_string(),
            description: "Demo model with ReLU + Sigmoid pipeline".to_string(),
            producer: "RunNX Format Converter".to_string(),
            onnx_version: "1.9.0".to_string(),
            domain: "ai.example".to_string(),
        },
        graph,
    );

    println!("✅ Created model: {}", model.name());
    println!("   Description: {}", model.metadata.description);
    println!("   Nodes: {}", model.graph.nodes.len());

    // Original model summary
    println!("\n📋 Original Model Summary:");
    println!("   Name: {}", model.name());
    println!("   Producer: {}", model.metadata.producer);
    println!("   Version: {}", model.metadata.version);
    println!("   Graph: {}", model.graph.name);
    println!(
        "   Operations: {} -> {}",
        model.graph.nodes[0].op_type, model.graph.nodes[1].op_type
    );

    // Step 1: Save as JSON
    println!("\n📄 Step 1: Saving as JSON format...");
    model.to_json_file("conversion_demo.json")?;
    let json_size = fs::metadata("conversion_demo.json")?.len();
    println!("✅ JSON saved: {json_size} bytes");

    // Step 2: Convert JSON -> ONNX Binary
    println!("\n🔧 Step 2: Converting JSON to ONNX binary...");
    let loaded_from_json = model::Model::from_json_file("conversion_demo.json")?;
    loaded_from_json.to_onnx_file("conversion_demo.onnx")?;
    let onnx_size = fs::metadata("conversion_demo.onnx")?.len();
    println!("✅ ONNX binary saved: {onnx_size} bytes");

    // Step 3: Verify round-trip conversion
    println!("\n🔍 Step 3: Verifying round-trip conversion...");
    let loaded_from_onnx = model::Model::from_onnx_file("conversion_demo.onnx")?;

    // Verify integrity
    assert_eq!(model.name(), loaded_from_json.name());
    assert_eq!(model.name(), loaded_from_onnx.name());
    assert_eq!(model.graph.nodes.len(), loaded_from_onnx.graph.nodes.len());
    assert_eq!(
        model.metadata.description,
        loaded_from_onnx.metadata.description
    );

    println!("✅ Round-trip conversion successful!");
    println!("   Original name: {}", model.name());
    println!("   JSON->Model name: {}", loaded_from_json.name());
    println!("   ONNX->Model name: {}", loaded_from_onnx.name());

    // Step 4: Test auto-detection
    println!("\n🎯 Step 4: Testing auto-detection...");
    let auto_json = model::Model::from_file("conversion_demo.json")?;
    let auto_onnx = model::Model::from_file("conversion_demo.onnx")?;

    assert_eq!(auto_json.name(), auto_onnx.name());
    println!("✅ Auto-detection works for both formats!");

    // Step 5: File size comparison
    println!("\n📊 Step 5: File size comparison:");
    println!("   JSON format:  {json_size} bytes");
    println!("   ONNX binary:  {onnx_size} bytes");
    let compression_ratio = (json_size as f64 - onnx_size as f64) / json_size as f64 * 100.0;
    println!("   Compression:  {compression_ratio:.1}% smaller (ONNX vs JSON)");

    // Step 6: Performance comparison (loading time)
    println!("\n⚡ Step 6: Performance comparison:");

    let start = std::time::Instant::now();
    let _ = model::Model::from_json_file("conversion_demo.json")?;
    let json_load_time = start.elapsed();

    let start = std::time::Instant::now();
    let _ = model::Model::from_onnx_file("conversion_demo.onnx")?;
    let onnx_load_time = start.elapsed();

    println!("   JSON load time:  {json_load_time:?}");
    println!("   ONNX load time:  {onnx_load_time:?}");

    if onnx_load_time < json_load_time {
        let speedup = json_load_time.as_nanos() as f64 / onnx_load_time.as_nanos() as f64;
        println!("   ONNX is {speedup:.1}x faster to load");
    }

    // Step 7: Cross-format validation
    println!("\n🔄 Step 7: Cross-format validation:");

    // Save ONNX model back to JSON
    loaded_from_onnx.to_json_file("roundtrip.json")?;
    let roundtrip_json = model::Model::from_json_file("roundtrip.json")?;

    // Save JSON model back to ONNX
    loaded_from_json.to_onnx_file("roundtrip.onnx")?;
    let roundtrip_onnx = model::Model::from_onnx_file("roundtrip.onnx")?;

    assert_eq!(roundtrip_json.name(), roundtrip_onnx.name());
    assert_eq!(
        roundtrip_json.graph.nodes.len(),
        roundtrip_onnx.graph.nodes.len()
    );

    println!("✅ Cross-format validation successful!");
    println!("   All conversions preserve model integrity");

    // Cleanup
    println!("\n🧹 Cleaning up temporary files...");
    let files_to_remove = [
        "conversion_demo.json",
        "conversion_demo.onnx",
        "roundtrip.json",
        "roundtrip.onnx",
    ];

    for file in &files_to_remove {
        let _ = fs::remove_file(file);
    }

    println!("\n🎉 Format conversion demo completed successfully!");
    print!("🚀 RunNX supports seamless conversion between JSON and ONNX binary formats");
    println!(" with automatic format detection!");

    Ok(())
}
