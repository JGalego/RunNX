use runnx::*;
use std::collections::HashMap;

fn main() -> runnx::Result<()> {
    println!("🎯 RunNX YOLO Operator Support Demo");
    println!("===================================");

    // Test basic YOLO operators
    test_yolo_operators()?;

    // Create a simplified YOLO-like model structure
    create_yolo_model_demo()?;

    println!("\n🎉 YOLO demo completed successfully!");
    println!("RunNX now supports YOLO-essential operators! 🚀");

    Ok(())
}

fn test_yolo_operators() -> runnx::Result<()> {
    println!("\n🔧 Testing YOLO Operators...");

    // Test Concat operator
    println!("  📎 Testing Concat operator...");
    let tensor1 = tensor::Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    let tensor2 = tensor::Tensor::from_shape_vec(&[2, 2], vec![5.0, 6.0, 7.0, 8.0])?;
    let inputs = vec![tensor1, tensor2];
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());

    let concat_result =
        operators::execute_operator(&operators::OperatorType::Concat, &inputs, &attrs)?;
    println!(
        "     ✅ Concat: Input shape [2,2] + [2,2] -> Output shape {:?}",
        concat_result[0].shape()
    );

    // Test Softmax operator
    println!("  🔥 Testing Softmax operator...");
    let logits = tensor::Tensor::from_shape_vec(&[1, 3], vec![2.0, 1.0, 0.1])?;
    let softmax_inputs = vec![logits];
    let softmax_attrs = HashMap::new();

    let softmax_result = operators::execute_operator(
        &operators::OperatorType::Softmax,
        &softmax_inputs,
        &softmax_attrs,
    )?;
    let sum: f32 = softmax_result[0].data().iter().sum();
    println!("     ✅ Softmax: Sum of probabilities = {sum:.6} (should be ~1.0)");

    // Test MaxPool operator
    println!("  🏊 Testing MaxPool operator...");
    let feature_map = tensor::Tensor::zeros(&[1, 64, 32, 32]); // Typical YOLO feature map
    let pool_inputs = vec![feature_map];
    let mut pool_attrs = HashMap::new();
    pool_attrs.insert("kernel_shape".to_string(), "2,2".to_string());
    pool_attrs.insert("strides".to_string(), "2,2".to_string());

    let pool_result =
        operators::execute_operator(&operators::OperatorType::MaxPool, &pool_inputs, &pool_attrs)?;
    println!(
        "     ✅ MaxPool: Input [1,64,32,32] -> Output {:?}",
        pool_result[0].shape()
    );

    // Test Upsample operator
    println!("  📈 Testing Upsample operator...");
    let small_map = tensor::Tensor::ones(&[1, 32, 16, 16]);
    let upsample_inputs = vec![small_map];
    let mut upsample_attrs = HashMap::new();
    upsample_attrs.insert("mode".to_string(), "nearest".to_string());

    let upsample_result = operators::execute_operator(
        &operators::OperatorType::Upsample,
        &upsample_inputs,
        &upsample_attrs,
    )?;
    println!(
        "     ✅ Upsample: Input [1,32,16,16] -> Output {:?}",
        upsample_result[0].shape()
    );

    // Test NonMaxSuppression operator
    println!("  🎯 Testing NonMaxSuppression operator...");
    let boxes = tensor::Tensor::from_shape_vec(&[1, 100, 4], vec![0.0; 400])?; // 100 bounding boxes
    let scores = tensor::Tensor::ones(&[1, 80, 100]); // 80 classes, 100 boxes
    let nms_inputs = vec![boxes, scores];
    let nms_attrs = HashMap::new();

    let nms_result = operators::execute_operator(
        &operators::OperatorType::NonMaxSuppression,
        &nms_inputs,
        &nms_attrs,
    )?;
    println!(
        "     ✅ NMS: Input boxes [1,100,4] + scores [1,80,100] -> Selected indices {:?}",
        nms_result[0].shape()
    );

    println!("  ✅ All YOLO operators tested successfully!");
    Ok(())
}

fn create_yolo_model_demo() -> runnx::Result<()> {
    println!("\n🏗️  Creating YOLO-like Model Structure...");

    // Create a simplified YOLO model graph
    let mut graph = graph::Graph::new("yolo_demo".to_string());

    // Input: RGB image (batch_size=1, channels=3, height=640, width=640)
    let input_spec = graph::TensorSpec::new(
        "images".to_string(),
        vec![Some(1), Some(3), Some(640), Some(640)],
    );
    graph.add_input(input_spec);

    // Layer 1: Initial Conv + SiLU (Sigmoid * x)
    let conv1_node = graph::Node::new(
        "conv1".to_string(),
        "Conv".to_string(),
        vec!["images".to_string()],
        vec!["conv1_out".to_string()],
    );
    graph.add_node(conv1_node);

    let sigmoid1_node = graph::Node::new(
        "sigmoid1".to_string(),
        "Sigmoid".to_string(),
        vec!["conv1_out".to_string()],
        vec!["sigmoid1_out".to_string()],
    );
    graph.add_node(sigmoid1_node);

    let silu1_node = graph::Node::new(
        "silu1".to_string(),
        "Mul".to_string(),
        vec!["conv1_out".to_string(), "sigmoid1_out".to_string()],
        vec!["silu1_out".to_string()],
    );
    graph.add_node(silu1_node);

    // Layer 2: MaxPool for downsampling
    let pool1_node = graph::Node::new(
        "pool1".to_string(),
        "MaxPool".to_string(),
        vec!["silu1_out".to_string()],
        vec!["pool1_out".to_string()],
    );
    graph.add_node(pool1_node);

    // Feature Pyramid Network (FPN) - Upsample for multi-scale features
    let upsample1_node = graph::Node::new(
        "upsample1".to_string(),
        "Upsample".to_string(),
        vec!["pool1_out".to_string()],
        vec!["upsampled_features".to_string()],
    );
    graph.add_node(upsample1_node);

    // Concatenate features from different scales
    let concat_node = graph::Node::new(
        "feature_concat".to_string(),
        "Concat".to_string(),
        vec!["silu1_out".to_string(), "upsampled_features".to_string()],
        vec!["concat_features".to_string()],
    );
    graph.add_node(concat_node);

    // Detection head - classification scores
    let cls_conv_node = graph::Node::new(
        "cls_conv".to_string(),
        "Conv".to_string(),
        vec!["concat_features".to_string()],
        vec!["cls_logits".to_string()],
    );
    graph.add_node(cls_conv_node);

    // Apply softmax to classification scores
    let softmax_node = graph::Node::new(
        "softmax".to_string(),
        "Softmax".to_string(),
        vec!["cls_logits".to_string()],
        vec!["cls_probs".to_string()],
    );
    graph.add_node(softmax_node);

    // Detection head - bounding box regression
    let bbox_conv_node = graph::Node::new(
        "bbox_conv".to_string(),
        "Conv".to_string(),
        vec!["concat_features".to_string()],
        vec!["bbox_preds".to_string()],
    );
    graph.add_node(bbox_conv_node);

    // Output: Detection results
    let output_spec = graph::TensorSpec::new(
        "detections".to_string(),
        vec![Some(1), Some(25200), Some(85)], // Typical YOLO output: [batch, anchors, classes+coords]
    );
    graph.add_output(output_spec);

    // Final concatenation for output
    let output_concat_node = graph::Node::new(
        "output_concat".to_string(),
        "Concat".to_string(),
        vec!["bbox_preds".to_string(), "cls_probs".to_string()],
        vec!["detections".to_string()],
    );
    graph.add_node(output_concat_node);

    // Create model with metadata
    let model = model::Model::with_metadata(
        model::ModelMetadata {
            name: "yolo_demo_v1".to_string(),
            version: "1.0.0".to_string(),
            description: "A simplified YOLO-like object detection model demonstrating RunNX YOLO operator support".to_string(),
            producer: "RunNX YOLO Demo".to_string(),
            onnx_version: "1.12.0".to_string(),
            domain: "ai.onnx".to_string(),
        },
        graph,
    );

    println!("✅ YOLO Model Structure Created:");
    println!("   📛 Model: {}", model.name());
    println!("   📊 Graph: {}", model.graph.name);
    println!(
        "   📥 Inputs: {} (images: [1,3,640,640])",
        model.graph.inputs.len()
    );
    println!(
        "   📤 Outputs: {} (detections: [1,25200,85])",
        model.graph.outputs.len()
    );
    println!(
        "   🔗 Nodes: {} (Conv, SiLU, MaxPool, Upsample, Concat, Softmax)",
        model.graph.nodes.len()
    );

    // Save the model in both formats
    println!("\n💾 Saving YOLO model...");
    model.to_json_file("yolo_demo.json")?;
    model.to_onnx_file("yolo_demo.onnx")?;
    println!("   ✅ Saved: yolo_demo.json");
    println!("   ✅ Saved: yolo_demo.onnx");

    // Test loading the model
    println!("\n🔄 Testing model loading...");
    let loaded_model = model::Model::from_file("yolo_demo.json")?;
    println!("   ✅ Loaded model: {}", loaded_model.name());
    println!("   📊 Nodes loaded: {}", loaded_model.graph.nodes.len());

    // Show typical YOLO operator flow
    println!("\n🔄 YOLO Operator Flow:");
    println!("   1. 📷 Input Image [1,3,640,640]");
    println!("   2. 🧠 Conv2D + SiLU Activation");
    println!("   3. 🏊 MaxPool Downsampling");
    println!("   4. 📈 Upsample for FPN");
    println!("   5. 📎 Concat Multi-scale Features");
    println!("   6. 🎯 Classification + BBox Heads");
    println!("   7. 🔥 Softmax for Class Probabilities");
    println!("   8. 📤 Output Detections [1,25200,85]");

    // Cleanup
    let _ = std::fs::remove_file("yolo_demo.json");
    let _ = std::fs::remove_file("yolo_demo.onnx");

    Ok(())
}

/// Demonstrate YOLO inference pipeline (conceptual)
fn _demonstrate_yolo_inference() -> runnx::Result<()> {
    println!("\n🚀 YOLO Inference Pipeline Demo (Conceptual):");

    // 1. Preprocessing
    println!("   1. 📷 Preprocess: Resize image to 640x640, normalize");

    // 2. Forward pass through backbone
    println!("   2. 🧠 Backbone: Extract multi-scale features");

    // 3. Feature Pyramid Network
    println!("   3. 🏗️  FPN: Upsample and concat features");

    // 4. Detection heads
    println!("   4. 🎯 Heads: Predict classes and bounding boxes");

    // 5. Post-processing
    println!("   5. 🔧 Post-process: Apply NMS, filter by confidence");

    // 6. Output
    println!("   6. 📊 Output: Final detections with boxes and labels");

    Ok(())
}
