//! YOLOv8 Object Detection Example
//!
//! This example demonstrates how to use RunNX to load a YOLOv8n model
//! and perform object detection on real images.

use image::{imageops::FilterType, ImageReader};
use runnx::*;
use std::collections::HashMap;

fn main() -> runnx::Result<()> {
    println!("🎯 RunNX YOLOv8 Object Detection Demo");
    println!("====================================");

    // Initialize logging to see operator warnings
    env_logger::init();

    // Step 1: Load the YOLOv8n model
    let model = load_yolov8n_model()?;
    println!("✅ YOLOv8n model loaded successfully!");

    // Step 2: Load and prepare real image
    let input_tensor = load_and_prepare_image("assets/example.jpg")?;
    println!(
        "✅ Real image loaded and prepared: shape {:?}",
        input_tensor.shape()
    );

    // Step 3: Run inference
    run_inference(&model, input_tensor)?;

    // Step 4: Show model architecture details
    analyze_model_architecture(&model)?;

    // Step 5: Demonstrate YOLOv8 output format
    demonstrate_yolo_output_format()?;

    println!("\n🎉 YOLOv8 object detection demo completed!");
    println!("This demonstrates RunNX's capability to load and work with YOLOv8 models using real images.");

    Ok(())
}

/// Load the YOLOv8n model from file
fn load_yolov8n_model() -> runnx::Result<model::Model> {
    let model_path = "yolov8n.onnx";

    if !std::path::Path::new(model_path).exists() {
        return Err(error::OnnxError::model_load_error(
            "YOLOv8n model file (yolov8n.onnx) not found in current directory. 
            Please download it from: https://github.com/ultralytics/assets/releases/"
                .to_string(),
        ));
    }

    println!("📂 Loading YOLOv8n model from {model_path}...");
    let model = model::Model::from_onnx_file(model_path)?;

    println!("   📛 Model name: {}", model.name());
    println!("   📊 Graph: {}", model.graph.name);
    println!("   🔗 Nodes: {}", model.graph.nodes.len());
    println!("   📥 Inputs: {}", model.graph.inputs.len());
    println!("   📤 Outputs: {}", model.graph.outputs.len());

    Ok(model)
}

/// Load and prepare a real image for YOLOv8 inference
fn load_and_prepare_image(image_path: &str) -> runnx::Result<tensor::Tensor> {
    println!("🖼️  Loading image from: {image_path}");

    // Check if image exists
    if !std::path::Path::new(image_path).exists() {
        println!("⚠️  Image file not found, falling back to dummy image");
        return prepare_dummy_image();
    }

    // Load the image
    let img = ImageReader::open(image_path)
        .map_err(|e| error::OnnxError::model_load_error(format!("Failed to open image: {e}")))?
        .decode()
        .map_err(|e| error::OnnxError::model_load_error(format!("Failed to decode image: {e}")))?;

    println!(
        "   📐 Original image size: {}x{}",
        img.width(),
        img.height()
    );

    // YOLOv8 expects input shape: [batch_size, channels, height, width]
    // For YOLOv8n: [1, 3, 640, 640]
    let target_size = 640usize;
    let batch_size = 1usize;
    let channels = 3usize; // RGB

    // Resize image to 640x640 (maintaining aspect ratio with padding would be better for real use)
    let resized_img =
        img.resize_exact(target_size as u32, target_size as u32, FilterType::Lanczos3);
    let rgb_img = resized_img.to_rgb8();

    println!("   🔄 Resized to: {target_size}x{target_size}");

    // Convert to tensor format: [batch, channels, height, width]
    let mut image_data = Vec::with_capacity(batch_size * channels * target_size * target_size);

    // YOLOv8 expects CHW format (Channels, Height, Width)
    for c in 0..channels {
        for h in 0..target_size {
            for w in 0..target_size {
                let pixel = rgb_img.get_pixel(w as u32, h as u32);
                let value = pixel[c] as f32 / 255.0; // Normalize to [0, 1]
                image_data.push(value);
            }
        }
    }

    let shape = [batch_size, channels, target_size, target_size];
    let tensor = tensor::Tensor::from_shape_vec(&shape, image_data)?;

    println!("   📏 Tensor shape: {:?}", tensor.shape());
    println!("   🎨 Pixel range: [0.0, 1.0] (normalized)");
    println!("   ✅ Image preprocessing completed");

    Ok(tensor)
}

/// Prepare a dummy RGB image for testing (fallback)
fn prepare_dummy_image() -> runnx::Result<tensor::Tensor> {
    println!("🖼️  Preparing dummy RGB image...");

    // YOLOv8 expects input shape: [batch_size, channels, height, width]
    // For YOLOv8n: [1, 3, 640, 640]
    let batch_size = 1;
    let channels = 3; // RGB
    let height = 640;
    let width = 640;

    let total_elements = batch_size * channels * height * width;

    // Create a synthetic image with gradient pattern
    let mut image_data = Vec::with_capacity(total_elements);

    for _b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    // Create a gradient pattern that varies by channel
                    let value = match c {
                        0 => (h as f32 / height as f32) * 255.0, // Red channel: vertical gradient
                        1 => (w as f32 / width as f32) * 255.0, // Green channel: horizontal gradient
                        2 => ((h + w) as f32 / (height + width) as f32) * 255.0, // Blue channel: diagonal gradient
                        _ => 128.0,
                    };

                    // Normalize to [0, 1] range (typical for YOLO preprocessing)
                    image_data.push(value / 255.0);
                }
            }
        }
    }

    let shape = [batch_size, channels, height, width];
    let tensor = tensor::Tensor::from_shape_vec(&shape, image_data)?;

    println!("   📏 Input shape: {:?}", tensor.shape());
    println!("   🎨 Pixel range: [0.0, 1.0] (normalized)");

    Ok(tensor)
}

/// Run inference on the model
fn run_inference(model: &model::Model, input_tensor: tensor::Tensor) -> runnx::Result<()> {
    println!("🔮 Running YOLOv8 inference...");

    // Get input name from model
    let input_name = if let Some(input_spec) = model.graph.inputs.first() {
        input_spec.name.clone()
    } else {
        "images".to_string() // Default YOLOv8 input name
    };

    println!("   📥 Input name: '{input_name}'");
    println!(
        "   ⏱️  Starting inference with {} operations...",
        model.graph.nodes.len()
    );

    // Prepare inputs for the model as HashMap
    let mut inputs = HashMap::new();
    inputs.insert(input_name, input_tensor);

    // Add timing for inference
    let start_time = std::time::Instant::now();

    // Note: This is where actual inference would happen
    match model.run(&inputs) {
        Ok(outputs) => {
            let duration = start_time.elapsed();
            println!("✅ Inference completed successfully in {duration:?}!");

            for (name, tensor) in outputs.iter() {
                println!("   📤 Output '{}': shape {:?}", name, tensor.shape());

                if name == "output0" || name.contains("output") {
                    analyze_yolo_output(tensor)?;
                }
            }
        }
        Err(e) => {
            let duration = start_time.elapsed();
            println!("⚠️  Inference failed after {duration:?}:");
            println!("   Error: {e}");
            println!("   This may indicate operator implementation issues.");

            // For debugging, return the error instead of continuing
            return Err(e);
        }
    }

    Ok(())
}

/// Analyze YOLOv8 output tensor format
fn analyze_yolo_output(output: &tensor::Tensor) -> runnx::Result<()> {
    let shape = output.shape();
    println!("🔍 Analyzing YOLOv8 output format:");

    if shape.len() >= 3 {
        let batch_size = shape[0];
        let num_features = shape[1];
        let num_detections = shape[2];

        println!("   📊 Batch size: {batch_size}");
        println!("   🔢 Features per detection: {num_features}");
        println!("   🎯 Number of detection anchors: {num_detections}");

        if num_features == 84 {
            println!("   📝 Feature breakdown for COCO dataset:");
            println!("      - Bounding box coordinates (x, y, w, h): 4 values");
            println!("      - Object classes (COCO): 80 values");
            println!("      - Total: 4 + 80 = 84 features per detection");
        }

        println!("   💡 Post-processing steps needed:");
        println!("      1. Apply sigmoid to coordinates and confidence");
        println!("      2. Scale coordinates to image size");
        println!("      3. Apply Non-Maximum Suppression (NMS)");
        println!("      4. Filter by confidence threshold");
    }

    Ok(())
}

/// Demonstrate what YOLOv8 output format looks like
fn demonstrate_yolo_output_format() -> runnx::Result<()> {
    println!("📋 YOLOv8 Output Format Demonstration:");

    // Create a synthetic output tensor to show the format
    let batch_size = 1;
    let num_features = 84; // 4 bbox coords + 80 COCO classes
    let num_anchors = 8400; // Typical for YOLOv8n at 640x640

    println!("   📐 Expected output shape: [{batch_size}, {num_features}, {num_anchors}]");
    println!("   🎯 Detection format per anchor:");
    println!("      - [0-3]: Bounding box (center_x, center_y, width, height)");
    println!("      - [4-83]: Class probabilities for 80 COCO classes");

    // Show COCO class names (first 10)
    let coco_classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
    ];

    println!("   📚 COCO classes (first 10):");
    for (i, class) in coco_classes.iter().enumerate() {
        println!("      {i}: {class}");
    }
    println!("      ... (70 more classes)");

    Ok(())
}

/// Analyze the model architecture in detail
fn analyze_model_architecture(model: &model::Model) -> runnx::Result<()> {
    println!("\n🏗️  YOLOv8 Model Architecture Analysis");
    println!("=====================================");

    // Analyze operators used in the model
    let mut operator_counts = HashMap::new();
    for node in &model.graph.nodes {
        *operator_counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }

    println!("🧩 Operators used in YOLOv8n model:");
    for (op_type, count) in operator_counts.iter() {
        let status = match op_type.as_str() {
            "Conv" | "Sigmoid" | "Mul" | "Add" | "Concat" | "Reshape" | "Transpose" | "Div"
            | "Sub" | "Identity" | "Cast" | "Shape" => "✅",
            "Slice" | "MaxPool" | "Upsample" | "Softmax" | "NonMaxSuppression"
            | "BatchNormalization" | "Split" | "Gather" | "ConstantOfShape" | "Unsqueeze"
            | "Squeeze" | "Pad" | "Exp" | "Sqrt" | "Pow" | "ReduceMean" | "Resize" => "🚧",
            _ => "❓",
        };
        println!("   {status} {op_type}: {count} uses");
    }

    // Show model metadata
    println!("\n📋 Model Metadata:");
    println!("   📛 Name: {}", model.metadata.name);
    println!("   📝 Description: {}", model.metadata.description);
    println!("   🏭 Producer: {}", model.metadata.producer);
    println!("   📦 Version: {}", model.metadata.version);
    println!("   🔧 ONNX Version: {}", model.metadata.onnx_version);

    // Show input/output details
    println!("\n📥 Model Inputs:");
    for input in &model.graph.inputs {
        println!(
            "   🖼️  '{}': {:?} ({})",
            input.name, input.dimensions, input.dtype
        );
    }

    println!("\n📤 Model Outputs:");
    for output in &model.graph.outputs {
        println!(
            "   🎯 '{}': {:?} ({})",
            output.name, output.dimensions, output.dtype
        );
    }

    // Analyze initializers (weights)
    println!("\n⚖️  Model Weights:");
    println!(
        "   📊 Number of weight tensors: {}",
        model.graph.initializers.len()
    );

    let mut total_params = 0;
    for (name, tensor) in model.graph.initializers.iter().take(5) {
        let params = tensor.len();
        total_params += params;
        println!(
            "   📦 '{}': shape {:?} ({} parameters)",
            name,
            tensor.shape(),
            params
        );
    }

    if model.graph.initializers.len() > 5 {
        println!(
            "   ... and {} more weight tensors",
            model.graph.initializers.len() - 5
        );
    }

    println!(
        "   🔢 Estimated total parameters: ~{}M",
        (total_params as f32 / 1_000_000.0).round()
    );

    Ok(())
}

/// Create a simple object detection pipeline demonstration
fn _demonstrate_object_detection_pipeline() -> runnx::Result<()> {
    println!("\n🔄 Object Detection Pipeline Overview");
    println!("====================================");

    println!("1. 📷 Image Preprocessing:");
    println!("   - Resize image to 640x640");
    println!("   - Normalize pixel values to [0, 1]");
    println!("   - Convert to tensor format [1, 3, 640, 640]");

    println!("\n2. 🧠 Model Inference:");
    println!("   - Feed preprocessed image to YOLOv8 model");
    println!("   - Get raw predictions [1, 84, 8400]");

    println!("\n3. 🔍 Post-processing:");
    println!("   - Decode bounding box coordinates");
    println!("   - Apply confidence thresholding");
    println!("   - Perform Non-Maximum Suppression (NMS)");
    println!("   - Map class indices to class names");

    println!("\n4. 📊 Results:");
    println!("   - List of detected objects with:");
    println!("     • Bounding box coordinates");
    println!("     • Class name and confidence score");
    println!("     • Optional: Draw bounding boxes on image");

    Ok(())
}
