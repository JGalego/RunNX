# Usage Examples

This guide provides comprehensive examples of using RunNX for various machine learning tasks and use cases, including computer vision applications.

## Table of Contents

- [Basic Model Operations](#basic-model-operations)
- [Computer Vision Applications](#computer-vision-applications)
- [Format Conversion](#format-conversion)
- [Graph Visualization](#graph-visualization)
- [Async Operations](#async-operations)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)

## Basic Model Operations

### Loading Models

```rust
use runnx::{Model, Tensor};

// Auto-detection based on file extension
let model = Model::from_file("model.onnx")?;

// Explicit format specification
let json_model = Model::from_json_file("model.json")?;
let onnx_model = Model::from_onnx_file("model.onnx")?;

// Loading from memory
let model_data = std::fs::read("model.onnx")?;
let model = Model::from_onnx_bytes(&model_data)?;
```

### Creating Input Tensors

```rust
use ndarray::{Array2, Array4};

// From ndarray
let input_2d = Tensor::from_array(Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?);

// Image-like 4D tensor (batch_size, channels, height, width)
let image_input = Tensor::from_array(Array4::zeros((1, 3, 224, 224)));

// From Vec with shape
let data = vec![1.0, 2.0, 3.0, 4.0];
let tensor = Tensor::from_shape_vec(vec![2, 2], data)?;
```

### Running Inference

```rust
// Single input model
let input = Tensor::from_array(ndarray::array![[1.0, 2.0, 3.0]]);
let outputs = model.run(&[("input", input)])?;
let result = outputs.get("output").unwrap();

// Multiple inputs
let input1 = Tensor::from_array(ndarray::array![[1.0, 2.0]]);
let input2 = Tensor::from_array(ndarray::array![[3.0, 4.0]]);
let outputs = model.run(&[
    ("input1", input1),
    ("input2", input2),
])?;

// Accessing results
for (name, tensor) in outputs.iter() {
    println!("Output '{}': shape {:?}", name, tensor.shape());
    println!("Data: {:?}", tensor.data());
}
```

## Format Conversion

### JSON to ONNX Binary

```rust
use runnx::Model;

// Load JSON model
let model = Model::from_json_file("model.json")?;

// Save as ONNX binary
model.to_onnx_file("model.onnx")?;

println!("Converted JSON to ONNX binary format");
```

### ONNX Binary to JSON

```rust
// Load ONNX binary model
let model = Model::from_onnx_file("model.onnx")?;

// Save as JSON (human-readable)
model.to_json_file("model.json")?;

println!("Converted ONNX binary to JSON format");
```

## Computer Vision Applications

RunNX supports various computer vision models including classification, object detection, and segmentation. Here's a comprehensive example using object detection:

### Complete Detection Pipeline

The easiest way to get started with computer vision models:

```bash
# Example: Object detection with YOLOv8 (demonstrates broader model support)
# Download YOLOv8n model (if you don't have it)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx

# Run the complete detection example  
cargo run --example yolov8_detect_and_draw

# Expected output demonstrates RunNX capabilities:
# Model loaded successfully: 224 nodes
# Preprocessing image: assets/bus.jpg -> [1, 3, 640, 640] 
# Running inference...
# Inference completed in 45.67ms
# 
# Detection Results:
# 🚌 bus: 89.2% confidence at [18, 137, 558, 437]
# 🚗 car: 76.8% confidence at [666, 158, 805, 213]
# 👤 person: 65.3% confidence at [49, 398, 98, 536]
# 
# Total detections: 3
# Output saved to: assets/bus_with_detections.jpg
```

### Programmatic Usage

```rust
use runnx::{Model, Tensor};
use image::{ImageReader, DynamicImage};
use std::collections::HashMap;

fn main() -> runnx::Result<()> {
    // Load any compatible computer vision model
    let model = Model::from_onnx_file("vision_model.onnx")?;  // Works with various CV models
    println!("Model loaded: {} nodes", model.graph.nodes.len());
    
    // Load and preprocess image (example with object detection model)
    let (input_tensor, original_image, original_size) = load_and_prepare_image("assets/bus.jpg")?;
    
    // Run inference
    let mut inputs = HashMap::new();
    inputs.insert("images", input_tensor);  // Input name varies by model
    let outputs = model.run(&inputs)?;
    
    // Post-process results (varies by model type)
    let detections = post_process_outputs(&outputs, original_size, 0.25, 0.45)?;
    
    // Print results
    for (i, detection) in detections.iter().enumerate() {
        println!("Detection {}: {} ({:.1}% confidence)", 
                 i + 1, detection.class_name, detection.confidence * 100.0);
    }
    
    Ok(())
}

#[derive(Debug, Clone)]
struct Detection {
    bbox: [f32; 4], // [x1, y1, x2, y2] in image coordinates
    confidence: f32,
    class_name: String,
}

fn load_and_prepare_image(path: &str) -> runnx::Result<(Tensor, DynamicImage, (u32, u32))> {
    // Load original image
    let original_image = ImageReader::open(path)
        .map_err(|e| runnx::error::OnnxError::runtime_error(format!("Failed to open image: {e}")))?
        .decode()
        .map_err(|e| runnx::error::OnnxError::runtime_error(format!("Failed to decode image: {e}")))?;

    let original_size = (original_image.width(), original_image.height());
    
    // Resize to model input size (example: 640x640 for detection models)
    let resized = original_image.resize_exact(640, 640, image::imageops::FilterType::Lanczos3);
    let rgb_image = resized.to_rgb8();

    // Convert to tensor format [1, 3, H, W] (common CV format)
    let mut input_data = vec![0.0f32; 1 * 3 * 640 * 640];
    let (width, height) = (640, 640);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_image.get_pixel(x, y);
            let base_idx = (y * width + x) as usize;
            
            // Normalize to [0, 1] and arrange in CHW format
            input_data[base_idx] = pixel[0] as f32 / 255.0;                    // R channel
            input_data[width * height + base_idx] = pixel[1] as f32 / 255.0;   // G channel  
            input_data[2 * width * height + base_idx] = pixel[2] as f32 / 255.0; // B channel
        }
    }

    let input_tensor = Tensor::from_shape_vec(vec![1, 3, 640, 640], input_data)?;
    Ok((input_tensor, original_image, original_size))
}

fn post_process_outputs(
    outputs: &HashMap<String, Tensor>,
    original_size: (u32, u32),
    conf_threshold: f32,
    iou_threshold: f32,
) -> runnx::Result<Vec<Detection>> {
    // Note: Post-processing varies significantly by model type
    // This example shows object detection post-processing
    let output = outputs.get("output0")  // Output name varies by model
        .ok_or_else(|| runnx::error::OnnxError::runtime_error("No output found".to_string()))?;

    // Example: Object detection output format [1, 84, 8400]
    // 84 = 4 (bbox) + 80 (classes for COCO dataset)
    let data = output.data();
    let shape = output.shape();
    
    if shape.len() != 3 || shape[0] != 1 || shape[1] != 84 {
        return Err(runnx::error::OnnxError::runtime_error(
            format!("Unexpected output shape: {:?}", shape)
        ));
    }

    let num_detections = shape[2];
    let mut detections = Vec::new();

    // COCO class names (subset for brevity)
    let class_names = vec![
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        // ... (add more COCO classes as needed)
    ];

    // Extract detections
    for i in 0..num_detections {
        // Get bbox coordinates (center_x, center_y, width, height)
        let center_x = data[i];
        let center_y = data[num_detections + i];
        let width = data[2 * num_detections + i];
        let height = data[3 * num_detections + i];

        // Find the class with highest confidence
        let mut max_conf = 0.0f32;
        let mut best_class = 0;
        
        for class_id in 0..80 {
            let conf = data[(4 + class_id) * num_detections + i];
            if conf > max_conf {
                max_conf = conf;
                best_class = class_id;
            }
        }

        // Filter by confidence threshold
        if max_conf >= conf_threshold {
            // Convert to corner coordinates and scale to original image size
            let scale_x = original_size.0 as f32 / 640.0;
            let scale_y = original_size.1 as f32 / 640.0;

            let x1 = (center_x - width / 2.0) * scale_x;
            let y1 = (center_y - height / 2.0) * scale_y;
            let x2 = (center_x + width / 2.0) * scale_x;
            let y2 = (center_y + height / 2.0) * scale_y;

            let class_name = class_names.get(best_class)
                .unwrap_or(&"unknown")
                .to_string();

            detections.push(Detection {
                bbox: [x1, y1, x2, y2],
                confidence: max_conf,
                class_name,
            });
        }
    }

    // Apply Non-Maximum Suppression (simplified version)
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    
    let mut final_detections = Vec::new();
    for detection in detections {
        let mut should_keep = true;
        
        for existing in &final_detections {
            if iou(&detection.bbox, &existing.bbox) > iou_threshold {
                should_keep = false;
                break;
            }
        }
        
        if should_keep {
            final_detections.push(detection);
        }
    }

    Ok(final_detections)
}

fn iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let union = area1 + area2 - intersection;

    intersection / union
}
```

### Visualization with Bounding Boxes

To draw bounding boxes on detected objects:

```rust
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

fn draw_detections_on_image(
    mut image: DynamicImage, 
    detections: &[Detection]
) -> runnx::Result<DynamicImage> {
    let mut rgb_image = image.to_rgb8();
    
    for detection in detections {
        let bbox = &detection.bbox;
        let rect = Rect::at(bbox[0] as i32, bbox[1] as i32)
            .of_size((bbox[2] - bbox[0]) as u32, (bbox[3] - bbox[1]) as u32);
        
        // Draw bounding box in red
        draw_hollow_rect_mut(&mut rgb_image, rect, Rgb([255, 0, 0]));
        
        println!("📍 {}: {:.1}% confidence at [{:.0}, {:.0}, {:.0}, {:.0}]",
                 detection.class_name,
                 detection.confidence * 100.0,
                 bbox[0], bbox[1], bbox[2], bbox[3]);
    }
    
    Ok(DynamicImage::ImageRgb8(rgb_image))
}
```

## Format Conversion
        let x_center = data[i];
        let y_center = data[num_detections + i];
        let width = data[2 * num_detections + i];
        let height = data[3 * num_detections + i];
        
        // Convert center format to corner format
        let x1 = x_center - width / 2.0;
        let y1 = y_center - height / 2.0;
        let x2 = x_center + width / 2.0;
        let y2 = y_center + height / 2.0;
        
        // Find class with highest confidence (remaining 80 values for COCO)
        let mut max_conf = 0.0;
        let mut class_id = 0;
        
        for class_idx in 0..80 {
            let conf = data[(4 + class_idx) * num_detections + i];
            if conf > max_conf {
                max_conf = conf;
                class_id = class_idx;
            }
        }
        
        // Only keep detections above confidence threshold
        if max_conf > 0.5 {
            detections.push(Detection {
                bbox: [x1, y1, x2, y2],
                confidence: max_conf,
                class_id,
            });
        }
    }
    
    Ok(detections)
}
```

## Graph Visualization

### Terminal Visualization

```rust
use runnx::Model;

// Load model
let model = Model::from_file("model.onnx")?;

// Print beautiful terminal visualization
model.print_graph();

// Print model summary
model.print_summary();
```

### DOT Format Export

```rust
// Generate DOT format for Graphviz
let dot_content = model.to_dot()?;

// Save to file
std::fs::write("model_graph.dot", dot_content)?;

// Generate PNG using Graphviz (requires `dot` command)
use std::process::Command;
Command::new("dot")
    .args(&["-Tpng", "model_graph.dot", "-o", "model_graph.png"])
    .output()?;
```

### Custom Graph Styling

```rust
// Generate DOT with custom styling
let dot_content = model.to_dot_with_style(&GraphStyle {
    node_color: "lightblue",
    edge_color: "darkblue", 
    font_name: "Arial",
    ..Default::default()
})?;
```

## Custom Operators

### Implementing a Custom Operator

```rust
use runnx::{Operator, Tensor, OperatorError};

struct CustomReluOperator;

impl Operator for CustomReluOperator {
    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, OperatorError> {
        if inputs.len() != 1 {
            return Err(OperatorError::InvalidInputCount);
        }
        
        let input = inputs[0];
        let data = input.data();
        
        // Apply ReLU: max(0, x)
        let result_data: Vec<f32> = data.iter()
            .map(|&x| x.max(0.0))
            .collect();
        
        let result = Tensor::from_shape_vec(input.shape().to_vec(), result_data)?;
        Ok(vec![result])
    }
    
    fn operator_type(&self) -> &str {
        "CustomRelu"
    }
}
```

### Using Custom Operators

```rust
// Register custom operator
let mut model = Model::from_file("model.onnx")?;
model.register_operator("CustomRelu", Box::new(CustomReluOperator))?;

// Use in inference
let outputs = model.run(&[("input", input_tensor)])?;
```

## Async Operations

### Async Model Loading

```rust
use runnx::Model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model asynchronously
    let model = Model::from_file_async("large_model.onnx").await?;
    
    println!("Model loaded asynchronously");
    Ok(())
}
```

### Async Inference

```rust
// Run inference asynchronously
async fn run_inference_async(model: &Model, input: Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let outputs = model.run_async(&[("input", input)]).await?;
    
    for (name, tensor) in outputs.iter() {
        println!("Output '{}': {:?}", name, tensor.shape());
    }
    
    Ok(())
}
```

### Batch Processing

```rust
use futures::future::join_all;

async fn process_batch(model: &Model, inputs: Vec<Tensor>) -> Result<Vec<TensorMap>, Box<dyn std::error::Error>> {
    let futures: Vec<_> = inputs.into_iter()
        .map(|input| model.run_async(&[("input", input)]))
        .collect();
    
    let results = join_all(futures).await;
    
    // Collect successful results
    let outputs: Result<Vec<_>, _> = results.into_iter().collect();
    outputs
}
```

## Error Handling

### Comprehensive Error Handling

```rust
use runnx::{Model, Tensor, RunNXError};

fn robust_inference(model_path: &str, input_data: Vec<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Load model with error handling
    let model = match Model::from_file(model_path) {
        Ok(m) => m,
        Err(RunNXError::IoError(e)) => {
            eprintln!("Failed to load model file: {}", e);
            return Err(e.into());
        }
        Err(RunNXError::ParseError(e)) => {
            eprintln!("Failed to parse model: {}", e);
            return Err(e.into());
        }
        Err(e) => {
            eprintln!("Unexpected error loading model: {}", e);
            return Err(e.into());
        }
    };
    
    // Create tensor with validation
    let input_tensor = match Tensor::from_shape_vec(vec![1, input_data.len()], input_data) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create input tensor: {}", e);
            return Err(e.into());
        }
    };
    
    // Run inference with error handling
    let outputs = match model.run(&[("input", input_tensor)]) {
        Ok(outputs) => outputs,
        Err(RunNXError::InferenceError(e)) => {
            eprintln!("Inference failed: {}", e);
            return Err(e.into());
        }
        Err(e) => {
            eprintln!("Unexpected error during inference: {}", e);
            return Err(e.into());
        }
    };
    
    // Extract output safely
    let result = outputs.get("output")
        .ok_or("Model has no 'output' tensor")?;
    
    Ok(result.data().to_vec())
}
```

### Logging and Debugging

```rust
use log::{info, warn, error, debug};

fn debug_inference(model: &Model, input: Tensor) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting inference with input shape: {:?}", input.shape());
    
    // Enable debug logging
    env_logger::init();
    
    debug!("Input tensor stats: min={:.3}, max={:.3}, mean={:.3}", 
           input.data().iter().fold(f32::INFINITY, |a, &b| a.min(b)),
           input.data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
           input.data().iter().sum::<f32>() / input.data().len() as f32);
    
    let start = std::time::Instant::now();
    let outputs = model.run(&[("input", input)])?;
    let elapsed = start.elapsed();
    
    info!("Inference completed in {:?}", elapsed);
    
    for (name, tensor) in outputs.iter() {
        debug!("Output '{}': shape={:?}, dtype={:?}", 
               name, tensor.shape(), tensor.dtype());
    }
    
    Ok(())
}
```

## Performance Optimization

### Memory-Efficient Processing

```rust
// Process large datasets in chunks
fn process_large_dataset(model: &Model, data: &[Vec<f32>], chunk_size: usize) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    
    for chunk in data.chunks(chunk_size) {
        // Process chunk
        let batch_results = process_batch_sync(model, chunk)?;
        results.extend(batch_results);
        
        // Optional: force garbage collection
        // std::mem::drop(batch_results);
    }
    
    Ok(results)
}

fn process_batch_sync(model: &Model, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    
    for input_data in inputs {
        let tensor = Tensor::from_shape_vec(vec![1, input_data.len()], input_data.clone())?;
        let outputs = model.run(&[("input", tensor)])?;
        let result = outputs.get("output").unwrap();
        results.push(result.data().to_vec());
    }
    
    Ok(results)
}
```

## CLI Examples

### Basic CLI Usage

```bash
# Run inference
./target/release/runnx-runner --model model.onnx --input input.json

# Show model information
./target/release/runnx-runner --model model.onnx --summary

# Visualize model graph
./target/release/runnx-runner --model model.onnx --graph

# Export DOT format
./target/release/runnx-runner --model model.onnx --dot model.dot
```

### Advanced CLI Workflows

```bash
# Complete model analysis pipeline
./target/release/runnx-runner --model yolov8n.onnx --summary --graph --dot yolo.dot

# Convert DOT to various formats
dot -Tpng yolo.dot -o yolo.png
dot -Tsvg yolo.dot -o yolo.svg
dot -Tpdf yolo.dot -o yolo.pdf
```

## Next Steps

- **[Graph Visualization Guide](graph-visualization.md)** - Detailed visualization features
- **[API Reference](../api/)** - Complete API documentation
- **[Development Guide](../development/)** - Contributing and development setup
- **[Formal Verification](../FORMAL_VERIFICATION.md)** - Mathematical verification system
