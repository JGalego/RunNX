# Quick Start Guide

Get up and running with RunNX in just a few minutes! This guide will walk you through the basics of using RunNX for ONNX model inference.

## Prerequisites

RunNX requires the Protocol Buffers compiler (`protoc`) to build. Install it for your platform:

### Ubuntu/Debian
```bash
sudo apt-get install protobuf-compiler
```

### macOS
```bash
brew install protobuf
```

### Windows
```bash
choco install protoc
```

## Installation

### As a Library

Add RunNX to your `Cargo.toml`:

```toml
[dependencies]
runnx = "0.2.0"
```

### From Source

```bash
git clone https://github.com/jgalego/runnx.git
cd runnx
cargo build --release
```

## Basic Usage

### Loading and Running Models

```rust
use runnx::{Model, Tensor};

// Load a model (supports both JSON and ONNX binary formats)
let model = Model::from_file("model.onnx")?;  // Auto-detects format

// Create input tensor
let input = Tensor::from_array(ndarray::array![[1.0, 2.0, 3.0]]);

// Run inference
let outputs = model.run(&[("input", input)])?;

// Get results
let result = outputs.get("output").unwrap();
println!("Result: {:?}", result.data());
```

### Format Support

RunNX supports both JSON and ONNX binary formats with automatic detection:

```rust
// Auto-detection based on file extension
let json_model = Model::from_file("model.json")?;   // JSON format
let onnx_model = Model::from_file("model.onnx")?;   // Binary ONNX

// Explicit format specification
let json_model = Model::from_json_file("model.json")?;
let onnx_model = Model::from_onnx_file("model.onnx")?;
```

### Saving Models

```rust
let model = /* ... create or load model ... */;

// Save in different formats
model.to_file("output.onnx")?;        // Auto-detects format from extension
model.to_onnx_file("binary.onnx")?;   // Explicit binary ONNX format
model.to_json_file("readable.json")?; // Explicit JSON format
```

## Command Line Usage

RunNX includes a command-line runner for quick model testing:

# Quick Start Guide

Get up and running with RunNX in just a few minutes! This guide will walk you through the basics of using RunNX for ONNX model inference, including the powerful YOLOv8 object detection capabilities.

## Prerequisites

RunNX requires the Protocol Buffers compiler (`protoc`) to build. Install it for your platform:

### Ubuntu/Debian
```bash
sudo apt-get install protobuf-compiler
```

### macOS
```bash
brew install protobuf
```

### Windows
```bash
choco install protoc
```

## Installation

### As a Library

Add RunNX to your `Cargo.toml`:

```toml
[dependencies]
runnx = "0.2.0"
```

### From Source

```bash
git clone https://github.com/jgalego/runnx.git
cd runnx
cargo build --release
```

## Basic Usage

### Loading and Running Models

```rust
use runnx::{Model, Tensor};

// Load a model (supports both JSON and ONNX binary formats)
let model = Model::from_file("model.onnx")?;  // Auto-detects format

// Create input tensor
let input = Tensor::from_array(ndarray::array![[1.0, 2.0, 3.0]]);

// Run inference
let outputs = model.run(&[("input", input)])?;

// Get results
let result = outputs.get("output").unwrap();
println!("Result: {:?}", result.data());
```

### Computer Vision Example

RunNX supports various computer vision models including classification, object detection, and segmentation:

```bash
# Example: Object detection with YOLOv8 (one of many supported model types)
# Download a sample model (optional)
wget https://huggingface.co/unity/inference-engine-yolo/resolve/ed7f4daf9263d0d31be1d60b9d67c8baea721d60/yolov8n.onnx

# Run object detection example
cargo run --example yolov8_detect_and_draw

# Expected workflow:
# 1. Model loaded successfully: 224 nodes
# 2. Preprocessing image: assets/bus.jpg -> [1, 3, 640, 640]
# 3. Running inference...
# 4. Inference completed in ~45ms
# 5. Detection results with bounding boxes
# 6. Output saved to: assets/bus_with_detections.jpg
```

**What this demonstrates:**
- ✅ **Model compatibility**: Works with various ONNX model architectures
- ✅ **Complete pipeline**: Load, preprocess, infer, post-process, visualize  
- ✅ **Performance**: Real-time inference with timing information
- ✅ **Production ready**: Comprehensive error handling and logging

### Format Support

RunNX supports both JSON and ONNX binary formats with automatic detection:

```rust
// Auto-detection based on file extension
let json_model = Model::from_file("model.json")?;   // JSON format
let onnx_model = Model::from_file("model.onnx")?;   // Binary ONNX

// Explicit format specification
let json_model = Model::from_json_file("model.json")?;
let onnx_model = Model::from_onnx_file("model.onnx")?;
```

### Saving Models

```rust
let model = /* ... create or load model ... */;

// Save in different formats
model.to_file("output.onnx")?;        // Auto-detects format from extension
model.to_onnx_file("binary.onnx")?;   // Explicit binary ONNX format
model.to_json_file("readable.json")?; // Explicit JSON format
```

## Command Line Usage

RunNX includes a command-line runner for quick model testing:

### Basic Inference

```bash
# Run inference (supports both .onnx and .json files)
cargo run --bin runnx-runner -- --model model.onnx --input input.json
cargo run --bin runnx-runner -- --model model.json --input input.json
```

### Model Visualization

```bash
# Show model summary and terminal graph visualization
cargo run --bin runnx-runner -- --model model.onnx --summary --graph

# Generate Graphviz DOT file for professional diagrams
cargo run --bin runnx-runner -- --model model.onnx --dot graph.dot
```

### Async Support

```bash
# Run with async support enabled
cargo run --features async --bin runnx-runner -- --model model.onnx --input input.json
```

## Example: YOLO Object Detection

RunNX supports essential operators for YOLO models:

```rust
use runnx::{Model, Tensor};

// Load YOLO model
let model = Model::from_file("yolov8n.onnx")?;

// Prepare image input (1x3x640x640 for YOLOv8)
let image_tensor = Tensor::from_array(image_array);

// Run detection
let outputs = model.run(&[("images", image_tensor)])?;

// Post-process results
let detections = outputs.get("output0").unwrap();
println!("Detections shape: {:?}", detections.shape());
```

## Next Steps

Now that you have RunNX running, explore these guides:

- **[Installation Guide](installation.md)** - Detailed installation for all platforms
- **[Usage Examples](examples.md)** - Comprehensive examples and tutorials
- **[Graph Visualization Guide](graph-visualization.md)** - Learn about model visualization
- **[API Reference](../api/)** - Detailed API documentation

## Getting Help

- **Documentation**: Browse the [docs/](../README.md) directory
- **Examples**: Check out working examples in [examples/](../../examples/)
- **Issues**: Report problems on [GitHub Issues](https://github.com/jgalego/runnx/issues)
- **API Docs**: Run `cargo doc --open` for detailed API documentation

## What's Next?

- Explore [graph visualization capabilities](graph-visualization.md)
- Learn about [formal verification](../FORMAL_VERIFICATION.md)
- Check out [YOLO examples](examples.md#yolo-object-detection)
- Read about [format compatibility](../FORMAT_COMPATIBILITY.md)
