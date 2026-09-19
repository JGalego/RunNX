# Command Line Interface (CLI) Reference

RunNX provides a powerful command-line interface through the `runnx-runner` binary. This guide covers all CLI options and usage patterns.

## Installation

The CLI is available through several methods:

```bash
# Install from crates.io
cargo install runnx

# Build from source
git clone https://github.com/jgalego/runnx.git
cd runnx
cargo build --release
# Binary available at: ./target/release/runnx-runner

# Development build
cargo build
# Binary available at: ./target/debug/runnx-runner
```

## Basic Syntax

```bash
runnx-runner [OPTIONS] --model <MODEL_PATH>
```

## Options Reference

### Required Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model <PATH>` | Path to the ONNX model file (`.onnx` or `.json`) | `--model yolov8n.onnx` |

### Input/Output Options

| Option | Description | Example |
|--------|-------------|---------|
| `--input <PATH>` | Path to input data JSON file | `--input input.json` |
| `--output <PATH>` | Path to save output results | `--output results.json` |

### Visualization Options

| Option | Description | Example |
|--------|-------------|---------|
| `--summary` | Display model summary information | `--summary` |
| `--graph` | Show terminal graph visualization | `--graph` |
| `--dot <PATH>` | Export Graphviz DOT format to file | `--dot model.dot` |

### Runtime Options

| Option | Description | Example |
|--------|-------------|---------|
| `--verbose` | Enable verbose logging | `--verbose` |

### Help Options

| Option | Description |
|--------|-------------|
| `--help` | Show help information |
| `--version` | Show version information |

## Usage Examples

### Basic Model Inference

```bash
# Run inference with ONNX binary model
runnx-runner --model model.onnx --input input.json

# Run inference with JSON model
runnx-runner --model model.json --input input.json

# Save output to file
runnx-runner --model model.onnx --input input.json --output results.json
```

### Model Analysis

```bash
# Show model summary
runnx-runner --model model.onnx --summary

# Show terminal graph visualization
runnx-runner --model model.onnx --graph

# Combined summary and graph
runnx-runner --model model.onnx --summary --graph
```

### Graph Export

```bash
# Export DOT format
runnx-runner --model model.onnx --dot model.dot

# Export and show summary
runnx-runner --model model.onnx --summary --dot model.dot

# Complete analysis with all outputs
runnx-runner --model model.onnx --summary --graph --dot model.dot
```

### Advanced Usage

```bash
# Verbose logging
runnx-runner --model model.onnx --input input.json --verbose

# Build the library with its asynchronous API enabled
cargo build --features async
```

## Input File Format

### Input JSON Structure

The input file should contain tensor data in JSON format:

```json
{
  "input_name": {
    "shape": [1, 3, 224, 224],
    "data": [0.1, 0.2, 0.3, ...]
  },
  "another_input": {
    "shape": [1, 100],
    "data": [1.0, 2.0, 3.0, ...]
  }
}
```

### Example Input Files

**Simple classification input:**
```json
{
  "input": {
    "shape": [1, 784],
    "data": [0.0, 0.1, 0.2, 0.3, ...]
  }
}
```

**YOLO object detection input:**
```json
{
  "images": {
    "shape": [1, 3, 640, 640],
    "data": [0.485, 0.456, 0.406, ...]
  }
}
```

**Multi-input model:**
```json
{
  "image": {
    "shape": [1, 3, 224, 224],
    "data": [...]
  },
  "metadata": {
    "shape": [1, 10],
    "data": [1.0, 0.0, 1.0, ...]
  }
}
```

## Output File Format

### Output JSON Structure

Results are saved in a structured JSON format:

```json
{
  "outputs": {
    "output_name": {
      "shape": [1, 1000],
      "data": [0.001, 0.002, 0.997, ...]
    }
  },
  "metadata": {
    "model": "model.onnx",
    "inference_time_ms": 45.2,
    "timestamp": "2025-08-13T10:30:00Z"
  }
}
```

### Example Outputs

**Classification result:**
```json
{
  "outputs": {
    "probabilities": {
      "shape": [1, 1000],
      "data": [0.001, 0.002, 0.997, ...]
    }
  },
  "metadata": {
    "model": "resnet50.onnx",
    "inference_time_ms": 23.4,
    "top_class": 285,
    "confidence": 0.997
  }
}
```

**YOLO detection result:**
```json
{
  "outputs": {
    "output0": {
      "shape": [1, 84, 8400],
      "data": [...]
    }
  },
  "metadata": {
    "model": "yolov8n.onnx",
    "inference_time_ms": 45.2,
    "detections_found": 5
  }
}
```

## Model Summary Output

When using `--summary`, the CLI displays comprehensive model information:

```
┌─────────────────────────────────────────┐
│          MODEL SUMMARY: yolov8n         │
└─────────────────────────────────────────┘

📊 GENERAL INFORMATION:
   ├─ Format: ONNX Binary
   ├─ IR Version: 8
   ├─ Producer: pytorch
   └─ Model Version: 1.0

📥 INPUTS (1):
   └─ images: [1 × 3 × 640 × 640] (float32)

📤 OUTPUTS (1):
   └─ output0: [1 × 84 × 8400] (float32)

⚙️  OPERATORS (22 unique types):
   ├─ Conv: 22 instances
   ├─ Relu: 20 instances
   ├─ MaxPool: 3 instances
   ├─ Concat: 6 instances
   ├─ Upsample: 2 instances
   └─ ... (17 more types)

📊 STATISTICS:
   ├─ Total nodes: 168
   ├─ Parameters: 3,151,904
   ├─ Model size: ~12.4 MB
   └─ Estimated FLOPS: 8.7 G
```

## Graph Visualization Output

Terminal graph visualization shows the model structure:

```
┌────────────────────────────────────────┐
│         GRAPH: yolov8n_detection       │
└────────────────────────────────────────┘

🔄 COMPUTATION FLOW:
   │
   ├─ Step 1: backbone_stem
   │  ┌─ Operation: Conv
   │  ├─ Inputs: images, conv1_weight, conv1_bias
   │  ├─ Outputs: conv1_output
   │  └─ Attributes: kernel_shape=[6,6], strides=[2,2]
   │
   ├─ Step 2: backbone_block1
   │  ┌─ Operation: Conv
   │  └─ Feature extraction: 320×320×32
   ...
```

## Workflows and Pipelines

### Model Development Workflow

```bash
#!/bin/bash
# Model development pipeline

MODEL_PATH="model.onnx"
INPUT_PATH="test_input.json"

echo "=== Model Analysis ==="
runnx-runner --model "$MODEL_PATH" --summary

echo "=== Graph Visualization ==="
runnx-runner --model "$MODEL_PATH" --graph

echo "=== Test Inference ==="
runnx-runner --model "$MODEL_PATH" --input "$INPUT_PATH" --verbose

echo "=== Export Documentation ==="
runnx-runner --model "$MODEL_PATH" --dot "docs/model_graph.dot"
```

### Batch Processing

```bash
#!/bin/bash
# Batch inference script

MODEL="model.onnx"
INPUT_DIR="inputs/"
OUTPUT_DIR="outputs/"

mkdir -p "$OUTPUT_DIR"

for input_file in "$INPUT_DIR"*.json; do
    filename=$(basename "$input_file" .json)
    echo "Processing: $filename"
    
    runnx-runner \
        --model "$MODEL" \
        --input "$input_file" \
      --output "$OUTPUT_DIR/${filename}_result.json" \
      > /dev/null
done

echo "Batch processing complete"
```

### Model Validation

```bash
#!/bin/bash
# Model validation script

validate_model() {
    local model_path=$1
    
    echo "Validating: $model_path"
    
    # Check model loads successfully
    if runnx-runner --model "$model_path" --summary > /dev/null 2>&1; then
        echo "✅ Model loads successfully"
    else
        echo "❌ Model failed to load"
        return 1
    fi
    
    # Check graph structure
    if runnx-runner --model "$model_path" --graph > /dev/null 2>&1; then
        echo "✅ Graph visualization works"
    else
        echo "⚠️  Graph visualization failed"
    fi
    
    echo "Model validation complete"
}

# Validate all models in directory
for model in models/*.onnx; do
    validate_model "$model"
    echo "---"
done
```

## Error Handling

### Common CLI Errors

**Model not found:**
```
Error: Failed to load model: No such file or directory (os error 2)
```
*Solution: Check the model path is correct*

**Invalid input format:**
```
Error: Failed to parse input JSON: expected value at line 1 column 1
```
*Solution: Verify input JSON syntax*

**Missing required input:**
```
Error: Model requires input 'images' but none provided
```
*Solution: Add required input to JSON file*

**Output permission denied:**
```
Error: Permission denied (os error 13)
```
*Solution: Check output directory permissions*

### Debugging with Verbose Mode

```bash
# Enable detailed logging
RUST_LOG=debug runnx-runner --model model.onnx --input input.json --verbose
```

This shows:
- Model loading details
- Tensor shape information
- Operator execution steps
- Performance timing
- Memory usage

## Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/model-test.yml
name: Model Testing

on: [push, pull_request]

jobs:
  test-models:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          
      - name: Install dependencies
        run: sudo apt-get install protobuf-compiler
        
      - name: Build RunNX
        run: cargo build --release
        
      - name: Test model loading
        run: |
          ./target/release/runnx-runner --model models/test.onnx --summary
          
      - name: Validate model inference
        run: |
          ./target/release/runnx-runner --model models/test.onnx --input test_data/input.json
```

### Docker Integration

```dockerfile
FROM rust:1.70 as builder

# Install protobuf
RUN apt-get update && apt-get install -y protobuf-compiler

# Build RunNX
COPY . /app
WORKDIR /app
RUN cargo build --release

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/runnx-runner /usr/local/bin/

ENTRYPOINT ["runnx-runner"]
```

### Python Integration

```python
import subprocess
import json

def run_inference(model_path, input_data):
    """Run RunNX inference from Python"""
    
    # Prepare input file
    with open("temp_input.json", "w") as f:
        json.dump(input_data, f)
    
    # Run inference
    result = subprocess.run([
        "runnx-runner",
        "--model", model_path,
        "--input", "temp_input.json",
        "--output", "temp_output.json"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed: {result.stderr}")
    
    # Load results
    with open("temp_output.json", "r") as f:
        return json.load(f)

# Example usage
input_data = {
    "images": {
        "shape": [1, 3, 640, 640],
        "data": [0.5] * (3 * 640 * 640)
    }
}

results = run_inference("yolov8n.onnx", input_data)
print(f"Inference completed: {results['metadata']['inference_time_ms']}ms")
```

## Performance Tips

### Optimization Flags

```bash
# Release build for better performance
cargo build --release

# Use release binary
./target/release/runnx-runner --model model.onnx --input input.json

# Build without optional execution backends
cargo build --release --no-default-features
```

### Large Model Handling

```bash
# For very large models, increase stack size
RUST_MIN_STACK=16777216 runnx-runner --model large_model.onnx --summary

# Redirect progress output in automation
runnx-runner --model model.onnx --input input.json --output results.json > /dev/null
```

## Next Steps

- **[Usage Examples](examples.md)** - Comprehensive code examples
- **[API Reference](../api/)** - Library API documentation
- **[Development Guide](../development/)** - Contributing to RunNX
- **[Graph Visualization](graph-visualization.md)** - Detailed visualization guide
