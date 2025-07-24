# Format Compatibility Guide

RunNX supports both human-readable JSON format and the standard binary ONNX protobuf format, providing flexibility for different use cases.

## Supported Formats

### JSON Format (.json)
- **Purpose**: Human-readable format for debugging and inspection
- **Advantages**: 
  - Easy to read and modify manually
  - Version control friendly (text diffs)
  - No binary dependencies for viewing
- **Disadvantages**:
  - Larger file sizes
  - Slower parsing for large models

### Binary ONNX Format (.onnx) 
- **Purpose**: Standard ONNX protobuf format for production use
- **Advantages**:
  - Compact binary representation
  - Faster loading and parsing
  - Interoperable with other ONNX runtimes
  - Industry standard format
- **Disadvantages**:
  - Not human-readable
  - Requires tools to inspect contents

## File Format Auto-Detection

RunNX automatically detects the format based on file extension:

```rust
use runnx::Model;

// Auto-detection based on extension
let json_model = Model::from_file("model.json")?;  // → JSON format
let onnx_model = Model::from_file("model.onnx")?;  // → Binary ONNX format
```

For explicit format control:

```rust
// Explicit format specification
let json_model = Model::from_json_file("model.json")?;
let onnx_model = Model::from_onnx_file("model.onnx")?;
```

## Converting Between Formats

### JSON to ONNX
```rust
use runnx::Model;

// Load from JSON
let model = Model::from_json_file("model.json")?;

// Save as binary ONNX
model.to_onnx_file("model.onnx")?;
```

### ONNX to JSON
```rust
use runnx::Model;

// Load from binary ONNX  
let model = Model::from_onnx_file("model.onnx")?;

// Save as JSON
model.to_json_file("model.json")?;
```

### Auto-save with Extension Detection
```rust
use runnx::Model;

let model = /* ... your model ... */;

// Format determined by file extension
model.to_file("output.json")?;  // → JSON format
model.to_file("output.onnx")?;  // → Binary ONNX format
```

## File Size Comparison

The binary ONNX format is typically more compact:

| Model Size | JSON Format | ONNX Binary | Compression Ratio |
|------------|-------------|-------------|-------------------|
| Small      | ~500 bytes  | ~200 bytes  | ~60% smaller      |
| Medium     | ~50 KB      | ~20 KB      | ~60% smaller      |
| Large      | ~5 MB       | ~2 MB       | ~60% smaller      |

*Actual compression varies based on model complexity and tensor data.*

## Performance Characteristics

### Loading Performance
- **ONNX Binary**: Faster parsing due to binary format
- **JSON**: Slower parsing due to text deserialization

### Memory Usage
- **ONNX Binary**: Lower memory overhead during parsing
- **JSON**: Higher memory usage due to intermediate string representations

## Compatibility Matrix

| Operation | JSON Format | ONNX Binary | Notes |
|-----------|-------------|-------------|-------|
| Model Loading | ✅ | ✅ | Both support all operators |
| Model Saving | ✅ | ✅ | Lossless conversion |
| Auto-detection | ✅ | ✅ | Based on file extension |
| Manual Format | ✅ | ✅ | Explicit format methods |
| Interoperability | ❌ | ✅ | ONNX binary works with other runtimes |

## Best Practices

### Use JSON Format When:
- 🔍 **Debugging models**: Need to inspect model structure manually
- 📝 **Development**: Iterating on model definitions
- 🔄 **Version Control**: Want meaningful diffs in git
- 📚 **Documentation**: Including model examples in docs

### Use ONNX Binary When:
- 🚀 **Production**: Deploying models for inference
- 📦 **Distribution**: Sharing models with other tools/runtimes
- 💾 **Storage**: Minimizing file size and bandwidth
- ⚡ **Performance**: Requiring fast model loading

## Migration Guide

### From JSON-only to Dual Format

If you have existing code using only JSON format:

```rust
// Old code (JSON only)
let model = Model::from_json_file("model.json")?;

// New code (format auto-detection)
let model = Model::from_file("model.json")?;  // Still works!
let model = Model::from_file("model.onnx")?;  // Now also works!
```

### Batch Conversion Script

Convert multiple models:

```bash
# Convert all JSON models to ONNX binary
cargo run --example convert_formats -- --input-dir ./json_models --output-dir ./onnx_models --format onnx

# Convert all ONNX models to JSON  
cargo run --example convert_formats -- --input-dir ./onnx_models --output-dir ./json_models --format json
```

## Troubleshooting

### Common Issues

**Q: "Format detection failed"**
- Ensure file has correct extension (`.json` or `.onnx`)
- For non-standard extensions, use explicit format methods

**Q: "ONNX parsing error"** 
- Verify file is valid ONNX protobuf format
- Try opening with explicit `from_onnx_file()` for better error messages

**Q: "JSON parsing error"**
- Check JSON syntax validity
- Ensure all required fields are present
- Use JSON validator tools if needed

**Q: "File size too large"**
- Consider using ONNX binary format for smaller files
- Check if model has unnecessary large tensors

### Getting Help

For format-specific issues:
1. Check the error message for specific parsing failures
2. Try loading with explicit format methods for detailed diagnostics
3. Validate your files with external tools (ONNX checker, JSON validator)
4. Create minimal reproduction cases

## Examples

See the following examples for practical usage:
- `examples/onnx_demo.rs` - Comprehensive format compatibility demo
- `examples/simple_model.rs` - Basic model operations
- `examples/format_conversion.rs` - Converting between formats
