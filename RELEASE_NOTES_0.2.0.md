# RunNX 0.2.0 Release Notes

*Released on July 25, 2025*

We're excited to announce RunNX 0.2.0, a major release that brings comprehensive graph visualization capabilities and enhanced YOLO model support to the RunNX ONNX runtime.

## 🎉 What's New

### 📊 Complete Graph Visualization System

The highlight of this release is a comprehensive graph visualization system that makes understanding ONNX model structures easier than ever:

**Terminal Visualization**
- Beautiful ASCII art representations with Unicode symbols
- Dynamic title box sizing that adapts to any graph name length
- Rich information display including shapes, data types, and statistics
- Topological sorting showing correct execution order
- Robust cycle detection for complex graphs

**Professional Diagram Export**
- Graphviz DOT format export for publication-quality diagrams
- Support for PNG, SVG, and PDF output formats
- Color-coded visual elements for different node types
- Professional layouts suitable for papers and presentations

**CLI Integration**
- `--graph` flag for terminal visualization
- `--dot` flag for DOT file export
- Seamless workflow from model loading to visualization

### 🛠️ Enhanced YOLO Support

Complete operator support for YOLO-style object detection models:
- **Concat**: Tensor concatenation for feature fusion
- **Slice**: Tensor slicing operations
- **Upsample**: Feature map upsampling for FPN
- **MaxPool**: Max pooling operations
- **Softmax**: Classification probability computation
- **NonMaxSuppression**: Object detection post-processing

All YOLO operators include formal verification contracts ensuring mathematical correctness.

### 🗂️ Project Organization

- **Asset Management**: Dedicated `assets/` folder for clean project structure
- **Documentation**: Comprehensive examples and usage guides
- **Reference Materials**: Complete DOT format examples and explanations

## 🚀 Key Improvements

- **Visualization Quality**: Professional-grade diagrams suitable for academic papers
- **Developer Experience**: Rich terminal output makes debugging models intuitive
- **Code Organization**: Clean separation of concerns with dedicated graph module
- **Documentation**: Extensive examples covering all visualization capabilities
- **Testing**: Complete test coverage for new functionality

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
runnx = "0.2.0"
```

## 🎯 Quick Start with Visualization

```bash
# Terminal visualization
cargo run --bin runnx-runner -- --model model.onnx --graph

# Professional diagrams
cargo run --bin runnx-runner -- --model model.onnx --dot graph.dot
dot -Tpng graph.dot -o diagram.png
```

## 📊 Example Output

The new visualization system generates beautiful terminal output and professional diagrams showing:
- Input/output tensors with shapes and types
- Computation flow with step-by-step execution
- Operation summaries and statistics
- Color-coded DOT graphs for publications

## 🔧 Migration from 0.1.x

This release is fully backward compatible. Existing code continues to work unchanged, with new visualization capabilities available as opt-in features through CLI flags or programmatic methods.

## 🙏 Acknowledgments

Special thanks to all contributors and users who provided feedback that shaped this release. The graph visualization system was designed based on real-world needs for understanding and debugging ONNX models.

## 📋 Full Changelog

For complete details, see [CHANGELOG.md](CHANGELOG.md).

## � Release History

- **Current**: [v0.2.0](RELEASE_NOTES_0.2.0.md) - Graph Visualization & YOLO Support  
- **Previous**: [v0.1.1](docs/releases/RELEASE_NOTES_0.1.1.md) - ONNX Binary Format & Formal Verification
- **All Releases**: [docs/releases/](docs/releases/) - Complete release history

## �🔗 Links

- [Documentation](https://docs.rs/runnx)
- [Repository](https://github.com/JGalego/runnx)
- [Issues](https://github.com/JGalego/runnx/issues)
- [Crates.io](https://crates.io/crates/runnx)

---

**Previous Release**: [0.1.1](docs/releases/RELEASE_NOTES_0.1.1.md) | **Next Release**: TBD
