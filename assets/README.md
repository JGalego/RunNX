# Assets

This directory contains static assets for the RunNX project documentation.

## Files

### Images
- **`runnx.jpg`** - Main RunNX logo and branding image used in README.md
- **`complex_graph.png`** - Example graph visualization showing a multi-task neural network with classification and segmentation branches

### Graph Data
- **`complex_graph.dot`** - Example Graphviz DOT format file demonstrating the structured graph data format used for generating professional diagrams

## Usage

These assets are referenced in the project documentation:

- The logo is displayed at the top of the main README
- The graph visualization serves as an example of RunNX's graph visualization capabilities
- The DOT file provides a reference implementation for users generating their own graph exports

## Generating New Assets

To regenerate the graph assets:

```bash
# Generate DOT file from a model
./target/debug/runnx-runner --model model.onnx --dot assets/graph.dot

# Convert DOT to PNG using Graphviz
dot -Tpng assets/graph.dot -o assets/graph.png

# Convert DOT to SVG for vector graphics
dot -Tsvg assets/graph.dot -o assets/graph.svg
```

## File Formats

- **JPG**: Compressed raster format for photos and complex images
- **PNG**: Lossless raster format for diagrams and images with transparency
- **DOT**: Graphviz graph description language for generating diagrams
