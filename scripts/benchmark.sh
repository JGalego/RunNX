#!/bin/bash

# RunNX Performance Benchmark Suite
# This script runs all benchmarks and generates a performance report

set -e

echo "🚀 RunNX Performance Benchmark Suite"
echo "======================================"

# Set benchmark configuration
export CRITERION_HOME="target/criterion"
mkdir -p target/criterion

echo ""
echo "📊 Running Tensor Operations Benchmarks..."
echo "-------------------------------------------"
cargo bench --bench inference_benchmark

echo ""
echo "🔧 Running ONNX Operator Benchmarks..."
echo "---------------------------------------"
cargo bench --bench operator_benchmark

echo ""
echo "🎯 Running Computer Vision Benchmarks..."
echo "-----------------------------------------"
cargo bench --bench cv_benchmark

echo ""
echo "📈 Benchmark Summary"
echo "===================="
echo "All benchmarks completed successfully!"
echo ""
echo "📁 Detailed reports available in: target/criterion/"
echo "🌐 Open target/criterion/report/index.html for detailed analysis"
echo ""

# Check if criterion reports exist and provide links
if [ -d "target/criterion/report" ]; then
    echo "📊 Available benchmark reports:"
    echo "  - Tensor Operations: target/criterion/tensor_operations/report/index.html"
    echo "  - Activation Functions: target/criterion/activation_functions/report/index.html"
    echo "  - Tensor Creation: target/criterion/tensor_creation/report/index.html"
    echo "  - Model Inference: target/criterion/model_inference/report/index.html"
    echo "  - Basic Operators: target/criterion/basic_operators/report/index.html"
    echo "  - Computer Vision: target/criterion/computer_vision_operators/report/index.html"
    echo "  - YOLO Post-processing: target/criterion/yolo_postprocessing/report/index.html"
fi

echo ""
echo "✅ Benchmark suite completed!"
