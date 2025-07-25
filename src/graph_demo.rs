//! Demo program showing the graph visualization functionality

use runnx::{
    graph::{Graph, Node, TensorSpec},
    model::{Model, ModelMetadata},
    tensor::Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a more complex demo graph
    let mut graph = Graph::new("neural_network_demo".to_string());
    
    // Add inputs
    graph.add_input(TensorSpec::new("image_input".to_string(), vec![Some(1), Some(3), Some(224), Some(224)]));
    graph.add_input(TensorSpec::new("mask_input".to_string(), vec![Some(1), Some(1), Some(224), Some(224)]));
    
    // Add outputs
    graph.add_output(TensorSpec::new("classification".to_string(), vec![Some(1), Some(1000)]));
    graph.add_output(TensorSpec::new("segmentation".to_string(), vec![Some(1), Some(21), Some(224), Some(224)]));
    
    // Add some initializers (weights and biases)
    let conv_weights = Tensor::from_shape_vec(&[64, 3, 7, 7], vec![0.1; 64 * 3 * 7 * 7])?;
    let conv_bias = Tensor::from_shape_vec(&[64], vec![0.01; 64])?;
    let fc_weights = Tensor::from_shape_vec(&[1000, 512], vec![0.02; 1000 * 512])?;
    let fc_bias = Tensor::from_shape_vec(&[1000], vec![0.0; 1000])?;
    
    graph.add_initializer("conv1_weight".to_string(), conv_weights);
    graph.add_initializer("conv1_bias".to_string(), conv_bias);
    graph.add_initializer("fc_weight".to_string(), fc_weights);
    graph.add_initializer("fc_bias".to_string(), fc_bias);
    
    // Create nodes for a more complex network
    
    // Convolutional layer
    let mut conv_node = Node::new(
        "conv1".to_string(),
        "Conv".to_string(),
        vec!["image_input".to_string(), "conv1_weight".to_string(), "conv1_bias".to_string()],
        vec!["conv1_output".to_string()]
    );
    conv_node.add_attribute("kernel_shape", "[7, 7]");
    conv_node.add_attribute("strides", "[2, 2]");
    conv_node.add_attribute("pads", "[3, 3, 3, 3]");
    graph.add_node(conv_node);
    
    // ReLU activation
    let relu_node = Node::new(
        "relu1".to_string(),
        "Relu".to_string(),
        vec!["conv1_output".to_string()],
        vec!["relu1_output".to_string()]
    );
    graph.add_node(relu_node);
    
    // Max pooling
    let mut maxpool_node = Node::new(
        "maxpool1".to_string(),
        "MaxPool".to_string(),
        vec!["relu1_output".to_string()],
        vec!["maxpool1_output".to_string()]
    );
    maxpool_node.add_attribute("kernel_shape", "[3, 3]");
    maxpool_node.add_attribute("strides", "[2, 2]");
    graph.add_node(maxpool_node);
    
    // Global average pooling
    let gap_node = Node::new(
        "global_avg_pool".to_string(),
        "GlobalAveragePool".to_string(),
        vec!["maxpool1_output".to_string()],
        vec!["gap_output".to_string()]
    );
    graph.add_node(gap_node);
    
    // Flatten
    let flatten_node = Node::new(
        "flatten".to_string(),
        "Flatten".to_string(),
        vec!["gap_output".to_string()],
        vec!["flatten_output".to_string()]
    );
    graph.add_node(flatten_node);
    
    // Fully connected layer for classification
    let fc_node = Node::new(
        "fc_classifier".to_string(),
        "MatMul".to_string(),
        vec!["flatten_output".to_string(), "fc_weight".to_string()],
        vec!["fc_output".to_string()]
    );
    graph.add_node(fc_node);
    
    // Add bias
    let add_bias_node = Node::new(
        "add_bias".to_string(),
        "Add".to_string(),
        vec!["fc_output".to_string(), "fc_bias".to_string()],
        vec!["classification".to_string()]
    );
    graph.add_node(add_bias_node);
    
    // Segmentation branch (simplified)
    let mut upsample_node = Node::new(
        "upsample".to_string(),
        "Upsample".to_string(),
        vec!["maxpool1_output".to_string()],
        vec!["upsample_output".to_string()]
    );
    upsample_node.add_attribute("mode", "bilinear");
    upsample_node.add_attribute("scales", "[1.0, 1.0, 4.0, 4.0]");
    graph.add_node(upsample_node);
    
    // Mask attention
    let mask_attention_node = Node::new(
        "mask_attention".to_string(),
        "Mul".to_string(),
        vec!["upsample_output".to_string(), "mask_input".to_string()],
        vec!["masked_features".to_string()]
    );
    graph.add_node(mask_attention_node);
    
    // Final convolution for segmentation
    let seg_conv_node = Node::new(
        "seg_conv".to_string(),
        "Conv".to_string(),
        vec!["masked_features".to_string()],
        vec!["segmentation".to_string()]
    );
    graph.add_node(seg_conv_node);
    
    // Create model
    let metadata = ModelMetadata {
        name: "Complex Neural Network Demo".to_string(),
        version: "1.0".to_string(),
        description: "Demo of multi-task neural network with classification and segmentation".to_string(),
        producer: "RunNX Graph Demo".to_string(),
        onnx_version: "1.9.0".to_string(),
        domain: "ai.demo".to_string(),
    };
    
    let model = Model::with_metadata(metadata, graph);
    
    println!("=== Complex Neural Network Demo ===\n");
    
    // Show model summary
    println!("📋 MODEL SUMMARY:");
    println!("{}", model.summary());
    
    // Show graph visualization
    println!("🎨 GRAPH VISUALIZATION:");
    model.print_graph();
    
    // Save DOT format
    let dot_content = model.to_dot();
    std::fs::write("complex_graph.dot", &dot_content)?;
    println!("💾 DOT file saved to: complex_graph.dot");
    println!("   Generate PNG with: dot -Tpng complex_graph.dot -o complex_graph.png");
    
    Ok(())
}
