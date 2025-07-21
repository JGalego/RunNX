//! Computational graph representation
//!
//! This module defines the graph structure for ONNX models, including
//! nodes, edges, and the overall graph representation.

use crate::{error::{OnnxError, Result}, operators::OperatorType, tensor::Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in the computational graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier for the node
    pub name: String,
    /// Type of operation this node performs
    pub op_type: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Node attributes (parameters)
    pub attributes: HashMap<String, String>,
}

impl Node {
    /// Create a new node
    pub fn new(
        name: String,
        op_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            name,
            op_type,
            inputs,
            outputs,
            attributes: HashMap::new(),
        }
    }

    /// Add an attribute to the node
    pub fn add_attribute<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Get the operator type as enum
    pub fn get_operator_type(&self) -> Result<OperatorType> {
        OperatorType::from_str(&self.op_type)
    }
}

/// Represents the computational graph of an ONNX model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// Graph name
    pub name: String,
    /// List of nodes in execution order
    pub nodes: Vec<Node>,
    /// Input tensor specifications
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications
    pub outputs: Vec<TensorSpec>,
    /// Initial values for parameters/constants
    pub initializers: HashMap<String, Tensor>,
}

/// Tensor specification with name and shape information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Name of the tensor
    pub name: String,
    /// Shape of the tensor (None for dynamic dimensions)
    pub shape: Vec<Option<usize>>,
    /// Data type (simplified to f32 for this implementation)
    pub dtype: String,
}

impl TensorSpec {
    /// Create a new tensor specification
    pub fn new(name: String, shape: Vec<Option<usize>>) -> Self {
        Self {
            name,
            shape,
            dtype: "float32".to_string(),
        }
    }

    /// Check if the tensor spec matches a given tensor
    pub fn matches_tensor(&self, tensor: &Tensor) -> bool {
        let tensor_shape = tensor.shape();
        
        if self.shape.len() != tensor_shape.len() {
            return false;
        }

        for (spec_dim, &tensor_dim) in self.shape.iter().zip(tensor_shape.iter()) {
            match spec_dim {
                Some(expected) => {
                    if *expected != tensor_dim {
                        return false;
                    }
                }
                None => {
                    // Dynamic dimension, any size is acceptable
                    continue;
                }
            }
        }

        true
    }
}

impl Graph {
    /// Create a new empty graph
    pub fn new(name: String) -> Self {
        Self {
            name,
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    /// Add an input specification
    pub fn add_input(&mut self, input_spec: TensorSpec) {
        self.inputs.push(input_spec);
    }

    /// Add an output specification
    pub fn add_output(&mut self, output_spec: TensorSpec) {
        self.outputs.push(output_spec);
    }

    /// Add an initializer (constant tensor)
    pub fn add_initializer(&mut self, name: String, tensor: Tensor) {
        self.initializers.insert(name, tensor);
    }

    /// Get input tensor names
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(|spec| spec.name.as_str()).collect()
    }

    /// Get output tensor names
    pub fn output_names(&self) -> Vec<&str> {
        self.outputs.iter().map(|spec| spec.name.as_str()).collect()
    }

    /// Validate the graph structure
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate node names
        let mut node_names = std::collections::HashSet::new();
        for node in &self.nodes {
            if !node_names.insert(&node.name) {
                return Err(OnnxError::graph_validation_error(format!(
                    "Duplicate node name: {}",
                    node.name
                )));
            }
        }

        // Check that all node inputs/outputs are valid tensor names
        let mut available_tensors = std::collections::HashSet::new();
        
        // Add input tensors
        for input in &self.inputs {
            available_tensors.insert(&input.name);
        }
        
        // Add initializer tensors
        for name in self.initializers.keys() {
            available_tensors.insert(name);
        }

        // Process nodes in order
        for node in &self.nodes {
            // Check that all inputs are available
            for input_name in &node.inputs {
                if !available_tensors.contains(input_name) {
                    return Err(OnnxError::graph_validation_error(format!(
                        "Node '{}' references unknown input tensor '{}'",
                        node.name, input_name
                    )));
                }
            }

            // Add outputs to available tensors
            for output_name in &node.outputs {
                available_tensors.insert(output_name);
            }

            // Validate operator type
            node.get_operator_type().map_err(|e| {
                OnnxError::graph_validation_error(format!(
                    "Node '{}' has invalid operator type '{}': {}",
                    node.name, node.op_type, e
                ))
            })?;
        }

        // Check that all outputs are available
        for output in &self.outputs {
            if !available_tensors.contains(&output.name) {
                return Err(OnnxError::graph_validation_error(format!(
                    "Graph output '{}' is not produced by any node",
                    output.name
                )));
            }
        }

        Ok(())
    }

    /// Perform topological sort to get execution order
    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0; n];
        let mut adjacency_list: Vec<Vec<usize>> = vec![vec![]; n];

        // Build adjacency list and in-degree count
        for (i, node) in self.nodes.iter().enumerate() {
            for output in &node.outputs {
                for (j, other_node) in self.nodes.iter().enumerate() {
                    if i != j && other_node.inputs.contains(output) {
                        adjacency_list[i].push(j);
                        in_degree[j] += 1;
                    }
                }
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut result = Vec::new();

        while let Some(current) = queue.pop() {
            result.push(current);

            for &neighbor in &adjacency_list[current] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push(neighbor);
                }
            }
        }

        if result.len() != n {
            return Err(OnnxError::graph_validation_error(
                "Graph contains cycles".to_string(),
            ));
        }

        Ok(result)
    }

    /// Create a simple linear graph for testing
    pub fn create_simple_linear() -> Self {
        let mut graph = Graph::new("simple_linear".to_string());

        // Add inputs
        graph.add_input(TensorSpec::new("input".to_string(), vec![Some(1), Some(3)]));
        
        // Add outputs
        graph.add_output(TensorSpec::new("output".to_string(), vec![Some(1), Some(2)]));

        // Add weight initializer
        let weights = Tensor::from_shape_vec(&[3, 2], vec![0.5, 0.3, 0.2, 0.4, 0.1, 0.6]).unwrap();
        let bias = Tensor::from_shape_vec(&[1, 2], vec![0.1, 0.2]).unwrap();
        
        graph.add_initializer("weights".to_string(), weights);
        graph.add_initializer("bias".to_string(), bias);

        // Add MatMul node
        let matmul_node = Node::new(
            "matmul".to_string(),
            "MatMul".to_string(),
            vec!["input".to_string(), "weights".to_string()],
            vec!["matmul_output".to_string()],
        );
        graph.add_node(matmul_node);

        // Add Add node (bias)
        let add_node = Node::new(
            "add_bias".to_string(),
            "Add".to_string(),
            vec!["matmul_output".to_string(), "bias".to_string()],
            vec!["output".to_string()],
        );
        graph.add_node(add_node);

        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_node_creation() {
        let mut node = Node::new(
            "test_node".to_string(),
            "Add".to_string(),
            vec!["input1".to_string(), "input2".to_string()],
            vec!["output".to_string()],
        );

        assert_eq!(node.name, "test_node");
        assert_eq!(node.op_type, "Add");
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.outputs.len(), 1);

        node.add_attribute("axis", "1");
        assert_eq!(node.attributes.get("axis"), Some(&"1".to_string()));
    }

    #[test]
    fn test_tensor_spec() {
        let spec = TensorSpec::new("test_tensor".to_string(), vec![Some(2), Some(3), None]);
        
        let matching_tensor = Tensor::zeros(&[2, 3, 5]); // 5 is dynamic
        let non_matching_tensor = Tensor::zeros(&[2, 4, 5]); // Wrong second dimension
        
        assert!(spec.matches_tensor(&matching_tensor));
        assert!(!spec.matches_tensor(&non_matching_tensor));
    }

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new("test_graph".to_string());
        
        graph.add_input(TensorSpec::new("input".to_string(), vec![Some(1), Some(3)]));
        graph.add_output(TensorSpec::new("output".to_string(), vec![Some(1), Some(1)]));
        
        let node = Node::new(
            "relu".to_string(),
            "Relu".to_string(),
            vec!["input".to_string()],
            vec!["output".to_string()],
        );
        graph.add_node(node);

        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.input_names(), vec!["input"]);
        assert_eq!(graph.output_names(), vec!["output"]);
    }

    #[test]
    fn test_graph_validation_success() {
        let mut graph = Graph::new("valid_graph".to_string());
        
        graph.add_input(TensorSpec::new("input".to_string(), vec![Some(1), Some(3)]));
        graph.add_output(TensorSpec::new("output".to_string(), vec![Some(1), Some(3)]));
        
        let node = Node::new(
            "relu".to_string(),
            "Relu".to_string(),
            vec!["input".to_string()],
            vec!["output".to_string()],
        );
        graph.add_node(node);

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_graph_validation_failure() {
        let mut graph = Graph::new("invalid_graph".to_string());
        
        // Missing input declaration
        graph.add_output(TensorSpec::new("output".to_string(), vec![Some(1), Some(3)]));
        
        let node = Node::new(
            "relu".to_string(),
            "Relu".to_string(),
            vec!["missing_input".to_string()], // References unknown input
            vec!["output".to_string()],
        );
        graph.add_node(node);

        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_simple_linear_graph() {
        let graph = Graph::create_simple_linear();
        
        assert!(graph.validate().is_ok());
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.initializers.len(), 2);
        
        // Test topological sort
        let order = graph.topological_sort().unwrap();
        assert_eq!(order.len(), 2);
        // MatMul should come before Add
        let matmul_pos = order.iter().position(|&i| graph.nodes[i].op_type == "MatMul").unwrap();
        let add_pos = order.iter().position(|&i| graph.nodes[i].op_type == "Add").unwrap();
        assert!(matmul_pos < add_pos);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = Graph::new("test_topo".to_string());
        
        // Create a simple chain: input -> relu -> sigmoid -> output
        graph.add_input(TensorSpec::new("input".to_string(), vec![Some(1), Some(3)]));
        graph.add_output(TensorSpec::new("output".to_string(), vec![Some(1), Some(3)]));
        
        let relu_node = Node::new(
            "relu".to_string(),
            "Relu".to_string(),
            vec!["input".to_string()],
            vec!["relu_out".to_string()],
        );
        graph.add_node(relu_node);
        
        let sigmoid_node = Node::new(
            "sigmoid".to_string(),
            "Sigmoid".to_string(),
            vec!["relu_out".to_string()],
            vec!["output".to_string()],
        );
        graph.add_node(sigmoid_node);

        let order = graph.topological_sort().unwrap();
        assert_eq!(order, vec![0, 1]); // relu first, then sigmoid
    }
}
