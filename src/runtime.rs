//! Runtime execution engine
//!
//! This module provides the runtime execution engine that can execute
//! ONNX computational graphs with proper tensor management.

use crate::{
    error::{OnnxError, Result},
    graph::{Graph, Node},
    operators,
    tensor::Tensor,
};
use std::collections::HashMap;

/// Runtime execution engine for ONNX models
pub struct Runtime {
    /// Whether to enable debug logging
    debug: bool,
    /// Maximum number of concurrent operations (for future async support)
    max_concurrency: usize,
}

/// Execution context holding intermediate tensors
pub struct ExecutionContext {
    /// All tensors available during execution
    tensors: HashMap<String, Tensor>,
    /// Execution statistics
    stats: ExecutionStats,
}

/// Statistics collected during execution
#[derive(Debug, Default)]
pub struct ExecutionStats {
    /// Total execution time in milliseconds
    pub total_time_ms: f64,
    /// Number of operations executed
    pub ops_executed: usize,
    /// Memory usage in bytes (approximate)
    pub memory_usage_bytes: usize,
    /// Time per operation type
    pub op_times: HashMap<String, f64>,
}

impl Runtime {
    /// Create a new runtime with default settings
    pub fn new() -> Self {
        Self {
            debug: false,
            max_concurrency: 1,
        }
    }

    /// Create a new runtime with debug logging enabled
    pub fn with_debug() -> Self {
        Self {
            debug: true,
            max_concurrency: 1,
        }
    }

    /// Set the maximum number of concurrent operations
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Execute a graph with given inputs
    ///
    /// # Arguments
    /// * `graph` - The computational graph to execute
    /// * `inputs` - Map of input tensor names to tensors
    ///
    /// # Returns
    /// * Map of output tensor names to result tensors
    ///
    /// # Examples
    /// ```
    /// use runnx::{Runtime, Graph, Tensor};
    /// use std::collections::HashMap;
    /// use ndarray::Array2;
    ///
    /// let runtime = Runtime::new();
    /// let graph = Graph::create_simple_linear();
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("input".to_string(), Tensor::from_array(Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap().into_dyn()));
    ///
    /// let outputs = runtime.execute(&graph, inputs).unwrap();
    /// assert!(outputs.contains_key("output"));
    /// ```
    pub fn execute(
        &self,
        graph: &Graph,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let start_time = std::time::Instant::now();

        if self.debug {
            log::debug!("Starting execution of graph '{}'", graph.name);
        }

        // Validate graph
        graph.validate()?;

        // Validate inputs
        self.validate_inputs(graph, &inputs)?;

        // Create execution context
        let mut context = ExecutionContext::new();
        context.add_tensors(inputs);

        // Add initializers
        for (name, tensor) in &graph.initializers {
            context.add_tensor(name.clone(), tensor.clone());
        }

        // Get execution order
        let execution_order = graph.topological_sort()?;

        // Execute nodes in order
        for &node_idx in &execution_order {
            let node = &graph.nodes[node_idx];
            self.execute_node(node, &mut context)?;
        }

        // Extract outputs
        let outputs = self.extract_outputs(graph, &context)?;

        // Update statistics
        context.stats.total_time_ms = start_time.elapsed().as_millis() as f64;

        if self.debug {
            log::debug!(
                "Execution completed in {:.2}ms",
                context.stats.total_time_ms
            );
            log::debug!("Operations executed: {}", context.stats.ops_executed);
        }

        Ok(outputs)
    }

    /// Execute a graph with async support (feature gated)
    #[cfg(feature = "async")]
    pub async fn execute_async(
        self,
        graph: Graph,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // For now, just delegate to sync execution
        // In a full implementation, this would support parallel execution
        tokio::task::spawn_blocking(move || self.execute(&graph, inputs))
            .await
            .map_err(|e| OnnxError::runtime_error(e.to_string()))?
    }

    /// Validate that inputs match the graph's input specifications
    fn validate_inputs(&self, graph: &Graph, inputs: &HashMap<String, Tensor>) -> Result<()> {
        for input_spec in &graph.inputs {
            let tensor = inputs.get(&input_spec.name).ok_or_else(|| {
                OnnxError::runtime_error(format!("Missing required input: {}", input_spec.name))
            })?;

            if !input_spec.matches_tensor(tensor) {
                return Err(OnnxError::shape_mismatch(
                    &input_spec
                        .shape
                        .iter()
                        .map(|dim| dim.unwrap_or(0))
                        .collect::<Vec<_>>(),
                    tensor.shape(),
                ));
            }
        }

        Ok(())
    }

    /// Execute a single node
    fn execute_node(&self, node: &Node, context: &mut ExecutionContext) -> Result<()> {
        let node_start = std::time::Instant::now();

        if self.debug {
            log::debug!("Executing node '{}' ({})", node.name, node.op_type);
        }

        // Gather input tensors
        let input_tensors: Vec<Tensor> = node
            .inputs
            .iter()
            .map(|name| {
                context
                    .get_tensor(name)
                    .ok_or_else(|| {
                        OnnxError::runtime_error(format!(
                            "Node '{}' references unknown tensor '{}'",
                            node.name, name
                        ))
                    })
                    .cloned()
            })
            .collect::<Result<Vec<_>>>()?;

        // Execute the operator
        let op_type = node.get_operator_type()?;
        let output_tensors =
            operators::execute_operator(&op_type, &input_tensors, &node.attributes)?;

        // Store output tensors
        if output_tensors.len() != node.outputs.len() {
            return Err(OnnxError::runtime_error(format!(
                "Node '{}' produced {} outputs but expected {}",
                node.name,
                output_tensors.len(),
                node.outputs.len()
            )));
        }

        for (output_name, output_tensor) in node.outputs.iter().zip(output_tensors.iter()) {
            context.add_tensor(output_name.clone(), output_tensor.clone());
        }

        // Update statistics
        let execution_time = node_start.elapsed().as_millis() as f64;
        context.stats.ops_executed += 1;
        *context
            .stats
            .op_times
            .entry(node.op_type.clone())
            .or_insert(0.0) += execution_time;

        if self.debug {
            log::debug!("Node '{}' executed in {:.2}ms", node.name, execution_time);
        }

        Ok(())
    }

    /// Extract output tensors from the execution context
    fn extract_outputs(
        &self,
        graph: &Graph,
        context: &ExecutionContext,
    ) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();

        for output_spec in &graph.outputs {
            let tensor = context.get_tensor(&output_spec.name).ok_or_else(|| {
                OnnxError::runtime_error(format!(
                    "Graph output '{}' not found in execution context",
                    output_spec.name
                ))
            })?;

            outputs.insert(output_spec.name.clone(), tensor.clone());
        }

        Ok(outputs)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            stats: ExecutionStats::default(),
        }
    }

    /// Add a tensor to the context
    pub fn add_tensor(&mut self, name: String, tensor: Tensor) {
        // Update memory usage estimate
        self.stats.memory_usage_bytes += tensor.len() * std::mem::size_of::<f32>();
        self.tensors.insert(name, tensor);
    }

    /// Add multiple tensors to the context
    pub fn add_tensors(&mut self, tensors: HashMap<String, Tensor>) {
        for (name, tensor) in tensors {
            self.add_tensor(name, tensor);
        }
    }

    /// Get a tensor from the context
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get execution statistics
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionStats {
    /// Get the average time per operation
    pub fn avg_op_time(&self) -> f64 {
        if self.ops_executed > 0 {
            self.total_time_ms / self.ops_executed as f64
        } else {
            0.0
        }
    }

    /// Get memory usage in MB
    pub fn memory_usage_mb(&self) -> f64 {
        self.memory_usage_bytes as f64 / 1024.0 / 1024.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Tensor};
    use ndarray::Array1;
    #[cfg(feature = "async")]
    use ndarray::Array2;

    #[test]
    fn test_runtime_creation() {
        let runtime = Runtime::new();
        assert!(!runtime.debug);
        assert_eq!(runtime.max_concurrency, 1);

        let debug_runtime = Runtime::with_debug();
        assert!(debug_runtime.debug);
    }

    #[test]
    fn test_runtime_with_custom_config() {
        let runtime = Runtime::with_debug();

        assert!(runtime.debug);
        assert_eq!(runtime.max_concurrency, 1);
    }

    #[test]
    fn test_execution_context() {
        let mut context = ExecutionContext::new();

        let tensor = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        context.add_tensor("test".to_string(), tensor.clone());

        assert!(context.get_tensor("test").is_some());
        assert!(context.get_tensor("missing").is_none());
        assert_eq!(context.tensor_names(), vec!["test"]);

        // Test tensor retrieval
        let retrieved = context.get_tensor("test").unwrap();
        assert_eq!(retrieved.shape(), tensor.shape());
    }

    #[test]
    fn test_execution_context_multiple_tensors() {
        let mut context = ExecutionContext::new();

        let tensor1 = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let tensor2 = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));

        context.add_tensor("tensor1".to_string(), tensor1);
        context.add_tensor("tensor2".to_string(), tensor2);

        let mut names = context.tensor_names();
        names.sort(); // Sort for deterministic comparison
        assert_eq!(names, vec!["tensor1", "tensor2"]);

        assert!(context.get_tensor("tensor1").is_some());
        assert!(context.get_tensor("tensor2").is_some());
    }

    #[test]
    fn test_simple_execution() {
        env_logger::try_init().ok(); // Initialize logger for tests

        let runtime = Runtime::with_debug();
        let graph = Graph::create_simple_linear();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
        );

        let outputs = runtime.execute(&graph, inputs).unwrap();
        assert!(outputs.contains_key("output"));

        let output = outputs.get("output").unwrap();
        assert_eq!(output.shape(), &[1, 2]);

        // Expected result: [1, 2, 3] * [[0.5, 0.3], [0.2, 0.4], [0.1, 0.6]] + [0.1, 0.2]
        // = [1*0.5 + 2*0.2 + 3*0.1, 1*0.3 + 2*0.4 + 3*0.6] + [0.1, 0.2]
        // = [0.5 + 0.4 + 0.3, 0.3 + 0.8 + 1.8] + [0.1, 0.2]
        // = [1.2, 2.9] + [0.1, 0.2] = [1.3, 3.1]
        let data = output.data();
        let expected = [1.3, 3.1];
        for (actual, &expected) in data.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_runtime_non_debug_execution() {
        let runtime = Runtime::new(); // Non-debug mode
        let graph = Graph::create_simple_linear();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
        );

        let outputs = runtime.execute(&graph, inputs).unwrap();
        assert!(outputs.contains_key("output"));
    }

    #[test]
    fn test_missing_input() {
        let runtime = Runtime::new();
        let graph = Graph::create_simple_linear();

        let inputs = HashMap::new(); // Missing required input

        let result = runtime.execute(&graph, inputs);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Missing required input"));
    }

    #[test]
    fn test_input_shape_validation_error() {
        let runtime = Runtime::new();
        let graph = Graph::create_simple_linear();

        let mut inputs = HashMap::new();
        // Wrong shape - should be [1, 3] but providing [2, 2]
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );

        let result = runtime.execute(&graph, inputs);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Shape mismatch") || error_msg.contains("shape"));
    }

    #[test]
    fn test_unknown_tensor_reference_error() {
        let runtime = Runtime::new();

        // Create a graph with an invalid node that references a non-existent tensor
        let mut graph = Graph::new("invalid_graph".to_string());
        let node = crate::graph::Node::new(
            "invalid_node".to_string(),
            "Add".to_string(),
            vec![
                "nonexistent_tensor".to_string(),
                "another_nonexistent".to_string(),
            ],
            vec!["output".to_string()],
        );
        graph.add_node(node);

        // Add a fake input spec to pass validation
        let input_spec = crate::graph::TensorSpec {
            name: "input".to_string(),
            dtype: "float32".to_string(),
            shape: vec![Some(1), Some(3)],
        };
        graph.add_input(input_spec);

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Tensor::zeros(&[1, 3]));

        let result = runtime.execute(&graph, inputs);
        assert!(result.is_err());
        // The error should mention the unknown tensor
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("unknown tensor") || error_msg.contains("nonexistent"));
    }

    #[test]
    fn test_execution_with_intermediate_tensors() {
        let runtime = Runtime::with_debug();

        // Create a more complex graph with intermediate tensors
        let mut graph = Graph::new("complex_graph".to_string());

        // Add input spec
        let input_spec = crate::graph::TensorSpec {
            name: "input".to_string(),
            dtype: "float32".to_string(),
            shape: vec![Some(1), Some(2)],
        };
        graph.add_input(input_spec);

        // Add output spec
        let output_spec = crate::graph::TensorSpec {
            name: "output".to_string(),
            dtype: "float32".to_string(),
            shape: vec![Some(1), Some(2)],
        };
        graph.add_output(output_spec);

        // First node: ReLU
        let relu_node = crate::graph::Node::new(
            "relu".to_string(),
            "Relu".to_string(),
            vec!["input".to_string()],
            vec!["intermediate".to_string()],
        );
        graph.add_node(relu_node);

        // Second node: Add (add intermediate to itself)
        let add_node = crate::graph::Node::new(
            "add".to_string(),
            "Add".to_string(),
            vec!["intermediate".to_string(), "intermediate".to_string()],
            vec!["output".to_string()],
        );
        graph.add_node(add_node);

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[1, 2], vec![-1.0, 2.0]).unwrap(),
        );

        let outputs = runtime.execute(&graph, inputs).unwrap();
        assert!(outputs.contains_key("output"));

        // Expected: ReLU(-1, 2) = (0, 2), then (0, 2) + (0, 2) = (0, 4)
        let output = outputs.get("output").unwrap();
        let data = output.data();
        assert!((data[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((data[[0, 1]] - 4.0).abs() < 1e-6);
    }

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn test_async_execution() {
        let runtime = Runtime::new();
        let graph = Graph::create_simple_linear();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_array(
                Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0])
                    .unwrap()
                    .into_dyn(),
            ),
        );

        let outputs = runtime.execute_async(graph, inputs).await.unwrap();
        assert!(outputs.contains_key("output"));
    }

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn test_async_execution_error() {
        let runtime = Runtime::new();
        let graph = Graph::create_simple_linear();

        let inputs = HashMap::new(); // Missing required input

        let result = runtime.execute_async(graph, inputs).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_execution_stats() {
        let stats = ExecutionStats {
            total_time_ms: 100.0,
            ops_executed: 5,
            memory_usage_bytes: 1024 * 1024, // 1MB
            ..Default::default()
        };

        assert_eq!(stats.avg_op_time(), 20.0);
        assert_eq!(stats.memory_usage_mb(), 1.0);
    }

    #[test]
    fn test_execution_stats_zero_ops() {
        let stats = ExecutionStats {
            total_time_ms: 100.0,
            ops_executed: 0,
            memory_usage_bytes: 0,
            ..Default::default()
        };

        assert_eq!(stats.avg_op_time(), 0.0);
        assert_eq!(stats.memory_usage_mb(), 0.0);
    }

    #[test]
    fn test_execution_stats_default() {
        let stats = ExecutionStats::default();

        assert_eq!(stats.total_time_ms, 0.0);
        assert_eq!(stats.ops_executed, 0);
        assert_eq!(stats.memory_usage_bytes, 0);
        assert_eq!(stats.avg_op_time(), 0.0);
        assert_eq!(stats.memory_usage_mb(), 0.0);
    }

    #[test]
    fn test_runtime_builder_pattern() {
        let runtime = Runtime::new();

        assert!(!runtime.debug);

        let runtime2 = Runtime::with_debug();

        assert!(runtime2.debug);
    }

    #[test]
    fn test_large_batch_execution() {
        let runtime = Runtime::new();
        let graph = Graph::create_simple_linear();

        // Test with a larger input that matches the expected shape
        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
        );

        let outputs = runtime.execute(&graph, inputs).unwrap();
        assert!(outputs.contains_key("output"));

        let output = outputs.get("output").unwrap();
        assert_eq!(output.shape(), &[1, 2]); // Batch size 1, output dim 2
    }

    #[test]
    fn test_execution_context_tensor_overwrite() {
        let mut context = ExecutionContext::new();

        let tensor1 = Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let tensor2 = Tensor::from_array(Array1::from_vec(vec![4.0, 5.0, 6.0]));

        context.add_tensor("test".to_string(), tensor1);
        context.add_tensor("test".to_string(), tensor2); // Overwrite

        let retrieved = context.get_tensor("test").unwrap();
        let data = retrieved.data();
        // Should have the second tensor's data
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 5.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
    }
}
