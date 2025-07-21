//! ONNX model representation and loading
//!
//! This module provides the main [`Model`] struct that represents
//! a complete ONNX model with its graph and metadata.

use crate::{
    error::{OnnxError, Result},
    graph::Graph,
    runtime::{ExecutionStats, Runtime},
    tensor::Tensor,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// ONNX model representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Computational graph
    pub graph: Graph,
}

/// Model metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Producer name (e.g., "PyTorch", "TensorFlow")
    pub producer: String,
    /// ONNX version used
    pub onnx_version: String,
    /// Domain (namespace) of the model
    pub domain: String,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "onnx-model".to_string(),
            version: "1.0".to_string(),
            description: "ONNX model".to_string(),
            producer: "RunNX".to_string(),
            onnx_version: "1.9.0".to_string(),
            domain: "".to_string(),
        }
    }
}

impl Model {
    /// Create a new model with the given graph
    pub fn new(graph: Graph) -> Self {
        Self {
            metadata: ModelMetadata {
                name: graph.name.clone(),
                ..Default::default()
            },
            graph,
        }
    }

    /// Create a model with custom metadata
    pub fn with_metadata(metadata: ModelMetadata, graph: Graph) -> Self {
        Self { metadata, graph }
    }

    /// Load a model from file (simplified JSON format for this implementation)
    ///
    /// In a full ONNX implementation, this would parse the protobuf format.
    /// For simplicity, we use JSON serialization.
    ///
    /// # Examples
    /// ```no_run
    /// use runnx::Model;
    ///
    /// let model = Model::from_file("model.json").unwrap();
    /// println!("Loaded model: {}", model.name());
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| {
            OnnxError::model_load_error(format!(
                "Failed to read model file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let model: Model = serde_json::from_str(&content).map_err(|e| {
            OnnxError::model_load_error(format!(
                "Failed to parse model file '{}': {}",
                path.display(),
                e
            ))
        })?;

        // Validate the loaded model
        model.validate()?;

        Ok(model)
    }

    /// Save the model to file (JSON format)
    ///
    /// # Examples
    /// ```no_run
    /// use runnx::{Model, Graph};
    ///
    /// let graph = Graph::create_simple_linear();
    /// let model = Model::new(graph);
    /// model.to_file("model.json").unwrap();
    /// ```
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let content = serde_json::to_string_pretty(self)?;

        fs::write(path, content).map_err(|e| {
            OnnxError::other(format!(
                "Failed to write model file '{}': {}",
                path.display(),
                e
            ))
        })
    }

    /// Get the model name
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get the model version
    pub fn version(&self) -> &str {
        &self.metadata.version
    }

    /// Get the model description
    pub fn description(&self) -> &str {
        &self.metadata.description
    }

    /// Get input tensor names
    pub fn input_names(&self) -> Vec<&str> {
        self.graph.input_names()
    }

    /// Get output tensor names
    pub fn output_names(&self) -> Vec<&str> {
        self.graph.output_names()
    }

    /// Get input specifications
    pub fn input_specs(&self) -> &[crate::graph::TensorSpec] {
        &self.graph.inputs
    }

    /// Get output specifications  
    pub fn output_specs(&self) -> &[crate::graph::TensorSpec] {
        &self.graph.outputs
    }

    /// Validate the model
    pub fn validate(&self) -> Result<()> {
        // Validate the computational graph
        self.graph.validate()?;

        // Additional model-level validations can be added here
        if self.metadata.name.is_empty() {
            return Err(OnnxError::model_load_error("Model name cannot be empty"));
        }

        Ok(())
    }

    /// Run inference on the model
    ///
    /// # Arguments
    /// * `inputs` - Map of input names to tensors
    ///
    /// # Returns
    /// * Map of output names to result tensors
    ///
    /// # Examples
    /// ```
    /// use runnx::{Model, Graph, Tensor};
    /// use std::collections::HashMap;
    /// use ndarray::Array2;
    ///
    /// let graph = Graph::create_simple_linear();
    /// let model = Model::new(graph);
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("input".to_string(), Tensor::from_array(Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap().into_dyn()));
    ///
    /// let outputs = model.run(&inputs).unwrap();
    /// assert!(outputs.contains_key("output"));
    /// ```
    pub fn run(&self, inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let runtime = Runtime::new();
        runtime.execute(&self.graph, inputs.clone())
    }

    /// Run inference with a custom runtime
    pub fn run_with_runtime(
        &self,
        inputs: &HashMap<String, Tensor>,
        runtime: &Runtime,
    ) -> Result<HashMap<String, Tensor>> {
        runtime.execute(&self.graph, inputs.clone())
    }

    /// Run inference and return execution statistics
    pub fn run_with_stats(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<(HashMap<String, Tensor>, ExecutionStats)> {
        let runtime = Runtime::with_debug();
        let outputs = runtime.execute(&self.graph, inputs.clone())?;

        // In a full implementation, we would return the actual stats from the runtime
        // For now, we return default stats
        let stats = ExecutionStats::default();

        Ok((outputs, stats))
    }

    /// Run inference with async support (feature gated)
    #[cfg(feature = "async")]
    pub async fn run_async(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let runtime = Runtime::new();
        runtime
            .execute_async(self.graph.clone(), inputs.clone())
            .await
    }

    /// Create a simple linear model for testing
    pub fn create_simple_linear() -> Self {
        let graph = Graph::create_simple_linear();
        Self::new(graph)
    }

    /// Get model summary as a formatted string
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("Model: {}\n", self.name()));
        summary.push_str(&format!("Version: {}\n", self.version()));
        summary.push_str(&format!("Description: {}\n", self.description()));
        summary.push_str(&format!("Producer: {}\n", self.metadata.producer));
        summary.push_str(&format!("ONNX Version: {}\n", self.metadata.onnx_version));
        summary.push_str(&format!("Domain: {}\n", self.metadata.domain));
        summary.push('\n');

        summary.push_str("Inputs:\n");
        for input_spec in &self.graph.inputs {
            summary.push_str(&format!(
                "  - {}: {:?} ({})\n",
                input_spec.name, input_spec.shape, input_spec.dtype
            ));
        }
        summary.push('\n');

        summary.push_str("Outputs:\n");
        for output_spec in &self.graph.outputs {
            summary.push_str(&format!(
                "  - {}: {:?} ({})\n",
                output_spec.name, output_spec.shape, output_spec.dtype
            ));
        }
        summary.push('\n');

        summary.push_str(&format!("Graph: {}\n", self.graph.name));
        summary.push_str(&format!("  Nodes: {}\n", self.graph.nodes.len()));
        summary.push_str(&format!(
            "  Initializers: {}\n",
            self.graph.initializers.len()
        ));

        summary.push_str("  Operations:\n");
        let mut op_counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for node in &self.graph.nodes {
            *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        }
        for (op_type, count) in op_counts {
            summary.push_str(&format!("    {op_type}: {count}\n"));
        }

        summary
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;
    use ndarray::Array2;
    use tempfile::NamedTempFile;

    #[test]
    fn test_model_creation() {
        let graph = Graph::create_simple_linear();
        let model = Model::new(graph);

        assert_eq!(model.name(), "simple_linear");
        assert_eq!(model.input_names(), vec!["input"]);
        assert_eq!(model.output_names(), vec!["output"]);
    }

    #[test]
    fn test_model_with_metadata() {
        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            description: "Test model for unit testing".to_string(),
            ..Default::default()
        };

        let graph = Graph::create_simple_linear();
        let model = Model::with_metadata(metadata, graph);

        assert_eq!(model.name(), "test_model");
        assert_eq!(model.description(), "Test model for unit testing");
    }

    #[test]
    fn test_model_validation() {
        let graph = Graph::create_simple_linear();
        let model = Model::new(graph);

        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_model_run() {
        let model = Model::create_simple_linear();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
        );

        let outputs = model.run(&inputs).unwrap();
        assert!(outputs.contains_key("output"));

        let output = outputs.get("output").unwrap();
        assert_eq!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_model_serialization() {
        let model = Model::create_simple_linear();

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        // Save and load the model
        model.to_file(file_path).unwrap();
        let loaded_model = Model::from_file(file_path).unwrap();

        assert_eq!(model.name(), loaded_model.name());
        assert_eq!(model.input_names(), loaded_model.input_names());
        assert_eq!(model.output_names(), loaded_model.output_names());
    }

    #[test]
    fn test_model_summary() {
        let model = Model::create_simple_linear();
        let summary = model.summary();

        assert!(summary.contains("Model: simple_linear"));
        assert!(summary.contains("Inputs:"));
        assert!(summary.contains("Outputs:"));
        assert!(summary.contains("Graph:"));
        assert!(summary.contains("MatMul: 1"));
        assert!(summary.contains("Add: 1"));
    }

    #[test]
    fn test_model_display() {
        let model = Model::create_simple_linear();
        let display_string = format!("{model}");
        let summary = model.summary();

        assert_eq!(display_string, summary);
    }

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn test_model_run_async() {
        let model = Model::create_simple_linear();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_array(
                Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0])
                    .unwrap()
                    .into_dyn(),
            ),
        );

        let outputs = model.run_async(&inputs).await.unwrap();
        assert!(outputs.contains_key("output"));
    }

    #[test]
    fn test_invalid_model_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        // Write invalid JSON
        std::fs::write(file_path, "invalid json").unwrap();

        let result = Model::from_file(file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_nonexistent_model_file() {
        let result = Model::from_file("/nonexistent/path/model.json");
        assert!(result.is_err());
    }
}
