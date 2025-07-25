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

    /// Load a model from file (supports both JSON and binary ONNX formats)
    ///
    /// This method automatically detects the file format:
    /// - Files with .onnx extension are treated as binary ONNX protobuf format
    /// - Files with .json extension are treated as JSON format
    /// - Other files are treated as JSON format by default
    ///
    /// # Examples
    /// ```no_run
    /// use runnx::Model;
    ///
    /// // Load from JSON format
    /// let model = Model::from_file("model.json").unwrap();
    /// println!("Loaded model: {}", model.name());
    ///
    /// // Load from binary ONNX format
    /// let model = Model::from_file("model.onnx").unwrap();
    /// println!("Loaded model: {}", model.name());
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Determine format based on file extension
        if let Some(extension) = path.extension() {
            if extension == "onnx" {
                return Self::from_onnx_file(path);
            }
        }

        // Default to JSON format
        Self::from_json_file(path)
    }

    /// Load a model from JSON file format
    ///
    /// # Examples
    /// ```no_run
    /// use runnx::Model;
    ///
    /// let model = Model::from_json_file("model.json").unwrap();
    /// println!("Loaded model: {}", model.name());
    /// ```
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
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
                "Failed to parse JSON model file '{}': {}",
                path.display(),
                e
            ))
        })?;

        // Validate the loaded model
        model.validate()?;

        Ok(model)
    }

    /// Load a model from binary ONNX protobuf file format
    ///
    /// # Examples
    /// ```no_run
    /// use runnx::Model;
    ///
    /// let model = Model::from_onnx_file("model.onnx").unwrap();
    /// println!("Loaded model: {}", model.name());
    /// ```
    pub fn from_onnx_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        crate::converter::load_onnx_model(path)
    }

    /// Save the model to file (supports both JSON and binary ONNX formats)
    ///
    /// This method automatically detects the file format based on extension:
    /// - Files with .onnx extension are saved as binary ONNX protobuf format
    /// - Files with .json extension are saved as JSON format
    /// - Other files are saved as JSON format by default
    ///
    /// # Examples
    /// ```no_run
    /// use runnx::{Model, Graph};
    ///
    /// let graph = Graph::create_simple_linear();
    /// let model = Model::new(graph);
    ///
    /// // Save as JSON
    /// model.to_file("model.json").unwrap();
    ///
    /// // Save as binary ONNX
    /// model.to_file("model.onnx").unwrap();
    /// ```
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Determine format based on file extension
        if let Some(extension) = path.extension() {
            if extension == "onnx" {
                return self.to_onnx_file(path);
            }
        }

        // Default to JSON format
        self.to_json_file(path)
    }

    /// Save the model to JSON file format
    ///
    /// # Examples
    /// ```no_run
    /// use runnx::{Model, Graph};
    ///
    /// let graph = Graph::create_simple_linear();
    /// let model = Model::new(graph);
    /// model.to_json_file("model.json").unwrap();
    /// ```
    pub fn to_json_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let content = serde_json::to_string_pretty(self)?;

        fs::write(path, content).map_err(|e| {
            OnnxError::other(format!(
                "Failed to write JSON model file '{}': {}",
                path.display(),
                e
            ))
        })
    }

    /// Save the model to binary ONNX protobuf file format
    ///
    /// # Examples
    /// ```no_run
    /// use runnx::{Model, Graph};
    ///
    /// let graph = Graph::create_simple_linear();
    /// let model = Model::new(graph);
    /// model.to_onnx_file("model.onnx").unwrap();
    /// ```
    pub fn to_onnx_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        crate::converter::save_onnx_model(self, path)
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
                input_spec.name, input_spec.dimensions, input_spec.dtype
            ));
        }
        summary.push('\n');

        summary.push_str("Outputs:\n");
        for output_spec in &self.graph.outputs {
            summary.push_str(&format!(
                "  - {}: {:?} ({})\n",
                output_spec.name, output_spec.dimensions, output_spec.dtype
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

    /// Print the model graph in a visual ASCII format
    pub fn print_graph(&self) {
        self.graph.print_graph();
    }

    /// Generate DOT format for the model graph (for use with Graphviz)
    ///
    /// # Examples
    /// ```no_run
    /// use runnx::Model;
    ///
    /// let model = Model::from_file("model.json").unwrap();
    /// let dot_content = model.to_dot();
    /// std::fs::write("graph.dot", dot_content).unwrap();
    /// // Then run: dot -Tpng graph.dot -o graph.png
    /// ```
    pub fn to_dot(&self) -> String {
        self.graph.to_dot()
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
    #[cfg(feature = "async")]
    use ndarray::Array2;
    use std::fs;
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
            version: "1.2.3".to_string(),
            producer: "test_producer".to_string(),
            onnx_version: "1.14.0".to_string(),
            domain: "test.domain".to_string(),
        };

        let graph = Graph::create_simple_linear();
        let model = Model::with_metadata(metadata, graph);

        assert_eq!(model.name(), "test_model");
        assert_eq!(model.description(), "Test model for unit testing");
        assert_eq!(model.version(), "1.2.3");
    }

    #[test]
    fn test_model_metadata_default() {
        let metadata = ModelMetadata::default();
        assert_eq!(metadata.name, "onnx-model");
        assert_eq!(metadata.description, "ONNX model");
        assert_eq!(metadata.version, "1.0");
        assert_eq!(metadata.producer, "RunNX");
        assert_eq!(metadata.onnx_version, "1.9.0");
        assert_eq!(metadata.domain, "");
    }

    #[test]
    fn test_model_accessors() {
        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            description: "Test description".to_string(),
            version: "2.0.0".to_string(),
            producer: "Test Producer".to_string(),
            onnx_version: "1.15.0".to_string(),
            domain: "ai.test".to_string(),
        };

        let graph = Graph::create_simple_linear();
        let model = Model::with_metadata(metadata, graph);

        assert_eq!(model.name(), "test_model");
        assert_eq!(model.description(), "Test description");
        assert_eq!(model.version(), "2.0.0");
        assert_eq!(model.input_names(), vec!["input"]);
        assert_eq!(model.output_names(), vec!["output"]);

        // Test specs accessors
        assert_eq!(model.input_specs().len(), 1);
        assert_eq!(model.output_specs().len(), 1);
        assert_eq!(model.input_specs()[0].name, "input");
        assert_eq!(model.output_specs()[0].name, "output");
    }

    #[test]
    fn test_model_validation() {
        let graph = Graph::create_simple_linear();
        let model = Model::new(graph);

        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_model_validation_empty_name() {
        let metadata = ModelMetadata {
            name: "".to_string(), // Empty name should cause validation error
            ..Default::default()
        };

        let graph = Graph::create_simple_linear();
        let model = Model::with_metadata(metadata, graph);

        let result = model.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("name cannot be empty"));
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
    fn test_model_run_with_runtime() {
        let model = Model::create_simple_linear();
        let runtime = Runtime::with_debug();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
        );

        let outputs = model.run_with_runtime(&inputs, &runtime).unwrap();
        assert!(outputs.contains_key("output"));
    }

    #[test]
    fn test_model_run_with_stats() {
        let model = Model::create_simple_linear();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
        );

        let (outputs, stats) = model.run_with_stats(&inputs).unwrap();
        assert!(outputs.contains_key("output"));
        assert_eq!(stats.total_time_ms, 0.0); // Default stats for now
    }

    #[test]
    fn test_model_run_error_missing_input() {
        let model = Model::create_simple_linear();
        let inputs = HashMap::new(); // Missing required input

        let result = model.run(&inputs);
        assert!(result.is_err());
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
        assert_eq!(model.description(), loaded_model.description());
        assert_eq!(model.version(), loaded_model.version());
    }

    #[test]
    fn test_model_serialization_custom_metadata() {
        let metadata = ModelMetadata {
            name: "custom_model".to_string(),
            description: "Custom test model".to_string(),
            version: "1.5.0".to_string(),
            producer: "Custom Producer".to_string(),
            onnx_version: "1.16.0".to_string(),
            domain: "custom.domain".to_string(),
        };

        let graph = Graph::create_simple_linear();
        let model = Model::with_metadata(metadata, graph);

        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        model.to_file(file_path).unwrap();
        let loaded_model = Model::from_file(file_path).unwrap();

        assert_eq!(loaded_model.name(), "custom_model");
        assert_eq!(loaded_model.description(), "Custom test model");
        assert_eq!(loaded_model.version(), "1.5.0");
    }

    #[test]
    fn test_model_to_file_error() {
        let model = Model::create_simple_linear();

        // Try to save to an invalid path (directory doesn't exist)
        let result = model.to_file("/nonexistent/directory/model.json");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to write JSON model file"));
    }

    #[test]
    fn test_model_summary() {
        let model = Model::create_simple_linear();
        let summary = model.summary();

        assert!(summary.contains("Model: simple_linear"));
        assert!(summary.contains("Version: 1.0"));
        assert!(summary.contains("Producer: RunNX"));
        assert!(summary.contains("ONNX Version: 1.9.0"));
        assert!(summary.contains("Inputs:"));
        assert!(summary.contains("Outputs:"));
        assert!(summary.contains("Graph:"));
        assert!(summary.contains("MatMul: 1"));
        assert!(summary.contains("Add: 1"));
        assert!(summary.contains("Operations:"));
    }

    #[test]
    fn test_model_summary_with_custom_metadata() {
        let metadata = ModelMetadata {
            name: "custom_model".to_string(),
            description: "A custom model for testing".to_string(),
            version: "2.1.0".to_string(),
            producer: "Test Suite".to_string(),
            onnx_version: "1.15.0".to_string(),
            domain: "test.models".to_string(),
        };

        let graph = Graph::create_simple_linear();
        let model = Model::with_metadata(metadata, graph);
        let summary = model.summary();

        assert!(summary.contains("Model: custom_model"));
        assert!(summary.contains("Version: 2.1.0"));
        assert!(summary.contains("Description: A custom model for testing"));
        assert!(summary.contains("Producer: Test Suite"));
        assert!(summary.contains("ONNX Version: 1.15.0"));
        assert!(summary.contains("Domain: test.models"));
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

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn test_model_run_async_error() {
        let model = Model::create_simple_linear();
        let inputs = HashMap::new(); // Missing required input

        let result = model.run_async(&inputs).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_model_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        // Write invalid JSON
        fs::write(file_path, "invalid json").unwrap();

        let result = Model::from_file(file_path);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("JSON error") || error_msg.contains("parse"));
    }

    #[test]
    fn test_model_file_with_invalid_model() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        // Write valid JSON but invalid model (missing required fields)
        let invalid_model = r#"{"metadata": {"name": ""}, "graph": {"nodes": []}}"#;
        fs::write(file_path, invalid_model).unwrap();

        let result = Model::from_file(file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_nonexistent_model_file() {
        let result = Model::from_file("/nonexistent/path/model.json");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("Failed to read model file") || error_msg.contains("No such file")
        );
    }

    #[test]
    fn test_model_clone_functionality() {
        // Test that models can be used in ways that require cloning
        let model = Model::create_simple_linear();
        let model_copy = model.clone();

        assert_eq!(model.name(), model_copy.name());
        assert_eq!(model.input_names(), model_copy.input_names());
        assert_eq!(model.output_names(), model_copy.output_names());

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
        );

        // Both models should produce the same output
        let outputs1 = model.run(&inputs).unwrap();
        let outputs2 = model_copy.run(&inputs).unwrap();

        assert_eq!(outputs1.len(), outputs2.len());
        for (key, tensor1) in &outputs1 {
            let tensor2 = outputs2.get(key).unwrap();
            assert_eq!(tensor1.shape(), tensor2.shape());
        }
    }

    #[test]
    fn test_model_serialization_round_trip_preserves_functionality() {
        let model = Model::create_simple_linear();

        // Test original model
        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
        );
        let original_outputs = model.run(&inputs).unwrap();

        // Serialize and deserialize
        let temp_file = NamedTempFile::new().unwrap();
        model.to_file(temp_file.path()).unwrap();
        let loaded_model = Model::from_file(temp_file.path()).unwrap();

        // Test loaded model produces same results
        let loaded_outputs = loaded_model.run(&inputs).unwrap();

        assert_eq!(original_outputs.len(), loaded_outputs.len());
        for (key, original_tensor) in &original_outputs {
            let loaded_tensor = loaded_outputs.get(key).unwrap();
            assert_eq!(original_tensor.shape(), loaded_tensor.shape());

            let original_data = original_tensor.data();
            let loaded_data = loaded_tensor.data();
            for (orig, loaded) in original_data.iter().zip(loaded_data.iter()) {
                assert!((orig - loaded).abs() < 1e-6);
            }
        }
    }
}
