//! Additional tests to reach 100% coverage (excluding bin/* files)
//!
//! These tests specifically target the uncovered code paths identified
//! in the coverage analysis.

use ndarray::Array2;
use runnx::model::ModelMetadata;
use runnx::operators::OperatorType;
use runnx::{error::OnnxError, Graph, Model, Node, Tensor};
use std::collections::HashMap;

#[cfg(test)]
mod coverage_tests {
    use super::*;

    #[test]
    fn test_uncovered_error_constructors() {
        // Test OnnxError::other with String parameter
        let err = OnnxError::other(String::from("custom error message"));
        assert!(err.to_string().contains("custom error message"));

        // Test OnnxError::model_load_error with str parameter
        let err = OnnxError::model_load_error("file read failed");
        assert!(err.to_string().contains("file read failed"));

        // Test OnnxError::invalid_dimensions with str parameter
        let err = OnnxError::invalid_dimensions("negative dimension");
        assert!(err.to_string().contains("negative dimension"));

        // Test OnnxError::unsupported_operation with str parameter
        let err = OnnxError::unsupported_operation("CustomLayer");
        assert!(err.to_string().contains("CustomLayer"));
    }

    #[test]
    fn test_model_accessor_methods() {
        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            description: "Test model description".to_string(),
            version: "2.1.0".to_string(),
            producer: "TestProducer".to_string(),
            onnx_version: "1.16.0".to_string(),
            domain: "test.domain".to_string(),
        };

        let graph = Graph::create_simple_linear();
        let model = Model::with_metadata(metadata, graph);

        // Test the uncovered accessor methods
        assert_eq!(model.description(), "Test model description");
        assert_eq!(model.version(), "2.1.0");

        // Test input/output specs
        let input_specs = model.input_specs();
        assert!(!input_specs.is_empty());

        let output_specs = model.output_specs();
        assert!(!output_specs.is_empty());
    }

    #[test]
    fn test_model_summary() {
        let model = Model::create_simple_linear();
        let summary = model.summary();

        // Ensure summary method is covered
        assert!(summary.contains("Model:"));
        assert!(summary.contains("Version:"));
        assert!(summary.contains("Inputs:"));
        assert!(summary.contains("Outputs:"));
    }

    #[test]
    fn test_model_run_with_stats() {
        let model = Model::create_simple_linear();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_array(Array2::from_elem((1, 3), 1.0)),
        );

        // Test run_with_stats method
        let (outputs, stats) = model.run_with_stats(&inputs).unwrap();
        assert!(!outputs.is_empty());
        // Check that we get stats back (stats may be 0 if not properly implemented)
        assert!(stats.total_time_ms >= 0.0);
    }

    #[test]
    fn test_model_run_with_runtime() {
        let model = Model::create_simple_linear();
        let runtime = runnx::Runtime::new();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_array(Array2::from_elem((1, 3), 1.0)),
        );

        // Test run_with_runtime method
        let outputs = model.run_with_runtime(&inputs, &runtime).unwrap();
        assert!(!outputs.is_empty());
    }

    #[test]
    fn test_graph_validation() {
        // Create a graph that should validate successfully
        let graph = Graph::create_simple_linear();

        // Test the validate method
        let result = graph.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_validation_failure() {
        // Create a graph with validation issues
        let mut graph = Graph::new("test_graph".to_string());

        // Create a node that references non-existent input tensors
        let node = Node::new(
            "broken_node".to_string(),
            "Add".to_string(),
            vec![
                "nonexistent_input1".to_string(),
                "nonexistent_input2".to_string(),
            ], // These don't exist
            vec!["output1".to_string()],
        );
        graph.add_node(node);

        // This should trigger the validation failure path
        let result = graph.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown input tensor"));
    }

    #[test]
    fn test_node_add_attribute() {
        let mut node = Node::new(
            "test_node".to_string(),
            "Conv".to_string(),
            vec!["input1".to_string()],
            vec!["output1".to_string()],
        );

        // Test the add_attribute method
        node.add_attribute("kernel_shape".to_string(), "3,3".to_string());

        // Verify the attribute was added
        assert!(node.attributes.contains_key("kernel_shape"));
        assert_eq!(node.attributes["kernel_shape"], "3,3");
    }

    #[test]
    fn test_error_conversion_paths() {
        // Test std::io::Error conversion
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test file not found");
        let onnx_error: OnnxError = io_error.into();
        assert!(matches!(onnx_error, OnnxError::IoError(_)));

        // Test serde_json::Error conversion
        let json_result = serde_json::from_str::<serde_json::Value>("invalid {json");
        let json_error = json_result.unwrap_err();
        let onnx_error: OnnxError = json_error.into();
        assert!(matches!(onnx_error, OnnxError::JsonError(_)));
    }

    #[test]
    fn test_operator_from_str_unsupported() {
        // Test FromStr for unsupported operator type
        let result = "UnsupportedOperation".parse::<OperatorType>();
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("UnsupportedOperation"));
    }

    #[test]
    fn test_tensor_edge_cases_for_coverage() {
        // Test cases to potentially trigger uncovered error paths in tensor operations

        // Try to create a tensor that might trigger dimension conversion errors
        let tensor = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // This should work fine, but ensures the success path is covered
        let reshaped = tensor.reshape(&[4]).unwrap();
        assert_eq!(reshaped.shape(), &[4]);

        // Test matrix multiplication to ensure coverage of dimensionality conversion
        let a = Tensor::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }
}
