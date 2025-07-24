//! Error handling for the ONNX runtime
//!
//! This module defines the main error types used throughout the crate.
//! All operations return [`Result<T, OnnxError>`] for consistent error handling.

/// Result type alias for ONNX operations
pub type Result<T> = std::result::Result<T, OnnxError>;

/// Main error type for ONNX runtime operations
#[derive(thiserror::Error, Debug)]
pub enum OnnxError {
    /// Tensor shape mismatch errors
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Invalid tensor dimensions
    #[error("Invalid dimensions: {message}")]
    InvalidDimensions { message: String },

    /// Unsupported operation
    #[error("Unsupported operation: {operation}")]
    UnsupportedOperation { operation: String },

    /// Model loading errors
    #[error("Model loading failed: {reason}")]
    ModelLoadError { reason: String },

    /// Graph validation errors
    #[error("Graph validation failed: {message}")]
    GraphValidationError { message: String },

    /// Runtime execution errors
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },

    /// I/O errors
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON parsing errors
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Protobuf errors
    #[error("Protobuf error: {0}")]
    ProtobufError(#[from] prost::DecodeError),

    /// Other errors
    #[error("Other error: {message}")]
    Other { message: String },
}

impl OnnxError {
    /// Create a new shape mismatch error
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        Self::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }

    /// Create a new invalid dimensions error
    pub fn invalid_dimensions<S: Into<String>>(message: S) -> Self {
        Self::InvalidDimensions {
            message: message.into(),
        }
    }

    /// Create a new unsupported operation error
    pub fn unsupported_operation<S: Into<String>>(operation: S) -> Self {
        Self::UnsupportedOperation {
            operation: operation.into(),
        }
    }

    /// Create a new model load error
    pub fn model_load_error<S: Into<String>>(reason: S) -> Self {
        Self::ModelLoadError {
            reason: reason.into(),
        }
    }

    /// Create a new graph validation error
    pub fn graph_validation_error<S: Into<String>>(message: S) -> Self {
        Self::GraphValidationError {
            message: message.into(),
        }
    }

    /// Create a new runtime error
    pub fn runtime_error<S: Into<String>>(message: S) -> Self {
        Self::RuntimeError {
            message: message.into(),
        }
    }

    /// Create a new other error
    pub fn other<S: Into<String>>(message: S) -> Self {
        Self::Other {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = OnnxError::shape_mismatch(&[2, 3], &[3, 2]);
        assert!(err.to_string().contains("Shape mismatch"));

        let err = OnnxError::unsupported_operation("CustomOp");
        assert!(err.to_string().contains("CustomOp"));

        let err = OnnxError::runtime_error("Test error");
        assert!(err.to_string().contains("Test error"));
    }

    #[test]
    fn test_error_display() {
        let err = OnnxError::invalid_dimensions("Invalid tensor shape");
        let error_string = format!("{err}");
        assert!(error_string.contains("Invalid dimensions"));
        assert!(error_string.contains("Invalid tensor shape"));
    }

    #[test]
    fn test_all_error_constructors() {
        // Test shape mismatch
        let err = OnnxError::shape_mismatch(&[1, 2, 3], &[3, 2, 1]);
        assert!(matches!(err, OnnxError::ShapeMismatch { .. }));
        assert!(err.to_string().contains("[1, 2, 3]"));
        assert!(err.to_string().contains("[3, 2, 1]"));

        // Test invalid dimensions
        let err = OnnxError::invalid_dimensions("dimension must be positive");
        assert!(matches!(err, OnnxError::InvalidDimensions { .. }));
        assert!(err.to_string().contains("dimension must be positive"));

        // Test unsupported operation
        let err = OnnxError::unsupported_operation("Conv3D");
        assert!(matches!(err, OnnxError::UnsupportedOperation { .. }));
        assert!(err.to_string().contains("Conv3D"));

        // Test model load error
        let err = OnnxError::model_load_error("file not found");
        assert!(matches!(err, OnnxError::ModelLoadError { .. }));
        assert!(err.to_string().contains("file not found"));

        // Test graph validation error
        let err = OnnxError::graph_validation_error("circular dependency detected");
        assert!(matches!(err, OnnxError::GraphValidationError { .. }));
        assert!(err.to_string().contains("circular dependency detected"));

        // Test runtime error
        let err = OnnxError::runtime_error("out of memory");
        assert!(matches!(err, OnnxError::RuntimeError { .. }));
        assert!(err.to_string().contains("out of memory"));

        // Test other error
        let err = OnnxError::other("unexpected error");
        assert!(matches!(err, OnnxError::Other { .. }));
        assert!(err.to_string().contains("unexpected error"));
    }

    #[test]
    fn test_error_from_conversions() {
        // Test IO error conversion
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let onnx_err: OnnxError = io_err.into();
        assert!(matches!(onnx_err, OnnxError::IoError(_)));
        assert!(onnx_err.to_string().contains("file not found"));

        // Test JSON error conversion
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let onnx_err: OnnxError = json_err.into();
        assert!(matches!(onnx_err, OnnxError::JsonError(_)));
    }

    #[test]
    fn test_error_debug_formatting() {
        let err = OnnxError::shape_mismatch(&[2, 3], &[4, 5]);
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("ShapeMismatch"));
        assert!(debug_str.contains("[2, 3]"));
        assert!(debug_str.contains("[4, 5]"));
    }

    #[test]
    fn test_result_type_alias() {
        // Test that our Result type alias works with functions
        fn returns_success() -> Result<i32> {
            Ok(42)
        }

        fn returns_error() -> Result<i32> {
            Err(OnnxError::other("test error"))
        }

        let success = returns_success();
        assert!(success.is_ok());
        assert_eq!(success.unwrap_or(0), 42);

        let failure = returns_error();
        assert!(failure.is_err());
    }

    #[test]
    fn test_string_conversions() {
        let err = OnnxError::invalid_dimensions(String::from("test string"));
        assert!(err.to_string().contains("test string"));

        let err = OnnxError::other("test str");
        assert!(err.to_string().contains("test str"));
    }
}
