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
    ProtobufError(#[from] protobuf::Error),

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
        let error_string = format!("{}", err);
        assert!(error_string.contains("Invalid dimensions"));
        assert!(error_string.contains("Invalid tensor shape"));
    }
}
