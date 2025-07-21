//! # RunNX: A Minimal ONNX Runtime in Rust
//!
//! This crate provides a minimal, verifiable implementation of an ONNX runtime
//! focused on educational purposes and simplicity while maintaining performance.
//!
//! ## Key Features
//!
//! - **Simple Architecture**: Easy to understand and extend
//! - **Type Safety**: Leverages Rust's type system for memory safety
//! - **Performance**: Efficient tensor operations using ndarray
//! - **Verifiable**: Comprehensive tests and clear documentation
//!
//! ## Quick Start
//!
//! ```rust
//! use runnx::{Model, Tensor};
//! use ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a simple tensor
//! let input = Tensor::from_array(Array2::from_elem((2, 3), 1.0));
//! let weights = Tensor::from_array(Array2::from_elem((3, 4), 0.5));
//!
//! // Perform matrix multiplication
//! let result = input.matmul(&weights)?;
//! println!("Result shape: {:?}", result.shape());
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture Overview
//!
//! The runtime consists of several core components:
//!
//! - [`Model`] - ONNX model representation
//! - [`Graph`] - Computational graph with nodes and edges
//! - [`Tensor`] - N-dimensional array with type safety
//! - [`operators`] - ONNX operation implementations
//! - [`runtime`] - Execution engine
//!
//! ## Error Handling
//!
//! All operations return [`Result`] types with descriptive error messages.
//! The main error type is [`OnnxError`] which covers various failure modes.

pub mod error;
pub mod graph;
pub mod model;
pub mod operators;
pub mod runtime;
pub mod tensor;

// Re-export main types
pub use error::{OnnxError, Result};
pub use graph::{Graph, Node};
pub use model::Model;
pub use tensor::Tensor;

// Re-export commonly used external types
pub use ndarray;

/// Version of the RunNX crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_basic_tensor_operations() {
        let a = Tensor::from_array(Array2::from_elem((2, 3), 1.0));
        let b = Tensor::from_array(Array2::from_elem((2, 3), 2.0));
        
        let result = a.add(&b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        
        // Verify all elements are 3.0 (1.0 + 2.0)
        let data = result.data();
        assert!(data.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Tensor::from_array(Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap());
        let b = Tensor::from_array(Array2::from_shape_vec((3, 2), vec![1., 2., 3., 4., 5., 6.]).unwrap());
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        
        // Expected result: [[22, 28], [49, 64]]
        let data = result.data();
        let expected = vec![22.0, 28.0, 49.0, 64.0];
        for (actual, &expected) in data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
