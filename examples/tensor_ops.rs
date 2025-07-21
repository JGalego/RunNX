//! Tensor operations example
//!
//! This example demonstrates various tensor operations available
//! in the onnx-rs-min library.

use onnx_rs_min::Tensor;
use ndarray::{Array1, Array2, Array3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("=== Tensor Operations Example ===\n");

    // Basic tensor creation
    println!("1. Tensor Creation");
    println!("==================");
    
    // From array
    let t1 = Tensor::from_array(Array1::from_vec(vec![1., 2., 3., 4.]));
    println!("From vector: {}", t1);
    
    // Zeros and ones
    let t2 = Tensor::zeros(&[2, 3]);
    println!("Zeros (2x3):\n{}", t2);
    
    let t3 = Tensor::ones(&[3, 2]);
    println!("Ones (3x2):\n{}", t3);
    
    // From shape and data
    let t4 = Tensor::from_shape_vec(&[2, 4], vec![1., 2., 3., 4., 5., 6., 7., 8.])?;
    println!("From shape and data (2x4):\n{}", t4);

    // Element-wise operations
    println!("\n2. Element-wise Operations");
    println!("==========================");
    
    let a = Tensor::from_array(Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.])?);
    let b = Tensor::from_array(Array2::from_shape_vec((2, 3), vec![2., 2., 2., 3., 3., 3.])?);
    
    println!("Tensor A (2x3):\n{}", a);
    println!("Tensor B (2x3):\n{}", b);
    
    // Addition
    let add_result = a.add(&b)?;
    println!("A + B:\n{}", add_result);
    
    // Multiplication
    let mul_result = a.mul(&b)?;
    println!("A * B (element-wise):\n{}", mul_result);

    // Matrix operations
    println!("\n3. Matrix Operations");
    println!("====================");
    
    let m1 = Tensor::from_array(Array2::from_shape_vec((3, 4), 
        vec![1., 2., 3., 4., 
             5., 6., 7., 8., 
             9., 10., 11., 12.])?);
    let m2 = Tensor::from_array(Array2::from_shape_vec((4, 2), 
        vec![1., 0., 
             0., 1., 
             1., 1., 
             2., 1.])?);
    
    println!("Matrix M1 (3x4):\n{}", m1);
    println!("Matrix M2 (4x2):\n{}", m2);
    
    // Matrix multiplication
    let matmul_result = m1.matmul(&m2)?;
    println!("M1 × M2 (matrix multiplication):\n{}", matmul_result);
    
    // Transpose
    let m1_t = m1.transpose()?;
    println!("M1 transposed (4x3):\n{}", m1_t);

    // Reshaping
    println!("\n4. Reshaping Operations");
    println!("=======================");
    
    let original = Tensor::from_shape_vec(&[2, 6], 
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])?;
    println!("Original (2x6):\n{}", original);
    
    // Reshape to different dimensions
    let reshaped1 = original.reshape(&[3, 4])?;
    println!("Reshaped to (3x4):\n{}", reshaped1);
    
    let reshaped2 = original.reshape(&[4, 3])?;
    println!("Reshaped to (4x3):\n{}", reshaped2);
    
    let reshaped3 = original.reshape(&[12])?;
    println!("Reshaped to (12,):\n{}", reshaped3);

    // Activation functions
    println!("\n5. Activation Functions");
    println!("=======================");
    
    let input = Tensor::from_array(Array1::from_vec(vec![-3., -1., -0.5, 0., 0.5, 1., 3.]));
    println!("Input: {}", input);
    
    // ReLU
    let relu_output = input.relu();
    println!("ReLU: {}", relu_output);
    
    // Sigmoid
    let sigmoid_output = input.sigmoid();
    println!("Sigmoid: {}", sigmoid_output);

    // Advanced operations
    println!("\n6. Advanced Examples");
    println!("====================");
    
    // 3D tensor operations
    let tensor_3d = Tensor::zeros(&[2, 3, 4]);
    println!("3D tensor shape: {:?}", tensor_3d.shape());
    println!("3D tensor dimensions: {}", tensor_3d.ndim());
    println!("3D tensor total elements: {}", tensor_3d.len());
    
    // Chaining operations
    println!("\nChaining operations example:");
    let start = Tensor::from_array(Array2::from_shape_vec((2, 2), vec![1., 2., 3., 4.])?);
    println!("Start: {}", start);
    
    // Apply ReLU then add a constant tensor
    let step1 = start.relu(); // ReLU (no-op for positive values)
    let constant = Tensor::from_array(Array2::from_shape_vec((2, 2), vec![0.1, 0.1, 0.1, 0.1])?);
    let step2 = step1.add(&constant)?;
    println!("After ReLU + 0.1: {}", step2);
    
    // Then apply sigmoid
    let final_result = step2.sigmoid();
    println!("After sigmoid: {}", final_result);

    // Performance demonstration
    println!("\n7. Performance Test");
    println!("===================");
    
    let large_a = Tensor::ones(&[100, 100]);
    let large_b = Tensor::ones(&[100, 100]);
    
    println!("Created two 100x100 tensors");
    
    let start_time = std::time::Instant::now();
    let _result = large_a.add(&large_b)?;
    let add_time = start_time.elapsed();
    
    println!("Element-wise addition of 100x100 tensors: {:?}", add_time);
    
    let large_c = Tensor::ones(&[100, 100]);
    let large_d = Tensor::ones(&[100, 100]);
    
    let start_time = std::time::Instant::now();
    let _result = large_c.matmul(&large_d)?;
    let matmul_time = start_time.elapsed();
    
    println!("Matrix multiplication of 100x100 tensors: {:?}", matmul_time);

    // Error handling demonstration
    println!("\n8. Error Handling");
    println!("=================");
    
    let incompatible_a = Tensor::zeros(&[2, 3]);
    let incompatible_b = Tensor::zeros(&[3, 2]);
    
    println!("Attempting to add incompatible tensors (2x3) + (3x2):");
    match incompatible_a.add(&incompatible_b) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error: {}", e),
    }
    
    println!("\nAttempting matrix multiplication with incompatible shapes (2x3) × (4x2):");
    let incompatible_c = Tensor::zeros(&[4, 2]);
    match incompatible_a.matmul(&incompatible_c) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error: {}", e),
    }
    
    println!("\nAttempting invalid reshape (6 elements -> 8 elements):");
    let small_tensor = Tensor::ones(&[2, 3]);
    match small_tensor.reshape(&[2, 4]) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error: {}", e),
    }

    // Tensor comparison
    println!("\n9. Tensor Comparison");
    println!("====================");
    
    let t_a = Tensor::from_array(Array1::from_vec(vec![1., 2., 3.]));
    let t_b = Tensor::from_array(Array1::from_vec(vec![1., 2., 3.]));
    let t_c = Tensor::from_array(Array1::from_vec(vec![1., 2., 4.]));
    
    println!("Tensor A: {}", t_a);
    println!("Tensor B: {}", t_b);
    println!("Tensor C: {}", t_c);
    
    println!("A == B: {}", t_a == t_b);
    println!("A == C: {}", t_a == t_c);
    
    println!("\n=== Tensor Operations Example Complete ===");
    Ok(())
}
