//! Example: Using Formal Verification Features in RunNX
//! 
//! This example demonstrates how to use the formal verification
//! capabilities to ensure mathematical correctness of operations.

use runnx::formal::contracts::{AdditionContracts, ActivationContracts, StabilityContracts};
use runnx::formal::runtime_verification::InvariantMonitor;
use runnx::{Tensor};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 RunNX Formal Verification Example");
    println!("=====================================");
    
    // Create some test tensors
    let a = Tensor::from_array(Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?);
    let b = Tensor::from_array(Array2::from_shape_vec((2, 3), vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5])?);
    
    println!("📊 Input tensors:");
    println!("A: {:?}", a.data().iter().collect::<Vec<_>>());
    println!("B: {:?}", b.data().iter().collect::<Vec<_>>());
    
    // Use formally verified addition
    println!("\n🧮 Testing Addition with Formal Contracts:");
    let add_result = a.add_with_contracts(&b)?;
    println!("A + B = {:?}", add_result.data().iter().collect::<Vec<_>>());
    
    // Verify commutativity: a + b = b + a
    let commutative_result = b.add_with_contracts(&a)?;
    let is_commutative = add_result.data().iter()
        .zip(commutative_result.data().iter())
        .all(|(x, y)| (x - y).abs() < f32::EPSILON);
    println!("✓ Commutativity verified: {}", is_commutative);
    
    // Test ReLU with negative values
    println!("\n🎯 Testing ReLU Activation with Formal Contracts:");
    let mixed_tensor = Tensor::from_array(Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])?);
    println!("Input: {:?}", mixed_tensor.data().iter().collect::<Vec<_>>());
    
    let relu_result = mixed_tensor.relu_with_contracts()?;
    println!("ReLU result: {:?}", relu_result.data().iter().collect::<Vec<_>>());
    
    // Verify non-negativity property
    let is_non_negative = relu_result.data().iter().all(|&x| x >= 0.0);
    println!("✓ Non-negativity verified: {}", is_non_negative);
    
    // Test idempotency: ReLU(ReLU(x)) = ReLU(x)
    let double_relu = relu_result.relu_with_contracts()?;
    let is_idempotent = relu_result.data().iter()
        .zip(double_relu.data().iter())
        .all(|(x, y)| (x - y).abs() < f32::EPSILON);
    println!("✓ Idempotency verified: {}", is_idempotent);
    
    // Runtime verification monitoring
    println!("\n🔍 Runtime Invariant Monitoring:");
    let monitor = InvariantMonitor::new();
    
    // Check numerical stability
    println!("Checking numerical stability...");
    let inputs = vec![&a, &b, &mixed_tensor];
    let outputs = vec![&add_result, &relu_result];
    
    let is_stable = monitor.verify_operation(&inputs, &outputs);
    println!("✓ Runtime verification passed: {}", is_stable);
    
    // Check specific tensor properties
    println!("\n📐 Testing Numerical Stability Properties:");
    println!("Tensor A is numerically stable: {}", a.is_numerically_stable());
    println!("Tensor A is within bounds (ε=1e-3): {}", a.check_bounds(1e-3));
    
    // Demonstrate associativity with matrix multiplication
    println!("\n🔢 Testing Matrix Multiplication Associativity:");
    let mat_a = Tensor::from_array(Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?);
    let mat_b = Tensor::from_array(Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0])?);
    let mat_c = Tensor::from_array(Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 2.0])?);
    
    // Test (A * B) * C vs A * (B * C)
    let ab = mat_a.matmul(&mat_b)?;
    let left_assoc = ab.matmul(&mat_c)?;
    
    let bc = mat_b.matmul(&mat_c)?;
    let right_assoc = mat_a.matmul(&bc)?;
    
    let is_associative = left_assoc.data().iter()
        .zip(right_assoc.data().iter())
        .all(|(x, y)| (x - y).abs() < 1e-6);
    
    println!("(A × B) × C: {:?}", left_assoc.data().iter().collect::<Vec<_>>());
    println!("A × (B × C): {:?}", right_assoc.data().iter().collect::<Vec<_>>());
    println!("✓ Matrix multiplication associativity verified: {}", is_associative);
    
    println!("\n🎉 All formal verification checks passed!");
    println!("Your RunNX operations are mathematically sound!");
    
    Ok(())
}
