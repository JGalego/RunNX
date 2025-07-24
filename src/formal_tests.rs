#[cfg(test)]
mod formal_verification_tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::formal::contracts::*;
    use ndarray::Array2;
    use proptest::prelude::*;
    
    // Property-based test for addition commutativity
    proptest! {
        #[test]
        fn test_add_commutativity(
            a in prop::array::uniform32(prop::num::f32::NORMAL, 2..10),
            b in prop::array::uniform32(prop::num::f32::NORMAL, 2..10)
        ) {
            let shape = [2, a.len() / 2];
            let tensor_a = Tensor::from_array(Array2::from_shape_vec(shape, a.to_vec()).unwrap());
            let tensor_b = Tensor::from_array(Array2::from_shape_vec(shape, b.to_vec()).unwrap());
            
            let result1 = tensor_a.add(&tensor_b).unwrap();
            let result2 = tensor_b.add(&tensor_a).unwrap();
            
            // Check commutativity: a + b = b + a
            assert_eq!(result1.data(), result2.data());
        }
    }

    // Property-based test for softmax probability distribution
    proptest! {
        #[test]
        fn test_softmax_probability_distribution(
            data in prop::collection::vec(prop::num::f32::NORMAL, 1..20)
        ) {
            if let Ok(tensor) = Tensor::from_shape_vec(&[data.len()], data) {
                if let Ok(softmax_result) = tensor.softmax() {
                    // Test 1: Sum should equal 1.0
                    let sum: f32 = softmax_result.data().iter().sum();
                    prop_assert!((sum - 1.0).abs() < 1e-5, "Softmax sum should be 1.0, got: {}", sum);
                    
                    // Test 2: All values should be positive
                    for &value in softmax_result.data().iter() {
                        prop_assert!(value > 0.0, "Softmax values should be positive, got: {}", value);
                        prop_assert!(value < 1.0, "Softmax values should be < 1.0, got: {}", value);
                    }
                }
            }
        }
    }

    // Property-based test for softmax numerical stability
    proptest! {
        #[test]
        fn test_softmax_numerical_stability(
            data in prop::collection::vec(prop::num::f32::NORMAL, 1..10),
            shift in prop::num::f32::NORMAL
        ) {
            if let Ok(tensor) = Tensor::from_shape_vec(&[data.len()], data.clone()) {
                // Apply shift to all elements
                let shifted_data: Vec<f32> = data.iter().map(|&x| x + shift).collect();
                if let Ok(shifted_tensor) = Tensor::from_shape_vec(&[data.len()], shifted_data) {
                    
                    if let (Ok(softmax1), Ok(softmax2)) = (tensor.softmax(), shifted_tensor.softmax()) {
                        // Softmax should be invariant under constant shifts
                        for (a, b) in softmax1.data().iter().zip(softmax2.data().iter()) {
                            prop_assert!((a - b).abs() < 1e-5, 
                                "Softmax should be stable under shifts: {} vs {}", a, b);
                        }
                    }
                }
            }
        }
    }

    // Property-based test for ReLU idempotency
    proptest! {
        #[test]
        fn test_relu_idempotency(
            data in prop::collection::vec(prop::num::f32::NORMAL, 1..20)
        ) {
            if let Ok(tensor) = Tensor::from_shape_vec(&[data.len()], data) {
                let relu1 = tensor.relu();
                let relu2 = relu1.relu();
                
                // ReLU should be idempotent: ReLU(ReLU(x)) = ReLU(x)
                for (a, b) in relu1.data().iter().zip(relu2.data().iter()) {
                    prop_assert_eq!(*a, *b, "ReLU should be idempotent");
                }
                
                // All ReLU outputs should be non-negative
                for &value in relu1.data().iter() {
                    prop_assert!(value >= 0.0, "ReLU output should be non-negative, got: {}", value);
                }
            }
        }
    }

    // Property-based test for tensor shape preservation
    proptest! {
        #[test]
        fn test_unary_operations_shape_preservation(
            rows in 1usize..5,
            cols in 1usize..5,
            data in prop::collection::vec(prop::num::f32::NORMAL, 1..25)
        ) {
            if rows * cols <= data.len() {
                let shape = [rows, cols];
                if let Ok(tensor) = Tensor::from_shape_vec(&shape, data[..rows*cols].to_vec()) {
                    // Test ReLU shape preservation
                    let relu_result = tensor.relu();
                    prop_assert_eq!(relu_result.shape(), tensor.shape());
                    
                    // Test Sigmoid shape preservation
                    let sigmoid_result = tensor.sigmoid();
                    prop_assert_eq!(sigmoid_result.shape(), tensor.shape());
                    
                    // Test Softmax shape preservation
                    if let Ok(softmax_result) = tensor.softmax() {
                        prop_assert_eq!(softmax_result.shape(), tensor.shape());
                    }
                }
            }
        }
    }

    // Property-based test for YOLO operator contract validation
    proptest! {
        #[test]
        fn test_yolo_operator_contracts_validation(
            dim1 in 1usize..5,
            dim2 in 1usize..5,
            data in prop::collection::vec(prop::num::f32::NORMAL, 1..25)
        ) {
            if dim1 * dim2 <= data.len() {
                let shape = [dim1, dim2];
                if let Ok(tensor) = Tensor::from_shape_vec(&shape, data[..dim1*dim2].to_vec()) {
                    
                    // Test softmax contracts
                    let softmax_result = tensor.softmax_with_contracts();
                    if softmax_result.is_ok() {
                        let result = softmax_result.unwrap();
                        let sum: f32 = result.data().iter().sum();
                        prop_assert!((sum - 1.0).abs() < 1e-5);
                    }
                    
                    // Test concat contracts with invalid parameters
                    let other_tensor = if let Ok(t) = Tensor::from_shape_vec(&[dim1 + 1, dim2], vec![0.0; (dim1 + 1) * dim2]) {
                        let concat_result = tensor.concat_with_contracts(&t, 0);
                        prop_assert!(concat_result.is_err(), "Concat should fail with incompatible shapes");
                        Some(t)
                    } else {
                        None
                    };
                    
                    // Test slice contracts with invalid parameters
                    let slice_result = tensor.slice_with_contracts(&[0, 0], &[0, 1], None, None); // Invalid: start >= end
                    prop_assert!(slice_result.is_err(), "Slice should fail with invalid ranges");
                    
                    // Test upsample contracts with invalid parameters
                    let upsample_result = tensor.upsample_with_contracts(&[2.0]); // Wrong number of scale factors
                    prop_assert!(upsample_result.is_err(), "Upsample should fail with wrong number of scale factors");
                }
            }
        }
    }
    
    // Property-based test for matrix multiplication associativity
    proptest! {
        #[test]
        fn test_matmul_associativity(
            a_data in prop::collection::vec(prop::num::f32::NORMAL, 4..16),
            b_data in prop::collection::vec(prop::num::f32::NORMAL, 4..16),
            c_data in prop::collection::vec(prop::num::f32::NORMAL, 4..16)
        ) {
            // Create compatible matrix dimensions
            let dim = 2;
            if a_data.len() >= dim * dim && b_data.len() >= dim * dim && c_data.len() >= dim * dim {
                let tensor_a = Tensor::from_array(
                    Array2::from_shape_vec((dim, dim), a_data[..dim*dim].to_vec()).unwrap()
                );
                let tensor_b = Tensor::from_array(
                    Array2::from_shape_vec((dim, dim), b_data[..dim*dim].to_vec()).unwrap()
                );
                let tensor_c = Tensor::from_array(
                    Array2::from_shape_vec((dim, dim), c_data[..dim*dim].to_vec()).unwrap()
                );
                
                // Test associativity: (a * b) * c = a * (b * c)
                let left = tensor_a.matmul(&tensor_b).unwrap().matmul(&tensor_c).unwrap();
                let right = tensor_a.matmul(&tensor_b.matmul(&tensor_c).unwrap()).unwrap();
                
                // Allow small numerical errors
                for (l, r) in left.data().iter().zip(right.data().iter()) {
                    prop_assert!((l - r).abs() < 1e-6);
                }
            }
        }
    }
    
    // Property-based test for ReLU properties
    proptest! {
        #[test]
        fn test_relu_properties(
            data in prop::collection::vec(prop::num::f32::NORMAL, 4..16)
        ) {
            let shape = [data.len() / 2, 2];
            let tensor = Tensor::from_array(Array2::from_shape_vec(shape, data).unwrap());
            let relu_result = tensor.relu().unwrap();
            
            // Test idempotency: ReLU(ReLU(x)) = ReLU(x)
            let double_relu = relu_result.relu().unwrap();
            assert_eq!(relu_result.data(), double_relu.data());
            
            // Test non-negativity: all outputs >= 0
            for &value in relu_result.data() {
                prop_assert!(value >= 0.0);
            }
            
            // Test monotonicity preservation
            for (original, activated) in tensor.data().iter().zip(relu_result.data().iter()) {
                if *original > 0.0 {
                    prop_assert_eq!(*original, *activated);
                } else {
                    prop_assert_eq!(*activated, 0.0);
                }
            }
        }
    }
    
    // Property-based test for numerical stability
    proptest! {
        #[test]
        fn test_numerical_stability(
            data in prop::collection::vec(-1000.0f32..1000.0f32, 4..16)
        ) {
            let shape = [2, data.len() / 2];
            let tensor = Tensor::from_array(Array2::from_shape_vec(shape, data).unwrap());
            
            // Test that operations don't produce NaN or infinity
            let result = tensor.add(&tensor).unwrap();
            for &value in result.data() {
                prop_assert!(value.is_finite());
            }
            
            // Test sigmoid bounds
            let sigmoid_result = tensor.sigmoid().unwrap();
            for &value in sigmoid_result.data() {
                prop_assert!(value > 0.0 && value < 1.0);
            }
        }
    }
}