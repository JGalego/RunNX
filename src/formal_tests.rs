#[cfg(test)]
mod formal_verification_tests {
    use super::*;
    use crate::tensor::Tensor;
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