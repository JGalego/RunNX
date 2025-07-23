
#[cfg(test)]
mod operator_property_tests {{
    use super::*;
    use crate::tensor::Tensor;
    use proptest::prelude::*;
    
    // Property test for addition commutativity
    proptest! {{
        #[test]
        fn test_add_commutativity(
            a in prop::collection::vec(any::<f32>(), 1..100),
            shape in prop::collection::vec(1usize..10, 1..4)
        ) {{
            let tensor_a = Tensor::new(a.clone(), shape.clone()).unwrap();
            let tensor_b = Tensor::new(a.clone(), shape.clone()).unwrap();
            
            let result1 = tensor_a.add(&tensor_b).unwrap();
            let result2 = tensor_b.add(&tensor_a).unwrap();
            
            // Commutativity: a + b == b + a
            prop_assert_eq!(result1.data(), result2.data());
        }}
    }}
    
    // Property test for ReLU non-negativity
    proptest! {{
        #[test]
        fn test_relu_non_negative(
            data in prop::collection::vec(any::<f32>(), 1..100),
            shape in prop::collection::vec(1usize..10, 1..4)
        ) {{
            let tensor = Tensor::new(data, shape).unwrap();
            let result = tensor.relu().unwrap();
            
            // Non-negativity: all outputs >= 0
            for &value in result.data() {{
                prop_assert!(value >= 0.0);
            }}
        }}
    }}
    
    // Property test for matrix multiplication associativity
    proptest! {{
        #[test]
        fn test_matmul_associativity(
            m in 1usize..10,
            n in 1usize..10,
            p in 1usize..10,
            q in 1usize..10
        ) {{
            let a_data: Vec<f32> = (0..m*n).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..n*p).map(|i| i as f32).collect();
            let c_data: Vec<f32> = (0..p*q).map(|i| i as f32).collect();
            
            let a = Tensor::new(a_data, vec![m, n]).unwrap();
            let b = Tensor::new(b_data, vec![n, p]).unwrap();
            let c = Tensor::new(c_data, vec![p, q]).unwrap();
            
            // (A * B) * C
            let ab = a.matmul(&b).unwrap();
            let ab_c = ab.matmul(&c).unwrap();
            
            // A * (B * C)
            let bc = b.matmul(&c).unwrap();
            let a_bc = a.matmul(&bc).unwrap();
            
            // Associativity: (A * B) * C == A * (B * C)
            for (i, (&v1, &v2)) in ab_c.data().iter().zip(a_bc.data().iter()).enumerate() {{
                prop_assert!((v1 - v2).abs() < 1e-5, "Mismatch at index {{}}: {{}} vs {{}}", i, v1, v2);
            }}
        }}
    }}
}}
