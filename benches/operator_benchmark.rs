use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use runnx::operators::*;
use runnx::*;
use std::collections::HashMap;
use std::hint::black_box;

fn bench_basic_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_operators");

    // Add operator benchmark
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("add", size), size, |b, &size| {
            let input1 = Tensor::ones(&[size]);
            let input2 = Tensor::ones(&[size]);
            let inputs = vec![input1, input2];
            let attributes = HashMap::new();

            b.iter(|| {
                black_box(execute_operator(&OperatorType::Add, &inputs, &attributes).unwrap())
            });
        });
    }

    // Mul operator benchmark
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("mul", size), size, |b, &size| {
            let input1 = Tensor::ones(&[size]);
            let input2 = Tensor::ones(&[size]);
            let inputs = vec![input1, input2];
            let attributes = HashMap::new();

            b.iter(|| {
                black_box(execute_operator(&OperatorType::Mul, &inputs, &attributes).unwrap())
            });
        });
    }

    // MatMul operator benchmark
    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |b, &size| {
            let input1 = Tensor::ones(&[size, size]);
            let input2 = Tensor::ones(&[size, size]);
            let inputs = vec![input1, input2];
            let attributes = HashMap::new();

            b.iter(|| {
                black_box(execute_operator(&OperatorType::MatMul, &inputs, &attributes).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_activation_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_operators");

    // ReLU operator benchmark
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("relu", size), size, |b, &size| {
            let input = Tensor::from_shape_vec(
                &[size],
                (0..size)
                    .map(|i| (i as f32) - (size as f32) / 2.0)
                    .collect(),
            )
            .unwrap();
            let inputs = vec![input];
            let attributes = HashMap::new();

            b.iter(|| {
                black_box(execute_operator(&OperatorType::Relu, &inputs, &attributes).unwrap())
            });
        });
    }

    // Sigmoid operator benchmark
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("sigmoid", size), size, |b, &size| {
            let input =
                Tensor::from_shape_vec(&[size], (0..size).map(|i| (i as f32) / 100.0).collect())
                    .unwrap();
            let inputs = vec![input];
            let attributes = HashMap::new();

            b.iter(|| {
                black_box(execute_operator(&OperatorType::Sigmoid, &inputs, &attributes).unwrap())
            });
        });
    }

    // Softmax operator benchmark
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("softmax", size), size, |b, &size| {
            let input =
                Tensor::from_shape_vec(&[1, size], (0..size).map(|i| (i as f32) / 10.0).collect())
                    .unwrap();
            let inputs = vec![input];
            let mut attributes = HashMap::new();
            attributes.insert("axis".to_string(), "1".to_string());

            b.iter(|| {
                black_box(execute_operator(&OperatorType::Softmax, &inputs, &attributes).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_shape_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_operators");

    // Reshape operator benchmark
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("reshape", size), size, |b, &size| {
            let input = Tensor::ones(&[size, 1]);
            let shape_tensor = Tensor::from_shape_vec(&[1], vec![size as f32]).unwrap();
            let inputs = vec![input, shape_tensor];
            let attributes = HashMap::new();

            b.iter(|| {
                black_box(execute_operator(&OperatorType::Reshape, &inputs, &attributes).unwrap())
            });
        });
    }

    // Transpose operator benchmark
    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("transpose", size), size, |b, &size| {
            let input = Tensor::ones(&[size, size]);
            let inputs = vec![input];
            let mut attributes = HashMap::new();
            attributes.insert("perm".to_string(), "[1,0]".to_string());

            b.iter(|| {
                black_box(execute_operator(&OperatorType::Transpose, &inputs, &attributes).unwrap())
            });
        });
    }

    // Concat operator benchmark
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("concat", size), size, |b, &size| {
            let input1 = Tensor::ones(&[size]);
            let input2 = Tensor::ones(&[size]);
            let inputs = vec![input1, input2];
            let mut attributes = HashMap::new();
            attributes.insert("axis".to_string(), "0".to_string());

            b.iter(|| {
                black_box(execute_operator(&OperatorType::Concat, &inputs, &attributes).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_pooling_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("pooling_operators");

    // MaxPool operator benchmark
    for size in [16, 32, 64].iter() {
        group.bench_with_input(BenchmarkId::new("maxpool", size), size, |b, &size| {
            let input = Tensor::ones(&[1, 1, size, size]);
            let inputs = vec![input];
            let mut attributes = HashMap::new();
            attributes.insert("kernel_shape".to_string(), "[2,2]".to_string());
            attributes.insert("strides".to_string(), "[2,2]".to_string());

            b.iter(|| {
                black_box(execute_operator(&OperatorType::MaxPool, &inputs, &attributes).unwrap())
            });
        });
    }

    group.finish();
}

criterion_group!(
    operator_benches,
    bench_basic_operators,
    bench_activation_operators,
    bench_shape_operators,
    bench_pooling_operators
);
criterion_main!(operator_benches);
