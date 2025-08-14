use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use runnx::*;
use std::collections::HashMap;
use std::hint::black_box;

fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");

    // Test different tensor sizes for add operation
    for size in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::new("add", size), size, |bencher, &size| {
            let a = Tensor::ones(&[size, size]);
            let b = Tensor::ones(&[size, size]);
            bencher.iter(|| black_box(a.add(&b).unwrap()));
        });
    }

    // Test different tensor sizes for matrix multiplication
    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bencher, &size| {
            let a = Tensor::ones(&[size, size]);
            let b_tensor = Tensor::ones(&[size, size]);
            bencher.iter(|| black_box(a.matmul(&b_tensor).unwrap()));
        });
    }

    // Test element-wise multiplication
    for size in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::new("mul", size), size, |bencher, &size| {
            let a = Tensor::ones(&[size, size]);
            let b_tensor = Tensor::ones(&[size, size]);
            bencher.iter(|| black_box(a.mul(&b_tensor).unwrap()));
        });
    }

    group.finish();
}

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");

    // ReLU benchmark
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("relu", size), size, |bencher, &size| {
            let tensor = Tensor::from_shape_vec(
                &[size],
                (0..size)
                    .map(|i| (i as f32) - (size as f32) / 2.0)
                    .collect(),
            )
            .unwrap();
            bencher.iter(|| black_box(tensor.relu()));
        });
    }

    // Sigmoid benchmark
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("sigmoid", size), size, |bencher, &size| {
            let tensor =
                Tensor::from_shape_vec(&[size], (0..size).map(|i| (i as f32) / 100.0).collect())
                    .unwrap();
            bencher.iter(|| black_box(tensor.sigmoid()));
        });
    }

    group.finish();
}

fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    // Zeros creation
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("zeros", size), size, |bencher, &size| {
            bencher.iter(|| black_box(Tensor::zeros(&[size])));
        });
    }

    // Ones creation
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("ones", size), size, |bencher, &size| {
            bencher.iter(|| black_box(Tensor::ones(&[size])));
        });
    }

    // From shape vec creation
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("from_shape_vec", size),
            size,
            |bencher, &size| {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                bencher.iter(|| black_box(Tensor::from_shape_vec(&[size], data.clone()).unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_tensor_reshape_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_reshape_transpose");

    // Reshape benchmarks
    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("reshape", size), size, |bencher, &size| {
            let tensor = Tensor::ones(&[size, size]);
            bencher.iter(|| black_box(tensor.reshape(&[size * size]).unwrap()));
        });
    }

    // Transpose benchmarks
    for size in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("transpose", size),
            size,
            |bencher, &size| {
                let tensor = Tensor::ones(&[size, size]);
                bencher.iter(|| black_box(tensor.transpose().unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_model_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_inference");

    // Simple linear model
    let graph = Graph::create_simple_linear();
    let model = Model::new(graph);

    let mut inputs = HashMap::new();
    inputs.insert(
        "input".to_string(),
        Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap(),
    );

    group.bench_function("simple_linear", |bencher| {
        bencher.iter(|| black_box(model.run(&inputs).unwrap()));
    });

    // Test different batch sizes for inference
    for batch_size in [1, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_inference", batch_size),
            batch_size,
            |bencher, &batch_size| {
                let mut batch_inputs = HashMap::new();
                batch_inputs.insert(
                    "input".to_string(),
                    Tensor::from_shape_vec(&[batch_size, 3], vec![1.0; batch_size * 3]).unwrap(),
                );
                bencher.iter(|| black_box(model.run(&batch_inputs).unwrap()));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_operations,
    bench_activation_functions,
    bench_tensor_creation,
    bench_tensor_reshape_transpose,
    bench_model_inference
);
criterion_main!(benches);
