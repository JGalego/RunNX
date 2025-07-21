use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use runnx::*;

fn bench_tensor_operations(c: &mut Criterion) {
    // Simple tensor add benchmark
    c.bench_function("tensor_add_10x10", |bencher| {
        let a = Tensor::ones(&[10, 10]);
        let b = Tensor::ones(&[10, 10]);
        bencher.iter(|| {
            black_box(a.add(&b).unwrap())
        });
    });
    
    // Matrix multiplication benchmark
    c.bench_function("tensor_matmul_10x10", |bencher| {
        let a = Tensor::ones(&[10, 10]);
        let b = Tensor::ones(&[10, 10]);
        bencher.iter(|| {
            black_box(a.matmul(&b).unwrap())
        });
    });
}

fn bench_model_inference(c: &mut Criterion) {
    let graph = Graph::create_simple_linear();
    let model = Model::new(graph);
    
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("input".to_string(), Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap());
    
    c.bench_function("model_inference", |bencher| {
        bencher.iter(|| {
            black_box(model.run(&inputs).unwrap())
        });
    });
}

criterion_group!(benches, bench_tensor_operations, bench_model_inference);
criterion_main!(benches);
