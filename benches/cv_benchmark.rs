use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use runnx::operators::*;
use runnx::*;
use std::collections::HashMap;
use std::hint::black_box;

fn bench_computer_vision_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("computer_vision_operators");

    // Conv operator benchmark - typical YOLO conv layers
    for (channels, size) in [(3, 416), (32, 208), (64, 104), (128, 52)].iter() {
        group.bench_with_input(
            BenchmarkId::new("conv", format!("{channels}ch_{size}x{size}")),
            &(channels, size),
            |bencher, &(channels, size)| {
                let input = Tensor::ones(&[1, *channels, *size, *size]);
                let weights = Tensor::ones(&[32, *channels, 3, 3]);
                let bias = Tensor::ones(&[32]);
                let inputs = vec![input, weights, bias];
                let mut attributes = HashMap::new();
                attributes.insert("strides".to_string(), "[1,1]".to_string());
                attributes.insert("pads".to_string(), "[1,1,1,1]".to_string());

                bencher.iter(|| {
                    black_box(execute_operator(&OperatorType::Conv, &inputs, &attributes).unwrap())
                });
            },
        );
    }

    // MaxPool operator benchmark - YOLO pooling layers
    for size in [416, 208, 104, 52, 26, 13].iter() {
        group.bench_with_input(BenchmarkId::new("maxpool", size), size, |bencher, &size| {
            let input = Tensor::ones(&[1, 64, size, size]);
            let inputs = vec![input];
            let mut attributes = HashMap::new();
            attributes.insert("kernel_shape".to_string(), "[2,2]".to_string());
            attributes.insert("strides".to_string(), "[2,2]".to_string());

            bencher.iter(|| {
                black_box(execute_operator(&OperatorType::MaxPool, &inputs, &attributes).unwrap())
            });
        });
    }

    // Resize/Upsample operator benchmark - for YOLO FPN
    for (input_size, scale) in [(13, 2), (26, 2), (52, 2)].iter() {
        group.bench_with_input(
            BenchmarkId::new(
                "upsample",
                format!("{input_size}x{input_size}_scale{scale}"),
            ),
            &(input_size, scale),
            |bencher, &(input_size, scale)| {
                let input = Tensor::ones(&[1, 256, *input_size, *input_size]);
                let inputs = vec![input];
                let mut attributes = HashMap::new();
                attributes.insert(
                    "scales".to_string(),
                    format!("[1.0,1.0,{scale}.0,{scale}.0]"),
                );

                bencher.iter(|| {
                    black_box(
                        execute_operator(&OperatorType::Upsample, &inputs, &attributes).unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_yolo_postprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("yolo_postprocessing");

    // Softmax for classification head
    for num_classes in [80, 91, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("softmax_classes", num_classes),
            num_classes,
            |bencher, &num_classes| {
                let input = Tensor::from_shape_vec(
                    &[1, num_classes],
                    (0..num_classes).map(|i| (i as f32) / 10.0).collect(),
                )
                .unwrap();
                let inputs = vec![input];
                let mut attributes = HashMap::new();
                attributes.insert("axis".to_string(), "1".to_string());

                bencher.iter(|| {
                    black_box(
                        execute_operator(&OperatorType::Softmax, &inputs, &attributes).unwrap(),
                    )
                });
            },
        );
    }

    // Sigmoid for objectness and bbox regression
    for num_anchors in [3, 9, 27].iter() {
        group.bench_with_input(
            BenchmarkId::new("sigmoid_anchors", num_anchors),
            num_anchors,
            |bencher, &num_anchors| {
                let input = Tensor::from_shape_vec(
                    &[1, num_anchors * 4],
                    (0..(num_anchors * 4)).map(|i| (i as f32) / 100.0).collect(),
                )
                .unwrap();
                let inputs = vec![input];
                let attributes = HashMap::new();

                bencher.iter(|| {
                    black_box(
                        execute_operator(&OperatorType::Sigmoid, &inputs, &attributes).unwrap(),
                    )
                });
            },
        );
    }

    // NonMaxSuppression for final detection filtering
    for num_boxes in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("nms", num_boxes),
            num_boxes,
            |bencher, &num_boxes| {
                let boxes = Tensor::from_shape_vec(
                    &[num_boxes, 4],
                    (0..(num_boxes * 4)).map(|i| (i as f32) % 100.0).collect(),
                )
                .unwrap();
                let scores = Tensor::from_shape_vec(
                    &[num_boxes],
                    (0..num_boxes)
                        .map(|i| (i as f32) / (num_boxes as f32))
                        .collect(),
                )
                .unwrap();
                let inputs = vec![boxes, scores];
                let mut attributes = HashMap::new();
                attributes.insert("iou_threshold".to_string(), "0.5".to_string());
                attributes.insert("score_threshold".to_string(), "0.1".to_string());

                bencher.iter(|| {
                    black_box(
                        execute_operator(&OperatorType::NonMaxSuppression, &inputs, &attributes)
                            .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    // Batch normalization - common in YOLO backbones
    for (batch_size, channels) in [(1, 32), (4, 64), (8, 128), (16, 256)].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_norm", format!("batch{batch_size}_ch{channels}")),
            &(batch_size, channels),
            |bencher, &(batch_size, channels)| {
                let input = Tensor::ones(&[*batch_size, *channels, 64, 64]);
                let scale = Tensor::ones(&[*channels]);
                let bias = Tensor::ones(&[*channels]);
                let mean = Tensor::zeros(&[*channels]);
                let var = Tensor::ones(&[*channels]);
                let inputs = vec![input, scale, bias, mean, var];
                let mut attributes = HashMap::new();
                attributes.insert("epsilon".to_string(), "1e-5".to_string());

                bencher.iter(|| {
                    black_box(
                        execute_operator(&OperatorType::BatchNormalization, &inputs, &attributes)
                            .unwrap(),
                    )
                });
            },
        );
    }

    // Concat for feature map concatenation (FPN)
    for num_feature_maps in [2, 3, 4].iter() {
        group.bench_with_input(
            BenchmarkId::new("concat_features", num_feature_maps),
            num_feature_maps,
            |bencher, &num_feature_maps| {
                let mut inputs = Vec::new();
                for _ in 0..num_feature_maps {
                    inputs.push(Tensor::ones(&[1, 256, 52, 52]));
                }
                let mut attributes = HashMap::new();
                attributes.insert("axis".to_string(), "1".to_string());

                bencher.iter(|| {
                    black_box(
                        execute_operator(&OperatorType::Concat, &inputs, &attributes).unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_shape_manipulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_manipulation");

    // Reshape for flattening detection heads
    for (height, width) in [(13, 13), (26, 26), (52, 52)].iter() {
        group.bench_with_input(
            BenchmarkId::new("reshape_detection_head", format!("{height}x{width}")),
            &(height, width),
            |bencher, &(height, width)| {
                let input = Tensor::ones(&[1, 255, *height, *width]); // 255 = 3 * (80 + 5) for YOLO
                let new_shape =
                    Tensor::from_shape_vec(&[3], vec![1.0, (255 * height * width) as f32, 1.0])
                        .unwrap();
                let inputs = vec![input, new_shape];
                let attributes = HashMap::new();

                bencher.iter(|| {
                    black_box(
                        execute_operator(&OperatorType::Reshape, &inputs, &attributes).unwrap(),
                    )
                });
            },
        );
    }

    // Transpose for changing data layout
    for size in [13, 26, 52].iter() {
        group.bench_with_input(
            BenchmarkId::new("transpose", size),
            size,
            |bencher, &size| {
                let input = Tensor::ones(&[1, 255, size, size]);
                let inputs = vec![input];
                let mut attributes = HashMap::new();
                attributes.insert("perm".to_string(), "[0,2,3,1]".to_string());

                bencher.iter(|| {
                    black_box(
                        execute_operator(&OperatorType::Transpose, &inputs, &attributes).unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    cv_benches,
    bench_computer_vision_operators,
    bench_yolo_postprocessing,
    bench_batch_operations,
    bench_shape_manipulation
);
criterion_main!(cv_benches);
