//! YOLOv8 Object Detection with Bounding Box Visualization
//!
//! This example demonstrates complete object detection pipeline:
//! 1. Load YOLOv8n ONNX model
//! 2. Process input image
//! 3. Run inference
//! 4. Post-process results (NMS, thresholding)
//! 5. Draw bounding boxes on the image
//! 6. Save the output image

use image::{imageops::FilterType, DynamicImage, ImageReader, Rgb};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use runnx::*;
use std::collections::HashMap;
use std::time::Instant;

/// Detection result with bounding box and class information
#[derive(Debug, Clone)]
struct Detection {
    bbox: [f32; 4], // [x1, y1, x2, y2] in original image coordinates
    confidence: f32,
    class_name: String,
}

fn main() -> runnx::Result<()> {
    println!("🎯 RunNX YOLOv8 Object Detection with Bounding Box Visualization");
    println!("================================================================");

    // Initialize logging
    env_logger::init();

    // Step 1: Load the YOLOv8n model
    let model = load_yolov8n_model()?;
    println!("✅ YOLOv8n model loaded successfully!");

    // Step 2: Load and prepare the bus image
    let image_path = "assets/bus.jpg";
    let (input_tensor, original_image, original_size) = load_and_prepare_image(image_path)?;
    println!(
        "✅ Image loaded and prepared: shape {:?}",
        input_tensor.shape()
    );

    // Step 3: Run inference
    let start_time = Instant::now();
    let outputs = run_inference(&model, input_tensor)?;
    let inference_time = start_time.elapsed();
    println!(
        "✅ Inference completed successfully in {:.2}s!",
        inference_time.as_secs_f32()
    );

    // Step 4: Post-process detections
    let conf_threshold = 0.25; // Confidence threshold
    let iou_threshold = 0.45; // NMS IoU threshold

    let detections =
        post_process_detections(&outputs, original_size, conf_threshold, iou_threshold)?;

    // Step 5: Display results
    print_detection_results(&detections);

    // Step 6: Draw bounding boxes on the image
    if !detections.is_empty() {
        let output_image = draw_detections_on_image(original_image, &detections)?;

        // Save the output image
        let output_path = "assets/bus_with_detections.jpg";
        output_image.save(output_path).map_err(|e| {
            error::OnnxError::runtime_error(format!("Failed to save output image: {e}"))
        })?;

        println!("💾 Output image saved to: {output_path}");
    } else {
        println!("❌ No objects detected - no output image generated");
    }

    println!("\n🎉 YOLOv8 object detection demo completed!");
    println!("This demonstrates RunNX's capability to perform complete object detection");
    println!("with real images, including post-processing and visualization.");

    Ok(())
}

/// Load the YOLOv8n model from file
fn load_yolov8n_model() -> runnx::Result<model::Model> {
    let model_path = "yolov8n.onnx";

    if !std::path::Path::new(model_path).exists() {
        return Err(error::OnnxError::model_load_error(
            "YOLOv8n model file (yolov8n.onnx) not found in current directory. 
            Please download it from: https://github.com/ultralytics/assets/releases/"
                .to_string(),
        ));
    }

    println!("📂 Loading YOLOv8n model from {model_path}...");
    let model = model::Model::from_onnx_file(model_path)?;

    println!("   📛 Model name: {}", model.name());
    println!("   📊 Graph: {}", model.graph.name);
    println!("   🔗 Nodes: {}", model.graph.nodes.len());
    println!("   📥 Inputs: {}", model.graph.inputs.len());
    println!("   📤 Outputs: {}", model.graph.outputs.len());

    Ok(model)
}

/// Load and prepare image for YOLOv8 inference
fn load_and_prepare_image(
    image_path: &str,
) -> runnx::Result<(tensor::Tensor, DynamicImage, (u32, u32))> {
    if !std::path::Path::new(image_path).exists() {
        return Err(error::OnnxError::runtime_error(format!(
            "Image file '{image_path}' not found"
        )));
    }

    println!("🖼️  Loading image from: {image_path}");

    // Load the image
    let original_image = ImageReader::open(image_path)
        .map_err(|e| error::OnnxError::runtime_error(format!("Failed to open image: {e}")))?
        .decode()
        .map_err(|e| error::OnnxError::runtime_error(format!("Failed to decode image: {e}")))?;

    let original_size = (original_image.width(), original_image.height());
    println!(
        "   📐 Original image size: {}x{}",
        original_size.0, original_size.1
    );

    // Prepare a square image for inference (YOLOv8 expects square input, preserve aspect ratio like Python)
    let length = original_size.0.max(original_size.1);
    println!("   📏 Square canvas size: {length}x{length}");

    // Create square canvas (like Python: image = np.zeros((length, length, 3), np.uint8))
    let mut square_image = DynamicImage::new_rgb8(length, length);

    // Place original image at top-left (like Python: image[0:height, 0:width] = original_image)
    // Use overlay to place the image
    image::imageops::overlay(&mut square_image, &original_image, 0, 0);

    // Resize square image to 640x640 (like Python: cv2.dnn.blobFromImage(image, ..., size=(640, 640)))
    let resized = square_image.resize_exact(640, 640, FilterType::Lanczos3);
    println!("   🔄 Resized to: 640x640");

    // Convert to RGB if needed
    let rgb_image = resized.to_rgb8();

    // Prepare tensor data: normalize to [0, 1] and convert to CHW format
    let mut tensor_data = Vec::with_capacity(3 * 640 * 640);

    // Convert from HWC to CHW format (channels first)
    for channel in 0..3 {
        for y in 0..640 {
            for x in 0..640 {
                let pixel = rgb_image.get_pixel(x, y);
                let normalized_value = pixel[channel] as f32 / 255.0;
                tensor_data.push(normalized_value);
            }
        }
    }

    // Check if image is essentially blank (which would explain lack of detections)
    let avg_value = tensor_data.iter().sum::<f32>() / tensor_data.len() as f32;
    let variance = tensor_data
        .iter()
        .map(|x| (x - avg_value).powi(2))
        .sum::<f32>()
        / tensor_data.len() as f32;
    println!("   📊 Image stats: avg={avg_value:.4}, variance={variance:.6}");

    if variance < 0.001 {
        println!("   ⚠️  WARNING: Very low variance - image might be mostly uniform!");
    }

    let input_tensor = tensor::Tensor::from_shape_vec(&[1, 3, 640, 640], tensor_data)?;

    println!("   📏 Tensor shape: {:?}", input_tensor.shape());
    println!("   🎨 Pixel range: [0.0, 1.0] (normalized)");
    println!("   ✅ Image preprocessing completed");

    Ok((input_tensor, original_image, original_size))
}

/// Run inference on the model
fn run_inference(
    model: &model::Model,
    input_tensor: tensor::Tensor,
) -> runnx::Result<HashMap<String, tensor::Tensor>> {
    println!("🔮 Running YOLOv8 inference...");
    println!("   📥 Input name: 'images'");

    // Debug: Check input tensor stats before inference
    let input_data = input_tensor.data();
    let input_slice = input_data.as_slice().unwrap();
    let input_min = input_slice.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let input_max = input_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let input_avg = input_slice.iter().sum::<f32>() / input_slice.len() as f32;
    println!(
        "   🔍 Input tensor stats: min={input_min:.6}, max={input_max:.6}, avg={input_avg:.6}"
    );

    let mut inputs = HashMap::new();
    inputs.insert("images".to_string(), input_tensor);

    println!("   ⚙️  Creating RunNX runtime...");
    let runtime = runtime::Runtime::with_debug();

    println!("   🚀 Executing inference...");
    let outputs = runtime.execute(&model.graph, inputs)?;

    for (name, tensor) in &outputs {
        println!("   📤 Output '{}': shape {:?}", name, tensor.shape());

        // Debug: Check if the output tensor contains meaningful data
        let output_data = tensor.data();
        let output_slice = output_data.as_slice().unwrap();
        let output_min = output_slice.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let output_max = output_slice
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let output_avg = output_slice.iter().sum::<f32>() / output_slice.len() as f32;
        println!("   🔍 Output '{name}' stats: min={output_min:.6}, max={output_max:.6}, avg={output_avg:.6}");

        // Check if output looks like it's just anchor grid data
        let first_10 = &output_slice[0..10.min(output_slice.len())];
        println!("   🔍 Output '{name}' first 10 values: {first_10:?}");

        // Look for signs that inference actually ran
        let has_variety = output_max - output_min > 10.0; // Real inference should have wide range
        let has_large_values = output_max > 5.0 || output_min < -5.0; // Logits should have large values
        println!("   🧪 Output '{name}' analysis: variety={has_variety}, large_values={has_large_values}");
    }

    Ok(outputs)
}

/// Post-process YOLOv8 raw outputs to extract detections
fn post_process_detections(
    outputs: &HashMap<String, runnx::Tensor>,
    original_size: (u32, u32),
    conf_threshold: f32,
    iou_threshold: f32,
) -> runnx::Result<Vec<Detection>> {
    println!("🔍 Post-processing detections...");

    // Get the output tensor - YOLOv8 typically outputs to "output0"
    let output_tensor = outputs
        .get("output0")
        .or_else(|| outputs.values().next())
        .ok_or_else(|| error::OnnxError::runtime_error("No output tensor found".to_string()))?;

    let output_shape = output_tensor.shape();
    println!("   📊 Output shape: {output_shape:?}");

    // Expected shape: [1, 84, 8400] (batch=1, features=84, anchors=8400)
    if output_shape.len() != 3 || output_shape[0] != 1 {
        return Err(error::OnnxError::runtime_error(format!(
            "Unexpected output shape: {output_shape:?}, expected [1, features, anchors]"
        )));
    }

    let num_features = output_shape[1];
    let num_anchors = output_shape[2];
    let num_classes = num_features - 4; // 4 bbox coordinates + class probabilities

    println!("   🎯 Number of anchors: {num_anchors}");
    println!("   📝 Number of classes: {num_classes}");
    println!("   🎚️  Confidence threshold: {conf_threshold}");

    let output_data = output_tensor.data();
    let mut detections = Vec::new();

    // Quick debug check of tensor data
    let flat_data = output_data.as_slice().unwrap();
    println!(
        "   🔍 First 10 raw tensor values: {:?}",
        &flat_data[0..10.min(flat_data.len())]
    );

    // Check if we're getting anchor grid pattern (indicating inference didn't run properly)
    let appears_to_be_grid = flat_data[0] == 0.00625 && flat_data[1] == 0.01875;
    if appears_to_be_grid {
        println!("   ⚠️  WARNING: Output appears to be anchor grid pattern - inference may not have run!");
    }

    let mut debug_printed = false;
    let mut max_seen_conf = 0.0f32;

    // Process each anchor
    for anchor_idx in 0..num_anchors {
        // The tensor has shape [1, 83, 8400] meaning [batch, feature, anchor]
        let cx = output_data[[0, 0, anchor_idx]]; // center_x
        let cy = output_data[[0, 1, anchor_idx]]; // center_y
        let w = output_data[[0, 2, anchor_idx]]; // width
        let h = output_data[[0, 3, anchor_idx]]; // height

        // Find the class with highest confidence
        let mut max_class_conf = 0.0;
        let mut max_class_id = 0;

        for class_idx in 0..num_classes {
            let class_conf = output_data[[0, 4 + class_idx, anchor_idx]];
            // YOLOv8 class confidences are already probabilities (0-1), find the maximum
            if class_conf > max_class_conf {
                max_class_conf = class_conf;
                max_class_id = class_idx;
            }
        }

        // No sigmoid needed - class confidences are already probabilities

        // Track maximum confidence seen
        if max_class_conf > max_seen_conf {
            max_seen_conf = max_class_conf;
        }

        // Debug: Print details for first few anchors only
        if !debug_printed && anchor_idx < 3 {
            println!("   🧪 Debug anchor {anchor_idx}: cx={cx:.6}, cy={cy:.6}, w={w:.6}, h={h:.6}, max_conf={max_class_conf:.6}");

            if anchor_idx == 2 {
                debug_printed = true;
            }
        }

        // Apply confidence threshold
        if max_class_conf >= conf_threshold {
            // Store raw normalized coordinates for NMS (like Python approach)
            // box format: [x1, y1, width, height] where x1,y1 are top-left normalized coords
            let x1 = cx - (w / 2.0); // left edge
            let y1 = cy - (h / 2.0); // top edge

            let class_name = get_coco_class_name(max_class_id);

            detections.push(Detection {
                bbox: [x1, y1, w, h], // Store as [x1, y1, width, height] in normalized coords
                confidence: max_class_conf,
                class_name,
            });
        }
    }

    println!("   📊 Maximum confidence seen: {max_seen_conf:.6}");
    println!("   ✅ Found {} detections before NMS", detections.len());

    // Apply Non-Maximum Suppression (NMS) - work with normalized coordinates
    let filtered_detections = apply_nms(detections, iou_threshold);

    println!("   ✅ {} detections after NMS", filtered_detections.len());

    // Convert from normalized coordinates to final image coordinates (like Python approach)
    let final_detections = convert_to_image_coordinates(filtered_detections, original_size);

    Ok(final_detections)
}

/// Convert detections from normalized coordinates to image coordinates
fn convert_to_image_coordinates(
    detections: Vec<Detection>,
    original_size: (u32, u32),
) -> Vec<Detection> {
    // Following Python approach exactly
    let length = original_size.0.max(original_size.1) as f32; // Max dimension for square padding
    let scale_factor = length / 640.0; // Same as Python: scale_factor = length / 640.0

    detections.into_iter().enumerate().map(|(i, mut detection)| {
        // detection.bbox is currently [x1, y1, width, height] in normalized coords
        let x1 = detection.bbox[0];
        let y1 = detection.bbox[1];
        let width = detection.bbox[2];
        let height = detection.bbox[3];

        // Following Python conversion exactly:
        // center_x = (box[0] + box[2]/2) * 640 * scale_factor
        // center_y = (box[1] + box[3]/2) * 640 * scale_factor
        // box_width = box[2] * 640 * scale_factor
        // box_height = box[3] * 640 * scale_factor
        let center_x = (x1 + width / 2.0) * 640.0 * scale_factor;
        let center_y = (y1 + height / 2.0) * 640.0 * scale_factor;
        let box_width = width * 640.0 * scale_factor;
        let box_height = height * 640.0 * scale_factor;

        // Convert to corner coordinates (same as Python)
        let final_x1 = center_x - box_width / 2.0;
        let final_y1 = center_y - box_height / 2.0;
        let final_x2 = center_x + box_width / 2.0;
        let final_y2 = center_y + box_height / 2.0;

        // Debug first detection only
        if i == 0 {
            println!("   🔍 Debug coordinate conversion for detection {i}: norm_x1={x1:.6}, norm_y1={y1:.6}, norm_w={width:.6}, norm_h={height:.6}");
            println!("      scale_factor={scale_factor:.3}, center_x={center_x:.1}, center_y={center_y:.1}, box_width={box_width:.1}, box_height={box_height:.1}");
            println!("      final coords: ({final_x1:.1}, {final_y1:.1}) to ({final_x2:.1}, {final_y2:.1})");
        }

        // Ensure coordinates are within original image bounds (same as Python)
        // Python: x1 = max(0, min(x1, width - 1))
        let final_x1 = final_x1.max(0.0).min(original_size.0 as f32 - 1.0);
        let final_y1 = final_y1.max(0.0).min(original_size.1 as f32 - 1.0);
        let final_x2 = final_x2.max(0.0).min(original_size.0 as f32 - 1.0);
        let final_y2 = final_y2.max(0.0).min(original_size.1 as f32 - 1.0);

        // Debug after clipping
        if i == 0 {
            println!("      after clipping: ({final_x1:.1}, {final_y1:.1}) to ({final_x2:.1}, {final_y2:.1})");
        }
        // Update detection with final coordinates in [x1, y1, x2, y2] format
        detection.bbox = [final_x1, final_y1, final_x2, final_y2];
        detection
    }).collect()
}

/// Apply Non-Maximum Suppression to remove overlapping detections
fn apply_nms(mut detections: Vec<Detection>, nms_threshold: f32) -> Vec<Detection> {
    // Sort by confidence (highest first)
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(detections[i].clone());

        // Suppress overlapping detections
        for j in (i + 1)..detections.len() {
            if suppressed[j] {
                continue;
            }

            let iou = calculate_iou(&detections[i].bbox, &detections[j].bbox);
            if iou > nms_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// Calculate Intersection over Union (IoU) between two bounding boxes
/// Boxes are in [x1, y1, width, height] format (normalized coordinates)
fn calculate_iou(bbox1: &[f32; 4], bbox2: &[f32; 4]) -> f32 {
    // Convert from [x1, y1, width, height] to [x1, y1, x2, y2]
    let box1_x2 = bbox1[0] + bbox1[2];
    let box1_y2 = bbox1[1] + bbox1[3];
    let box2_x2 = bbox2[0] + bbox2[2];
    let box2_y2 = bbox2[1] + bbox2[3];

    let x1_inter = bbox1[0].max(bbox2[0]);
    let y1_inter = bbox1[1].max(bbox2[1]);
    let x2_inter = box1_x2.min(box2_x2);
    let y2_inter = box1_y2.min(box2_y2);

    if x2_inter <= x1_inter || y2_inter <= y1_inter {
        return 0.0;
    }

    let intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter);
    let area1 = bbox1[2] * bbox1[3]; // width * height
    let area2 = bbox2[2] * bbox2[3]; // width * height
    let union = area1 + area2 - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Draw detections on the original image
fn draw_detections_on_image(
    original_image: DynamicImage,
    detections: &[Detection],
) -> runnx::Result<DynamicImage> {
    println!("🎨 Drawing {} detections on image...", detections.len());

    let mut image = original_image.to_rgb8();
    let color = Rgb([0, 255, 0]); // Green for all detections

    for detection in detections.iter() {
        // Ensure coordinates are within image bounds and make sense
        let x1 = detection.bbox[0].max(0.0) as i32;
        let y1 = detection.bbox[1].max(0.0) as i32;
        let x2 = detection.bbox[2].min(image.width() as f32) as i32;
        let y2 = detection.bbox[3].min(image.height() as f32) as i32;

        // Ensure we have a valid rectangle (width and height > 0)
        if x2 <= x1 || y2 <= y1 {
            continue; // Skip invalid boxes
        }

        // Draw bounding box with consistent thickness
        // Instead of expanding outward (which gets clipped at edges),
        // we'll draw multiple rectangles with proper bounds checking
        let thickness = 3; // Total thickness in pixels

        for t in 0..thickness {
            // Calculate rectangle bounds for this thickness layer
            let rect_x1 = (x1 - t).max(0);
            let rect_y1 = (y1 - t).max(0);
            let rect_x2 = (x2 + t).min(image.width() as i32);
            let rect_y2 = (y2 + t).min(image.height() as i32);

            // Only draw if we have a valid rectangle
            if rect_x2 > rect_x1 && rect_y2 > rect_y1 {
                let rect_width = (rect_x2 - rect_x1) as u32;
                let rect_height = (rect_y2 - rect_y1) as u32;

                let rect = Rect::at(rect_x1, rect_y1).of_size(rect_width, rect_height);
                draw_hollow_rect_mut(&mut image, rect, color);
            }
        }

        println!(
            "   📦 {} ({:.1}%) at [{}, {}, {}, {}]",
            detection.class_name,
            detection.confidence * 100.0,
            x1,
            y1,
            x2,
            y2
        );
    }

    Ok(DynamicImage::ImageRgb8(image))
}

/// Print detection results in a formatted table
fn print_detection_results(detections: &[Detection]) {
    if detections.is_empty() {
        println!("   ❌ No objects detected");
        return;
    }

    println!("\n📋 Detection Results ({} objects):", detections.len());
    println!("   ┌─────────────────┬────────────┬─────────────────────────────────┐");
    println!("   │ Class           │ Confidence │ Bounding Box (x1,y1,x2,y2)      │");
    println!("   ├─────────────────┼────────────┼─────────────────────────────────┤");

    for detection in detections.iter() {
        let bbox_width = detection.bbox[2] - detection.bbox[0];
        let bbox_height = detection.bbox[3] - detection.bbox[1];

        println!(
            "   │ {:15} │ {:8.1}%  │ ({:4.0},{:4.0},{:4.0},{:4.0}) {:4.0}x{:4.0} │",
            detection.class_name,
            detection.confidence * 100.0,
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3],
            bbox_width,
            bbox_height
        );
    }

    println!("   └─────────────────┴────────────┴─────────────────────────────────┘");
}

/// Get COCO class name by index
fn get_coco_class_name(class_id: usize) -> String {
    let coco_classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ];

    if class_id < coco_classes.len() {
        coco_classes[class_id].to_string()
    } else {
        format!("unknown_{class_id}")
    }
}
