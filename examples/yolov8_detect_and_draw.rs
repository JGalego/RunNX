//! YOLOv8 Object Detection with Visualization
//!
//! Demonstrates a complete object detection pipeline using YOLOv8n:
//! - Loads an ONNX model and processes input images
//! - Performs inference with proper preprocessing
//! - Post-processes results with NMS and confidence thresholding
//! - Draws bounding boxes and saves the output image

use image::{imageops::FilterType, DynamicImage, ImageReader, Rgb};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use runnx::*;
use std::collections::HashMap;
use std::time::Instant;

/// Detection result with bounding box and class information
#[derive(Debug, Clone)]
struct Detection {
    bbox: [f32; 4], // [x1, y1, x2, y2] in image coordinates
    confidence: f32,
    class_name: String,
}

fn main() -> runnx::Result<()> {
    if std::env::var("RUST_LOG").is_ok() {
        env_logger::init();
    }

    let model = load_yolov8n_model()?;
    let image_path = "assets/bus.jpg";
    let (input_tensor, original_image, original_size) = load_and_prepare_image(image_path)?;

    let start_time = Instant::now();
    let outputs = run_inference(&model, input_tensor)?;
    let inference_time = start_time.elapsed();

    let conf_threshold = 0.25;
    let iou_threshold = 0.45;
    let detections =
        post_process_detections(&outputs, original_size, conf_threshold, iou_threshold)?;

    print_detection_results(&detections);

    if !detections.is_empty() {
        let output_image = draw_detections_on_image(original_image, &detections)?;
        let output_path = "assets/bus_with_detections.jpg";
        output_image.save(output_path).map_err(|e| {
            error::OnnxError::runtime_error(format!("Failed to save output image: {e}"))
        })?;
        println!("Output saved to: {output_path}");
    } else {
        println!("No objects detected");
    }

    println!("Detection completed in {:.2}ms", inference_time.as_millis());

    Ok(())
}

/// Load the YOLOv8n model from file
fn load_yolov8n_model() -> runnx::Result<model::Model> {
    let model_path = "yolov8n.onnx";

    if !std::path::Path::new(model_path).exists() {
        return Err(error::OnnxError::model_load_error(
            "YOLOv8n model file (yolov8n.onnx) not found. Download from: https://github.com/ultralytics/assets/releases/".to_string(),
        ));
    }

    model::Model::from_onnx_file(model_path)
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

    let original_image = ImageReader::open(image_path)
        .map_err(|e| error::OnnxError::runtime_error(format!("Failed to open image: {e}")))?
        .decode()
        .map_err(|e| error::OnnxError::runtime_error(format!("Failed to decode image: {e}")))?;

    let original_size = (original_image.width(), original_image.height());

    // Create square canvas for proper aspect ratio preservation
    let length = original_size.0.max(original_size.1);
    let mut square_image = DynamicImage::new_rgb8(length, length);
    image::imageops::overlay(&mut square_image, &original_image, 0, 0);

    // Resize to YOLOv8 input size
    let resized = square_image.resize_exact(640, 640, FilterType::Lanczos3);
    let rgb_image = resized.to_rgb8();

    // Convert to tensor format: normalize to [0,1] and convert HWC to CHW
    let mut tensor_data = Vec::with_capacity(3 * 640 * 640);
    for channel in 0..3 {
        for y in 0..640 {
            for x in 0..640 {
                let pixel = rgb_image.get_pixel(x, y);
                let normalized_value = pixel[channel] as f32 / 255.0;
                tensor_data.push(normalized_value);
            }
        }
    }

    let input_tensor = tensor::Tensor::from_shape_vec(&[1, 3, 640, 640], tensor_data)?;
    Ok((input_tensor, original_image, original_size))
}

/// Run inference on the model
fn run_inference(
    model: &model::Model,
    input_tensor: tensor::Tensor,
) -> runnx::Result<HashMap<String, tensor::Tensor>> {
    let mut inputs = HashMap::new();
    inputs.insert("images".to_string(), input_tensor);

    let runtime = runtime::Runtime::new();
    runtime.execute(&model.graph, inputs)
}

/// Post-process YOLOv8 raw outputs to extract detections
fn post_process_detections(
    outputs: &HashMap<String, runnx::Tensor>,
    original_size: (u32, u32),
    conf_threshold: f32,
    iou_threshold: f32,
) -> runnx::Result<Vec<Detection>> {
    let output_tensor = outputs
        .get("output0")
        .or_else(|| outputs.values().next())
        .ok_or_else(|| error::OnnxError::runtime_error("No output tensor found".to_string()))?;

    let output_shape = output_tensor.shape();
    if output_shape.len() != 3 || output_shape[0] != 1 {
        return Err(error::OnnxError::runtime_error(format!(
            "Unexpected output shape: {output_shape:?}, expected [1, features, anchors]"
        )));
    }

    let num_features = output_shape[1];
    let num_anchors = output_shape[2];
    let num_classes = num_features - 4;

    let output_data = output_tensor.data();
    let mut detections = Vec::new();

    // Process each anchor
    for anchor_idx in 0..num_anchors {
        let cx = output_data[[0, 0, anchor_idx]];
        let cy = output_data[[0, 1, anchor_idx]];
        let w = output_data[[0, 2, anchor_idx]];
        let h = output_data[[0, 3, anchor_idx]];

        // Find the class with highest confidence
        let mut max_class_conf = 0.0;
        let mut max_class_id = 0;

        for class_idx in 0..num_classes {
            let class_conf = output_data[[0, 4 + class_idx, anchor_idx]];
            if class_conf > max_class_conf {
                max_class_conf = class_conf;
                max_class_id = class_idx;
            }
        }

        // Apply confidence threshold
        if max_class_conf >= conf_threshold {
            let x1 = cx - (w / 2.0);
            let y1 = cy - (h / 2.0);
            let class_name = get_coco_class_name(max_class_id);

            detections.push(Detection {
                bbox: [x1, y1, w, h],
                confidence: max_class_conf,
                class_name,
            });
        }
    }

    // Apply Non-Maximum Suppression
    let filtered_detections = apply_nms(detections, iou_threshold);

    // Convert from normalized coordinates to image coordinates
    let final_detections = convert_to_image_coordinates(filtered_detections, original_size);

    Ok(final_detections)
}

/// Convert detections from normalized coordinates to image coordinates
fn convert_to_image_coordinates(
    detections: Vec<Detection>,
    original_size: (u32, u32),
) -> Vec<Detection> {
    let length = original_size.0.max(original_size.1) as f32;
    let scale_factor = length / 640.0;

    detections
        .into_iter()
        .map(|mut detection| {
            let x1 = detection.bbox[0];
            let y1 = detection.bbox[1];
            let width = detection.bbox[2];
            let height = detection.bbox[3];

            // Convert to center coordinates
            let center_x = (x1 + width / 2.0) * 640.0 * scale_factor;
            let center_y = (y1 + height / 2.0) * 640.0 * scale_factor;
            let box_width = width * 640.0 * scale_factor;
            let box_height = height * 640.0 * scale_factor;

            // Convert to corner coordinates
            let final_x1 = center_x - box_width / 2.0;
            let final_y1 = center_y - box_height / 2.0;
            let final_x2 = center_x + box_width / 2.0;
            let final_y2 = center_y + box_height / 2.0;

            // Clamp to image bounds
            let final_x1 = final_x1.max(0.0).min(original_size.0 as f32 - 1.0);
            let final_y1 = final_y1.max(0.0).min(original_size.1 as f32 - 1.0);
            let final_x2 = final_x2.max(0.0).min(original_size.0 as f32 - 1.0);
            let final_y2 = final_y2.max(0.0).min(original_size.1 as f32 - 1.0);

            detection.bbox = [final_x1, final_y1, final_x2, final_y2];
            detection
        })
        .collect()
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
    let area1 = bbox1[2] * bbox1[3];
    let area2 = bbox2[2] * bbox2[3];
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
    let mut image = original_image.to_rgb8();
    let color = Rgb([0, 255, 0]); // Green for all detections

    for detection in detections.iter() {
        let x1 = detection.bbox[0].max(0.0) as i32;
        let y1 = detection.bbox[1].max(0.0) as i32;
        let x2 = detection.bbox[2].min(image.width() as f32) as i32;
        let y2 = detection.bbox[3].min(image.height() as f32) as i32;

        // Ensure we have a valid rectangle
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        // Draw bounding box with thickness
        let thickness = 3;
        for t in 0..thickness {
            let rect_x1 = (x1 - t).max(0);
            let rect_y1 = (y1 - t).max(0);
            let rect_x2 = (x2 + t).min(image.width() as i32);
            let rect_y2 = (y2 + t).min(image.height() as i32);

            if rect_x2 > rect_x1 && rect_y2 > rect_y1 {
                let rect_width = (rect_x2 - rect_x1) as u32;
                let rect_height = (rect_y2 - rect_y1) as u32;
                let rect = Rect::at(rect_x1, rect_y1).of_size(rect_width, rect_height);
                draw_hollow_rect_mut(&mut image, rect, color);
            }
        }
    }

    Ok(DynamicImage::ImageRgb8(image))
}

/// Print detection results in a formatted table
fn print_detection_results(detections: &[Detection]) {
    if detections.is_empty() {
        println!("No objects detected");
        return;
    }

    println!("Detected {} objects:", detections.len());
    for detection in detections.iter() {
        println!(
            "  {} ({:.1}%) at [{:.0}, {:.0}, {:.0}, {:.0}]",
            detection.class_name,
            detection.confidence * 100.0,
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3]
        );
    }
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
