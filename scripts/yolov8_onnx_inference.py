# pylint: disable=line-too-long,no-member,too-many-statements,too-many-locals,too-many-arguments,too-many-positional-arguments,unused-variable
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "opencv-python",
#   "onnxruntime",
#   "numpy"
# ]
# ///

"""
Baseline script for YOLO inference
"""

import argparse

import cv2.dnn
import numpy as np

# COCO dataset class names (80 classes)
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    print(f"Drawing box: ({x}, {y}) to ({x_plus_w}, {y_plus_h}) for class {class_id}")

    # Ensure coordinates are within image bounds
    img_h, img_w = img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    x_plus_w = max(0, min(x_plus_w, img_w - 1))
    y_plus_h = max(0, min(y_plus_h, img_h - 1))

    # Use bright colors that are clearly visible
    color = (0, 255, 0)  # Bright green
    thickness = 3

    label = f"{CLASSES[class_id]} ({confidence:.2f})"

    # Draw rectangle
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, thickness)

    # Draw label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)

    # Draw label text
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def main(onnx_model, input_image):
    """
    Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Args:
        onnx_model (str): Path to the ONNX model.
        input_image (str): Path to the input image.

    Returns:
        list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
    """
    # Load the ONNX model
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # Read the input image
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape
    print(f"Original image size: {width}x{height}")

    # Prepare a square image for inference (YOLOv8 expects square input)
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor from square image to original
    scale_factor = length / 640.0

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (min_score, max_score, min_class_loc, (x, max_class_index)) = cv2.minMaxLoc(classes_scores)
        if max_score >= 0.25:
            # Raw outputs from model
            raw_cx = outputs[0][i][0]
            raw_cy = outputs[0][i][1]
            raw_w = outputs[0][i][2]
            raw_h = outputs[0][i][3]

            if len(boxes) == 0:  # Print debug info for first detection only
                print(f"Raw model outputs: cx={raw_cx:.6f}, cy={raw_cy:.6f}, w={raw_w:.6f}, h={raw_h:.6f}")
                print(f"Model outputs are normalized to 0-1 range, scaling by 640 then by {scale_factor:.3f}")

            box = [
                raw_cx - (0.5 * raw_w),  # x1 (left edge)
                raw_cy - (0.5 * raw_h),  # y1 (top edge)
                raw_w,                   # width
                raw_h,                   # height
            ]
            boxes.append(box)
            scores.append(max_score)
            class_ids.append(max_class_index)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    print(f"Found {len(boxes)} detections before NMS")
    print(f"Found {len(result_boxes) if len(result_boxes) > 0 else 0} detections after NMS")

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    if len(result_boxes) > 0:
        for i in result_boxes.flatten():
            box = boxes[i]
            detection = {
                "class_id": class_ids[i],
                "class_name": CLASSES[class_ids[i]],
                "confidence": scores[i],
                "box": box,
            }
            print(detection)
            detections.append(detection)

            # The model outputs are normalized to 0-1 range relative to the 640x640 input
            # First convert to 640x640 coordinate system, then scale to actual image size
            scale_factor = length / 640.0

            # Convert normalized coordinates to 640x640 pixel coordinates, then scale
            center_x = (box[0] + box[2]/2) * 640 * scale_factor
            center_y = (box[1] + box[3]/2) * 640 * scale_factor
            box_width = box[2] * 640 * scale_factor
            box_height = box[3] * 640 * scale_factor

            # Convert to corner coordinates
            x1 = int(center_x - box_width / 2)
            y1 = int(center_y - box_height / 2)
            x2 = int(center_x + box_width / 2)
            y2 = int(center_y + box_height / 2)

            # Ensure coordinates are within original image bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            print(f"Corner coordinates: ({x1}, {y1}) to ({x2}, {y2})")

            draw_bounding_box(
                original_image,
                class_ids[i],
                scores[i],
                x1, y1, x2, y2,
            )
    else:
        print("No detections found after NMS")

    # Save the image with bounding boxes
    input_filename = input_image.split('/')[-1].split('.')[0]  # Get filename without extension
    output_path = f"assets/{input_filename}_with_detections.jpg"
    cv2.imwrite(output_path, original_image)
    print(f"Output image saved as '{output_path}'")

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", default="assets/example.jpg", help="Path to input image.")
    args = parser.parse_args()
    main(args.model, args.img)
