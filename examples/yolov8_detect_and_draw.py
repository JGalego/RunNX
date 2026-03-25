"""
YOLOv8 Object Detection with Visualization
Mirrors examples/yolov8_detect_and_draw.rs using the ultralytics Python package.

Usage:
    python examples/yolov8_detect_and_draw.py [--image assets/bus.jpg] [--model yolov8n.pt]
                                               [--conf 0.25] [--iou 0.45]
                                               [--output assets/bus_with_detections.jpg]
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 detect and draw")
    parser.add_argument("--image",  default="assets/bus.jpg",                     help="Input image path")
    parser.add_argument("--model",  default="yolov8n.pt",                         help="Model file (.pt or .onnx)")
    parser.add_argument("--conf",   default=0.25, type=float,                     help="Confidence threshold")
    parser.add_argument("--iou",    default=0.45, type=float,                     help="NMS IoU threshold")
    parser.add_argument("--output", default="assets/bus_with_detections.jpg",     help="Output image path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file '{image_path}' not found")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Download with: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\""
        )

    model = YOLO(str(model_path))

    start = time.perf_counter()
    results = model(
        str(image_path),
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = results[0]
    boxes = result.boxes

    if len(boxes) == 0:
        print("No objects detected")
    else:
        print(f"Detected {len(boxes)} objects:")
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            print(f"  {cls_name} ({conf*100:.1f}%) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(output_path))
    print(f"Output saved to: {output_path}")

    print(f"Detection completed in {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    main()
