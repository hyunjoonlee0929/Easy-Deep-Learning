"""Image/Video detection utilities."""

from __future__ import annotations

import tempfile
import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _load_yolo(model_name: str = "yolov8n.pt"):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for YOLO models.") from exc
    return YOLO(model_name)


def _load_torchvision_fasterrcnn():
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise RuntimeError("torch/torchvision required for Faster R-CNN.") from exc

    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model, weights


def _draw_boxes(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for det in detections:
        box = det["box"]
        score = det.get("score", 0.0)
        label = det.get("label", "")
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{label} {score:.2f}", fill="red")
    return image


def detect_image_pil(
    image: Image.Image,
    model_type: str = "yolo",
    conf: float = 0.25,
    model_name: str = "yolov8n.pt",
) -> tuple[Image.Image, list[dict[str, Any]]]:
    image = image.convert("RGB")
    detections: list[dict[str, Any]] = []

    if model_type == "yolo":
        model = _load_yolo(model_name)
        results = model(image, conf=conf)
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                detections.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "score": float(b.conf[0]),
                        "label": model.names[int(b.cls[0])],
                    }
                )
    else:
        model, weights = _load_torchvision_fasterrcnn()
        import torch
        from torchvision.transforms import functional as F

        tensor = F.to_tensor(image)
        with torch.no_grad():
            outputs = model([tensor])[0]
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        boxes = outputs["boxes"].cpu().numpy()
        categories = weights.meta.get("categories", [])
        for box, score, label in zip(boxes, scores, labels):
            if float(score) < conf:
                continue
            label_name = categories[label] if label < len(categories) else str(label)
            detections.append(
                {
                    "box": box.tolist(),
                    "score": float(score),
                    "label": label_name,
                }
            )

    image = _draw_boxes(image, detections)
    return image, detections


def detect_image_bytes(
    image_bytes: bytes,
    model_type: str = "yolo",
    conf: float = 0.25,
    model_name: str = "yolov8n.pt",
) -> tuple[Image.Image, list[dict[str, Any]]]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return detect_image_pil(image, model_type=model_type, conf=conf, model_name=model_name)


def detect_video_bytes(
    video_bytes: bytes,
    model_type: str = "yolo",
    conf: float = 0.25,
    model_name: str = "yolov8n.pt",
    frame_stride: int = 10,
    max_frames: int = 30,
) -> tuple[list[Image.Image], list[list[dict[str, Any]]]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for video detection.") from exc

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    all_dets = []
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotated, dets = detect_image_pil(
                image=img,
                model_type=model_type,
                conf=conf,
                model_name=model_name,
            )
            frames.append(annotated)
            all_dets.append(dets)
        idx += 1
    cap.release()
    return frames, all_dets
