#!/usr/bin/env python3
"""Evaluate fixed deployment thresholds after detector training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def ground_truth_boxes(label: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    if not label.exists():
        return boxes
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, x, y, w, h = (float(value) for value in line.split())
        boxes.append([(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height])
    return boxes


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_image(predictions: list[tuple[list[float], float]], truths: list[list[float]], threshold: float) -> tuple[int, int, int]:
    matches: set[int] = set()
    true_positives = 0
    false_positives = 0
    for box, confidence in sorted(predictions, key=lambda item: item[1], reverse=True):
        if confidence < threshold:
            continue
        candidates = [(iou(box, truth), index) for index, truth in enumerate(truths) if index not in matches]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= 0.5:
            matches.add(best_index)
            true_positives += 1
        else:
            false_positives += 1
    return true_positives, false_positives, len(truths) - len(matches)


def main() -> None:
    args = arguments()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights.resolve()))
    image_records = []

    for split in ("train", "val", "test"):
        images = sorted((dataset / "images" / split).glob("*.jpg"))
        results = model.predict([str(path) for path in images], conf=0.001, imgsz=args.imgsz, device=args.device, verbose=False)
        for image_path, result in zip(images, results):
            image = cv2.imread(str(image_path))
            truths = ground_truth_boxes(dataset / "labels" / split / f"{image_path.stem}.txt", image.shape[1], image.shape[0])
            predictions = [
                (list(map(float, box)), float(confidence))
                for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())
            ]
            image_records.append({"split": split, "path": str(image_path), "truths": truths, "predictions": predictions})

    rows = []
    for threshold in args.thresholds:
        test_tp = test_fp = test_fn = 0
        negative_fp_images = negative_fp_boxes = 0
        for record in image_records:
            if record["split"] == "test":
                tp, fp, fn = match_image(record["predictions"], record["truths"], threshold)
                test_tp += tp
                test_fp += fp
                test_fn += fn
            if not record["truths"]:
                count = sum(confidence >= threshold for _, confidence in record["predictions"])
                negative_fp_boxes += count
                negative_fp_images += count > 0
        precision = test_tp / (test_tp + test_fp) if test_tp + test_fp else 0.0
        recall = test_tp / (test_tp + test_fn) if test_tp + test_fn else 0.0
        rows.append(
            {
                "threshold": threshold,
                "test_true_positives_iou50": test_tp,
                "test_false_positives_iou50": test_fp,
                "test_false_negatives_iou50": test_fn,
                "test_precision_iou50": precision,
                "test_recall_iou50": recall,
                "all_negative_false_positive_images": negative_fp_images,
                "all_negative_false_positive_boxes": negative_fp_boxes,
            }
        )

    suitable = [
        row
        for row in rows
        if row["test_false_positives_iou50"] == 0
        and row["test_false_negatives_iou50"] == 0
        and row["all_negative_false_positive_boxes"] == 0
    ]
    recommended = min(suitable, key=lambda row: row["threshold"])["threshold"] if suitable else None
    report = {"iou_match_threshold": 0.5, "recommended_deployment_threshold": recommended, "rows": rows}
    (output / "deployment_threshold_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
