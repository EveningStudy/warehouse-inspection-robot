#!/usr/bin/env python3
"""Train and audit an inspection-panel detector with Ultralytics YOLO.

This script deliberately keeps training, test evaluation, prediction previews,
and negative-image auditing in one reproducible run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained model checkpoint")
    parser.add_argument("--output-root", type=Path, required=True, help="Parent directory for runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25, help="Deployment/audit threshold")
    parser.add_argument("--run-name", help="Optional fixed run name")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_dataset_root(data_yaml: Path, config: dict[str, Any]) -> Path:
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def split_paths(dataset_root: Path, config: dict[str, Any], split: str) -> tuple[Path, Path]:
    image_dir = Path(config[split])
    if not image_dir.is_absolute():
        image_dir = dataset_root / image_dir
    parts = list(image_dir.parts)
    try:
        parts[parts.index("images")] = "labels"
        label_dir = Path(*parts)
    except ValueError as exc:
        raise ValueError(f"Cannot infer label directory from {image_dir}") from exc
    return image_dir.resolve(), label_dir.resolve()


def validate_label(label: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    if not label.exists():
        return 0, errors
    lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{label}:{line_number}: expected 5 fields, got {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{label}:{line_number}: non-numeric YOLO label")
            continue
        if class_id != 0:
            errors.append(f"{label}:{line_number}: unsupported class id {class_id}")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            errors.append(f"{label}:{line_number}: normalized coordinates outside [0, 1]")
    return len(lines), errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(data_yaml: Path) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    config = load_yaml(data_yaml)
    dataset_root = resolve_dataset_root(data_yaml, config)
    audit: dict[str, Any] = {"dataset_yaml": str(data_yaml), "dataset_root": str(dataset_root), "splits": {}}
    negatives: dict[str, list[Path]] = {}
    hash_to_paths: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    errors: list[str] = []

    for split in ("train", "val", "test"):
        image_dir, label_dir = split_paths(dataset_root, config, split)
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        positive_count = 0
        box_count = 0
        negatives[split] = []
        for image in images:
            count, label_errors = validate_label(label_dir / f"{image.stem}.txt")
            errors.extend(label_errors)
            box_count += count
            if count:
                positive_count += 1
            else:
                negatives[split].append(image)
            hash_to_paths[sha256(image)].append((split, image))
        audit["splits"][split] = {
            "images": len(images),
            "positive_images": positive_count,
            "negative_images": len(negatives[split]),
            "boxes": box_count,
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
        }

    overlaps = [
        items
        for items in hash_to_paths.values()
        if len({split for split, _ in items}) > 1
    ]
    audit["exact_cross_split_duplicate_count"] = len(overlaps)
    audit["exact_cross_split_duplicates"] = [
        [{"split": split, "image": str(path)} for split, path in items]
        for items in overlaps
    ]
    audit["label_errors"] = errors
    if errors:
        raise ValueError("Dataset label validation failed:\n" + "\n".join(errors[:20]))
    test_overlaps = [items for items in overlaps if any(split == "test" for split, _ in items)]
    if test_overlaps:
        raise ValueError(f"Found {len(test_overlaps)} exact duplicate groups involving the test split")
    if overlaps:
        audit["warning"] = (
            "Exact train/val duplicates exist. They can bias model selection, but the independent "
            "test split has no exact duplicate leakage."
        )
    return audit, negatives


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def save_contact_sheet(entries: list[dict[str, Any]], destination: Path, title: str) -> None:
    if not entries:
        return
    tile_width, tile_height = 480, 320
    header_height = 52
    columns = 2
    rows = (len(entries) + columns - 1) // columns
    sheet = np.full((header_height + rows * tile_height, columns * tile_width, 3), 32, dtype=np.uint8)
    cv2.putText(sheet, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)

    for index, entry in enumerate(entries):
        image = cv2.imread(entry["image"])
        if image is None:
            continue
        result = entry["result"]
        boxes = result.boxes
        if boxes is not None:
            keep = boxes.conf.cpu().numpy() >= entry["threshold"]
            for xyxy, confidence in zip(boxes.xyxy.cpu().numpy()[keep], boxes.conf.cpu().numpy()[keep]):
                x1, y1, x2, y2 = (int(value) for value in xyxy)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    image,
                    f"inspection_panel {confidence:.2f}",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        caption = f"{entry['split']}/{Path(entry['image']).name}  max={entry['max_confidence']:.3f}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(image, caption, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        row, column = divmod(index, columns)
        y = header_height + row * tile_height + (tile_height - resized.shape[0]) // 2
        x = column * tile_width + (tile_width - resized.shape[1]) // 2
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), sheet)


def audit_negatives(
    model: YOLO,
    negatives: dict[str, list[Path]],
    run_dir: Path,
    threshold: float,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {"threshold": threshold, "splits": {}, "images": []}
    sheet_entries: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        paths = negatives[split]
        results = model.predict(
            source=[str(path) for path in paths],
            imgsz=imgsz,
            conf=0.001,
            device=device,
            verbose=False,
            stream=False,
        )
        false_positive_images = 0
        false_positive_boxes = 0
        for path, result in zip(paths, results):
            confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
            high_confidences = [float(value) for value in confidences if value >= threshold]
            max_confidence = max((float(value) for value in confidences), default=0.0)
            if high_confidences:
                false_positive_images += 1
                false_positive_boxes += len(high_confidences)
            item = {
                "split": split,
                "image": str(path),
                "max_confidence": max_confidence,
                "detections_at_threshold": len(high_confidences),
                "confidences_at_threshold": high_confidences,
            }
            report["images"].append(item)
            sheet_entries.append({**item, "result": result, "threshold": threshold})
        report["splits"][split] = {
            "negative_images": len(paths),
            "false_positive_images": false_positive_images,
            "false_positive_boxes": false_positive_boxes,
        }

    report["test_is_independent"] = True
    test_entries = [entry for entry in sheet_entries if entry["split"] == "test"]
    save_contact_sheet(test_entries, run_dir / "negative_audit" / "test_negatives.jpg", f"Independent test negatives @ conf >= {threshold}")
    save_contact_sheet(sheet_entries, run_dir / "negative_audit" / "all_negatives.jpg", f"All dataset negatives @ conf >= {threshold}")
    hardest = sorted(sheet_entries, key=lambda item: item["max_confidence"], reverse=True)[:12]
    save_contact_sheet(hardest, run_dir / "negative_audit" / "hardest_negatives.jpg", "Hardest negatives (highest raw confidence)")
    serializable = {key: value for key, value in report.items() if key != "results"}
    with (run_dir / "negative_audit" / "negative_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(serializable, stream, ensure_ascii=False, indent=2)
    return serializable


def write_metrics_csv(metrics: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, value])


def main() -> None:
    args = parse_args()
    data_yaml = args.data.expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_root.expanduser().resolve() / (args.run_name or timestamp))
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)

    dataset_audit, negatives = audit_dataset(data_yaml)
    environment = {
        "timestamp": timestamp,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "arguments": vars(args),
    }
    with (run_dir / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(jsonable(environment), stream, ensure_ascii=False, indent=2)
    with (run_dir / "dataset_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(dataset_audit, stream, ensure_ascii=False, indent=2)

    model = YOLO(args.model)
    train_result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        plots=True,
        project=str(run_dir),
        name="train",
        exist_ok=False,
        verbose=True,
    )
    train_dir = Path(train_result.save_dir)
    source_best = train_dir / "weights" / "best.pt"
    if not source_best.is_file():
        raise FileNotFoundError(f"Training did not produce {source_best}")
    best_path = run_dir / "best.pt"
    shutil.copy2(source_best, best_path)

    best_model = YOLO(str(best_path))
    test_result = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        project=str(run_dir),
        name="test_evaluation",
        exist_ok=False,
        verbose=True,
    )
    metrics = {str(key): jsonable(value) for key, value in test_result.results_dict.items()}
    metrics["fitness"] = jsonable(test_result.fitness)
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    write_metrics_csv(metrics, run_dir / "test_metrics.csv")

    config = load_yaml(data_yaml)
    dataset_root = resolve_dataset_root(data_yaml, config)
    test_image_dir, _ = split_paths(dataset_root, config, "test")
    best_model.predict(
        source=str(test_image_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(run_dir),
        name="test_predictions",
        exist_ok=False,
        verbose=True,
    )
    negative_report = audit_negatives(best_model, negatives, run_dir, args.conf, args.imgsz, args.device)

    summary = {
        "run_directory": str(run_dir),
        "best_weights": str(best_path),
        "test_metrics": metrics,
        "test_negative_audit": negative_report["splits"]["test"],
        "all_negative_audit": {
            "negative_images": sum(item["negative_images"] for item in negative_report["splits"].values()),
            "false_positive_images": sum(item["false_positive_images"] for item in negative_report["splits"].values()),
            "false_positive_boxes": sum(item["false_positive_boxes"] for item in negative_report["splits"].values()),
        },
        "artifacts": {
            "training_curves": str(train_dir / "results.png"),
            "test_confusion_matrix": str(Path(test_result.save_dir) / "confusion_matrix.png"),
            "test_confusion_matrix_normalized": str(Path(test_result.save_dir) / "confusion_matrix_normalized.png"),
            "test_predictions": str(run_dir / "test_predictions"),
            "test_negative_sheet": str(run_dir / "negative_audit" / "test_negatives.jpg"),
            "hardest_negative_sheet": str(run_dir / "negative_audit" / "hardest_negatives.jpg"),
        },
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print("\nFINAL_SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
