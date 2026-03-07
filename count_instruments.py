"""
count_instruments.py
--------------------
Runs the trained YOLOv11 model on one image, a folder of images, or a live
camera feed, and prints a per-class count of detected surgical instruments.

Usage examples:
    # count instruments in a single image
    python count_instruments.py --source data/Scalpel/bisturi1.jpg

    # count across an entire folder
    python count_instruments.py --source data/Scalpel/

    # use a custom trained model (default looks for best.pt from train.py)
    python count_instruments.py --source <path> --weights runs/detect/scrubtech_v1/weights/best.pt

    # live webcam (source=0)
    python count_instruments.py --source 0

    # save annotated output images/video
    python count_instruments.py --source <path> --save
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()

CLASS_NAMES = [
    "Scalpel",
    "Straight Dissection Clamp",
    "Straight Mayo Scissor",
    "Curved Mayo Scissor",
]

DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "scrubtech_v1" / "weights" / "best.pt"


def resolve_weights(weights_arg: str | None) -> Path:
    if weights_arg:
        p = Path(weights_arg)
        if not p.exists():
            sys.exit(f"[error] Weights not found: {p}")
        return p
    if DEFAULT_WEIGHTS.exists():
        return DEFAULT_WEIGHTS
    sys.exit(
        "[error] No trained weights found.\n"
        "  Run  python train.py  first, or pass --weights <path>."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True,
                   help="Image file, folder, video file, or webcam index (0).")
    p.add_argument("--weights", default=None,
                   help="Path to model weights (.pt). Defaults to best.pt from train.py.")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Detection confidence threshold (0–1).")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold.")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--device",  default="",
                   help="cuda device or cpu — empty = auto.")
    p.add_argument("--save",    action="store_true",
                   help="Save annotated results to runs/count/.")
    p.add_argument("--show",    action="store_true",
                   help="Display results in a window (requires a display).")
    return p.parse_args()


def count_detections(results) -> dict[str, int]:
    """Aggregate per-class detection counts across all result frames."""
    totals: dict[str, int] = defaultdict(int)
    for r in results:
        if r.boxes is None:
            continue
        for cls_id in r.boxes.cls.cpu().int().tolist():
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            totals[name] += 1
    return dict(totals)


def print_summary(counts: dict[str, int], source: str) -> None:
    total = sum(counts.values())
    print(f"\n{'═'*50}")
    print(f"  Instrument count — {source}")
    print(f"{'─'*50}")
    for name in CLASS_NAMES:
        n = counts.get(name, 0)
        bar = "█" * n + "░" * max(0, 20 - n)
        print(f"  {name:<32}  {n:>3}  {bar}")
    print(f"{'─'*50}")
    print(f"  {'TOTAL':<32}  {total:>3}")
    print(f"{'═'*50}\n")


def run_on_image(model: YOLO, source: Path, args) -> dict[str, int]:
    results = model.predict(
        source  = str(source),
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        device  = args.device if args.device else None,
        save    = args.save,
        show    = args.show,
        project = str(ROOT / "runs" / "count"),
        name    = "results",
        exist_ok= True,
        verbose = False,
    )
    return count_detections(results)


def run_on_folder(model: YOLO, folder: Path, args) -> dict[str, int]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in extensions)
    if not images:
        sys.exit(f"[error] No images found in {folder}")

    print(f"  Processing {len(images)} images in {folder} …")

    # predict on the whole folder in one call (most efficient)
    results = model.predict(
        source  = str(folder),
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        device  = args.device if args.device else None,
        save    = args.save,
        show    = args.show,
        project = str(ROOT / "runs" / "count"),
        name    = "results",
        exist_ok= True,
        verbose = False,
        stream  = True,   # memory-efficient for large folders
    )
    return count_detections(results)


def main():
    args = parse_args()
    weights = resolve_weights(args.weights)

    print(f"\nLoading model: {weights}")
    model = YOLO(str(weights))

    source = args.source
    try:
        source_int = int(source)       # webcam index
        is_webcam = True
    except ValueError:
        source_int = None
        is_webcam = False

    if is_webcam:
        print("  Running on webcam — press Q to quit.")
        results = model.predict(
            source  = source_int,
            conf    = args.conf,
            iou     = args.iou,
            imgsz   = args.imgsz,
            device  = args.device if args.device else None,
            show    = True,
            stream  = True,
            verbose = False,
        )
        counts = count_detections(results)
        print_summary(counts, f"webcam {source_int}")
        return

    source_path = Path(source)
    if not source_path.exists():
        sys.exit(f"[error] Source not found: {source_path}")

    if source_path.is_dir():
        counts = run_on_folder(model, source_path, args)
        print_summary(counts, str(source_path))
    else:
        counts = run_on_image(model, source_path, args)
        print_summary(counts, source_path.name)

    if args.save:
        print(f"  Annotated results saved to: {ROOT / 'runs' / 'count' / 'results'}")


if __name__ == "__main__":
    main()
