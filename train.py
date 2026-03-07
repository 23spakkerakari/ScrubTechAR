"""
train.py
--------
Fine-tunes a YOLOv11 model on the 4-class surgical instrument dataset.

Prerequisites:
    1. pip install -r requirements.txt
    2. python setup_dataset.py          # builds dataset/ and dataset.yaml

Usage:
    python train.py                     # default settings (recommended first run)
    python train.py --model yolo11m.pt  # larger model
    python train.py --epochs 150 --batch 32 --imgsz 640

After training the best weights are saved to:
    runs/detect/scrubtech_v1/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT       = Path(__file__).parent.resolve()
YAML_PATH  = ROOT / "dataset.yaml"
RUNS_DIR   = ROOT / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="yolo11n.pt",
        help="Base YOLOv11 checkpoint. Options: yolo11n.pt / yolo11s.pt / yolo11m.pt / "
             "yolo11l.pt / yolo11x.pt  (auto-downloaded on first run)."
    )
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=16,
                   help="Batch size (-1 = auto).")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--device",  default="",
                   help="cuda device (0, 0,1, cpu) — empty = auto-detect.")
    p.add_argument("--workers", type=int,   default=8)
    p.add_argument("--name",    default="scrubtech_v1",
                   help="Experiment name inside runs/detect/.")
    p.add_argument("--resume",  action="store_true",
                   help="Resume an interrupted training run.")
    return p.parse_args()


def main():
    args = parse_args()

    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"{YAML_PATH} not found.\n"
            "Run  python setup_dataset.py  first."
        )

    dataset_dir = ROOT / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"{dataset_dir} not found.\n"
            "Run  python setup_dataset.py  first."
        )

    model = YOLO(args.model)

    print(f"\n{'─'*60}")
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {YAML_PATH}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Img size: {args.imgsz}")
    print(f"  Device  : {args.device or 'auto'}")
    print(f"{'─'*60}\n")

    results = model.train(
        data      = str(YAML_PATH),
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        device    = args.device if args.device else None,
        workers   = args.workers,
        project   = str(RUNS_DIR / "detect"),
        name      = args.name,
        resume    = args.resume,
        # augmentation — helpful for a small surgical dataset
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 10.0,
        translate = 0.1,
        scale     = 0.5,
        flipud    = 0.1,
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.1,
        # early stopping
        patience  = 20,
        # save the best and last checkpoints
        save      = True,
        save_period = 10,
    )

    best = RUNS_DIR / "detect" / args.name / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights saved to: {best}")
    print(f"\nTo count instruments in an image run:")
    print(f"  python count_instruments.py --source <image_or_folder> --weights {best}")


if __name__ == "__main__":
    main()
