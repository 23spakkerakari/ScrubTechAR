"""
setup_dataset.py
----------------
Scans data/ for images, pairs each with its matching annotation from
Labels/label object names/, then splits into train / val / test sets
and writes the standard YOLO directory layout:

    dataset/
        images/train/   images/val/   images/test/
        labels/train/   labels/val/   labels/test/

It also writes dataset.yaml so train.py can find everything.

Usage:
    python setup_dataset.py [--train 0.8] [--val 0.1] [--test 0.1] [--seed 42]
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path

# ── paths relative to this script ────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
DATA_DIR   = ROOT / "data"
LABELS_DIR = ROOT / "Labels" / "label object names"
DATASET    = ROOT / "dataset"
YAML_PATH  = ROOT / "dataset.yaml"

# ── class map  (must stay consistent with the label files) ───────────────────
CLASS_NAMES = {
    0: "Scalpel",
    1: "Straight Dissection Clamp",
    2: "Straight Mayo Scissor",
    3: "Curved Mayo Scissor",
}

def collect_pairs() -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) for every image that has a matching label."""
    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[str] = []

    for img_path in sorted(DATA_DIR.rglob("*.jpg")):
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    if missing_labels:
        print(f"  [warn] {len(missing_labels)} images have no matching label and will be skipped.")
        for name in missing_labels[:10]:
            print(f"         {name}")
        if len(missing_labels) > 10:
            print(f"         … and {len(missing_labels) - 10} more")

    return pairs


def split(pairs: list, train_r: float, val_r: float, seed: int):
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def copy_split(pairs: list[tuple[Path, Path]], split_name: str) -> None:
    img_dir = DATASET / "images" / split_name
    lbl_dir = DATASET / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_dir / img_path.name)
        shutil.copy2(lbl_path, lbl_dir / lbl_path.name)


def write_yaml() -> None:
    config = {
        "path": str(DATASET),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_NAMES),
        "names": list(CLASS_NAMES.values()),
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  Wrote {YAML_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--val",   type=float, default=0.10)
    parser.add_argument("--test",  type=float, default=0.10)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete and rebuild the dataset/ folder from scratch.")
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "train + val + test must sum to 1.0"

    if DATASET.exists():
        if args.overwrite:
            shutil.rmtree(DATASET)
            print(f"  Removed existing {DATASET}")
        else:
            print(f"  dataset/ already exists — pass --overwrite to rebuild.")
            return

    print("Collecting image / label pairs …")
    pairs = collect_pairs()
    print(f"  Found {len(pairs)} matched pairs")

    train_pairs, val_pairs, test_pairs = split(pairs, args.train, args.val, args.seed)

    print(f"\nSplitting  train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

    print("\nCopying files …")
    copy_split(train_pairs, "train")
    copy_split(val_pairs,   "val")
    copy_split(test_pairs,  "test")

    write_yaml()

    # ── per-class summary ────────────────────────────────────────────────────
    print("\nPer-class image counts (train split):")
    prefix_to_class = {
        "bisturi":      "Scalpel",
        "pinca":        "Straight Dissection Clamp",
        "tesourareta":  "Straight Mayo Scissor",
        "tesouracurva": "Curved Mayo Scissor",
        "separado":     "Mixed (multiple instruments)",
    }
    counts: dict[str, int] = {}
    for img_path, _ in train_pairs:
        stem = img_path.stem.rstrip("0123456789")
        label = prefix_to_class.get(stem, stem)
        counts[label] = counts.get(label, 0) + 1
    for name, cnt in sorted(counts.items()):
        print(f"    {name:<35} {cnt}")

    print("\nDone!  Run  python train.py  to start training.")


if __name__ == "__main__":
    main()
