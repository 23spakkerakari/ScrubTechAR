# ScrubTechAR

An XR software that allows surgeons to keep track of surgical tools during/before/after an OR, to reduce up to 90 minutes per surgery.

---

## Surgical Instruments Detected

| Class ID | Instrument | Training images |
|---|---|---|
| 0 | Scalpel | 550 |
| 1 | Straight Dissection Clamp | 460 |
| 2 | Straight Mayo Scissor | 450 |
| 3 | Curved Mayo Scissor | 550 |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Workflow

### 1 — Build the dataset

Matches images in `data/` with YOLO labels in `Labels/label object names/`, then splits everything into `dataset/images/{train,val,test}` and `dataset/labels/{train,val,test}`:

```bash
python setup_dataset.py
# Optional flags:
#   --train 0.80  --val 0.10  --test 0.10   (default 80/10/10 split)
#   --seed  42
#   --overwrite                              (rebuild from scratch)
```

### 2 — Train YOLOv11

```bash
python train.py
# Optional flags:
#   --model yolo11n.pt   (nano, default)
#   --model yolo11s.pt   (small)
#   --model yolo11m.pt   (medium — best accuracy/speed trade-off)
#   --epochs 100         (default)
#   --batch  16          (default)
#   --imgsz  640         (default)
```

Best weights are saved to `runs/detect/scrubtech_v1/weights/best.pt`.

### 3 — Count instruments

```bash
# Single image
python count_instruments.py --source data/Scalpel/bisturi1.jpg

# Entire folder
python count_instruments.py --source data/

# Save annotated output
python count_instruments.py --source data/ --save

# Live webcam
python count_instruments.py --source 0

# Custom weights
python count_instruments.py --source <path> --weights <path/to/best.pt>
```

Example output:
```
══════════════════════════════════════════════════
  Instrument count — data/
──────────────────────────────────────────────────
  Scalpel                            550  ██████████████████████
  Straight Dissection Clamp          460  ████████████████████
  Straight Mayo Scissor              450  ███████████████████
  Curved Mayo Scissor                550  ██████████████████████
──────────────────────────────────────────────────
  TOTAL                             2010
══════════════════════════════════════════════════
```

---

## Project Structure

```
ScrubTechAR/
├── data/
│   ├── Scalpel/                    550 images (bisturi*.jpg)
│   ├── Curved Mayo Scissor/        550 images (tesouracurva*.jpg)
│   ├── Straight Dissection Clamp/  460 images (pinca*.jpg)
│   └── Straight Mayo Scissor/      450 images (tesourareta*.jpg)
├── Labels/
│   ├── label object names/         YOLO annotations — 4-class instrument ID
│   └── label top-bottom/           YOLO annotations — 2-class orientation
├── dataset/                        ← created by setup_dataset.py
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── runs/                           ← created by train.py / count_instruments.py
│   └── detect/scrubtech_v1/weights/best.pt
├── dataset.yaml                    YOLO dataset config
├── setup_dataset.py                Dataset preparation script
├── train.py                        YOLOv11 training script
├── count_instruments.py            Inference + instrument counting
└── requirements.txt
```
