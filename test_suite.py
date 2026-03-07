import os
import time
import json
from collections import defaultdict, Counter

import cv2
from ultralytics import YOLO


MODEL_PATH = r"runs/detect/scrubtech_v12/weights/best.pt"
TEST_IMAGE_DIR = "dataset/images/test"
OUTPUT_DIR = "data/test_results"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

# Set to True to also print low-confidence detections that were suppressed.
# Useful for diagnosing domain shift — if top predictions are correct class
# but confidence is low (e.g. 0.05–0.20), the model recognises the object
# but the test images look too different from training data.
DIAGNOSTIC_MODE = True
DIAGNOSTIC_CONF = 0.01


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(model_path)

def get_image_paths(folder: str):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name.lower())[1]
        if ext in valid_exts:
            files.append(os.path.join(folder, name))
    return sorted(files)

def draw_counts_panel(image, counts_dict):
    panel = image.copy()
    x0, y0 = 10, 10
    line_h = 28
    box_w = 260
    box_h = max(40, 20 + line_h * max(1, len(counts_dict)))

    cv2.rectangle(panel, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    alpha = 0.55
    image = cv2.addWeighted(panel, alpha, image, 1 - alpha, 0)

    if not counts_dict:
        cv2.putText(
            image, "No detections",
            (x0 + 10, y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        return image

    y = y0 + 28
    for cls_name, count in counts_dict.items():
        text = f"{cls_name}: {count}"
        cv2.putText(
            image, text,
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        y += line_h

    return image


# Classes that are visually similar enough that an overlapping detection
# should be treated as a duplicate of the same physical object.
MUTEX_GROUPS = [
    {"Straight Mayo Scissor", "Curved Mayo Scissor"},
]

def box_iou(a, b):
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

def deduplicate_across_classes(detections, iou_threshold=0.5):
    """
    Remove lower-confidence duplicates when two boxes from mutually-exclusive
    class groups overlap above iou_threshold. Detections is a list of
    (xyxy, conf, cls_name) tuples sorted by descending confidence.
    Returns the filtered list.
    """
    kept = []
    for det in detections:
        box, conf, cls_name = det
        suppress = False
        for kept_box, kept_conf, kept_cls in kept:
            if kept_conf <= conf:
                continue
            # Only suppress within a mutex group
            in_same_group = any(
                cls_name in grp and kept_cls in grp for grp in MUTEX_GROUPS
            )
            if in_same_group and box_iou(box, kept_box) >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(det)
    return kept


def summarize_confidences(conf_list):
    if not conf_list:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(conf_list),
        "max": max(conf_list),
        "mean": sum(conf_list) / len(conf_list),
    }

def run_test_suite():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    annotated_dir = os.path.join(OUTPUT_DIR, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    model = load_model(MODEL_PATH)
    class_name_map = model.names

    image_paths = get_image_paths(TEST_IMAGE_DIR)
    if not image_paths:
        raise ValueError(f"No images found in {TEST_IMAGE_DIR}")

    overall_class_counts = Counter()
    overall_confidences = []
    per_image_results = []
    latencies_ms = []

    print(f"Found {len(image_paths)} test images")
    print("Running inference...\n")

    for image_path in image_paths:
        image_name = os.path.basename(image_path)

        start = time.perf_counter()
        results = model.predict(
            source=image_path,
            conf=DIAGNOSTIC_CONF if DIAGNOSTIC_MODE else CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies_ms.append(latency_ms)

        result = results[0]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping unreadable image: {image_name}")
            continue

        class_counts = Counter()
        confs_this_image = []

        suppressed_detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            # Separate detections that clear the confidence threshold
            above = []
            below = []
            for box, conf, cls_id in zip(xyxy, confs, clss):
                cls_name = class_name_map[cls_id]
                entry = (tuple(map(int, box)), float(conf), cls_name)
                if conf >= CONF_THRESHOLD:
                    above.append(entry)
                else:
                    below.append(entry)

            # Cross-class deduplication: if two overlapping boxes belong to
            # the same mutex group (e.g. both scissor variants), keep only
            # the one with higher confidence.
            above.sort(key=lambda x: -x[1])
            above = deduplicate_across_classes(above)

            for box, conf, cls_name in above:
                x1, y1, x2, y2 = box
                class_counts[cls_name] += 1
                overall_class_counts[cls_name] += 1
                confs_this_image.append(conf)
                overall_confidences.append(conf)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    image, label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

            for box, conf, cls_name in below:
                suppressed_detections.append((cls_name, conf))
                if DIAGNOSTIC_MODE:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 1)
                    cv2.putText(
                        image, f"({cls_name} {conf:.2f})",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1
                    )

        image = draw_counts_panel(image, dict(class_counts))

        save_path = os.path.join(annotated_dir, image_name)
        cv2.imwrite(save_path, image)

        per_image_results.append({
            "image": image_name,
            "latency_ms": latency_ms,
            "num_detections": sum(class_counts.values()),
            "class_counts": dict(class_counts),
            "confidence_summary": summarize_confidences(confs_this_image),
            "annotated_output": save_path,
        })

        line = (
            f"{image_name} | latency={latency_ms:.1f} ms | "
            f"detections={sum(class_counts.values())} | counts={dict(class_counts)}"
        )
        if DIAGNOSTIC_MODE and suppressed_detections:
            top = sorted(suppressed_detections, key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{n} {c:.3f}" for n, c in top)
            line += f" | SUPPRESSED (below {CONF_THRESHOLD}): {top_str}"
        print(line)

    summary = {
        "model_path": MODEL_PATH,
        "test_dir": TEST_IMAGE_DIR,
        "num_images": len(per_image_results),
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "img_size": IMG_SIZE,
        "overall_class_counts": dict(overall_class_counts),
        "overall_confidence_summary": summarize_confidences(overall_confidences),
        "latency_ms": {
            "min": min(latencies_ms) if latencies_ms else None,
            "max": max(latencies_ms) if latencies_ms else None,
            "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
        },
        "per_image_results": per_image_results,
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Annotated images saved to: {annotated_dir}")
    print(f"Summary JSON saved to: {summary_path}")


if __name__ == "__main__":
    run_test_suite()
