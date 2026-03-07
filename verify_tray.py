"""
verify_tray.py
--------------
Detects surgical instruments in an image (or folder/webcam), then compares
the detected per-instrument counts against the expected counts for a given
surgery. Reports PASS / LOW / HIGH / MISSING per instrument and an overall
tray status.

Usage:
    python verify_tray.py --surgery mastectomy --source tray_photo.jpg
    python verify_tray.py --surgery "aortic valve replacement" --source tray_photo.jpg
    python verify_tray.py --surgery "hip replacement" --source tray_photos/ --save
    python verify_tray.py --surgery mastectomy --source 0          # live webcam
    python verify_tray.py --list-surgeries                         # show available surgeries

Available surgery names (case-insensitive, partial match OK):
    mastectomy
    aortic valve replacement
    hip replacement
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT          = Path(__file__).parent.resolve()
SURGERIES_DIR = ROOT / "surgeries"
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "scrubtech_v1" / "weights" / "best.pt"

CLASS_NAMES = [
    "Scalpel",
    "Straight Dissection Clamp",
    "Straight Mayo Scissor",
    "Curved Mayo Scissor",
]

# ── ANSI colours (gracefully disabled on Windows) ────────────────────────────
try:
    import os
    _COLOR = os.name != "nt"
except Exception:
    _COLOR = False

GREEN  = "\033[92m" if _COLOR else ""
RED    = "\033[91m" if _COLOR else ""
YELLOW = "\033[93m" if _COLOR else ""
CYAN   = "\033[96m" if _COLOR else ""
BOLD   = "\033[1m"  if _COLOR else ""
RESET  = "\033[0m"  if _COLOR else ""


# ── Surgery file helpers ──────────────────────────────────────────────────────

def list_surgery_files() -> dict[str, Path]:
    """Return {normalised_name: path} for every .txt in surgeries/."""
    return {
        p.stem.replace("_", " ").lower(): p
        for p in sorted(SURGERIES_DIR.glob("*.txt"))
    }


def resolve_surgery(name: str) -> tuple[str, Path]:
    """Fuzzy-match a surgery name to a file. Returns (canonical_name, path)."""
    available = list_surgery_files()
    query = name.lower().strip()

    # exact match first
    if query in available:
        return query, available[query]

    # substring match
    matches = [(k, v) for k, v in available.items() if query in k or k in query]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(k for k, _ in matches)
        sys.exit(
            f"[error] '{name}' matches multiple surgeries: {names}\n"
            "  Be more specific."
        )

    # word-overlap fallback
    query_words = set(query.split())
    scored = [(len(query_words & set(k.split())), k, v) for k, v in available.items()]
    scored.sort(reverse=True)
    if scored and scored[0][0] > 0:
        _, k, v = scored[0]
        return k, v

    names = ", ".join(available.keys())
    sys.exit(
        f"[error] Surgery '{name}' not found.\n"
        f"  Available: {names}\n"
        f"  Run  python verify_tray.py --list-surgeries  for details."
    )


def parse_expected_counts(path: Path) -> dict[str, int]:
    """
    Extract per-class expected counts from a surgery .txt file.
    Looks for lines matching:   <ClassName>  :  <number>
    """
    pattern = re.compile(
        r"^\s*(Scalpel|Straight Dissection Clamp|Straight Mayo Scissor|Curved Mayo Scissor)"
        r"\s*:\s*(\d+)",
        re.IGNORECASE,
    )
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            # Normalise case to match CLASS_NAMES exactly
            raw_name = m.group(1).strip()
            for canonical in CLASS_NAMES:
                if canonical.lower() == raw_name.lower():
                    counts[canonical] = int(m.group(2))
                    break
    if not counts:
        sys.exit(
            f"[error] Could not parse expected counts from {path}.\n"
            "  Make sure the file contains lines like:\n"
            "    Scalpel                   :  3"
        )
    return counts


# ── Detection helpers ─────────────────────────────────────────────────────────

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


def count_detections(results) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for r in results:
        if r.boxes is None:
            continue
        for cls_id in r.boxes.cls.cpu().int().tolist():
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            totals[name] += 1
    return dict(totals)


def run_detection(model: YOLO, source, args) -> dict[str, int]:
    kwargs = dict(
        conf     = args.conf,
        iou      = args.iou,
        imgsz    = args.imgsz,
        device   = args.device if args.device else None,
        save     = args.save,
        project  = str(ROOT / "runs" / "verify"),
        name     = "results",
        exist_ok = True,
        verbose  = False,
    )
    # For folders and webcams use streaming to avoid memory issues
    if isinstance(source, int) or (isinstance(source, Path) and source.is_dir()):
        kwargs["stream"] = True
    results = model.predict(source=str(source) if isinstance(source, Path) else source,
                            **kwargs)
    return count_detections(results)


# ── Report ────────────────────────────────────────────────────────────────────

STATUS_OK      = f"{GREEN}{BOLD}  OK  {RESET}"
STATUS_LOW     = f"{RED}{BOLD} LOW  {RESET}"
STATUS_HIGH    = f"{YELLOW}{BOLD} HIGH {RESET}"
STATUS_MISSING = f"{RED}{BOLD} MISS {RESET}"


def _status_label(expected: int, detected: int) -> str:
    if expected == 0 and detected == 0:
        return STATUS_OK
    if detected == 0 and expected > 0:
        return STATUS_MISSING
    if detected == expected:
        return STATUS_OK
    if detected < expected:
        return STATUS_LOW
    return STATUS_HIGH


def print_report(
    surgery_name: str,
    source_label: str,
    expected: dict[str, int],
    detected: dict[str, int],
) -> bool:
    """Print the verification report. Returns True if tray is READY."""
    W = 66

    exp_total = sum(expected.get(n, 0) for n in CLASS_NAMES)
    det_total = sum(detected.get(n, 0) for n in CLASS_NAMES)
    diff_total = det_total - exp_total

    issues = []
    rows = []
    for name in CLASS_NAMES:
        exp = expected.get(name, 0)
        det = detected.get(name, 0)
        diff = det - exp
        status = _status_label(exp, det)
        rows.append((name, exp, det, diff, status))
        if det != exp:
            issues.append((name, exp, det, diff))

    tray_ready = len(issues) == 0

    # ── header ──────────────────────────────────────────────────────────────
    title = f"SURGICAL TRAY VERIFICATION — {surgery_name.upper()}"
    print(f"\n{BOLD}{'═'*W}{RESET}")
    print(f"{BOLD}  {title:<{W-4}}{RESET}")
    print(f"{BOLD}{'─'*W}{RESET}")
    print(f"  Source : {source_label}")
    print(f"{'─'*W}")

    # ── per-instrument table ─────────────────────────────────────────────────
    hdr = f"  {'Instrument':<30}  {'Exp':>4}  {'Det':>4}  {'Diff':>5}  {'Status'}"
    print(f"{BOLD}{hdr}{RESET}")
    print(f"{'─'*W}")

    for name, exp, det, diff, status in rows:
        diff_str = f"{diff:+d}" if diff != 0 else "  —  "
        print(f"  {name:<30}  {exp:>4}  {det:>4}  {diff_str:>5}  {status}")

    print(f"{'─'*W}")
    total_diff_str = f"{diff_total:+d}" if diff_total != 0 else "  —  "
    print(f"  {'TOTAL':<30}  {exp_total:>4}  {det_total:>4}  {total_diff_str:>5}")
    print(f"{'═'*W}")

    # ── tray status banner ───────────────────────────────────────────────────
    if tray_ready:
        banner = f"{GREEN}{BOLD}  ✓  TRAY READY — all instrument counts match{RESET}"
    else:
        low    = [n for n, e, d, _ in issues if d < e and d > 0]
        miss   = [n for n, e, d, _ in issues if d == 0 and e > 0]
        high   = [n for n, e, d, _ in issues if d > e]
        parts  = []
        if miss:
            parts.append(f"{len(miss)} instrument type(s) not detected")
        if low:
            parts.append(f"{len(low)} instrument type(s) below expected count")
        if high:
            parts.append(f"{len(high)} instrument type(s) above expected count")
        summary = " | ".join(parts)
        banner = f"{RED}{BOLD}  ✗  TRAY INCOMPLETE — {summary}{RESET}"

    print(banner)

    # ── detail on issues ─────────────────────────────────────────────────────
    if issues:
        print(f"\n{BOLD}  Issue details:{RESET}")
        for name, exp, det, diff in issues:
            if det == 0 and exp > 0:
                msg = f"    {RED}✗ {name}: NONE detected — need {exp}{RESET}"
            elif det < exp:
                msg = f"    {RED}✗ {name}: {det} detected, need {exp}  ({abs(diff)} missing){RESET}"
            else:
                msg = f"    {YELLOW}! {name}: {det} detected, expected {exp}  ({diff} extra){RESET}"
            print(msg)

    print(f"{'═'*W}\n")
    return tray_ready


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Verify a surgical instrument tray against a known surgery's requirements."
    )
    p.add_argument("--surgery", default=None,
                   help="Surgery name (partial match OK). E.g. 'mastectomy', 'hip replacement'.")
    p.add_argument("--source",  default=None,
                   help="Image file, folder of images, or webcam index (0).")
    p.add_argument("--weights", default=None,
                   help="Path to model weights (.pt). Defaults to best.pt from train.py.")
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.45)
    p.add_argument("--imgsz",  type=int,   default=640)
    p.add_argument("--device", default="",
                   help="cuda device index or 'cpu'. Empty = auto-detect.")
    p.add_argument("--save",   action="store_true",
                   help="Save annotated detection images to runs/verify/results/.")
    p.add_argument("--list-surgeries", action="store_true",
                   help="Print all available surgery profiles and exit.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── --list-surgeries ─────────────────────────────────────────────────────
    if args.list_surgeries:
        available = list_surgery_files()
        if not available:
            print(f"[info] No surgery files found in {SURGERIES_DIR}")
            return
        print(f"\nAvailable surgery profiles in {SURGERIES_DIR}:\n")
        for name, path in available.items():
            expected = parse_expected_counts(path)
            total = sum(expected.values())
            print(f"  {name}")
            for cls in CLASS_NAMES:
                print(f"      {cls:<30}  {expected.get(cls, 0):>3}")
            print(f"      {'TOTAL':<30}  {total:>3}")
            print()
        return

    # ── validate required args ────────────────────────────────────────────────
    if not args.surgery:
        sys.exit("[error] --surgery is required. Run --list-surgeries to see options.")
    if args.source is None:
        sys.exit("[error] --source is required (image path, folder, or webcam index).")

    # ── resolve surgery ───────────────────────────────────────────────────────
    surgery_key, surgery_path = resolve_surgery(args.surgery)
    expected = parse_expected_counts(surgery_path)

    print(f"\nSurgery  : {surgery_key.title()}")
    print(f"Profile  : {surgery_path}")
    print(f"Expected : ", end="")
    parts = [f"{CLASS_NAMES[i]}×{expected.get(CLASS_NAMES[i], 0)}" for i in range(len(CLASS_NAMES))]
    print("  ".join(parts))

    # ── resolve source ────────────────────────────────────────────────────────
    try:
        source = int(args.source)
        source_label = f"webcam {source}"
    except ValueError:
        source = Path(args.source)
        if not source.exists():
            sys.exit(f"[error] Source not found: {source}")
        source_label = str(source)

    # ── load model ────────────────────────────────────────────────────────────
    weights = resolve_weights(args.weights)
    print(f"Weights  : {weights}\n")
    model = YOLO(str(weights))

    # ── detect ────────────────────────────────────────────────────────────────
    print("Running detection …")
    detected = run_detection(model, source, args)

    # ── report ────────────────────────────────────────────────────────────────
    tray_ready = print_report(surgery_key, source_label, expected, detected)

    if args.save:
        print(f"  Annotated images saved to: {ROOT / 'runs' / 'verify' / 'results'}\n")

    sys.exit(0 if tray_ready else 1)


if __name__ == "__main__":
    main()
