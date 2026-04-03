from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import torch
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
except ImportError:
    torch = None
    F = None
    SegformerForSemanticSegmentation = None
    SegformerImageProcessor = None


CLASS_MAP: Dict[int, str] = {
    0: "background",
    1: "road",
    2: "sidewalk",
    3: "building",
    4: "wall",
    5: "fence",
    6: "pole",
    7: "traffic light",
    8: "traffic sign",
    9: "vegetation",
    10: "terrain",
    11: "sky",
    12: "person",
    13: "rider",
    14: "car",
    15: "truck",
    16: "bus",
    17: "train",
    18: "motorcycle",
    19: "bicycle",
}


# -----------------------------
# Global runtime configuration
# -----------------------------
RUN_GENERATE_MASKS_FROM_IMAGES = True
RUN_IMAGE_ROOT = Path("archive/IDD_RESIZED/image_archive")
RUN_SEG_MODEL_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
RUN_GENERATED_MASK_ROOT = Path("dataset/dataset/generated_multiclass_masks")
RUN_OVERWRITE_GENERATED_MASKS = False

RUN_MASK_ROOT = Path("archive/IDD_RESIZED/mask_archive")
RUN_MAX_MASKS = 500
RUN_MIN_AREA_RATIO = 0.005
RUN_MIN_NON_BG_CLASSES = 2
RUN_FAST_DOWNSAMPLE_FACTOR = 2
RUN_OUTPUT_JSONL = Path("dataset/dataset/multiclass_structured_scenes_500.jsonl")
RUN_ALLOW_BINARY_FALLBACK = True


def extract_index(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def find_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def normalize_label(name: str) -> str:
    return str(name).strip().lower().replace("-", " ").replace("_", " ")


def cityscapes_label_to_project_id(label_name: str) -> int:
    normalized = normalize_label(label_name)
    mapping = {
        "road": 1,
        "sidewalk": 2,
        "building": 3,
        "wall": 4,
        "fence": 5,
        "pole": 6,
        "traffic light": 7,
        "traffic sign": 8,
        "vegetation": 9,
        "terrain": 10,
        "sky": 11,
        "person": 12,
        "rider": 13,
        "car": 14,
        "truck": 15,
        "bus": 16,
        "train": 17,
        "motorcycle": 18,
        "bicycle": 19,
    }
    return mapping.get(normalized, 0)


def generate_multiclass_masks_from_images(
    image_root: Path,
    out_mask_root: Path,
    model_id: str,
    max_masks: int,
    overwrite: bool,
) -> Path:
    if torch is None or F is None or SegformerForSemanticSegmentation is None or SegformerImageProcessor is None:
        raise ImportError(
            "Missing packages for mask generation. Install: torch transformers"
        )

    image_paths = find_images(image_root)
    if not image_paths:
        raise RuntimeError(f"No images found under: {image_root}")

    use_n = min(len(image_paths), max(1, int(max_masks)))
    selected = image_paths[:use_n]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading segmentation model: {model_id}")
    print(f"Segmentation device: {device}")

    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()

    out_mask_root.mkdir(parents=True, exist_ok=True)
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    saved = 0
    for i, image_path in enumerate(selected, start=1):
        idx = extract_index(image_path.name)
        if idx is None:
            idx = i - 1
        out_path = out_mask_root / f"Mask_{idx}.png"
        if out_path.exists() and not overwrite:
            continue

        image = Image.open(image_path).convert("RGB")
        enc = processor(images=image, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

        remapped = np.zeros_like(pred, dtype=np.uint8)
        for src_id in np.unique(pred):
            label_name = id2label.get(int(src_id), "background")
            dst_id = cityscapes_label_to_project_id(label_name)
            remapped[pred == int(src_id)] = np.uint8(dst_id)

        Image.fromarray(remapped, mode="L").save(out_path)
        saved += 1

        if i == 1 or i % 50 == 0 or i == use_n:
            print(f"Generated masks: {i}/{use_n}")

    print(f"Saved/updated masks: {saved}")
    print(f"Mask output directory: {out_mask_root}")
    return out_mask_root


def area_bucket(ratio: float) -> str:
    if ratio < 0.02:
        return "small"
    if ratio < 0.15:
        return "medium"
    return "large"


def location_bucket(mask: np.ndarray) -> str:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return "center-middle"

    h, w = mask.shape
    cx = float(xs.mean()) / max(float(w), 1.0)
    cy = float(ys.mean()) / max(float(h), 1.0)

    if cx < (1.0 / 3.0):
        xloc = "left"
    elif cx < (2.0 / 3.0):
        xloc = "center"
    else:
        xloc = "right"

    if cy < (1.0 / 3.0):
        yloc = "top"
    elif cy < (2.0 / 3.0):
        yloc = "middle"
    else:
        yloc = "bottom"

    return f"{xloc}-{yloc}"


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask
    return mask[::factor, ::factor]


def connected_components_count(binary_mask: np.ndarray) -> int:
    if cv2 is not None:
        num_labels, _ = cv2.connectedComponents(binary_mask.astype(np.uint8), connectivity=8)
        return max(int(num_labels) - 1, 0)

    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    count = 0
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if binary_mask[y, x] == 0 or visited[y, x] == 1:
                continue
            count += 1
            q = deque([(y, x)])
            visited[y, x] = 1
            while q:
                cy, cx = q.popleft()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if binary_mask[ny, nx] == 0 or visited[ny, nx] == 1:
                        continue
                    visited[ny, nx] = 1
                    q.append((ny, nx))

    return count


def describe_scene(structured: List[Dict[str, object]]) -> str:
    if not structured:
        return "No non-background classes were detected in this segmentation map."

    major = [row for row in structured if row["area"] in ("large", "medium")]
    if not major:
        major = structured[:3]

    first_parts = []
    for row in major[:4]:
        first_parts.append(
            f"{row['class']} ({row['count']} instances, {row['area']}, {row['location']})"
        )
    sentence1 = "Detected classes: " + "; ".join(first_parts) + "."

    remaining = structured[len(major):]
    if remaining:
        extras = ", ".join(str(r["class"]) for r in remaining[:6])
        sentence2 = f"Additional small classes: {extras}."
        return sentence1 + " " + sentence2

    return sentence1


def analyze_mask(
    mask_path: Path,
    class_map: Dict[int, str],
    min_area_ratio: float,
    min_non_bg_classes: int,
    fast_downsample_factor: int,
    allow_binary_fallback: bool,
) -> Tuple[bool, Dict[str, object]]:
    arr = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    total = float(arr.size)

    unique_ids = np.unique(arr)
    non_bg_ids = [int(cid) for cid in unique_ids if int(cid) != 0]
    mask_mode = "multiclass"

    if len(non_bg_ids) < min_non_bg_classes:
        if allow_binary_fallback and len(non_bg_ids) >= 1:
            mask_mode = "binary-fallback"
        else:
            return False, {
                "mask_path": str(mask_path),
                "id": extract_index(mask_path.name),
                "num_non_bg_classes": len(non_bg_ids),
                "structured": [],
                "description": "Skipped: mask does not meet multiclass requirement.",
            }

    arr_fast = downsample_mask(arr, fast_downsample_factor)

    structured: List[Dict[str, object]] = []
    for cid in non_bg_ids:
        full_binary = (arr == cid)
        pixels = int(full_binary.sum())
        if pixels == 0:
            continue

        ratio = float(pixels) / total
        if ratio < min_area_ratio:
            continue

        fast_binary = (arr_fast == cid)
        structured.append(
            {
                "class_id": cid,
                "class": class_map.get(cid, f"class_{cid}"),
                "count": connected_components_count(fast_binary),
                "area": area_bucket(ratio),
                "location": location_bucket(fast_binary),
                "area_ratio": round(ratio, 5),
            }
        )

    structured.sort(key=lambda x: float(x["area_ratio"]), reverse=True)
    if not structured:
        return False, {
            "mask_path": str(mask_path),
            "id": extract_index(mask_path.name),
            "num_non_bg_classes": len(non_bg_ids),
            "structured": [],
            "description": "Skipped: no classes passed area threshold.",
        }

    description = describe_scene(structured)
    if mask_mode == "binary-fallback":
        description = "Binary fallback mode: " + description

    return True, {
        "mask_path": str(mask_path),
        "id": extract_index(mask_path.name),
        "num_non_bg_classes": len(non_bg_ids),
        "mask_mode": mask_mode,
        "structured": [
            {
                "class": row["class"],
                "count": row["count"],
                "area": row["area"],
                "location": row["location"],
            }
            for row in structured
        ],
        "description": description,
        "class_details": structured,
    }


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    mask_root = RUN_MASK_ROOT
    if RUN_GENERATE_MASKS_FROM_IMAGES:
        mask_root = generate_multiclass_masks_from_images(
            image_root=RUN_IMAGE_ROOT,
            out_mask_root=RUN_GENERATED_MASK_ROOT,
            model_id=RUN_SEG_MODEL_ID,
            max_masks=RUN_MAX_MASKS,
            overwrite=RUN_OVERWRITE_GENERATED_MASKS,
        )

    if not mask_root.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_root}")

    mask_paths = sorted(mask_root.glob("Mask_*.png"))
    if not mask_paths:
        raise RuntimeError(f"No masks found in: {mask_root}")

    use_n = min(len(mask_paths), max(1, int(RUN_MAX_MASKS)))
    selected = mask_paths[:use_n]

    rows: List[Dict[str, object]] = []
    skipped = 0
    mode_counts = {"multiclass": 0, "binary-fallback": 0}

    for mask_path in selected:
        ok, result = analyze_mask(
            mask_path=mask_path,
            class_map=CLASS_MAP,
            min_area_ratio=float(RUN_MIN_AREA_RATIO),
            min_non_bg_classes=int(RUN_MIN_NON_BG_CLASSES),
            fast_downsample_factor=max(1, int(RUN_FAST_DOWNSAMPLE_FACTOR)),
            allow_binary_fallback=bool(RUN_ALLOW_BINARY_FALLBACK),
        )
        if ok:
            rows.append(result)
            mode = str(result.get("mask_mode", "multiclass"))
            if mode in mode_counts:
                mode_counts[mode] += 1
        else:
            skipped += 1

    write_jsonl(RUN_OUTPUT_JSONL, rows)

    print(f"Processed masks: {use_n}")
    print(f"Valid multiclass masks: {len(rows)}")
    print(f"- Pure multiclass: {mode_counts['multiclass']}")
    print(f"- Binary fallback: {mode_counts['binary-fallback']}")
    print(f"Skipped masks: {skipped}")
    print(f"Saved output: {RUN_OUTPUT_JSONL}")

    for sample in rows[:3]:
        print("-")
        print(Path(str(sample["mask_path"])).name)
        print(sample["description"])


if __name__ == "__main__":
    main()
