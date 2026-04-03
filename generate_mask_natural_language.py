from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Global runtime configuration
# -----------------------------
MASK_ROOT = Path("archive/IDD_RESIZED/mask_archive")
IMAGE_ROOT = Path("archive/IDD_RESIZED/image_archive")
OUTPUT_JSONL = Path("dataset/mask_natural_language.jsonl")
OUTPUT_CSV = Path("dataset/mask_natural_language.csv")

MAX_MASKS = None  # Set an int to cap processing, e.g. 2000
MIN_AREA_RATIO = 0.005
CAPTIONS_PER_MASK = 4

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

DYNAMIC_CLASSES = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}

INFRA_CLASSES = {
    "traffic light",
    "traffic sign",
    "pole",
    "fence",
    "wall",
    "building",
}

SURFACE_CLASSES = {
    "road",
    "sidewalk",
    "terrain",
    "sky",
    "vegetation",
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_index(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def find_masks(mask_root: Path) -> List[Path]:
    if not mask_root.exists():
        return []
    return sorted(mask_root.glob("Mask_*.png"))


def build_image_lookup(image_root: Path) -> Dict[int, Path]:
    lookup: Dict[int, Path] = {}
    if not image_root.exists():
        return lookup

    for path in image_root.glob("Image_*.png"):
        idx = extract_index(path.name)
        if idx is not None:
            lookup[idx] = path
    return lookup


def is_binary_mask(arr: np.ndarray) -> bool:
    unique = np.unique(arr)
    return len(unique) <= 2 and np.all(np.isin(unique, np.array([0, 1], dtype=np.uint8)))


def extract_binary_metrics(arr: np.ndarray) -> Dict[str, float]:
    h, w = arr.shape
    fg = (arr > 0).astype(np.uint8)
    total = float(h * w)
    fg_count = float(fg.sum())

    if fg_count == 0:
        return {
            "foreground_ratio": 0.0,
            "top_ratio": 0.0,
            "mid_ratio": 0.0,
            "bottom_ratio": 0.0,
            "left_ratio": 0.0,
            "right_ratio": 0.0,
            "center_x": 0.5,
            "row_continuity": 0.0,
            "bottom_width": 0.0,
            "mid_width": 0.0,
        }

    ys, xs = np.where(fg > 0)
    top_end = max(1, int(0.33 * h))
    mid_end = max(top_end + 1, int(0.66 * h))
    left_end = max(1, int(0.5 * w))

    top_ratio = float(fg[:top_end, :].sum()) / total
    mid_ratio = float(fg[top_end:mid_end, :].sum()) / total
    bottom_ratio = float(fg[mid_end:, :].sum()) / total

    left_ratio = float(fg[:, :left_end].sum()) / total
    right_ratio = float(fg[:, left_end:].sum()) / total

    row_presence = (fg.sum(axis=1) > 0).astype(np.float32)
    row_continuity = float(row_presence.mean())

    bottom_band = fg[int(0.8 * h) :, :]
    mid_band = fg[int(0.45 * h) : int(0.65 * h), :]
    bottom_width = float(bottom_band.mean()) if bottom_band.size > 0 else 0.0
    mid_width = float(mid_band.mean()) if mid_band.size > 0 else 0.0

    return {
        "foreground_ratio": fg_count / total,
        "top_ratio": top_ratio,
        "mid_ratio": mid_ratio,
        "bottom_ratio": bottom_ratio,
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "center_x": float(xs.mean()) / float(w),
        "row_continuity": row_continuity,
        "bottom_width": bottom_width,
        "mid_width": mid_width,
    }


def extract_mask_objects(mask_path: Path) -> List[Tuple[str, float]]:
    arr = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    total = float(arr.size)
    counts = np.bincount(arr.reshape(-1), minlength=256)

    objects: List[Tuple[str, float]] = []
    for cid, count in enumerate(counts):
        if count == 0:
            continue
        name = CLASS_MAP.get(cid)
        if not name or name == "background":
            continue

        ratio = float(count) / total
        if ratio < MIN_AREA_RATIO:
            continue
        objects.append((name, ratio))

    objects.sort(key=lambda x: x[1], reverse=True)
    return objects


def format_list(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def phrase_ratio(ratio: float) -> str:
    pct = ratio * 100.0
    if pct >= 35:
        return "dominant"
    if pct >= 15:
        return "prominent"
    if pct >= 5:
        return "visible"
    return "sparse"


def build_sentences(objects: List[Tuple[str, float]]) -> List[str]:
    if not objects:
        return [
            "A segmented urban scene with limited semantic evidence.",
            "An unlabeled-looking street mask with weak object presence.",
            "A road scene where major classes are not clearly visible.",
        ]

    names = [name for name, _ in objects]
    dynamic = [name for name, _ in objects if name in DYNAMIC_CLASSES]
    infra = [name for name, _ in objects if name in INFRA_CLASSES]
    surfaces = [name for name, _ in objects if name in SURFACE_CLASSES]

    top_names = [name for name, _ in objects[:3]]
    top_desc = format_list(top_names)

    s1 = f"A segmented street scene featuring {top_desc}."

    if surfaces:
        top_surface = surfaces[0]
        top_surface_ratio = next(r for n, r in objects if n == top_surface)
        surface_phrase = phrase_ratio(top_surface_ratio)
        s2 = f"The mask shows a {surface_phrase} {top_surface} region with structured urban layout."
    else:
        s2 = "The mask indicates an urban layout with mixed object regions."

    if dynamic:
        dyn_desc = format_list(dynamic[:3])
        if infra:
            inf_desc = format_list(infra[:2])
            s3 = f"Traffic participants like {dyn_desc} appear with supporting infrastructure such as {inf_desc}."
        else:
            s3 = f"Traffic participants such as {dyn_desc} are present in the scene."
    else:
        if infra:
            inf_desc = format_list(infra[:3])
            s3 = f"The scene is mainly structural, with elements like {inf_desc}."
        else:
            s3 = "The scene is dominated by static street surfaces and background structure."

    s4 = "Semantic regions suggest an urban driving context with clear lane-side boundaries."
    candidates = [s1, s2, s3, s4]
    return candidates[: max(1, CAPTIONS_PER_MASK)]


def build_binary_sentences(metrics: Dict[str, float]) -> List[str]:
    fg_ratio = metrics["foreground_ratio"]
    bg_ratio = 1.0 - fg_ratio
    top_ratio = metrics["top_ratio"]
    mid_ratio = metrics["mid_ratio"]
    bottom_ratio = metrics["bottom_ratio"]
    center_x = metrics["center_x"]
    row_continuity = metrics["row_continuity"]
    bottom_width = metrics["bottom_width"]
    mid_width = metrics["mid_width"]

    # ROAD EXTENT (foreground)
    if fg_ratio < 0.05:
        extent_phrase = "very limited drivable area"
    elif fg_ratio < 0.15:
        extent_phrase = "a narrow drivable region"
    elif fg_ratio < 0.35:
        extent_phrase = "a moderate drivable region"
    else:
        extent_phrase = "a broad drivable region"

    # NON-ROAD EXTENT (background)
    if bg_ratio > 0.85:
        bg_phrase = "heavily congested with non-road elements"
    elif bg_ratio > 0.65:
        bg_phrase = "dominated by sky, buildings, and structures"
    elif bg_ratio > 0.45:
        bg_phrase = "balanced between road and non-road regions"
    else:
        bg_phrase = "sparse with minimal occlusion from above"

    # LATERAL POSITIONING
    if center_x < 0.42:
        lateral_phrase = "leans toward the left side"
    elif center_x > 0.58:
        lateral_phrase = "leans toward the right side"
    else:
        lateral_phrase = "stays near the center"

    # DEPTH STRUCTURE
    if bottom_ratio > mid_ratio > top_ratio:
        depth_phrase = "expands in the foreground and tapers into the distance"
    elif top_ratio > 0.05:
        depth_phrase = "remains visible into the far field"
    else:
        depth_phrase = "is concentrated near the lower field of view"

    # CONTINUITY
    if row_continuity > 0.85:
        continuity_phrase = "continuous along most image rows"
    elif row_continuity > 0.55:
        continuity_phrase = "present across a moderate vertical span"
    else:
        continuity_phrase = "fragmented across limited vertical bands"

    # WIDTH DISTRIBUTION
    if bottom_width > max(mid_width + 0.05, 0.2):
        width_phrase = "wider near the bottom than in the middle"
    elif mid_width > max(bottom_width + 0.05, 0.2):
        width_phrase = "broader through the mid-frame than the lower frame"
    else:
        width_phrase = "with similar spread across lower and middle bands"

    # COMPOSITE SENTENCES (ROAD + NON-ROAD)
    s1 = f"A binary street mask with {extent_phrase}. Non-road regions are {bg_phrase}."
    s2 = f"The drivable segment {lateral_phrase} and {depth_phrase}."
    s3 = f"Road coverage is {continuity_phrase}, {width_phrase}. Background comprises {'{:.0f}%'.format(bg_ratio*100)} non-drivable area."
    s4 = f"This scene represents an urban context where {'{:.0f}%'.format(fg_ratio*100)} is navigable road amid {'{:.0f}%'.format(bg_ratio*100)} of structural obstruction."
    return [s1, s2, s3, s4][: max(1, CAPTIONS_PER_MASK)]


def build_keywords(objects: List[Tuple[str, float]], max_keywords: int = 8) -> List[str]:
    keywords = [name for name, _ in objects[:max_keywords]]
    if not keywords:
        return ["urban", "street", "segmentation"]
    return keywords


def build_binary_keywords(metrics: Dict[str, float]) -> List[str]:
    fg_ratio = metrics["foreground_ratio"]
    bg_ratio = 1.0 - fg_ratio
    
    keys = ["road", "drivable-area", "segmentation", "binary-mask"]
    
    # Lateral bias
    if metrics["center_x"] < 0.42:
        keys.append("left-biased")
    elif metrics["center_x"] > 0.58:
        keys.append("right-biased")
    else:
        keys.append("centered")

    # Continuity
    if metrics["row_continuity"] > 0.8:
        keys.append("continuous")
    else:
        keys.append("fragmented")

    # Road extent
    if fg_ratio > 0.3:
        keys.append("wide-road")
    elif fg_ratio < 0.12:
        keys.append("narrow-road")
    else:
        keys.append("medium-road")
    
    # Non-road (background) extent
    if bg_ratio > 0.68:
        keys.append("congested-scene")
    elif bg_ratio > 0.45:
        keys.append("balanced-composition")
    else:
        keys.append("open-scene")
    
    return keys


def save_jsonl(path: Path, rows: List[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "mask_path",
                "image_path",
                "primary_caption",
                "keywords",
                "sentences",
                "classes_present",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "mask_path": row["mask_path"],
                    "image_path": row["image_path"],
                    "primary_caption": row["primary_caption"],
                    "keywords": "; ".join(row["keywords"]),
                    "sentences": " || ".join(row["sentences"]),
                    "classes_present": "; ".join(row["classes_present"]),
                }
            )


def main() -> None:
    mask_paths = find_masks(MASK_ROOT)
    if not mask_paths:
        raise RuntimeError(f"No masks found in: {MASK_ROOT}")

    if MAX_MASKS is not None:
        mask_paths = mask_paths[: min(len(mask_paths), int(MAX_MASKS))]

    image_lookup = build_image_lookup(IMAGE_ROOT)
    rows: List[dict] = []

    binary_count = 0

    for mask_path in tqdm(mask_paths, desc="Generating natural language from masks"):
        idx = extract_index(mask_path.name)
        arr = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        binary_mode = is_binary_mask(arr)

        if binary_mode:
            binary_count += 1
            metrics = extract_binary_metrics(arr)
            sentences = build_binary_sentences(metrics)
            keywords = build_binary_keywords(metrics)
            ratios = {"road": round(metrics["foreground_ratio"], 5)}
            classes_present = ["road"] if metrics["foreground_ratio"] > 0 else []
        else:
            objects = extract_mask_objects(mask_path)
            sentences = build_sentences(objects)
            keywords = build_keywords(objects)
            ratios = {name: round(ratio, 5) for name, ratio in objects}
            classes_present = [name for name, _ in objects]
            metrics = {}

        image_path = ""
        if idx is not None and idx in image_lookup:
            image_path = str(image_lookup[idx])

        rows.append(
            {
                "id": idx,
                "mask_path": str(mask_path),
                "image_path": image_path,
                "classes_present": classes_present,
                "class_ratios": ratios,
                "mask_metrics": metrics,
                "mask_mode": "binary" if binary_mode else "multiclass",
                "keywords": keywords,
                "sentences": sentences,
                "primary_caption": sentences[0],
            }
        )

    save_jsonl(OUTPUT_JSONL, rows)
    save_csv(OUTPUT_CSV, rows)

    print("Mask language generation complete.")
    print(f"Total masks processed: {len(rows)}")
    print(f"Saved JSONL: {OUTPUT_JSONL}")
    print(f"Saved CSV:   {OUTPUT_CSV}")
    print(f"Binary masks detected: {binary_count}/{len(rows)}")
    print("Sample outputs:")
    for sample in rows[:5]:
        print(f"- {Path(sample['mask_path']).name} -> {sample['primary_caption']}")


if __name__ == "__main__":
    main()
