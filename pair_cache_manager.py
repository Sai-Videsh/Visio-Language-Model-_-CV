from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


DEFAULT_IMAGE_ROOT = Path("archive/IDD_RESIZED/image_archive")
DEFAULT_CAPTIONS_JSONL = Path("dataset/synthetic_captions_blip.jsonl")
DEFAULT_PAIRS_JSONL = Path("dataset/cached_image_text_pairs.jsonl")
DEFAULT_PAIRS_MD = Path("dataset/cached_image_text_pairs.md")
DEFAULT_MASK_LANGUAGE_JSONL = Path("dataset/mask_natural_language.jsonl")
DEFAULT_HYBRID_PAIRS_JSONL = Path("dataset/cached_image_text_pairs_hybrid.jsonl")
DEFAULT_HYBRID_PAIRS_MD = Path("dataset/cached_image_text_pairs_hybrid.md")
DEFAULT_MAX_SAMPLES = 2500


def find_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def load_generated_caption_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Caption file not found: {path}")

    caption_map: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_name = str(row.get("image_name") or Path(str(row.get("image_path", ""))).name)
            captions = row.get("captions")
            if isinstance(captions, list):
                caption_list = [str(c).strip() for c in captions if str(c).strip()]
            else:
                single = str(row.get("caption", "")).strip()
                caption_list = [single] if single else []

            if not image_name or not caption_list:
                continue

            existing = caption_map.setdefault(image_name, [])
            existing_keys = {c.lower().strip() for c in existing}
            for caption in caption_list:
                key = caption.lower().strip()
                if key and key not in existing_keys:
                    existing.append(caption)
                    existing_keys.add(key)
    return caption_map


def extract_index(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def load_mask_language_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Mask language file not found: {path}")

    mask_map: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            image_name = Path(str(row.get("image_path", ""))).name
            if not image_name:
                idx = row.get("id")
                if idx is not None:
                    image_name = f"Image_{idx}.png"

            if not image_name:
                continue

            values: List[str] = []
            primary = str(row.get("primary_caption", "")).strip()
            if primary:
                values.append(primary)

            sentences = row.get("sentences", [])
            if isinstance(sentences, list):
                for s in sentences:
                    text = str(s).strip()
                    if text:
                        values.append(text)

            if not values:
                continue

            dedup = []
            seen = set()
            for v in values:
                key = v.lower().strip()
                if key and key not in seen:
                    dedup.append(v)
                    seen.add(key)

            if dedup:
                mask_map[image_name] = dedup

    return mask_map


def save_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_markdown(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Cached Image-Text Pairs\n\n")
        f.write(f"Total pairs: {len(rows)}\n\n")
        f.write("| # | Image | Caption |\n")
        f.write("|---|---|---|\n")
        for i, row in enumerate(rows, start=1):
            image_name = Path(str(row["image_path"])).name
            caption = str(row["caption"]).replace("|", "\\|")
            f.write(f"| {i} | {image_name} | {caption} |\n")


def load_pairs_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")

    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_and_cache_pairs(
    image_root: Path = DEFAULT_IMAGE_ROOT,
    captions_jsonl: Path = DEFAULT_CAPTIONS_JSONL,
    pairs_jsonl: Path = DEFAULT_PAIRS_JSONL,
    pairs_md: Path = DEFAULT_PAIRS_MD,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> List[dict]:
    images = find_images(image_root)
    if not images:
        raise RuntimeError(f"No images found at {image_root}")

    images = images[: min(len(images), max_samples)]
    caption_map = load_generated_caption_map(captions_jsonl)

    rows: List[dict] = []
    missing = 0
    for img_path in images:
        captions = caption_map.get(img_path.name)
        if not captions:
            missing += 1
            continue

        for caption in captions:
            rows.append(
                {
                    "image_path": str(img_path),
                    "caption": caption,
                    "caption_score": None,
                }
            )

    if not rows:
        raise RuntimeError("No cached pairs generated. Check caption file/image names.")

    save_jsonl(pairs_jsonl, rows)
    save_markdown(pairs_md, rows)

    if missing > 0:
        print(f"Warning: {missing} images were skipped due to missing captions.")

    print(f"Cached pairs JSONL: {pairs_jsonl}")
    print(f"Cached pairs Markdown: {pairs_md}")
    print(f"Total cached pairs: {len(rows)}")
    return rows


def build_and_cache_hybrid_pairs(
    image_root: Path = DEFAULT_IMAGE_ROOT,
    captions_jsonl: Path = DEFAULT_CAPTIONS_JSONL,
    mask_language_jsonl: Path = DEFAULT_MASK_LANGUAGE_JSONL,
    pairs_jsonl: Path = DEFAULT_HYBRID_PAIRS_JSONL,
    pairs_md: Path = DEFAULT_HYBRID_PAIRS_MD,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> List[dict]:
    images = find_images(image_root)
    if not images:
        raise RuntimeError(f"No images found at {image_root}")

    images = images[: min(len(images), max_samples)]
    blip_map = load_generated_caption_map(captions_jsonl)
    mask_map = load_mask_language_map(mask_language_jsonl)

    rows: List[dict] = []
    missing_blip = 0
    missing_mask = 0

    for img_path in images:
        image_name = img_path.name
        blip_captions = blip_map.get(image_name, [])
        mask_captions = mask_map.get(image_name, [])

        if not blip_captions:
            missing_blip += 1
        if not mask_captions:
            missing_mask += 1

        # BLIP-only rows
        for caption in blip_captions:
            rows.append(
                {
                    "image_path": str(img_path),
                    "caption": caption,
                    "caption_source": "blip",
                    "caption_score": None,
                }
            )

        # Mask-only rows
        for caption in mask_captions:
            rows.append(
                {
                    "image_path": str(img_path),
                    "caption": caption,
                    "caption_source": "mask",
                    "caption_score": None,
                }
            )

        # Hybrid rows (pair BLIP and mask text by cycling shorter list)
        if blip_captions and mask_captions:
            n = max(len(blip_captions), len(mask_captions))
            for i in range(n):
                b = blip_captions[i % len(blip_captions)]
                m = mask_captions[i % len(mask_captions)]
                hybrid = f"{b} Scene geometry: {m}"
                rows.append(
                    {
                        "image_path": str(img_path),
                        "caption": hybrid,
                        "caption_source": "hybrid",
                        "caption_score": None,
                    }
                )

    if not rows:
        raise RuntimeError("No hybrid pairs generated. Check BLIP and mask language files.")

    save_jsonl(pairs_jsonl, rows)
    save_markdown(pairs_md, rows)

    if missing_blip > 0:
        print(f"Warning: {missing_blip} images were missing BLIP captions.")
    if missing_mask > 0:
        print(f"Warning: {missing_mask} images were missing mask captions.")

    source_counts = {"blip": 0, "mask": 0, "hybrid": 0}
    for row in rows:
        src = str(row.get("caption_source", ""))
        if src in source_counts:
            source_counts[src] += 1

    print(f"Cached hybrid pairs JSONL: {pairs_jsonl}")
    print(f"Cached hybrid pairs Markdown: {pairs_md}")
    print(f"Total cached pairs: {len(rows)}")
    print(f"Source counts: {source_counts}")
    return rows


if __name__ == "__main__":
    build_and_cache_pairs()
