from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Global runtime configuration
# -----------------------------
IMAGE_ROOT = Path("archive/IDD_RESIZED/image_archive")
BLIP_JSONL = Path("dataset/synthetic_captions_blip.jsonl")
MASK_JSONL = Path("dataset/mask_natural_language.jsonl")
OUTPUT_JSONL = Path("dataset/hybrid_image_answers.jsonl")
OUTPUT_CSV = Path("dataset/hybrid_image_answers.csv")

MAX_IMAGES = None
MAX_BLIP_CAPTIONS = 3
MAX_MASK_SENTENCES = 3


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def extract_index(name: str) -> Optional[int]:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def load_blip_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"BLIP caption file not found: {path}")

    caption_map: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_name = str(row.get("image_name") or Path(str(row.get("image_path", ""))).name)
            caption = str(row.get("caption", "")).strip()
            if not image_name or not caption:
                continue

            existing = caption_map.setdefault(image_name, [])
            key = caption.lower().strip()
            if key and key not in {c.lower().strip() for c in existing}:
                existing.append(caption)
    return caption_map


def load_mask_map(path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Mask language file not found: {path}")

    mask_map: Dict[str, Dict[str, List[str]]] = {}
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

            primary = str(row.get("primary_caption", "")).strip()
            sentences = row.get("sentences", [])
            classes_present = row.get("classes_present", [])
            keywords = row.get("keywords", [])

            sentence_list: List[str] = []
            if primary:
                sentence_list.append(primary)
            if isinstance(sentences, list):
                for sentence in sentences:
                    text = str(sentence).strip()
                    if text:
                        sentence_list.append(text)

            unique_sentences: List[str] = []
            seen = set()
            for sentence in sentence_list:
                key = sentence.lower().strip()
                if key and key not in seen:
                    unique_sentences.append(sentence)
                    seen.add(key)

            mask_map[image_name] = {
                "sentences": unique_sentences,
                "classes_present": [str(x).strip() for x in classes_present if str(x).strip()] if isinstance(classes_present, list) else [],
                "keywords": [str(x).strip() for x in keywords if str(x).strip()] if isinstance(keywords, list) else [],
            }
    return mask_map


def join_sentences(parts: List[str]) -> str:
    clean_parts = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        clean_parts.append(text)
    return " ".join(clean_parts)


def build_answer(blip_captions: List[str], mask_sentences: List[str], classes_present: List[str], keywords: List[str]) -> str:
    blip_selected = blip_captions[:MAX_BLIP_CAPTIONS] if blip_captions else []
    mask_selected = mask_sentences[:MAX_MASK_SENTENCES] if mask_sentences else []

    blip_text = join_sentences(blip_selected)
    mask_text = join_sentences(mask_selected)

    class_text = ""
    if classes_present:
        class_text = f"Detected mask classes: {', '.join(classes_present[:6])}."
    elif keywords:
        class_text = f"Mask keywords: {', '.join(keywords[:6])}."

    answer_parts: List[str] = []
    if blip_text:
        answer_parts.append(f"Natural caption: {blip_text}")
    if mask_text:
        answer_parts.append(f"Mask geometry: {mask_text}")
    if class_text:
        answer_parts.append(class_text)

    if not answer_parts:
        return "No caption information available."

    return " ".join(answer_parts)


def build_records(image_root: Path, blip_jsonl: Path, mask_jsonl: Path, max_images: Optional[int]) -> List[dict]:
    images = find_images(image_root)
    if not images:
        raise RuntimeError(f"No images found under: {image_root}")

    if max_images is not None:
        images = images[: min(len(images), int(max_images))]

    blip_map = load_blip_map(blip_jsonl)
    mask_map = load_mask_map(mask_jsonl)

    rows: List[dict] = []
    for image_path in images:
        image_name = image_path.name
        blip_captions = blip_map.get(image_name, [])
        mask_info = mask_map.get(image_name, {"sentences": [], "classes_present": [], "keywords": []})

        answer = build_answer(
            blip_captions=blip_captions,
            mask_sentences=mask_info.get("sentences", []),
            classes_present=mask_info.get("classes_present", []),
            keywords=mask_info.get("keywords", []),
        )

        rows.append(
            {
                "image_name": image_name,
                "image_path": str(image_path),
                "blip_captions": blip_captions,
                "mask_sentences": mask_info.get("sentences", []),
                "classes_present": mask_info.get("classes_present", []),
                "keywords": mask_info.get("keywords", []),
                "hybrid_answer": answer,
            }
        )

    return rows


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
                "image_name",
                "image_path",
                "blip_captions",
                "mask_sentences",
                "classes_present",
                "keywords",
                "hybrid_answer",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_name": row["image_name"],
                    "image_path": row["image_path"],
                    "blip_captions": " || ".join(row["blip_captions"]),
                    "mask_sentences": " || ".join(row["mask_sentences"]),
                    "classes_present": "; ".join(row["classes_present"]),
                    "keywords": "; ".join(row["keywords"]),
                    "hybrid_answer": row["hybrid_answer"],
                }
            )


def find_record(rows: List[dict], image_query: str) -> Optional[dict]:
    query_path = Path(image_query)
    query_name = query_path.name
    for row in rows:
        if row["image_name"] == query_name or row["image_path"] == image_query:
            return row
    idx = extract_index(query_name)
    if idx is None:
        return None
    for row in rows:
        row_idx = extract_index(row["image_name"])
        if row_idx == idx:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine BLIP captions and binary mask language into hybrid image answers.")
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--blip-jsonl", type=Path, default=BLIP_JSONL)
    parser.add_argument("--mask-jsonl", type=Path, default=MASK_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    parser.add_argument("--image", type=str, default=None, help="Optional image path or image name to print a single hybrid answer.")
    args = parser.parse_args()

    rows = build_records(args.image_root, args.blip_jsonl, args.mask_jsonl, args.max_images)
    save_jsonl(args.output_jsonl, rows)
    save_csv(args.output_csv, rows)

    print(f"Built hybrid answers for {len(rows)} images.")
    print(f"Saved JSONL: {args.output_jsonl}")
    print(f"Saved CSV:   {args.output_csv}")

    if rows:
        print("\nSample hybrid answers:")
        for sample in rows[:5]:
            print(f"- {sample['image_name']} -> {sample['hybrid_answer']}")

    if args.image:
        row = find_record(rows, args.image)
        if row is None:
            print(f"\nNo record found for: {args.image}")
        else:
            print("\nRequested image answer:")
            print(f"Image: {row['image_name']}")
            print(f"Answer: {row['hybrid_answer']}")


if __name__ == "__main__":
    main()
