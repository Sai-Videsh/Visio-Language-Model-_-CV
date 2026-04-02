from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration


# -----------------------------
# Global runtime configuration
# -----------------------------
IMAGE_ROOT = Path("archive/IDD_RESIZED/image_archive")
OUTPUT_JSONL = Path("dataset/synthetic_captions_blip.jsonl")
OUTPUT_CSV = Path("dataset/synthetic_captions_blip.csv")

# BLIP captioning model
CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-large"

# Inference settings
MAX_IMAGES = 3000
MIN_CAPTIONS_PER_IMAGE = 3
MAX_CAPTIONS_PER_IMAGE = 5
SAMPLING_TEMPERATURE = 0.85
SAMPLING_TOP_P = 0.9
MAX_NEW_TOKENS = 30
MIN_NEW_TOKENS = 6
RANDOM_SEED = 42


def find_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def generate_caption(
    image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
) -> str:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        num_beams=1,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
    )

    text = processor.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


def save_jsonl(rows: List[Dict[str, str]], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(rows: List[Dict[str, str]], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "image_name", "caption_index", "caption"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_paths = find_images(IMAGE_ROOT)
    if not image_paths:
        raise RuntimeError(f"No images found under: {IMAGE_ROOT}")

    image_paths = image_paths[: min(len(image_paths), MAX_IMAGES)]
    print(f"Found {len(image_paths)} images")
    print(f"Loading caption model: {CAPTION_MODEL_ID}")

    processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID).to(device)
    model.eval()

    rows: List[Dict[str, str]] = []

    for path in tqdm(image_paths, desc="Generating captions"):
        image = Image.open(path).convert("RGB")
        count = random.randint(MIN_CAPTIONS_PER_IMAGE, MAX_CAPTIONS_PER_IMAGE)
        seen = set()
        captions: List[str] = []

        # Try a few extra generations to deduplicate sampled outputs.
        max_tries = count * 3
        tries = 0
        while len(captions) < count and tries < max_tries:
            tries += 1
            caption = generate_caption(image, processor, model, device)
            key = caption.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            captions.append(caption)

        if not captions:
            continue

        for i, caption in enumerate(captions, start=1):
            rows.append(
                {
                    "image_path": str(path),
                    "image_name": path.name,
                    "caption_index": str(i),
                    "caption": caption,
                }
            )

    save_jsonl(rows, OUTPUT_JSONL)
    save_csv(rows, OUTPUT_CSV)

    print("\nCaption generation complete.")
    print(f"Saved JSONL: {OUTPUT_JSONL}")
    print(f"Saved CSV:   {OUTPUT_CSV}")
    print(f"Generated {len(rows)} total image-caption rows from {len(image_paths)} images.")
    print("\nSample outputs:")
    for sample in rows[:5]:
        print(f"- {sample['image_name']} -> {sample['caption']}")


if __name__ == "__main__":
    main()
