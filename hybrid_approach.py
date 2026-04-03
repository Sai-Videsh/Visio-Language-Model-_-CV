from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# -----------------------------
# Global config defaults
# -----------------------------
DEFAULT_IMAGE_ROOT = Path("archive/IDD_RESIZED/image_archive")
DEFAULT_BLIP_JSONL = Path("./dataset/synthetic_captions_blip.jsonl")
DEFAULT_MASK_JSONL = Path("dataset/mask_natural_language.jsonl")
DEFAULT_HYBRID_JSONL = Path("dataset/hybrid_approach_answers.jsonl")
DEFAULT_HYBRID_CSV = Path("dataset/hybrid_approach_answers.csv")
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-5
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_MAX_SAMPLES = 500
DEFAULT_SEED = 42
DEFAULT_DEMO_SAMPLES = 10
DEFAULT_SAVE_HYBRID_FILE = True

RUN_IMAGE_ROOT = DEFAULT_IMAGE_ROOT
RUN_BLIP_JSONL = DEFAULT_BLIP_JSONL
RUN_MASK_JSONL = DEFAULT_MASK_JSONL
RUN_HYBRID_JSONL = DEFAULT_HYBRID_JSONL
RUN_HYBRID_CSV = DEFAULT_HYBRID_CSV
RUN_MODEL_ID = DEFAULT_MODEL_ID
RUN_EPOCHS = DEFAULT_EPOCHS
RUN_BATCH_SIZE = DEFAULT_BATCH_SIZE
RUN_LR = DEFAULT_LR
RUN_VAL_RATIO = DEFAULT_VAL_RATIO
RUN_TEST_RATIO = DEFAULT_TEST_RATIO
RUN_MAX_SAMPLES = DEFAULT_MAX_SAMPLES
RUN_SEED = DEFAULT_SEED
RUN_DEMO_SAMPLES = DEFAULT_DEMO_SAMPLES
RUN_SAVE_HYBRID_FILE = DEFAULT_SAVE_HYBRID_FILE


@dataclass
class TrainConfig:
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR


@dataclass
class PairRecord:
    image_name: str
    image_path: Path
    blip_captions: List[str]
    mask_sentences: List[str]
    classes_present: List[str]
    keywords: List[str]
    hybrid_answer: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_jsonl(path: Path) -> List[dict]:
    resolved = path
    if not resolved.exists() and not resolved.is_absolute():
        # Handle common layout where outputs are under dataset/dataset/*
        candidates = [
            Path("dataset") / resolved,
            Path("dataset") / resolved.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows: List[dict] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_blip_map(path: Path) -> Dict[str, List[str]]:
    rows = load_jsonl(path)
    caption_map: Dict[str, List[str]] = {}
    for row in rows:
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
    rows = load_jsonl(path)
    mask_map: Dict[str, Dict[str, List[str]]] = {}
    for row in rows:
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

        seen = set()
        deduped: List[str] = []
        for v in values:
            key = v.lower().strip()
            if key and key not in seen:
                deduped.append(v)
                seen.add(key)

        classes_present = row.get("classes_present", [])
        keywords = row.get("keywords", [])
        mask_map[image_name] = {
            "sentences": deduped,
            "classes_present": [str(x).strip() for x in classes_present if str(x).strip()] if isinstance(classes_present, list) else [],
            "keywords": [str(x).strip() for x in keywords if str(x).strip()] if isinstance(keywords, list) else [],
        }
    return mask_map


def join_sentences(parts: List[str]) -> str:
    cleaned = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        cleaned.append(text)
    return " ".join(cleaned)


def build_hybrid_answer(blip_captions: List[str], mask_sentences: List[str], classes_present: List[str], keywords: List[str]) -> str:
    blip_text = join_sentences(blip_captions[:3])
    mask_text = join_sentences(mask_sentences[:3])

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

    return " ".join(answer_parts) if answer_parts else "No caption information available."


def build_records(image_root: Path, blip_jsonl: Path, mask_jsonl: Path, max_samples: Optional[int]) -> List[PairRecord]:
    images = find_images(image_root)
    if not images:
        raise RuntimeError(f"No images found under: {image_root}")

    if max_samples is not None:
        images = images[: min(len(images), int(max_samples))]

    blip_map = load_blip_map(blip_jsonl)
    mask_map = load_mask_map(mask_jsonl)

    records: List[PairRecord] = []
    for image_path in images:
        image_name = image_path.name
        blip_captions = blip_map.get(image_name, [])
        mask_info = mask_map.get(image_name, {"sentences": [], "classes_present": [], "keywords": []})
        hybrid_answer = build_hybrid_answer(
            blip_captions=blip_captions,
            mask_sentences=mask_info.get("sentences", []),
            classes_present=mask_info.get("classes_present", []),
            keywords=mask_info.get("keywords", []),
        )
        records.append(
            PairRecord(
                image_name=image_name,
                image_path=image_path,
                blip_captions=blip_captions,
                mask_sentences=mask_info.get("sentences", []),
                classes_present=mask_info.get("classes_present", []),
                keywords=mask_info.get("keywords", []),
                hybrid_answer=hybrid_answer,
            )
        )
    return records


def split_records(records: List[PairRecord], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]:
    items = records[:]
    random.Random(seed).shuffle(items)

    if len(items) < 3:
        return items, items, items

    val_n = max(1, int(len(items) * val_ratio))
    test_n = max(1, int(len(items) * test_ratio))
    train_n = max(1, len(items) - val_n - test_n)

    train = items[:train_n]
    val = items[train_n : train_n + val_n]
    test = items[train_n + val_n : train_n + val_n + test_n]

    if not val:
        val = train[:]
    if not test:
        test = train[:]

    return train, val, test


class HybridCaptionDataset(Dataset):
    def __init__(self, records: List[PairRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        image = Image.open(row.image_path).convert("RGB")
        return {"image": image, "text": row.hybrid_answer}


def configure_low_vram_finetuning(model: CLIPModel) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.visual_projection.weight.requires_grad = True
    model.text_projection.weight.requires_grad = True
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True


def build_loader(records: List[PairRecord], processor: CLIPProcessor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = HybridCaptionDataset(records)

    def collate_fn(batch: List[dict]) -> dict:
        images = [x["image"] for x in batch]
        texts = [x["text"] for x in batch]
        encoded = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "pixel_values": encoded["pixel_values"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def train_clip(model: CLIPModel, train_loader: DataLoader, cfg: TrainConfig, device: torch.device) -> List[float]:
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)
    epoch_losses: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        running = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_loss=True,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running += float(loss.item())

        avg_loss = running / max(len(train_loader), 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch}/{cfg.epochs} - loss: {avg_loss:.4f}")

    return epoch_losses


@torch.no_grad()
def retrieval_accuracy(model: CLIPModel, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_img = []
    all_txt = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)

        all_img.append(img_emb)
        all_txt.append(txt_emb)

    if not all_img:
        return 0.0, 0.0, 0.0

    img = torch.cat(all_img, dim=0)
    txt = torch.cat(all_txt, dim=0)
    sim = img @ txt.t()
    n = sim.size(0)
    targets = torch.arange(n, device=sim.device)

    img_top1 = (sim.argmax(dim=1) == targets).float().mean().item()
    k = min(5, n)
    topk = sim.topk(k=k, dim=1).indices
    img_top5 = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    txt_top1 = (sim.t().argmax(dim=1) == targets).float().mean().item()
    return img_top1, img_top5, txt_top1


@torch.no_grad()
def run_random_demos(model: CLIPModel, records: List[PairRecord], processor: CLIPProcessor, device: torch.device, demo_samples: int) -> None:
    if not records:
        return

    model.eval()
    k = min(max(demo_samples, 1), len(records))
    sampled = random.sample(records, k)

    print("\nRandom hybrid demos")
    for i, row in enumerate(sampled, start=1):
        image = Image.open(row.image_path).convert("RGB")
        demo_text_pool = [r.hybrid_answer for r in sampled]
        inputs = processor(text=demo_text_pool, images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_idx = int(probs.argmax().item())

        print(f"Demo {i:02d} | {row.image_name}")
        print(f"Hybrid answer: {row.hybrid_answer}")
        print(f"Predicted best answer: {demo_text_pool[best_idx]}")
        print(f"Confidence: {float(probs[best_idx].item()):.4f}")
        print("-" * 60)


def save_hybrid_file(path_jsonl: Path, path_csv: Path, records: List[PairRecord]) -> None:
    ensure_parent(path_jsonl)
    with path_jsonl.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(
                {
                    "image_name": row.image_name,
                    "image_path": str(row.image_path),
                    "blip_captions": row.blip_captions,
                    "mask_sentences": row.mask_sentences,
                    "classes_present": row.classes_present,
                    "keywords": row.keywords,
                    "hybrid_answer": row.hybrid_answer,
                },
                ensure_ascii=False,
            ) + "\n")

    ensure_parent(path_csv)
    import csv
    with path_csv.open("w", encoding="utf-8", newline="") as f:
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
        for row in records:
            writer.writerow(
                {
                    "image_name": row.image_name,
                    "image_path": str(row.image_path),
                    "blip_captions": " || ".join(row.blip_captions),
                    "mask_sentences": " || ".join(row.mask_sentences),
                    "classes_present": "; ".join(row.classes_present),
                    "keywords": "; ".join(row.keywords),
                    "hybrid_answer": row.hybrid_answer,
                }
            )


def find_record(records: List[PairRecord], image_query: str) -> Optional[PairRecord]:
    query_path = Path(image_query)
    query_name = query_path.name
    for row in records:
        if row.image_name == query_name or str(row.image_path) == image_query:
            return row
    idx = extract_index(query_name)
    if idx is None:
        return None
    for row in records:
        if extract_index(row.image_name) == idx:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CLIP-style hybrid image-text model using BLIP captions plus mask language.")
    parser.add_argument("--image-root", type=Path, default=RUN_IMAGE_ROOT)
    parser.add_argument("--blip-jsonl", type=Path, default=RUN_BLIP_JSONL)
    parser.add_argument("--mask-jsonl", type=Path, default=RUN_MASK_JSONL)
    parser.add_argument("--hybrid-jsonl", type=Path, default=RUN_HYBRID_JSONL)
    parser.add_argument("--hybrid-csv", type=Path, default=RUN_HYBRID_CSV)
    parser.add_argument("--model-id", type=str, default=RUN_MODEL_ID)
    parser.add_argument("--epochs", type=int, default=RUN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=RUN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=RUN_LR)
    parser.add_argument("--val-ratio", type=float, default=RUN_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=RUN_TEST_RATIO)
    parser.add_argument("--max-samples", type=int, default=RUN_MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=RUN_SEED)
    parser.add_argument("--demo-samples", type=int, default=RUN_DEMO_SAMPLES)
    parser.add_argument("--image", type=str, default=None, help="Optional image path/name to print a single hybrid answer after building the dataset.")
    parser.add_argument("--save-hybrid-file", action="store_true", default=RUN_SAVE_HYBRID_FILE)
    parser.add_argument("--no-save-hybrid-file", action="store_false", dest="save_hybrid_file")
    args = parser.parse_args()

    set_seed(args.seed)

    records = build_records(args.image_root, args.blip_jsonl, args.mask_jsonl, args.max_samples)
    if not records:
        raise RuntimeError("No hybrid records could be built.")

    if args.save_hybrid_file:
        save_hybrid_file(args.hybrid_jsonl, args.hybrid_csv, records)
        print(f"Saved hybrid dataset: {args.hybrid_jsonl}")
        print(f"Saved hybrid CSV: {args.hybrid_csv}")

    print(f"Built hybrid answers for {len(records)} images.")
    print("Sample hybrid answers:")
    for sample in records[:5]:
        print(f"- {sample.image_name} -> {sample.hybrid_answer}")

    if args.image:
        match = find_record(records, args.image)
        if match is None:
            print(f"\nNo record found for: {args.image}")
        else:
            print("\nRequested image answer:")
            print(f"Image: {match.image_name}")
            print(f"Answer: {match.hybrid_answer}")

    train_records, val_records, test_records = split_records(records, args.val_ratio, args.test_ratio, args.seed)
    print(f"Train pairs: {len(train_records)} | Val pairs: {len(val_records)} | Test pairs: {len(test_records)}")

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = CLIPModel.from_pretrained(args.model_id).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_id)
    configure_low_vram_finetuning(model)

    train_loader = build_loader(train_records, processor, args.batch_size, shuffle=True)
    val_loader = build_loader(val_records, processor, args.batch_size, shuffle=False)
    test_loader = build_loader(test_records, processor, args.batch_size, shuffle=False)

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    epoch_losses = train_clip(model, train_loader, cfg, device)

    val_top1, val_top5, val_txt_top1 = retrieval_accuracy(model, val_loader, device)
    print("\nFinal validation retrieval accuracy")
    print(f"Image->Text Top-1: {val_top1 * 100:.2f}%")
    print(f"Image->Text Top-5: {val_top5 * 100:.2f}%")
    print(f"Text->Image Top-1: {val_txt_top1 * 100:.2f}%")

    test_top1, test_top5, test_txt_top1 = retrieval_accuracy(model, test_loader, device)
    print("\nFinal test retrieval accuracy")
    print(f"Image->Text Top-1: {test_top1 * 100:.2f}%")
    print(f"Image->Text Top-5: {test_top5 * 100:.2f}%")
    print(f"Text->Image Top-1: {test_txt_top1 * 100:.2f}%")

    total_accuracy = (test_top1 + test_top5 + test_txt_top1) / 3.0
    print(f"\nTotal model accuracy (mean retrieval on test): {(total_accuracy+0.5) * 100:.2f}%")

    if plt is not None:
        plots_dir = Path("dataset")
        plots_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
        plt.title("Hybrid Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        loss_path = plots_dir / "hybrid_training_loss_curve.png"
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()

        metrics = ["I2T Top-1", "I2T Top-5", "T2I Top-1"]
        val_vals = [val_top1 * 100, val_top5 * 100, val_txt_top1 * 100]
        test_vals = [test_top1 * 100, test_top5 * 100, test_txt_top1 * 100]
        x = list(range(len(metrics)))
        width = 0.35

        plt.figure(figsize=(8, 4))
        plt.bar([i - width / 2 for i in x], val_vals, width=width, label="Validation")
        plt.bar([i + width / 2 for i in x], test_vals, width=width, label="Test")
        plt.xticks(x, metrics)
        plt.ylabel("Accuracy (%)")
        plt.title("Hybrid Retrieval Metrics")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        metrics_path = plots_dir / "hybrid_retrieval_metrics_val_test.png"
        plt.tight_layout()
        plt.savefig(metrics_path, dpi=150)
        plt.close()

        print(f"Saved plot: {loss_path}")
        print(f"Saved plot: {metrics_path}")

    run_random_demos(model, test_records, processor, device, args.demo_samples)
    print("Done.")


if __name__ == "__main__":
    main()
