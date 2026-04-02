from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from pair_cache_manager import build_and_cache_hybrid_pairs, load_pairs_jsonl


# -----------------------------
# Global config defaults
# -----------------------------
DEFAULT_IMAGE_ROOT = Path("archive/IDD_RESIZED/image_archive")
DEFAULT_OUTPUT_JSONL = Path("dataset/image_only_static_pairs.jsonl")
DEFAULT_CAPTIONS_JSONL = Path("dataset/synthetic_captions_blip.jsonl")
DEFAULT_MASK_LANGUAGE_JSONL = Path("dataset/mask_natural_language.jsonl")
DEFAULT_PAIRS_MD = Path("dataset/cached_image_text_pairs_hybrid.md")
DEFAULT_CACHED_PAIRS_JSONL = Path("dataset/cached_image_text_pairs_hybrid.jsonl")
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"

DEFAULT_MAX_SAMPLES = 2500
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-5
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_DEMO_SAMPLES = 10
DEFAULT_SEED = 42

# Edit these once here, then run: python image_only_static_vl.py
RUN_IMAGE_ROOT = DEFAULT_IMAGE_ROOT
RUN_OUTPUT_JSONL = DEFAULT_OUTPUT_JSONL
RUN_CAPTIONS_JSONL = DEFAULT_CAPTIONS_JSONL
RUN_MASK_LANGUAGE_JSONL = DEFAULT_MASK_LANGUAGE_JSONL
RUN_PAIRS_MD = DEFAULT_PAIRS_MD
RUN_CACHED_PAIRS_JSONL = DEFAULT_CACHED_PAIRS_JSONL
RUN_MAX_SAMPLES = DEFAULT_MAX_SAMPLES
RUN_EPOCHS = DEFAULT_EPOCHS
RUN_BATCH_SIZE = DEFAULT_BATCH_SIZE
RUN_LR = DEFAULT_LR
RUN_VAL_RATIO = DEFAULT_VAL_RATIO
RUN_TEST_RATIO = DEFAULT_TEST_RATIO
RUN_DEMO_SAMPLES = DEFAULT_DEMO_SAMPLES
RUN_SEED = DEFAULT_SEED
RUN_BUILD_PAIRS_IF_MISSING = True

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_records(records: List[dict], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[dict], List[dict], List[dict]]:
    items = records[:]
    random.Random(seed).shuffle(items)
    if len(items) < 3:
        return items, items, items

    val_n = max(1, int(len(items) * val_ratio))
    test_n = max(1, int(len(items) * test_ratio))
    train_n = max(1, len(items) - val_n - test_n)

    train_records = items[:train_n]
    val_records = items[train_n : train_n + val_n]
    test_records = items[train_n + val_n : train_n + val_n + test_n]

    if not val_records:
        val_records = train_records[:]
    if not test_records:
        test_records = train_records[:]
    return train_records, val_records, test_records


class ImageCaptionDataset(Dataset):
    def __init__(self, records: List[dict], processor: CLIPProcessor) -> None:
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        text = row["caption"]
        return {
            "image": image,
            "text": text,
        }


@dataclass
class TrainConfig:
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR


def configure_low_vram_finetuning(model: CLIPModel) -> None:
    # Freeze full model first
    for p in model.parameters():
        p.requires_grad = False

    # Train lightweight alignment heads only
    model.visual_projection.weight.requires_grad = True
    model.text_projection.weight.requires_grad = True
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True


def build_loader(records: List[dict], processor: CLIPProcessor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = ImageCaptionDataset(records, processor)

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
def run_demo(model: CLIPModel, records: List[dict], processor: CLIPProcessor, device: torch.device, demo_samples: int) -> None:
    model.eval()
    if not records:
        return

    show_n = min(max(demo_samples, 1), len(records))
    subset = records[:show_n]

    print("\nDemo predictions")
    for row in subset:
        image = Image.open(row["image_path"]).convert("RGB")
        demo_caption_pool = [r["caption"] for r in subset]
        inputs = processor(text=demo_caption_pool, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_idx = int(probs.argmax().item())

        print(f"Image: {Path(row['image_path']).name}")
        print(f"Ground truth caption: {row['caption']}")
        print(f"Predicted caption: {demo_caption_pool[best_idx]}")
        print(f"Conf: {float(probs[best_idx].item()):.4f}")
        print("-" * 50)


def main() -> None:
    set_seed(RUN_SEED)

    if RUN_BUILD_PAIRS_IF_MISSING and not RUN_CACHED_PAIRS_JSONL.exists():
        print("Hybrid cached pairs not found. Building from existing BLIP + mask language files...")
        build_and_cache_hybrid_pairs(
            image_root=RUN_IMAGE_ROOT,
            captions_jsonl=RUN_CAPTIONS_JSONL,
            mask_language_jsonl=RUN_MASK_LANGUAGE_JSONL,
            pairs_jsonl=RUN_CACHED_PAIRS_JSONL,
            pairs_md=RUN_PAIRS_MD,
            max_samples=RUN_MAX_SAMPLES,
        )

    all_records = load_pairs_jsonl(RUN_CACHED_PAIRS_JSONL)
    all_records = all_records[: min(len(all_records), RUN_MAX_SAMPLES)]
    if not all_records:
        raise RuntimeError(f"No pairs found in cached file: {RUN_CACHED_PAIRS_JSONL}")
    print(f"Loaded cached pairs: {len(all_records)} from {RUN_CACHED_PAIRS_JSONL}")

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = CLIPModel.from_pretrained(DEFAULT_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(DEFAULT_MODEL_ID)
    configure_low_vram_finetuning(model)

    train_records, val_records, test_records = split_records(all_records, RUN_VAL_RATIO, RUN_TEST_RATIO, RUN_SEED)
    print(f"Train pairs: {len(train_records)} | Val pairs: {len(val_records)} | Test pairs: {len(test_records)}")

    train_loader = build_loader(train_records, processor, RUN_BATCH_SIZE, shuffle=True)
    val_loader = build_loader(val_records, processor, RUN_BATCH_SIZE, shuffle=False)
    test_loader = build_loader(test_records, processor, RUN_BATCH_SIZE, shuffle=False)

    cfg = TrainConfig(epochs=RUN_EPOCHS, batch_size=RUN_BATCH_SIZE, lr=RUN_LR)
    epoch_losses = train_clip(model, train_loader, cfg, device)

    top1, top5, txt_top1 = retrieval_accuracy(model, val_loader, device)
    print("\nFinal validation retrieval accuracy")
    print(f"Image->Text Top-1: {top1 * 100:.2f}%")
    print(f"Image->Text Top-5: {top5 * 100:.2f}%")
    print(f"Text->Image Top-1: {txt_top1 * 100:.2f}%")

    test_top1, test_top5, test_txt_top1 = retrieval_accuracy(model, test_loader, device)
    print("\nFinal test retrieval accuracy")
    print(f"Image->Text Top-1: {test_top1 * 100:.2f}%")
    print(f"Image->Text Top-5: {test_top5 * 100:.2f}%")
    print(f"Text->Image Top-1: {test_txt_top1 * 100:.2f}%")

    total_accuracy = (test_top1 + test_top5 + test_txt_top1) / 3.0
    print(f"\nTotal model accuracy (mean retrieval on test): {total_accuracy * 100:.2f}%")

    if plt is not None:
        plots_dir = Path("dataset")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Pareto chart 1: loss trend over epochs
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        loss_path = plots_dir / "training_loss_curve.png"
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()

        # Pareto chart 2: retrieval metric comparison on val vs test
        metrics = ["I2T Top-1", "I2T Top-5", "T2I Top-1"]
        val_vals = [top1 * 100, top5 * 100, txt_top1 * 100]
        test_vals = [test_top1 * 100, test_top5 * 100, test_txt_top1 * 100]
        x = list(range(len(metrics)))
        width = 0.35

        plt.figure(figsize=(8, 4))
        plt.bar([i - width / 2 for i in x], val_vals, width=width, label="Validation")
        plt.bar([i + width / 2 for i in x], test_vals, width=width, label="Test")
        plt.xticks(x, metrics)
        plt.ylabel("Accuracy (%)")
        plt.title("Retrieval Metrics")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        metrics_path = plots_dir / "retrieval_metrics_val_test.png"
        plt.tight_layout()
        plt.savefig(metrics_path, dpi=150)
        plt.close()

        print(f"Saved plot: {loss_path}")
        print(f"Saved plot: {metrics_path}")

    run_demo(model, test_records, processor, device, RUN_DEMO_SAMPLES)
    print("Done.")


if __name__ == "__main__":
    main()
