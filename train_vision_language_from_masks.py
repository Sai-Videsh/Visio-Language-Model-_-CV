from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError as exc:
    raise ImportError("This script requires PyTorch and torchvision.") from exc

try:
    from transformers import AutoTokenizer, DistilBertModel
except ImportError as exc:
    raise ImportError("This script requires transformers. Install with: pip install transformers") from exc


# -----------------------------
# Global runtime configuration
# -----------------------------
RUN_STRUCTURED_JSONL = Path("dataset/dataset/multiclass_structured_scenes_500.jsonl")
RUN_IMAGE_ROOT = Path("archive/IDD_RESIZED/image_archive")
RUN_MASK_ROOT = Path("dataset/dataset/generated_multiclass_masks")
RUN_OUTPUT_DIR = Path("dataset/dataset/vlm_mask_text_runs")

RUN_TEXT_MODEL = "distilbert-base-uncased"
RUN_EMBED_DIM = 256
RUN_IMAGE_SIZE = 224
RUN_MAX_TEXT_LEN = 128
RUN_BATCH_SIZE = 16
RUN_EPOCHS = 8
RUN_LR = 2e-4
RUN_WEIGHT_DECAY = 1e-4
RUN_VAL_RATIO = 0.1
RUN_NUM_WORKERS = 0
RUN_SEED = 42
RUN_MAX_SAMPLES: int | None = None
RUN_NUM_DEMOS = 20


def extract_index(name: str) -> Optional[int]:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_images(root: Path) -> Dict[int, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    result: Dict[int, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        idx = extract_index(p.name)
        if idx is None:
            continue
        if idx not in result:
            result[idx] = p
    return result


def normalize_mask(mask: np.ndarray, max_class_id: int = 19) -> np.ndarray:
    x = mask.astype(np.float32) / float(max(1, max_class_id))
    return np.clip(x, 0.0, 1.0)


def resize_image(arr: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    pil = Image.fromarray(arr)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return np.array(pil.resize((size, size), resample=resample))


def compose_text(row: dict) -> str:
    description = str(row.get("description", "")).strip()
    structured = row.get("structured", [])
    fragments = []
    if isinstance(structured, list):
        for item in structured[:8]:
            cls = str(item.get("class", "unknown"))
            count = item.get("count", "?")
            area = str(item.get("area", "unknown"))
            loc = str(item.get("location", "unknown"))
            fragments.append(f"{cls}: count={count}, area={area}, loc={loc}")

    if fragments:
        details = " | ".join(fragments)
        return f"{description} Structured: {details}."
    return description


@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    text: str


def build_samples(
    jsonl_path: Path,
    image_root: Path,
    default_mask_root: Path,
) -> List[Sample]:
    rows = read_jsonl(jsonl_path)
    image_map = find_images(image_root)

    samples: List[Sample] = []
    for row in rows:
        idx = row.get("id")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue

        image_path = image_map.get(idx)
        if image_path is None:
            continue

        mask_path_row = row.get("mask_path", "")
        mask_path = Path(str(mask_path_row)) if str(mask_path_row).strip() else (default_mask_root / f"Mask_{idx}.png")
        if not mask_path.exists():
            mask_path = default_mask_root / f"Mask_{idx}.png"
        if not mask_path.exists():
            continue

        text = compose_text(row)
        if not text:
            continue

        samples.append(Sample(image_path=image_path, mask_path=mask_path, text=text))

    return samples


class MaskTextDataset(Dataset):
    def __init__(self, samples: List[Sample], image_size: int):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        s = self.samples[idx]
        img4 = preprocess_sample_image(s, self.image_size)
        return img4, s.text


def preprocess_sample_image(sample: Sample, image_size: int) -> torch.Tensor:
    rgb = np.array(Image.open(sample.image_path).convert("RGB"))
    mask = np.array(Image.open(sample.mask_path).convert("L"), dtype=np.uint8)

    rgb = resize_image(rgb, image_size, is_mask=False)
    mask = resize_image(mask, image_size, is_mask=True)
    mask = normalize_mask(mask)

    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    rgb_t = (rgb_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    mask_t = torch.from_numpy(mask).unsqueeze(0).float()
    return torch.cat([rgb_t, mask_t], dim=0)


class BatchCollator:
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __call__(self, batch):
        images, texts = zip(*batch)
        images_t = torch.stack(images, dim=0)
        tokenized = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return images_t, tokenized


class VisionLanguageModel(nn.Module):
    def __init__(self, text_model_name: str, embed_dim: int):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)

        backbone.conv1 = new_conv
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        text_dim = int(self.text_encoder.config.hidden_size)

        self.image_proj = nn.Linear(feat_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def encode_image(self, image_4ch: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(image_4ch)
        emb = self.image_proj(feats)
        return F.normalize(emb, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        emb = self.text_proj(pooled)
        return F.normalize(emb, dim=-1)

    def forward(self, image_4ch: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.encode_image(image_4ch)
        txt = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)
        scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        return img, txt, scale


def clip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits_i2t = scale * img_emb @ txt_emb.t()
    logits_t2i = logits_i2t.t()
    target = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i = F.cross_entropy(logits_i2t, target)
    loss_t = F.cross_entropy(logits_t2i, target)
    return 0.5 * (loss_i + loss_t)


@torch.no_grad()
def retrieval_top1(model: VisionLanguageModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    img_embs = []
    txt_embs = []

    for images, tok in loader:
        images = images.to(device, non_blocking=True)
        input_ids = tok["input_ids"].to(device, non_blocking=True)
        attn = tok["attention_mask"].to(device, non_blocking=True)
        img, txt, _ = model(images, input_ids, attn)
        img_embs.append(img)
        txt_embs.append(txt)

    if not img_embs:
        return 0.0

    imgs = torch.cat(img_embs, dim=0)
    txts = torch.cat(txt_embs, dim=0)
    sims = imgs @ txts.t()
    pred = sims.argmax(dim=1)
    target = torch.arange(sims.size(0), device=sims.device)
    return float((pred == target).float().mean().item())


def split_samples(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    items = samples[:]
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    val = items[:n_val]
    train = items[n_val:]
    return train, val


@torch.no_grad()
def encode_texts_in_chunks(
    model: VisionLanguageModel,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_text_len: int,
    chunk_size: int = 64,
) -> torch.Tensor:
    model.eval()
    embs: List[torch.Tensor] = []
    for start in range(0, len(texts), max(1, int(chunk_size))):
        chunk = texts[start : start + chunk_size]
        tok = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        attn = tok["attention_mask"].to(device)
        txt = model.encode_text(input_ids=input_ids, attention_mask=attn)
        embs.append(txt)
    return torch.cat(embs, dim=0)


@torch.no_grad()
def generate_demo_predictions(
    model: VisionLanguageModel,
    tokenizer,
    demo_samples: List[Sample],
    candidate_texts: List[str],
    device: torch.device,
    image_size: int,
    max_text_len: int,
) -> List[Dict[str, object]]:
    model.eval()
    text_embs = encode_texts_in_chunks(
        model=model,
        tokenizer=tokenizer,
        texts=candidate_texts,
        device=device,
        max_text_len=max_text_len,
    )

    rows: List[Dict[str, object]] = []
    for s in demo_samples:
        img4 = preprocess_sample_image(s, image_size=image_size).unsqueeze(0).to(device)
        img_emb = model.encode_image(img4)
        sims = (img_emb @ text_embs.t()).squeeze(0)
        best_idx = int(torch.argmax(sims).item())
        pred_text = candidate_texts[best_idx]
        conf = float(torch.softmax(sims, dim=0)[best_idx].item())
        rows.append(
            {
                "image_path": str(s.image_path),
                "mask_path": str(s.mask_path),
                "true_text": s.text,
                "predicted_text": pred_text,
                "confidence": round(conf, 6),
                "is_correct": bool(pred_text == s.text),
            }
        )
    return rows


def train() -> None:
    torch.manual_seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    random.seed(RUN_SEED)

    jsonl_path = RUN_STRUCTURED_JSONL
    image_root = RUN_IMAGE_ROOT
    mask_root = RUN_MASK_ROOT
    out_dir = RUN_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(jsonl_path=jsonl_path, image_root=image_root, default_mask_root=mask_root)
    if RUN_MAX_SAMPLES is not None and RUN_MAX_SAMPLES > 0:
        samples = samples[: int(RUN_MAX_SAMPLES)]
    if len(samples) < 8:
        raise RuntimeError(f"Too few valid samples found: {len(samples)}")

    train_samples, val_samples = split_samples(samples, val_ratio=RUN_VAL_RATIO, seed=RUN_SEED)
    if len(train_samples) < 2 or len(val_samples) < 1:
        raise RuntimeError("Dataset split too small. Increase data size or adjust val_ratio.")

    tokenizer = AutoTokenizer.from_pretrained(RUN_TEXT_MODEL)

    train_ds = MaskTextDataset(train_samples, image_size=RUN_IMAGE_SIZE)
    val_ds = MaskTextDataset(val_samples, image_size=RUN_IMAGE_SIZE)

    if os.name == "nt" and RUN_NUM_WORKERS > 0:
        print("Windows detected: forcing RUN_NUM_WORKERS=0 to avoid DataLoader spawn pickling issues.")
        effective_num_workers = 0
    else:
        effective_num_workers = RUN_NUM_WORKERS

    train_loader = DataLoader(
        train_ds,
        batch_size=RUN_BATCH_SIZE,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=True,
        collate_fn=BatchCollator(tokenizer, RUN_MAX_TEXT_LEN),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=RUN_BATCH_SIZE,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=True,
        collate_fn=BatchCollator(tokenizer, RUN_MAX_TEXT_LEN),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionLanguageModel(text_model_name=RUN_TEXT_MODEL, embed_dim=RUN_EMBED_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RUN_LR, weight_decay=RUN_WEIGHT_DECAY)

    print(f"Using device: {device}")
    print(f"Total valid pairs: {len(samples)}")
    print(f"Train pairs: {len(train_samples)}")
    print(f"Val pairs: {len(val_samples)}")

    history = []
    best_val = -1.0

    for epoch in range(1, RUN_EPOCHS + 1):
        model.train()
        running = 0.0
        steps = 0

        for images, tok in train_loader:
            images = images.to(device, non_blocking=True)
            input_ids = tok["input_ids"].to(device, non_blocking=True)
            attn = tok["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            img_emb, txt_emb, scale = model(images, input_ids, attn)
            loss = clip_loss(img_emb, txt_emb, scale)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            steps += 1

        train_loss = running / max(1, steps)
        val_top1 = retrieval_top1(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_retrieval_top1": round(val_top1, 6),
        }
        history.append(row)
        print(json.dumps(row))

        if val_top1 > best_val:
            best_val = val_top1
            best_path = out_dir / "best_vlm_mask_text.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "structured_jsonl": str(RUN_STRUCTURED_JSONL),
                        "image_root": str(RUN_IMAGE_ROOT),
                        "mask_root": str(RUN_MASK_ROOT),
                        "output_dir": str(RUN_OUTPUT_DIR),
                        "text_model": RUN_TEXT_MODEL,
                        "embed_dim": RUN_EMBED_DIM,
                        "image_size": RUN_IMAGE_SIZE,
                        "max_text_len": RUN_MAX_TEXT_LEN,
                        "batch_size": RUN_BATCH_SIZE,
                        "epochs": RUN_EPOCHS,
                        "lr": RUN_LR,
                        "weight_decay": RUN_WEIGHT_DECAY,
                        "val_ratio": RUN_VAL_RATIO,
                        "num_workers": RUN_NUM_WORKERS,
                        "seed": RUN_SEED,
                        "max_samples": RUN_MAX_SAMPLES,
                    },
                    "metrics": row,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")

    hist_path = out_dir / "train_history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"Saved history: {hist_path}")

    final_val_acc = retrieval_top1(model, val_loader, device)
    print(f"Final validation retrieval top1 accuracy: {final_val_acc:.4f}")

    num_demos = max(1, min(int(RUN_NUM_DEMOS), len(val_samples)))
    demo_samples = val_samples[:num_demos]
    candidate_texts = [s.text for s in val_samples]
    demo_rows = generate_demo_predictions(
        model=model,
        tokenizer=tokenizer,
        demo_samples=demo_samples,
        candidate_texts=candidate_texts,
        device=device,
        image_size=RUN_IMAGE_SIZE,
        max_text_len=RUN_MAX_TEXT_LEN,
    )

    demo_correct = sum(1 for r in demo_rows if bool(r["is_correct"]))
    demo_acc = float(demo_correct) / float(max(1, len(demo_rows)))
    print(f"Demo retrieval top1 accuracy ({len(demo_rows)} images): {demo_acc:.4f}")

    demos_dir = out_dir / "demo_outputs"
    demos_dir.mkdir(parents=True, exist_ok=True)
    images_dir = demos_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(demo_rows):
        src = Path(str(row["image_path"]))
        dst = images_dir / f"demo_{i:02d}_{src.name}"
        if src.exists():
            shutil.copy2(src, dst)

    demo_jsonl = demos_dir / "demo_predictions_20.jsonl"
    with demo_jsonl.open("w", encoding="utf-8") as f:
        for row in demo_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "final_val_retrieval_top1": round(final_val_acc, 6),
        "best_val_retrieval_top1": round(best_val, 6),
        "demo_count": len(demo_rows),
        "demo_retrieval_top1": round(demo_acc, 6),
        "output_demo_jsonl": str(demo_jsonl),
        "output_demo_images_dir": str(images_dir),
    }
    metrics_path = out_dir / "final_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    demo_md = demos_dir / "demo_predictions_20.md"
    with demo_md.open("w", encoding="utf-8") as f:
        f.write("# Demo Predictions (20 images)\n\n")
        f.write(f"- Final val retrieval top1: {final_val_acc:.4f}\\n")
        f.write(f"- Demo retrieval top1: {demo_acc:.4f}\\n\n")
        for i, row in enumerate(demo_rows, start=1):
            f.write(f"## Demo {i}\\n")
            f.write(f"- Image: {row['image_path']}\\n")
            f.write(f"- Mask: {row['mask_path']}\\n")
            f.write(f"- Confidence: {row['confidence']}\\n")
            f.write(f"- Correct: {row['is_correct']}\\n")
            f.write(f"- Predicted text: {row['predicted_text']}\\n")
            f.write(f"- Ground truth text: {row['true_text']}\\n\\n")

    print(f"Saved demo JSONL: {demo_jsonl}")
    print(f"Saved demo markdown: {demo_md}")
    print(f"Saved final metrics: {metrics_path}")


if __name__ == "__main__":
    train()
