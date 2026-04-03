from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_CLASS_MAP: Dict[int, str] = {
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

# Global runtime defaults
DEFAULT_IMAGE_DIR = Path("archive/IDD_RESIZED/image_archive")
DEFAULT_MASK_DIR = Path("archive/IDD_RESIZED/mask_archive")
DEFAULT_PAIRS_OUT = Path("dataset/mask_caption_pairs.jsonl")
DEFAULT_MASK_LANGUAGE_JSONL = Path("dataset/mask_natural_language.jsonl")

DEFAULT_MAX_SAMPLES = 3000
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_DEMO_SAMPLES = 50
DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_index(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def find_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path, int]]:
    images = {}
    for p in image_dir.glob("Image_*.png"):
        idx = extract_index(p.name)
        if idx is not None:
            images[idx] = p

    masks = {}
    for p in mask_dir.glob("Mask_*.png"):
        idx = extract_index(p.name)
        if idx is not None:
            masks[idx] = p

    paired_idx = sorted(set(images.keys()) & set(masks.keys()))
    return [(images[i], masks[i], i) for i in paired_idx]


def load_mask_language_captions(jsonl_path: Path) -> Dict[int, str]:
    """Load pre-generated geometric captions from mask_natural_language.jsonl"""
    captions = {}
    if not jsonl_path.exists():
        return captions
    
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                img_id = row.get("id")
                caption = row.get("primary_caption")
                if img_id is not None and caption:
                    captions[img_id] = caption
    except Exception as e:
        print(f"Warning: Could not load mask_language captions from {jsonl_path}: {e}")
    
    return captions


def extract_objects_from_mask(mask_path: Path, class_map: Dict[int, str]) -> List[Tuple[str, float]]:
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask, dtype=np.uint8)
    total = arr.size
    counts = np.bincount(arr.flatten(), minlength=max(class_map.keys()) + 1)

    objects: List[Tuple[str, float]] = []
    for cid, count in enumerate(counts):
        name = class_map.get(int(cid))
        if not name or name == "background":
            continue
        ratio = float(count) / float(total)
        if ratio < 0.01:
            continue
        objects.append((name, ratio))

    objects.sort(key=lambda x: x[1], reverse=True)
    return objects


def generate_caption(objects: List[Tuple[str, float]]) -> str:
    if not objects:
        return "An urban street view with unclear semantic objects."

    # Common base surfaces should describe context, not be treated as main objects.
    base_surfaces = {"road", "sidewalk", "sky", "wall", "terrain"}
    dynamic_priority = {
        "person",
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    }

    named = [name for name, _ in objects]
    dynamic = [name for name, _ in objects if name in dynamic_priority]
    structure = [name for name, _ in objects if name not in dynamic_priority and name not in base_surfaces]

    if dynamic:
        top_dynamic = dynamic[:3]
        if len(top_dynamic) == 1:
            return f"An urban street scene featuring a {top_dynamic[0]}."
        if len(top_dynamic) == 2:
            return f"An urban street scene featuring {top_dynamic[0]} and {top_dynamic[1]}."
        return f"An urban street scene featuring {top_dynamic[0]}, {top_dynamic[1]}, and {top_dynamic[2]}."

    if structure:
        top_structure = structure[:2]
        if len(top_structure) == 1:
            return f"A road scene dominated by {top_structure[0]} with open street context."
        return f"A road scene with prominent {top_structure[0]} and {top_structure[1]}."

    # Fallback for surface-only masks
    if "road" in named and "sidewalk" in named:
        return "A road and sidewalk layout in an urban environment."
    if "road" in named:
        return "A drivable road segment in an urban street environment."

    return "A structured urban scene with segmented street regions."


def build_pairs(
    image_dir: Path,
    mask_dir: Path,
    class_map: Dict[int, str],
    max_samples: int,
    seed: int,
    mask_language_captions: Dict[int, str] | None = None,
) -> List[Dict[str, object]]:
    if mask_language_captions is None:
        mask_language_captions = {}
        
    pairs = find_pairs(image_dir, mask_dir)
    if not pairs:
        return []

    random.Random(seed).shuffle(pairs)
    pairs = pairs[: min(len(pairs), max_samples)]

    records: List[Dict[str, object]] = []
    for image_path, mask_path, idx in pairs:
        # Prefer geometric captions from mask_natural_language.jsonl
        if idx in mask_language_captions:
            caption = mask_language_captions[idx]
        else:
            # Fallback to generated caption for multiclass masks
            objects = extract_objects_from_mask(mask_path, class_map)
            caption = generate_caption(objects)
        
        records.append(
            {
                "id": idx,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "caption": caption,
            }
        )
    return records


def save_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SimpleTokenizer:
    def __init__(self, captions: List[str], min_freq: int = 1):
        counts: Dict[str, int] = {}
        for c in captions:
            for tok in self._tokenize(c):
                counts[tok] = counts.get(tok, 0) + 1

        self.pad = "<pad>"
        self.unk = "<unk>"
        vocab = [self.pad, self.unk]
        for tok, cnt in sorted(counts.items()):
            if cnt >= min_freq:
                vocab.append(tok)

        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = vocab

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        clean = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return [t for t in clean.split() if t]

    def encode(self, text: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        toks = self._tokenize(text)
        ids = [self.stoi.get(t, self.stoi[self.unk]) for t in toks][:max_len]
        attn = [1] * len(ids)

        while len(ids) < max_len:
            ids.append(self.stoi[self.pad])
            attn.append(0)

        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


class MaskCaptionDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, object]],
        tokenizer: SimpleTokenizer,
        class_map: Dict[int, str],
        image_size: int = 224,
        max_len: int = 24,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.class_map = class_map
        self.image_size = image_size
        self.max_len = max_len
        self.num_classes = max(class_map.keys()) + 1

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def _mask_hist(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        arr = np.array(mask, dtype=np.int64)
        hist = np.bincount(arr.flatten(), minlength=self.num_classes).astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        return torch.from_numpy(hist)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.records[idx]
        image = self._load_image(Path(str(row["image_path"])))
        mask_hist = self._mask_hist(Path(str(row["mask_path"])))
        input_ids, attention_mask = self.tokenizer.encode(str(row["caption"]), self.max_len)

        return {
            "image": image,
            "mask_hist": mask_hist,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.fc(h)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, heads: int = 4, layers: int = 2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(256, dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        pad_mask = attention_mask == 0
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled


class MaskAwareDualEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, proj_dim: int = 256):
        super().__init__()
        self.image_encoder = ImageEncoder(out_dim=proj_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, dim=proj_dim)
        self.mask_proj = nn.Linear(num_classes, proj_dim)

        self.image_proj = nn.Linear(proj_dim * 2, proj_dim)
        self.text_proj = nn.Linear(proj_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, image: torch.Tensor, mask_hist: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        mask_feat = self.mask_proj(mask_hist)
        fused = torch.cat([img_feat, mask_feat], dim=-1)
        z = self.image_proj(fused)
        return F.normalize(z, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        txt_feat = self.text_encoder(input_ids, attention_mask)
        z = self.text_proj(txt_feat)
        return F.normalize(z, dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        mask_hist: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_img = self.encode_image(image, mask_hist)
        z_txt = self.encode_text(input_ids, attention_mask)
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * (z_img @ z_txt.t())
        return logits, z_img, z_txt


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i + loss_t)


@dataclass
class TrainConfig:
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LEARNING_RATE


def compute_retrieval_metrics(
    model: MaskAwareDualEncoder,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    image_embeddings: List[torch.Tensor] = []
    text_embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask_hist = batch["mask_hist"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            z_img = model.encode_image(image, mask_hist)
            z_txt = model.encode_text(input_ids, attention_mask)
            image_embeddings.append(z_img)
            text_embeddings.append(z_txt)

    if not image_embeddings:
        return 0.0, 0.0, 0.0

    z_img_all = torch.cat(image_embeddings, dim=0)
    z_txt_all = torch.cat(text_embeddings, dim=0)
    sim = z_img_all @ z_txt_all.t()

    n = sim.size(0)
    targets = torch.arange(n, device=sim.device)

    pred_top1 = sim.argmax(dim=1)
    top1 = (pred_top1 == targets).float().mean().item()

    k = min(5, n)
    topk_idx = sim.topk(k=k, dim=1).indices
    top5 = (topk_idx == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    text_pred_top1 = sim.t().argmax(dim=1)
    text_top1 = (text_pred_top1 == targets).float().mean().item()

    return top1, top5, text_top1


def train(
    model: MaskAwareDualEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        running = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            mask_hist = batch["mask_hist"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            opt.zero_grad(set_to_none=True)
            logits, _, _ = model(image, mask_hist, input_ids, attention_mask)
            loss = contrastive_loss(logits)
            loss.backward()
            opt.step()

            running += float(loss.item())

        avg = running / max(len(train_loader), 1)
        val_top1, val_top5, val_text_top1 = compute_retrieval_metrics(model, val_loader, device)
        model.train()
        print(
            f"Epoch {epoch}/{cfg.epochs} - loss: {avg:.4f} | "
            f"val_img_top1: {val_top1 * 100:.2f}% | "
            f"val_img_top5: {val_top5 * 100:.2f}% | "
            f"val_txt_top1: {val_text_top1 * 100:.2f}%"
        )


def split_records(
    records: List[Dict[str, object]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Split records into train, test, val sets.
    
    Returns: (train_records, test_records, val_records)
    """
    items = records[:]
    random.Random(seed).shuffle(items)
    if len(items) < 3:
        return items, items, items

    val_n = max(1, int(len(items) * val_ratio))
    test_n = max(1, int(len(items) * test_ratio))
    train_n = max(1, len(items) - val_n - test_n)

    train_records = items[:train_n]
    test_records = items[train_n : train_n + test_n]
    val_records = items[train_n + test_n : train_n + test_n + val_n]

    if not test_records:
        test_records = train_records[:]
    if not val_records:
        val_records = train_records[:]
    return train_records, test_records, val_records


def retrieval_demo(
    model: MaskAwareDualEncoder,
    records: List[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    class_map: Dict[int, str],
    device: torch.device,
    demo_samples: int,
) -> None:
    model.eval()
    ds = MaskCaptionDataset(records, tokenizer, class_map)

    with torch.no_grad():
        use_count = min(max(demo_samples, 1), len(ds))
        batch = [ds[i] for i in range(use_count)]
        images = torch.stack([x["image"] for x in batch]).to(device)
        masks = torch.stack([x["mask_hist"] for x in batch]).to(device)
        input_ids = torch.stack([x["input_ids"] for x in batch]).to(device)
        attn = torch.stack([x["attention_mask"] for x in batch]).to(device)

        z_img = model.encode_image(images, masks)
        z_txt = model.encode_text(input_ids, attn)
        sim = z_img @ z_txt.t()

        print("\nInference demo")
        for q in range(use_count):
            best = int(sim[q].argmax().item())
            print(f"query image: {Path(str(records[q]['image_path'])).name}")
            print(f"predicted caption: {records[best]['caption']}")
            print(f"ground truth caption: {records[q]['caption']}")
            print("-" * 50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask-aware vision-language dual encoder training")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    p.add_argument("--pairs-out", type=Path, default=DEFAULT_PAIRS_OUT)
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    p.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    p.add_argument("--demo-samples", type=int, default=DEFAULT_DEMO_SAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.image_dir.exists() or not args.mask_dir.exists():
        raise FileNotFoundError("image-dir or mask-dir does not exist")

    print("Loading geometric captions from mask_natural_language.jsonl...")
    mask_language_captions = load_mask_language_captions(DEFAULT_MASK_LANGUAGE_JSONL)
    print(f"Loaded {len(mask_language_captions)} mask-generated captions")

    print("\nBuilding image-text pairs from masks...")
    records = build_pairs(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        class_map=DEFAULT_CLASS_MAP,
        max_samples=args.max_samples,
        seed=args.seed,
        mask_language_captions=mask_language_captions,
    )

    if not records:
        raise RuntimeError("No image-mask pairs were found")

    save_jsonl(args.pairs_out, records)
    print(f"Saved {len(records)} pairs to {args.pairs_out}")
    print("\nSample captions from pairs (first 10):")
    for i, row in enumerate(records[:10], start=1):
        image_name = Path(str(row["image_path"])).name
        print(f"{i:05d}. {image_name} -> {row['caption'][:60]}...")

    train_records, test_records, val_records = split_records(
        records, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"\nData split (80:10:10):")
    print(f"  Train pairs: {len(train_records)}")
    print(f"  Test pairs:  {len(test_records)}")
    print(f"  Val pairs:   {len(val_records)}")

    captions = [str(r["caption"]) for r in train_records]
    tokenizer = SimpleTokenizer(captions)

    train_dataset = MaskCaptionDataset(train_records, tokenizer, DEFAULT_CLASS_MAP)
    test_dataset = MaskCaptionDataset(test_records, tokenizer, DEFAULT_CLASS_MAP)
    val_dataset = MaskCaptionDataset(val_records, tokenizer, DEFAULT_CLASS_MAP)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = MaskAwareDualEncoder(
        vocab_size=len(tokenizer.itos),
        num_classes=max(DEFAULT_CLASS_MAP.keys()) + 1,
        proj_dim=256,
    ).to(device)

    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    train(model, train_loader, val_loader, train_cfg, device)

    val_top1, val_top5, val_text_top1 = compute_retrieval_metrics(model, val_loader, device)
    print("\nValidation accuracy")
    print(f"  Image->Text Top-1: {val_top1 * 100:.2f}%")
    print(f"  Image->Text Top-5: {val_top5 * 100:.2f}%")
    print(f"  Text->Image Top-1: {val_text_top1 * 100:.2f}%")

    test_top1, test_top5, test_text_top1 = compute_retrieval_metrics(model, test_loader, device)
    print("\nTest accuracy")
    print(f"  Image->Text Top-1: {test_top1 * 100:.2f}%")
    print(f"  Image->Text Top-5: {test_top5 * 100:.2f}%")
    print(f"  Text->Image Top-1: {test_text_top1 * 100:.2f}%")

    final_accuracy = (test_top1 + test_top5 + test_text_top1) / 3.0
    print(f"\nFinal Model Accuracy (mean test metrics): {final_accuracy * 100:.2f}%")

    retrieval_demo(model, val_records, tokenizer, DEFAULT_CLASS_MAP, device, demo_samples=args.demo_samples)
    print("Done.")


if __name__ == "__main__":
    main()
