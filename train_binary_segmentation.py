from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# -----------------------------
# Global config defaults
# -----------------------------
DEFAULT_IMAGE_ROOT = Path("archive/IDD_RESIZED/image_archive")
DEFAULT_MASK_ROOT = Path("archive/IDD_RESIZED/mask_archive")
DEFAULT_OUTPUT_DIR = Path("dataset")
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "binary_seg_unet_best.pt"
DEFAULT_PLOT_PATH = DEFAULT_OUTPUT_DIR / "binary_seg_training_curves.png"

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 4 #
DEFAULT_LR = 1e-3 # learning rate , can be decreased for longer training
DEFAULT_IMAGE_SIZE = 128
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_MAX_SAMPLES = 1000 #
DEFAULT_SEED = 42
DEFAULT_FAST_MODE = False
DEFAULT_FAST_EPOCHS = 15 #
DEFAULT_FAST_MAX_SAMPLES = 5000 #
DEFAULT_FAST_IMAGE_SIZE = 256 #
DEFAULT_DEMO_SAMPLES = 10 #
DEFAULT_DEMO_DIR = DEFAULT_OUTPUT_DIR / "binary_seg_demos"

RUN_IMAGE_ROOT = DEFAULT_IMAGE_ROOT
RUN_MASK_ROOT = DEFAULT_MASK_ROOT
RUN_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
RUN_CHECKPOINT = DEFAULT_CHECKPOINT
RUN_PLOT_PATH = DEFAULT_PLOT_PATH
RUN_EPOCHS = DEFAULT_EPOCHS
RUN_BATCH_SIZE = DEFAULT_BATCH_SIZE
RUN_LR = DEFAULT_LR
RUN_IMAGE_SIZE = DEFAULT_IMAGE_SIZE
RUN_VAL_RATIO = DEFAULT_VAL_RATIO
RUN_TEST_RATIO = DEFAULT_TEST_RATIO
RUN_MAX_SAMPLES = DEFAULT_MAX_SAMPLES
RUN_SEED = DEFAULT_SEED
RUN_FAST_MODE = DEFAULT_FAST_MODE
RUN_DEMO_SAMPLES = DEFAULT_DEMO_SAMPLES
RUN_DEMO_DIR = DEFAULT_DEMO_DIR


@dataclass
class PairRecord:
    image_path: Path
    mask_path: Path
    idx: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_index(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def find_pairs(image_root: Path, mask_root: Path) -> List[PairRecord]:
    image_map = {}
    for p in sorted(image_root.glob("Image_*.png")):
        idx = extract_index(p.name)
        if idx is not None:
            image_map[idx] = p

    mask_map = {}
    for p in sorted(mask_root.glob("Mask_*.png")):
        idx = extract_index(p.name)
        if idx is not None:
            mask_map[idx] = p

    common = sorted(set(image_map.keys()) & set(mask_map.keys()))
    return [PairRecord(image_map[i], mask_map[i], i) for i in common]


def split_pairs(
    pairs: List[PairRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]:
    items = pairs[:]
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


class BinarySegDataset(Dataset):
    def __init__(self, records: List[PairRecord], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]

        image = Image.open(row.image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(row.mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)

        image_arr = np.array(image, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image_arr).permute(2, 0, 1)

        # Standard CLIP-like normalization range is not required; simple [0,1] works for U-Net.
        mask_arr = np.array(mask, dtype=np.uint8)
        mask_t = torch.from_numpy(mask_arr).float()
        mask_t = (mask_t > 0).float().unsqueeze(0)

        return {
            "image": image_t,
            "mask": mask_t,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dsc = dice_loss(logits, targets)
    return 0.5 * bce + 0.5 * dsc


@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    total_inter = 0.0
    total_union = 0.0
    total_correct = 0.0
    total_pixels = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = segmentation_loss(logits, masks)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        inter = (preds * masks).sum().item()
        union = ((preds + masks) > 0).float().sum().item()

        correct = (preds == masks).float().sum().item()
        pixels = float(masks.numel())

        total_loss += float(loss.item())
        n += 1
        total_inter += inter
        total_union += union
        total_correct += correct
        total_pixels += pixels

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean_loss = total_loss / n
    iou = total_inter / max(total_union, 1.0)
    dice = (2.0 * total_inter) / max((total_union + total_inter), 1.0)
    pixel_acc = total_correct / max(total_pixels, 1.0)
    return mean_loss, iou, dice, pixel_acc


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, total_epochs: int) -> float:
    model.train()
    running = 0.0
    start = time.time()
    total_steps = max(len(loader), 1)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        loss.backward()
        optimizer.step()

        running += float(loss.item())

        if step == 1 or step % 25 == 0 or step == total_steps:
            elapsed = time.time() - start
            print(
                f"  Epoch {epoch}/{total_epochs} - step {step}/{total_steps} "
                f"loss={loss.item():.4f} elapsed={elapsed:.1f}s"
            )

    return running / max(len(loader), 1)


@torch.no_grad()
def run_random_demos(
    model: nn.Module,
    test_pairs: List[PairRecord],
    image_size: int,
    device: torch.device,
    out_dir: Path,
    demo_samples: int,
) -> None:
    if not test_pairs:
        print("No test pairs available for demo sampling.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    system_rng = random.SystemRandom()
    k = min(max(demo_samples, 1), len(test_pairs))
    sampled = system_rng.sample(test_pairs, k)

    model.eval()
    mean_iou = 0.0
    mean_pixel_acc = 0.0

    print(f"\nRandom demo evaluation ({k} samples)")
    for i, row in enumerate(sampled, start=1):
        image = Image.open(row.image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        mask = Image.open(row.mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)

        image_arr = np.array(image, dtype=np.float32) / 255.0
        mask_arr = (np.array(mask, dtype=np.uint8) > 0).astype(np.float32)

        image_t = torch.from_numpy(image_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0).to(device)

        logits = model(image_t)
        pred = (torch.sigmoid(logits) >= 0.5).float()

        inter = (pred * mask_t).sum().item()
        union = ((pred + mask_t) > 0).float().sum().item()
        iou = inter / max(union, 1.0)

        pixel_acc = (pred == mask_t).float().mean().item()
        mean_iou += iou
        mean_pixel_acc += pixel_acc

        print(
            f"Demo {i:02d} | Image_{row.idx}.png | IoU={iou * 100:.2f}% | "
            f"PixelAcc={pixel_acc * 100:.2f}%"
        )

        if plt is not None:
            pred_np = pred.squeeze().cpu().numpy()
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(image_arr)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(mask_arr, cmap="gray")
            axes[1].set_title("GT Mask")
            axes[1].axis("off")

            axes[2].imshow(pred_np, cmap="gray")
            axes[2].set_title("Pred Mask")
            axes[2].axis("off")

            fig.suptitle(f"Image_{row.idx}.png | IoU={iou * 100:.2f}%")
            fig.tight_layout()
            save_path = out_dir / f"demo_{i:02d}_image_{row.idx}.png"
            fig.savefig(save_path, dpi=140)
            plt.close(fig)

    mean_iou /= float(k)
    mean_pixel_acc /= float(k)
    print("\nRandom demo summary")
    print(f"Mean IoU (demo): {mean_iou * 100:.2f}%")
    print(f"Mean Pixel Accuracy (demo): {mean_pixel_acc * 100:.2f}%")
    print(f"Saved demo directory: {out_dir}")


def main() -> None:
    set_seed(RUN_SEED)

    run_epochs = RUN_EPOCHS
    run_image_size = RUN_IMAGE_SIZE
    run_max_samples = RUN_MAX_SAMPLES
    if RUN_FAST_MODE:
        run_epochs = min(run_epochs, DEFAULT_FAST_EPOCHS)
        run_image_size = min(run_image_size, DEFAULT_FAST_IMAGE_SIZE)
        if run_max_samples is None:
            run_max_samples = DEFAULT_FAST_MAX_SAMPLES
        else:
            run_max_samples = min(int(run_max_samples), DEFAULT_FAST_MAX_SAMPLES)
        print("Fast mode enabled: using reduced settings for quicker progress.")

    if not RUN_IMAGE_ROOT.exists() or not RUN_MASK_ROOT.exists():
        raise FileNotFoundError("Image root or mask root does not exist.")

    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(RUN_IMAGE_ROOT, RUN_MASK_ROOT)
    if not pairs:
        raise RuntimeError("No image-mask pairs found.")

    if run_max_samples is not None:
        pairs = pairs[: min(len(pairs), int(run_max_samples))]

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, RUN_VAL_RATIO, RUN_TEST_RATIO, RUN_SEED)

    print(f"Total pairs: {len(pairs)}")
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    train_ds = BinarySegDataset(train_pairs, run_image_size)
    val_ds = BinarySegDataset(val_pairs, run_image_size)
    test_ds = BinarySegDataset(test_pairs, run_image_size)

    train_loader = DataLoader(train_ds, batch_size=RUN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=RUN_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=RUN_BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = UNetSmall().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RUN_LR)

    train_curve: List[float] = []
    val_curve: List[float] = []
    val_iou_curve: List[float] = []

    best_iou = -1.0

    for epoch in range(1, run_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, run_epochs)
        val_loss, val_iou, val_dice, val_acc = eval_metrics(model, val_loader, device)

        train_curve.append(train_loss)
        val_curve.append(val_loss)
        val_iou_curve.append(val_iou)

        print(
            f"Epoch {epoch}/{run_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | val_dice={val_dice:.4f} | val_acc={val_acc:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_size": run_image_size,
                    "best_val_iou": best_iou,
                },
                RUN_CHECKPOINT,
            )

    checkpoint = torch.load(RUN_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_iou, test_dice, test_acc = eval_metrics(model, test_loader, device)

    print("\nBest checkpoint evaluation on test split")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test IoU: {test_iou * 100:.2f}%")
    print(f"Test Dice: {test_dice * 100:.2f}%")
    print(f"Test Pixel Accuracy: {test_acc * 100:.2f}%")
    print(f"Saved checkpoint: {RUN_CHECKPOINT}")

    if plt is not None:
        plt.figure(figsize=(8, 4))
        x = list(range(1, len(train_curve) + 1))
        plt.plot(x, train_curve, marker="o", label="Train Loss")
        plt.plot(x, val_curve, marker="o", label="Val Loss")
        plt.plot(x, val_iou_curve, marker="o", label="Val IoU")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Binary Segmentation Training Curves")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RUN_PLOT_PATH, dpi=150)
        plt.close()
        print(f"Saved plot: {RUN_PLOT_PATH}")

    run_random_demos(
        model=model,
        test_pairs=test_pairs,
        image_size=run_image_size,
        device=device,
        out_dir=RUN_DEMO_DIR,
        demo_samples=RUN_DEMO_SAMPLES,
    )


if __name__ == "__main__":
    main()
