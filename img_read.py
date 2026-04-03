from pathlib import Path

import numpy as np
from PIL import Image


# Change this to any mask you want to inspect.
DEMO_MASK_PATH = Path("archive/IDD_RESIZED/mask_archive/Mask_0.png")


def read_demo_mask(mask_path: Path) -> None:
	if not mask_path.exists():
		raise FileNotFoundError(f"Mask not found: {mask_path}")

	mask = Image.open(mask_path).convert("L")
	arr = np.array(mask, dtype=np.uint8)

	unique_vals, counts = np.unique(arr, return_counts=True)
	total = arr.size

	print(f"Mask path: {mask_path}")
	print(f"Shape (H, W): {arr.shape}")
	print(f"Dtype: {arr.dtype}")
	print(f"Unique class IDs: {unique_vals.tolist()}")

	print("Class distribution:")
	for cls_id, cls_count in zip(unique_vals.tolist(), counts.tolist()):
		ratio = (float(cls_count) / float(total)) * 100.0
		print(f"  class {cls_id}: {cls_count} pixels ({ratio:.2f}%)")


if __name__ == "__main__":
	read_demo_mask(DEMO_MASK_PATH)
