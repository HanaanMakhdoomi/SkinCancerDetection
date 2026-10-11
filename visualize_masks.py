from pathlib import Path
import random

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"E:\SkinCancer")

TRAIN_FILE = BASE_DIR / "splits" / "train.csv"

NUM_SAMPLES = 8
RANDOM_SEED = 42


# ============================================================
# LOAD TRAINING METADATA
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training split not found:\n{TRAIN_FILE}"
    )

df = pd.read_csv(TRAIN_FILE)

print(f"Training samples available: {len(df)}")


# ============================================================
# SELECT RANDOM SAMPLES
# ============================================================

random.seed(RANDOM_SEED)

sample_count = min(NUM_SAMPLES, len(df))

samples = random.sample(
    list(df.to_dict("records")),
    sample_count
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    sample_count,
    2,
    figsize=(10, 4 * sample_count)
)

# Handle case of only one sample
if sample_count == 1:
    axes = [axes]


# ============================================================
# DISPLAY IMAGES AND MASKS
# ============================================================

for row_index, sample in enumerate(samples):

    image_id = sample["image_id"]

    image_path = Path(sample["image_path"])
    mask_path = Path(sample["mask_path"])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found:\n{mask_path}"
        )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Load mask
    mask = Image.open(mask_path).convert("L")

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    axes[row_index][0].imshow(image)
    axes[row_index][0].set_title(
        f"{image_id} — Original Image"
    )
    axes[row_index][0].axis("off")

    # --------------------------------------------------------
    # Ground-truth mask
    # --------------------------------------------------------

    axes[row_index][1].imshow(
        mask,
        cmap="gray"
    )

    axes[row_index][1].set_title(
        f"{image_id} — Ground Truth Mask"
    )

    axes[row_index][1].axis("off")


# ============================================================
# SAVE FIGURE
# ============================================================

OUTPUT_DIR = BASE_DIR / "visual_checks"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "image_mask_pairs.png"

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print()
print("=" * 60)
print("VISUAL VERIFICATION COMPLETE")
print("=" * 60)

print(f"Samples displayed: {sample_count}")
print(f"Figure saved to:")
print(OUTPUT_FILE)