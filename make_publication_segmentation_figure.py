from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"E:\SkinCancer")

TRAIN_CSV = BASE_DIR / "splits" / "train.csv"
VAL_CSV = BASE_DIR / "splits" / "val.csv"
TEST_CSV = BASE_DIR / "splits" / "test.csv"

ORIGINAL_DIR = BASE_DIR / "ISIC-2017_Training_Data"
GT_MASK_DIR = BASE_DIR / "ISIC-2017_Training_Part1_GroundTruth"
PRED_MASK_ROOT = BASE_DIR / "segmentation" / "predicted_masks"
LESION_ROOT = BASE_DIR / "segmentation" / "lesion_focused"

OUTPUT_DIR = BASE_DIR / "segmentation" / "visualizations"
OUTPUT_FILE = OUTPUT_DIR / "publication_segmentation_figure.png"

RANDOM_SEED = 42
N_EXAMPLES = 4


# ============================================================
# DIRECTORY CHECKS
# ============================================================

for directory in [
    ORIGINAL_DIR,
    GT_MASK_DIR,
    PRED_MASK_ROOT,
    LESION_ROOT,
    TEST_CSV.parent,
]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Required path not found:\n{directory}"
        )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def find_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def find_original_image(image_id):
    return find_existing([
        ORIGINAL_DIR / f"{image_id}.jpg",
        ORIGINAL_DIR / f"{image_id}.jpeg",
        ORIGINAL_DIR / f"{image_id}.png",
    ])


def find_ground_truth_mask(image_id):
    return find_existing([
        GT_MASK_DIR / f"{image_id}_Segmentation.png",
        GT_MASK_DIR / f"{image_id}_segmentation.png",
        GT_MASK_DIR / f"{image_id}.png",
    ])


def find_predicted_mask(image_id):
    # The exact saved suffix can vary, so try the common forms.
    candidates = [
        PRED_MASK_ROOT / "test" / f"{image_id}.png",
        PRED_MASK_ROOT / "test" / f"{image_id}_Predicted.png",
        PRED_MASK_ROOT / "test" / f"{image_id}_predicted.png",
        PRED_MASK_ROOT / "test" / f"{image_id}_Mask.png",
        PRED_MASK_ROOT / "test" / f"{image_id}_mask.png",
    ]

    path = find_existing(candidates)

    if path is not None:
        return path

    # Fallback: search recursively for the exact image ID.
    matches = list(
        (PRED_MASK_ROOT / "test").glob(f"{image_id}*")
    )

    if matches:
        return matches[0]

    return None


def find_lesion_focused_image(image_id):
    candidates = [
        LESION_ROOT / "test" / f"{image_id}_Lesion.jpg",
        LESION_ROOT / "test" / f"{image_id}_Lesion.png",
        LESION_ROOT / "test" / f"{image_id}.jpg",
        LESION_ROOT / "test" / f"{image_id}.png",
    ]

    path = find_existing(candidates)

    if path is not None:
        return path

    matches = list(
        (LESION_ROOT / "test").glob(f"{image_id}*")
    )

    if matches:
        return matches[0]

    return None


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_mask(path):
    mask = Image.open(path).convert("L")

    # Convert any non-zero mask pixels to binary [0, 1].
    mask = np.asarray(mask)
    mask = (mask > 0).astype(np.uint8)

    return Image.fromarray(mask * 255)


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_CSV)

if "image_id" not in test_df.columns:
    raise ValueError(
        f"test.csv must contain image_id. "
        f"Found: {list(test_df.columns)}"
    )

if "melanoma" not in test_df.columns:
    raise ValueError(
        f"test.csv must contain melanoma. "
        f"Found: {list(test_df.columns)}"
    )

test_df["melanoma"] = (
    test_df["melanoma"]
    .astype(float)
    .astype(int)
)

melanoma_df = test_df[
    test_df["melanoma"] == 1
].copy()

non_melanoma_df = test_df[
    test_df["melanoma"] == 0
].copy()


# ============================================================
# SELECT FOUR EXAMPLES
# ============================================================

# Two melanoma and two non-melanoma examples.
# Keeping the seed fixed makes the figure reproducible.

rng = random.Random(RANDOM_SEED)

selected_melanoma = rng.sample(
    melanoma_df["image_id"].tolist(),
    min(2, len(melanoma_df))
)

selected_non_melanoma = rng.sample(
    non_melanoma_df["image_id"].tolist(),
    min(2, len(non_melanoma_df))
)

selected_ids = (
    selected_melanoma
    + selected_non_melanoma
)


# ============================================================
# VERIFY ALL FOUR EXAMPLES
# ============================================================

examples = []

for image_id in selected_ids:

    original_path = find_original_image(image_id)
    gt_path = find_ground_truth_mask(image_id)
    pred_path = find_predicted_mask(image_id)
    lesion_path = find_lesion_focused_image(image_id)

    missing = []

    if original_path is None:
        missing.append("original image")

    if gt_path is None:
        missing.append("ground-truth mask")

    if pred_path is None:
        missing.append("predicted mask")

    if lesion_path is None:
        missing.append("lesion-focused image")

    if missing:
        raise FileNotFoundError(
            f"\nCould not build figure for {image_id}.\n"
            f"Missing: {', '.join(missing)}"
        )

    label = int(
        test_df.loc[
            test_df["image_id"] == image_id,
            "melanoma"
        ].iloc[0]
    )

    examples.append({
        "image_id": image_id,
        "label": label,
        "original": original_path,
        "ground_truth": gt_path,
        "predicted": pred_path,
        "lesion": lesion_path,
    })


# ============================================================
# CREATE PUBLICATION FIGURE
# ============================================================

fig, axes = plt.subplots(
    N_EXAMPLES,
    4,
    figsize=(12, 11)
)

if N_EXAMPLES == 1:
    axes = np.expand_dims(axes, axis=0)


column_titles = [
    "Original Image",
    "Ground-Truth Mask",
    "Predicted Mask",
    "Lesion-Focused Image",
]


for column_index, title in enumerate(column_titles):

    axes[0, column_index].set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=10,
    )


for row_index, example in enumerate(examples):

    original = load_rgb(
        example["original"]
    )

    ground_truth = load_mask(
        example["ground_truth"]
    )

    predicted = load_mask(
        example["predicted"]
    )

    lesion = load_rgb(
        example["lesion"]
    )

    images = [
        original,
        ground_truth,
        predicted,
        lesion,
    ]

    labels = [
        "Original",
        "Ground Truth",
        "Prediction",
        "Lesion Focus",
    ]

    for col_index, image in enumerate(images):

        ax = axes[row_index, col_index]

        if col_index in [1, 2]:
            ax.imshow(
                image,
                cmap="gray"
            )
        else:
            ax.imshow(image)

        ax.axis("off")

    disease_name = (
        "Melanoma"
        if example["label"] == 1
        else "Non-melanoma"
    )

    axes[row_index, 0].text(
        -0.12,
        0.5,
        f"{example['image_id']}\n{disease_name}",
        transform=axes[row_index, 0].transAxes,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
    )


plt.suptitle(
    "Qualitative Evaluation of U-Net Lesion Segmentation",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout(
    rect=[0.10, 0.02, 1.0, 0.97]
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# PRINT RESULT
# ============================================================

print("=" * 70)
print("PUBLICATION SEGMENTATION FIGURE CREATED")
print("=" * 70)
print()
print("Examples used:")

for example in examples:
    disease_name = (
        "Melanoma"
        if example["label"] == 1
        else "Non-melanoma"
    )

    print(
        f"  {example['image_id']} "
        f"({disease_name})"
    )

print()
print("Figure:")
print(OUTPUT_FILE)
print()
print("Layout:")
print("  Original Image | Ground-Truth Mask | "
      "Predicted Mask | Lesion-Focused Image")
print()
print("=" * 70)
