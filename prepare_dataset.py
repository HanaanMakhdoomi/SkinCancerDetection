from pathlib import Path
import csv

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"E:\SkinCancer")

IMAGE_DIR = BASE_DIR / "ISIC-2017_Training_Data"
MASK_DIR = BASE_DIR / "ISIC-2017_Training_Part1_GroundTruth"
CSV_FILE = BASE_DIR / "ISIC-2017_Training_Part3_GroundTruth.csv"

OUTPUT_FILE = BASE_DIR / "dataset_metadata.csv"


# ============================================================
# CHECK THAT ALL REQUIRED PATHS EXIST
# ============================================================

print("=" * 60)
print("CHECKING DATASET PATHS")
print("=" * 60)

required_paths = {
    "Image directory": IMAGE_DIR,
    "Mask directory": MASK_DIR,
    "Ground-truth CSV": CSV_FILE,
}

for name, path in required_paths.items():
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )
    print(f"[OK] {name}: {path}")

print()


# ============================================================
# READ DIAGNOSIS CSV
# ============================================================

print("=" * 60)
print("READING DIAGNOSIS CSV")
print("=" * 60)

labels = {}

with open(
    CSV_FILE,
    "r",
    newline="",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    print("CSV columns:")
    print(reader.fieldnames)
    print()

    required_columns = {
        "image_id",
        "melanoma",
        "seborrheic_keratosis",
    }

    if reader.fieldnames is None:
        raise ValueError("CSV does not contain a header row.")

    missing_columns = required_columns - set(reader.fieldnames)

    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {missing_columns}"
        )

    for row_number, row in enumerate(reader, start=2):

        image_id = row["image_id"].strip()

        if not image_id:
            raise ValueError(
                f"Empty image_id found at CSV row {row_number}"
            )

        # ----------------------------------------------------
        # Melanoma label
        # CSV contains values such as 0.0 and 1.0
        # ----------------------------------------------------

        melanoma_value = row["melanoma"].strip()

        try:
            melanoma_float = float(melanoma_value)
        except ValueError:
            raise ValueError(
                f"Invalid melanoma value for {image_id}: "
                f"{melanoma_value}"
            )

        if melanoma_float not in {0.0, 1.0}:
            raise ValueError(
                f"Unexpected melanoma value for {image_id}: "
                f"{melanoma_value}"
            )

        melanoma_label = int(melanoma_float)

        # ----------------------------------------------------
        # Seborrheic keratosis label
        # ----------------------------------------------------

        sk_value = row["seborrheic_keratosis"].strip()

        try:
            sk_float = float(sk_value)
        except ValueError:
            raise ValueError(
                f"Invalid seborrheic_keratosis value for "
                f"{image_id}: {sk_value}"
            )

        if sk_float not in {0.0, 1.0}:
            raise ValueError(
                f"Unexpected seborrheic_keratosis value for "
                f"{image_id}: {sk_value}"
            )

        sk_label = int(sk_float)

        # ----------------------------------------------------
        # Store labels
        # ----------------------------------------------------

        labels[image_id] = {
            "melanoma": melanoma_label,
            "seborrheic_keratosis": sk_label,
        }


print(f"Diagnosis records found: {len(labels)}")
print()


# ============================================================
# FIND ALL JPEG IMAGES
# ============================================================

print("=" * 60)
print("SEARCHING FOR IMAGES")
print("=" * 60)

image_files = sorted(IMAGE_DIR.glob("*.jpg"))

print(f"JPEG images found: {len(image_files)}")
print()


# ============================================================
# CHECK IMAGE IDS FOR DUPLICATES
# ============================================================

image_ids = [image_path.stem for image_path in image_files]

duplicate_image_ids = {
    image_id
    for image_id in image_ids
    if image_ids.count(image_id) > 1
}

if duplicate_image_ids:
    raise ValueError(
        f"Duplicate image IDs found: "
        f"{sorted(duplicate_image_ids)}"
    )

print("[OK] No duplicate image IDs found.")
print()


# ============================================================
# MATCH IMAGE + MASK + LABEL
# ============================================================

print("=" * 60)
print("VERIFYING IMAGE / MASK / LABEL MATCHING")
print("=" * 60)

metadata = []

missing_masks = []
missing_labels = []

for image_path in image_files:

    image_id = image_path.stem

    # Expected segmentation mask filename
    mask_path = MASK_DIR / f"{image_id}_Segmentation.png"

    # --------------------------------------------------------
    # Check segmentation mask
    # --------------------------------------------------------

    if not mask_path.exists():
        missing_masks.append(image_id)
        continue

    # --------------------------------------------------------
    # Check diagnosis label
    # --------------------------------------------------------

    if image_id not in labels:
        missing_labels.append(image_id)
        continue

    # --------------------------------------------------------
    # Add valid record
    # --------------------------------------------------------

    metadata.append({
        "image_id": image_id,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "melanoma": labels[image_id]["melanoma"],
        "seborrheic_keratosis": labels[image_id][
            "seborrheic_keratosis"
        ],
    })


# ============================================================
# FIND LABELS THAT DON'T HAVE AN IMAGE
# ============================================================

image_id_set = set(image_ids)

labels_without_images = sorted(
    image_id
    for image_id in labels
    if image_id not in image_id_set
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

print(f"Total JPEG images       : {len(image_files)}")
print(f"Diagnosis records       : {len(labels)}")
print(f"Valid image-mask-labels : {len(metadata)}")
print(f"Missing masks           : {len(missing_masks)}")
print(f"Missing labels          : {len(missing_labels)}")
print(
    f"Labels without images   : "
    f"{len(labels_without_images)}"
)

print()


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

melanoma_count = sum(
    1
    for item in metadata
    if item["melanoma"] == 1
)

non_melanoma_count = sum(
    1
    for item in metadata
    if item["melanoma"] == 0
)

seborrheic_keratosis_count = sum(
    1
    for item in metadata
    if item["seborrheic_keratosis"] == 1
)

print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(f"Melanoma              : {melanoma_count}")
print(f"Non-melanoma          : {non_melanoma_count}")
print(
    f"Seborrheic keratosis  : "
    f"{seborrheic_keratosis_count}"
)

print()


# ============================================================
# SHOW FIRST 10 VALID RECORDS
# ============================================================

print("=" * 60)
print("FIRST 10 VALID SAMPLES")
print("=" * 60)

for item in metadata[:10]:

    print(
        f"{item['image_id']} | "
        f"Melanoma={item['melanoma']} | "
        f"SK={item['seborrheic_keratosis']}"
    )

print()


# ============================================================
# SAVE METADATA CSV
# ============================================================

print("=" * 60)
print("SAVING METADATA")
print("=" * 60)

fieldnames = [
    "image_id",
    "image_path",
    "mask_path",
    "melanoma",
    "seborrheic_keratosis",
]

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(metadata)


print(f"Metadata saved to:")
print(OUTPUT_FILE)
print()


# ============================================================
# REPORT ANY PROBLEMS
# ============================================================

if missing_masks:

    print("=" * 60)
    print("WARNING: MISSING MASKS")
    print("=" * 60)

    for image_id in missing_masks[:20]:
        print(image_id)

    if len(missing_masks) > 20:
        print(
            f"... and "
            f"{len(missing_masks) - 20} more"
        )

    print()


if missing_labels:

    print("=" * 60)
    print("WARNING: MISSING LABELS")
    print("=" * 60)

    for image_id in missing_labels[:20]:
        print(image_id)

    if len(missing_labels) > 20:
        print(
            f"... and "
            f"{len(missing_labels) - 20} more"
        )

    print()


if labels_without_images:

    print("=" * 60)
    print("WARNING: LABELS WITHOUT IMAGES")
    print("=" * 60)

    for image_id in labels_without_images[:20]:
        print(image_id)

    if len(labels_without_images) > 20:
        print(
            f"... and "
            f"{len(labels_without_images) - 20} more"
        )

    print()


# ============================================================
# FINAL STATUS
# ============================================================

print("=" * 60)

if (
    len(image_files) == 2000
    and len(labels) == 2000
    and len(metadata) == 2000
    and len(missing_masks) == 0
    and len(missing_labels) == 0
    and len(labels_without_images) == 0
):
    print("SUCCESS!")
    print(
        "All 2,000 images have matching "
        "segmentation masks and diagnosis labels."
    )
else:
    print("DATASET VERIFICATION NOT COMPLETE")
    print(
        "Please review the numbers above "
        "before continuing."
    )

print("=" * 60)
print()
print("Dataset preparation finished.")