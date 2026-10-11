from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"E:\SkinCancer")

METADATA_FILE = BASE_DIR / "dataset_metadata.csv"

SPLIT_DIR = BASE_DIR / "splits"

TRAIN_FILE = SPLIT_DIR / "train.csv"
VAL_FILE = SPLIT_DIR / "val.csv"
TEST_FILE = SPLIT_DIR / "test.csv"

RANDOM_SEED = 42


# ============================================================
# CHECK INPUT FILE
# ============================================================

print("=" * 70)
print("CHECKING INPUT")
print("=" * 70)

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE}"
    )

print(f"[OK] Metadata file found:")
print(METADATA_FILE)
print()


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("LOADING METADATA")
print("=" * 70)

df = pd.read_csv(METADATA_FILE)

print(f"Total records loaded: {len(df)}")
print()

required_columns = [
    "image_id",
    "image_path",
    "mask_path",
    "melanoma",
    "seborrheic_keratosis",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("[OK] All required columns are present.")
print()


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print("=" * 70)
print("VALIDATING METADATA")
print("=" * 70)

# Check duplicate image IDs
duplicate_ids = df[
    df["image_id"].duplicated(keep=False)
]["image_id"].unique()

if len(duplicate_ids) > 0:
    raise ValueError(
        f"Duplicate image IDs found: {duplicate_ids}"
    )

print("[OK] No duplicate image IDs.")

# Check melanoma labels
valid_labels = {0, 1}

actual_labels = set(
    df["melanoma"].astype(int).unique()
)

if not actual_labels.issubset(valid_labels):
    raise ValueError(
        f"Unexpected melanoma labels found: {actual_labels}"
    )

print("[OK] Melanoma labels are valid.")

# Check expected dataset size
if len(df) != 2000:
    raise ValueError(
        f"Expected 2000 records, but found {len(df)}."
    )

print("[OK] Dataset contains exactly 2000 records.")
print()


# ============================================================
# CHECK IF SPLITS ALREADY EXIST
# ============================================================

existing_files = [
    file
    for file in [TRAIN_FILE, VAL_FILE, TEST_FILE]
    if file.exists()
]

if existing_files:

    print("=" * 70)
    print("WARNING")
    print("=" * 70)

    print("One or more split files already exist:")

    for file in existing_files:
        print(f"  {file}")

    print()
    print(
        "The script will NOT overwrite existing splits."
    )
    print(
        "This protects our fixed experimental split."
    )
    print()

    raise FileExistsError(
        "Split files already exist. "
        "Delete them manually only if you intentionally "
        "want to create a new split."
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 70)
print("CREATING STRATIFIED SPLIT")
print("=" * 70)

print(f"Random seed: {RANDOM_SEED}")
print()


# ============================================================
# FIRST SPLIT
#
# 2000
#   ├── 70% train = 1400
#   └── 30% temporary = 600
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df["melanoma"],
)


# ============================================================
# SECOND SPLIT
#
# Temporary 600:
#   ├── 50% validation = 300
#   └── 50% test       = 300
#
# Therefore:
#   Train = 1400
#   Val   = 300
#   Test  = 300
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["melanoma"],
)


# ============================================================
# RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# VERIFY SPLIT SIZES
# ============================================================

print("Split sizes:")
print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Testing    : {len(test_df)}")
print()

expected_sizes = {
    "train": 1400,
    "val": 300,
    "test": 300,
}

if len(train_df) != expected_sizes["train"]:
    raise ValueError(
        f"Training set has {len(train_df)} samples."
    )

if len(val_df) != expected_sizes["val"]:
    raise ValueError(
        f"Validation set has {len(val_df)} samples."
    )

if len(test_df) != expected_sizes["test"]:
    raise ValueError(
        f"Test set has {len(test_df)} samples."
    )

print("[OK] Split sizes are correct.")
print()


# ============================================================
# CHECK FOR OVERLAP
# ============================================================

print("=" * 70)
print("CHECKING FOR DATA LEAKAGE")
print("=" * 70)

train_ids = set(train_df["image_id"])
val_ids = set(val_df["image_id"])
test_ids = set(test_df["image_id"])

train_val_overlap = train_ids.intersection(val_ids)
train_test_overlap = train_ids.intersection(test_ids)
val_test_overlap = val_ids.intersection(test_ids)

if train_val_overlap:
    raise ValueError(
        f"Train/validation overlap detected: "
        f"{train_val_overlap}"
    )

if train_test_overlap:
    raise ValueError(
        f"Train/test overlap detected: "
        f"{train_test_overlap}"
    )

if val_test_overlap:
    raise ValueError(
        f"Validation/test overlap detected: "
        f"{val_test_overlap}"
    )

print("[OK] No train/validation overlap.")
print("[OK] No train/test overlap.")
print("[OK] No validation/test overlap.")
print()


# ============================================================
# CHECK ALL IMAGE IDS ARE ACCOUNTED FOR
# ============================================================

all_original_ids = set(df["image_id"])

all_split_ids = (
    train_ids
    | val_ids
    | test_ids
)

if all_original_ids != all_split_ids:
    missing_ids = all_original_ids - all_split_ids
    extra_ids = all_split_ids - all_original_ids

    raise ValueError(
        f"Split mismatch detected.\n"
        f"Missing IDs: {missing_ids}\n"
        f"Extra IDs: {extra_ids}"
    )

print("[OK] Every original image appears exactly once.")
print()


# ============================================================
# CLASS DISTRIBUTION FUNCTION
# ============================================================

def print_class_distribution(
    name: str,
    data: pd.DataFrame
) -> None:

    melanoma_count = int(
        (data["melanoma"] == 1).sum()
    )

    non_melanoma_count = int(
        (data["melanoma"] == 0).sum()
    )

    total = len(data)

    melanoma_percentage = (
        melanoma_count / total
    ) * 100

    non_melanoma_percentage = (
        non_melanoma_count / total
    ) * 100

    print(f"{name}")
    print("-" * 70)
    print(
        f"Total samples     : {total}"
    )

    print(
        f"Melanoma          : "
        f"{melanoma_count} "
        f"({melanoma_percentage:.2f}%)"
    )

    print(
        f"Non-melanoma      : "
        f"{non_melanoma_count} "
        f"({non_melanoma_percentage:.2f}%)"
    )

    print()


# ============================================================
# PRINT CLASS DISTRIBUTIONS
# ============================================================

print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print_class_distribution(
    "FULL DATASET",
    df
)

print_class_distribution(
    "TRAINING SET",
    train_df
)

print_class_distribution(
    "VALIDATION SET",
    val_df
)

print_class_distribution(
    "TEST SET",
    test_df
)


# ============================================================
# SAVE SPLITS
# ============================================================

print("=" * 70)
print("SAVING FIXED SPLITS")
print("=" * 70)

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

val_df.to_csv(
    VAL_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

print(f"[OK] Training split saved:")
print(TRAIN_FILE)

print()

print(f"[OK] Validation split saved:")
print(VAL_FILE)

print()

print(f"[OK] Test split saved:")
print(TEST_FILE)

print()


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

saved_train = pd.read_csv(TRAIN_FILE)
saved_val = pd.read_csv(VAL_FILE)
saved_test = pd.read_csv(TEST_FILE)

if len(saved_train) != 1400:
    raise ValueError("Saved training split is incorrect.")

if len(saved_val) != 300:
    raise ValueError("Saved validation split is incorrect.")

if len(saved_test) != 300:
    raise ValueError("Saved test split is incorrect.")

saved_ids = (
    set(saved_train["image_id"])
    | set(saved_val["image_id"])
    | set(saved_test["image_id"])
)

if saved_ids != all_original_ids:
    raise ValueError(
        "Saved splits do not contain all original images."
    )

print("[OK] Saved training split verified.")
print("[OK] Saved validation split verified.")
print("[OK] Saved test split verified.")
print("[OK] All 2,000 images accounted for.")
print()

print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "The dataset split is now fixed and reproducible."
)

print()
print(
    "IMPORTANT: Do not regenerate these splits "
    "unless there is a genuine data problem."
)