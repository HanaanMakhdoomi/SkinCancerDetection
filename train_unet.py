from pathlib import Path
import random
import math
import json
import time
import os

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"E:\SkinCancer")

TRAIN_CSV = BASE_DIR / "splits" / "train.csv"
VAL_CSV = BASE_DIR / "splits" / "val.csv"
TEST_CSV = BASE_DIR / "splits" / "test.csv"

OUTPUT_DIR = BASE_DIR / "segmentation"

MODEL_DIR = OUTPUT_DIR / "models"
MASK_DIR = OUTPUT_DIR / "predicted_masks"
LESION_DIR = OUTPUT_DIR / "lesion_focused"
RESULTS_DIR = OUTPUT_DIR / "results"
VISUAL_DIR = OUTPUT_DIR / "visualizations"

MODEL_FILE = MODEL_DIR / "best_unet.pth"
LAST_CHECKPOINT_FILE = MODEL_DIR / "last_unet_checkpoint.pth"

HISTORY_FILE = RESULTS_DIR / "training_history.csv"
TEST_RESULTS_FILE = RESULTS_DIR / "test_metrics.json"
CONFIG_FILE = RESULTS_DIR / "configuration.json"


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

IMAGE_SIZE = 256

BATCH_SIZE = 4

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 6

RANDOM_SEED = 42

# Reduced from 32 to 16 for CPU training.
BASE_CHANNELS = 16

# Print progress every N batches.
PRINT_EVERY = 25

# Padding around the lesion for classification images.
CROP_PADDING_RATIO = 0.10

NUM_VISUAL_SAMPLES = 8


# ============================================================
# CPU CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

# Limit CPU threads to a reasonable value on Windows.
CPU_COUNT = os.cpu_count() or 4

THREADS = max(1, min(CPU_COUNT - 1, 8))

torch.set_num_threads(THREADS)

try:
    torch.set_num_interop_threads(
        min(4, THREADS)
    )
except RuntimeError:
    pass


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("U-NET LESION SEGMENTATION")
print("=" * 70)

print(f"Device          : {DEVICE}")
print(f"CPU threads     : {THREADS}")
print(f"Image size      : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epochs          : {NUM_EPOCHS}")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"Base channels   : {BASE_CHANNELS}")
print(f"Random seed     : {RANDOM_SEED}")
print()


# ============================================================
# CREATE DIRECTORIES
# ============================================================

directories = [
    MODEL_DIR,
    MASK_DIR / "train",
    MASK_DIR / "val",
    MASK_DIR / "test",
    LESION_DIR / "train",
    LESION_DIR / "val",
    LESION_DIR / "test",
    RESULTS_DIR,
    VISUAL_DIR,
]

for directory in directories:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CHECK INPUT FILES
# ============================================================

for file_path in [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ============================================================
# DATASET
# ============================================================

class LesionSegmentationDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int,
        training: bool,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.image_size = image_size
        self.training = training


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(
            row["image_path"]
        )

        mask_path = Path(
            row["mask_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )


        # ----------------------------------------------------
        # Training augmentation
        # ----------------------------------------------------

        if self.training:

            # Horizontal flip
            if random.random() < 0.5:

                image = TF.hflip(image)
                mask = TF.hflip(mask)


            # Vertical flip
            if random.random() < 0.5:

                image = TF.vflip(image)
                mask = TF.vflip(mask)


            # Small rotation
            angle = random.uniform(
                -10.0,
                10.0
            )

            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )


        # ----------------------------------------------------
        # Image tensor
        # ----------------------------------------------------

        image = TF.to_tensor(image)

        image = TF.normalize(
            image,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )


        # ----------------------------------------------------
        # Mask tensor
        # ----------------------------------------------------

        mask_array = np.array(
            mask,
            dtype=np.float32
        )

        mask_array = mask_array / 255.0

        mask_array = (
            mask_array >= 0.5
        ).astype(np.float32)

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0)


        return (
            image,
            mask_tensor,
            row["image_id"]
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA SPLITS")
print("=" * 70)

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)

print(
    f"Training   : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Testing    : {len(test_df)}"
)

print()


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = LesionSegmentationDataset(
    train_df,
    IMAGE_SIZE,
    training=True,
)

val_dataset = LesionSegmentationDataset(
    val_df,
    IMAGE_SIZE,
    training=False,
)

test_dataset = LesionSegmentationDataset(
    test_df,
    IMAGE_SIZE,
    training=False,
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# U-NET BUILDING BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )


    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=16,
    ):

        super().__init__()


        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16


        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.enc1 = DoubleConv(
            in_channels,
            c1,
        )

        self.enc2 = DoubleConv(
            c1,
            c2,
        )

        self.enc3 = DoubleConv(
            c2,
            c3,
        )

        self.enc4 = DoubleConv(
            c3,
            c4,
        )

        self.bottleneck = DoubleConv(
            c4,
            c5,
        )


        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            c5,
            c4,
            kernel_size=2,
            stride=2,
        )

        self.dec4 = DoubleConv(
            c4 * 2,
            c4,
        )


        self.up3 = nn.ConvTranspose2d(
            c4,
            c3,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv(
            c3 * 2,
            c3,
        )


        self.up2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            c2 * 2,
            c2,
        )


        self.up1 = nn.ConvTranspose2d(
            c2,
            c1,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            c1 * 2,
            c1,
        )


        # ----------------------------------------------------
        # Final segmentation layer
        # ----------------------------------------------------

        self.final_conv = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )


    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )


        # Decoder
        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)


        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)


        return self.final_conv(d1)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth


    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.view(
            probabilities.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()


    def forward(
        self,
        logits,
        targets
    ):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        return (
            0.5 * bce_loss
            + 0.5 * dice_loss
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets,
    threshold=0.5,
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()


    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )


    tp = (
        predictions * targets
    ).sum(dim=1)

    fp = (
        predictions
        * (1.0 - targets)
    ).sum(dim=1)

    fn = (
        (1.0 - predictions)
        * targets
    ).sum(dim=1)

    tn = (
        (1.0 - predictions)
        * (1.0 - targets)
    ).sum(dim=1)


    intersection = tp

    union = (
        tp + fp + fn
    )


    dice = (
        2.0 * tp + 1e-7
    ) / (
        2.0 * tp
        + fp
        + fn
        + 1e-7
    )


    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )


    precision = (
        tp + 1e-7
    ) / (
        tp + fp + 1e-7
    )


    recall = (
        tp + 1e-7
    ) / (
        tp + fn + 1e-7
    )


    specificity = (
        tn + 1e-7
    ) / (
        tn + fp + 1e-7
    )


    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "specificity": specificity.mean().item(),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
    }

    batch_count = 0


    with torch.no_grad():

        for images, masks, _ in loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks
            )

            metrics = calculate_metrics(
                logits,
                masks
            )

            total_loss += (
                loss.item()
            )

            for key in totals:

                totals[key] += (
                    metrics[key]
                )

            batch_count += 1


    if batch_count == 0:

        raise RuntimeError(
            "No validation/test batches."
        )


    average_loss = (
        total_loss
        / batch_count
    )


    average_metrics = {
        key: value / batch_count
        for key, value
        in totals.items()
    }


    return (
        average_loss,
        average_metrics
    )


# ============================================================
# CREATE MODEL
# ============================================================

print("=" * 70)
print("CREATING U-NET")
print("=" * 70)

model = UNet(
    base_channels=BASE_CHANNELS
).to(DEVICE)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Trainable parameters: "
    f"{parameter_count:,}"
)

print()


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = CombinedLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

configuration = {
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "early_stopping_patience":
        EARLY_STOPPING_PATIENCE,
    "random_seed": RANDOM_SEED,
    "base_channels": BASE_CHANNELS,
    "device": str(DEVICE),
    "cpu_threads": THREADS,
}


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        configuration,
        f,
        indent=4
    )


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = []

best_val_dice = -math.inf

epochs_without_improvement = 0


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    epoch_start = time.time()

    total_batches = len(
        train_loader
    )


    for batch_index, (
        images,
        masks,
        _
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            images
        )


        loss = criterion(
            logits,
            masks
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
        )


        # ----------------------------------------------------
        # LIVE PROGRESS
        # ----------------------------------------------------

        if (
            batch_index == 1
            or batch_index % PRINT_EVERY == 0
            or batch_index == total_batches
        ):

            current_loss = (
                running_loss
                / batch_index
            )

            elapsed = (
                time.time()
                - epoch_start
            )

            batches_per_second = (
                batch_index
                / elapsed
                if elapsed > 0
                else 0
            )

            remaining_batches = (
                total_batches
                - batch_index
            )

            eta = (
                remaining_batches
                / batches_per_second
                if batches_per_second > 0
                else 0
            )


            print(
                f"\r"
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Batch {batch_index:03d}/{total_batches} | "
                f"Loss {current_loss:.4f} | "
                f"ETA {eta / 60:.1f} min",
                end="",
                flush=True
            )


    print()


    train_loss = (
        running_loss
        / total_batches
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_loss, val_metrics = (
        evaluate_model(
            model,
            val_loader,
            criterion
        )
    )


    val_dice = val_metrics[
        "dice"
    ]


    scheduler.step(
        val_dice
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    epoch_time = (
        time.time()
        - epoch_start
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_dice": val_metrics["dice"],
        "val_iou": val_metrics["iou"],
        "val_precision":
            val_metrics["precision"],
        "val_recall":
            val_metrics["recall"],
        "val_specificity":
            val_metrics["specificity"],
        "learning_rate": current_lr,
        "epoch_time_seconds":
            epoch_time,
    }


    history.append(
        history_row
    )


    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )


    # --------------------------------------------------------
    # EPOCH SUMMARY
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_metrics['dice']:.4f} | "
        f"Val IoU: {val_metrics['iou']:.4f} | "
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f} | "
        f"Time: {epoch_time / 60:.1f} min"
    )


    # --------------------------------------------------------
    # SAVE LAST CHECKPOINT
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "best_val_dice":
                best_val_dice,
            "config":
                configuration,
        },
        LAST_CHECKPOINT_FILE
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        epochs_without_improvement = 0


        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "val_dice":
                    val_dice,
                "config":
                    configuration,
            },
            MODEL_FILE
        )


        print(
            f"  [BEST MODEL SAVED] "
            f"Validation Dice = "
            f"{val_dice:.4f}"
        )


    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation Dice: "
    f"{best_val_dice:.4f}"
)

print(
    f"Best model:\n{MODEL_FILE}"
)

print()


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Best model came from epoch "
    f"{checkpoint['epoch']}"
)

print()


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_metrics = (
    evaluate_model(
        model,
        test_loader,
        criterion
    )
)


print(
    f"Test Loss        : "
    f"{test_loss:.4f}"
)

print(
    f"Test Dice        : "
    f"{test_metrics['dice']:.4f}"
)

print(
    f"Test IoU         : "
    f"{test_metrics['iou']:.4f}"
)

print(
    f"Test Precision   : "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Test Recall      : "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"Test Specificity : "
    f"{test_metrics['specificity']:.4f}"
)


with open(
    TEST_RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "test_loss": test_loss,
            **test_metrics,
        },
        f,
        indent=4
    )


print()


# ============================================================
# IMAGE/MASK PREDICTION FUNCTION
# ============================================================

def predict_mask(
    model,
    image_path
):

    original_image = (
        Image.open(
            image_path
        ).convert("RGB")
    )


    resized = TF.resize(
        original_image,
        [IMAGE_SIZE, IMAGE_SIZE],
        interpolation=InterpolationMode.BILINEAR,
    )


    tensor = TF.to_tensor(
        resized
    )


    tensor = TF.normalize(
        tensor,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )


    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)


    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.sigmoid(
            logits
        )[0, 0]


        predicted_mask = (
            probabilities >= 0.5
        ).cpu().numpy().astype(
            np.uint8
        )


    return (
        original_image,
        predicted_mask
    )


# ============================================================
# CREATE LESION-FOCUSED IMAGE
# ============================================================

def create_lesion_focused_image(
    original_image,
    mask,
    padding_ratio=0.10
):

    original_array = np.array(
        original_image
    )


    width, height = (
        original_image.size
    )


    # Convert predicted mask to original resolution
    mask_image = Image.fromarray(
        (mask * 255).astype(
            np.uint8
        )
    )


    mask_image = mask_image.resize(
        (width, height),
        resample=Image.Resampling.NEAREST
    )


    original_mask = np.array(
        mask_image
    ) >= 128


    ys, xs = np.where(
        original_mask
    )


    # --------------------------------------------------------
    # Fallback if model predicts no lesion
    # --------------------------------------------------------

    if (
        len(xs) == 0
        or len(ys) == 0
    ):

        return original_image.copy()


    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())


    box_width = (
        x_max - x_min + 1
    )

    box_height = (
        y_max - y_min + 1
    )


    padding_x = int(
        box_width
        * padding_ratio
    )

    padding_y = int(
        box_height
        * padding_ratio
    )


    x_min = max(
        0,
        x_min - padding_x
    )

    y_min = max(
        0,
        y_min - padding_y
    )

    x_max = min(
        width - 1,
        x_max + padding_x
    )

    y_max = min(
        height - 1,
        y_max + padding_y
    )


    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    cropped = original_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]


    cropped_mask = original_mask[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]


    # --------------------------------------------------------
    # Suppress outside-lesion pixels
    # --------------------------------------------------------

    focused = cropped.copy()

    focused[
        ~cropped_mask
    ] = 0


    return Image.fromarray(
        focused.astype(
            np.uint8
        )
    )


# ============================================================
# GENERATE MASKS AND LESION IMAGES
# ============================================================

def generate_outputs(
    split_name,
    dataframe
):

    mask_output_dir = (
        MASK_DIR / split_name
    )

    lesion_output_dir = (
        LESION_DIR / split_name
    )


    total = len(
        dataframe
    )


    print()
    print("=" * 70)
    print(
        f"GENERATING "
        f"{split_name.upper()} OUTPUTS"
    )
    print("=" * 70)


    for index, row in enumerate(
        dataframe.itertuples(
            index=False
        ),
        start=1
    ):

        image_id = row.image_id

        image_path = Path(
            row.image_path
        )


        predicted_mask_path = (
            mask_output_dir
            / f"{image_id}_Predicted.png"
        )


        lesion_image_path = (
            lesion_output_dir
            / f"{image_id}_Lesion.jpg"
        )


        # Resume safely if already generated
        if (
            predicted_mask_path.exists()
            and lesion_image_path.exists()
        ):

            continue


        original_image, predicted_mask = (
            predict_mask(
                model,
                image_path
            )
        )


        Image.fromarray(
            (
                predicted_mask
                * 255
            ).astype(np.uint8)
        ).save(
            predicted_mask_path
        )


        lesion_image = (
            create_lesion_focused_image(
                original_image,
                predicted_mask,
                padding_ratio=
                    CROP_PADDING_RATIO
            )
        )


        lesion_image.save(
            lesion_image_path,
            quality=95
        )


        if (
            index == 1
            or index % 100 == 0
            or index == total
        ):

            print(
                f"Processed "
                f"{index}/{total}"
            )


# ============================================================
# GENERATE ALL PREDICTED OUTPUTS
# ============================================================

generate_outputs(
    "train",
    train_df
)

generate_outputs(
    "val",
    val_df
)

generate_outputs(
    "test",
    test_df
)


# ============================================================
# VISUALIZE TEST PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("CREATING FINAL VISUAL CHECK")
print("=" * 70)


random.seed(
    RANDOM_SEED
)


visual_samples = random.sample(
    list(
        test_df.to_dict(
            "records"
        )
    ),
    min(
        NUM_VISUAL_SAMPLES,
        len(test_df)
    )
)


fig, axes = plt.subplots(
    len(visual_samples),
    3,
    figsize=(
        12,
        4 * len(visual_samples)
    )
)


if len(visual_samples) == 1:

    axes = np.array(
        [axes]
    )


for row_index, sample in enumerate(
    visual_samples
):

    image_id = sample[
        "image_id"
    ]


    original_path = Path(
        sample["image_path"]
    )


    predicted_mask_path = (
        MASK_DIR
        / "test"
        / f"{image_id}_Predicted.png"
    )


    lesion_path = (
        LESION_DIR
        / "test"
        / f"{image_id}_Lesion.jpg"
    )


    original_image = (
        Image.open(
            original_path
        ).convert("RGB")
    )


    predicted_mask = (
        Image.open(
            predicted_mask_path
        ).convert("L")
    )


    lesion_image = (
        Image.open(
            lesion_path
        ).convert("RGB")
    )


    axes[row_index][0].imshow(
        original_image
    )

    axes[row_index][0].set_title(
        f"{image_id}\nOriginal"
    )

    axes[row_index][0].axis(
        "off"
    )


    axes[row_index][1].imshow(
        predicted_mask,
        cmap="gray"
    )

    axes[row_index][1].set_title(
        "U-Net predicted mask"
    )

    axes[row_index][1].axis(
        "off"
    )


    axes[row_index][2].imshow(
        lesion_image
    )

    axes[row_index][2].set_title(
        "Lesion-focused image"
    )

    axes[row_index][2].axis(
        "off"
    )


plt.tight_layout()


visualization_file = (
    VISUAL_DIR
    / "segmentation_results.png"
)


plt.savefig(
    visualization_file,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print(
    f"Visualization saved to:\n"
    f"{visualization_file}"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("U-NET STAGE COMPLETE")
print("=" * 70)

print()
print("Best U-Net:")
print(MODEL_FILE)

print()
print("Predicted masks:")
print(MASK_DIR)

print()
print("Lesion-focused images:")
print(LESION_DIR)

print()
print("Training history:")
print(HISTORY_FILE)

print()
print("Test metrics:")
print(TEST_RESULTS_FILE)

print()
print("Final visualization:")
print(visualization_file)

print()
print("=" * 70)
print("DONE")
print("=" * 70)