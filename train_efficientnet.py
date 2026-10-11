from pathlib import Path
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"E:\SkinCancer")

TRAIN_CSV = BASE_DIR / "splits" / "train.csv"
VAL_CSV = BASE_DIR / "splits" / "val.csv"
TEST_CSV = BASE_DIR / "splits" / "test.csv"

SEGMENTATION_DIR = (
    BASE_DIR / "segmentation" / "lesion_focused"
)

OUTPUT_DIR = BASE_DIR / "classification" / "efficientnetv2"

MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOT_DIR = RESULTS_DIR / "plots"
PREDICTION_DIR = RESULTS_DIR / "predictions"

BEST_MODEL_FILE = (
    MODEL_DIR / "best_efficientnetv2_s.pth"
)

LAST_CHECKPOINT_FILE = (
    MODEL_DIR / "last_checkpoint.pth"
)

CONFIG_FILE = (
    RESULTS_DIR / "configuration.json"
)

VAL_PREDICTIONS_FILE = (
    PREDICTION_DIR / "validation_predictions.csv"
)

TEST_PREDICTIONS_FILE = (
    PREDICTION_DIR / "test_predictions.csv"
)

METRICS_FILE = (
    RESULTS_DIR / "test_metrics.json"
)

THRESHOLD_FILE = (
    RESULTS_DIR / "optimal_threshold.json"
)

HISTORY_FILE = (
    RESULTS_DIR / "training_history.csv"
)

CONFUSION_MATRIX_FILE = (
    PLOT_DIR / "confusion_matrix.png"
)

ROC_CURVE_FILE = (
    PLOT_DIR / "roc_curve.png"
)

PR_CURVE_FILE = (
    PLOT_DIR / "precision_recall_curve.png"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 4

# Stage 1:
# Only the new classification head is trained.
HEAD_EPOCHS = 3

# Stage 2:
# Last EfficientNet blocks are unfrozen and fine-tuned.
FINETUNE_EPOCHS = 9

HEAD_LEARNING_RATE = 1e-3

BACKBONE_LEARNING_RATE = 1e-5

CLASSIFIER_LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 4

GRADIENT_CLIP_VALUE = 1.0

RANDOM_SEED = 42

PRINT_EVERY = 25

# Number of EfficientNet feature blocks to unfreeze
# during fine-tuning.
UNFREEZE_LAST_FEATURE_BLOCKS = 3

# Automatically continue from a checkpoint if present.
RESUME = True


# ============================================================
# CPU SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

CPU_COUNT = os.cpu_count() or 4

CPU_THREADS = max(
    1,
    min(CPU_COUNT - 1, 8)
)

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(
        min(4, CPU_THREADS)
    )
except RuntimeError:
    pass


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# DIRECTORIES
# ============================================================

for directory in [
    MODEL_DIR,
    RESULTS_DIR,
    PLOT_DIR,
    PREDICTION_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("EFFICIENTNETV2-S MELANOMA CLASSIFICATION")
print("=" * 70)

print(f"Device                 : {DEVICE}")
print(f"CPU threads            : {CPU_THREADS}")
print(f"Image size             : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Batch size             : {BATCH_SIZE}")
print(f"Head epochs            : {HEAD_EPOCHS}")
print(f"Fine-tuning epochs     : {FINETUNE_EPOCHS}")
print(f"Head learning rate     : {HEAD_LEARNING_RATE}")
print(f"Backbone learning rate : {BACKBONE_LEARNING_RATE}")
print(f"Random seed            : {RANDOM_SEED}")
print()


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
            f"Required split file not found:\n"
            f"{file_path}"
        )


# ============================================================
# LOAD SPLITS
# ============================================================

print("=" * 70)
print("LOADING FIXED DATA SPLITS")
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

print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Testing    : {len(test_df)}")
print()


# ============================================================
# CREATE PATH TO LESION-FOCUSED IMAGES
# ============================================================

def add_classification_paths(
    dataframe,
    split_name,
):

    dataframe = dataframe.copy()

    classification_paths = []

    for image_id in dataframe["image_id"]:

        image_path = (
            SEGMENTATION_DIR
            / split_name
            / f"{image_id}_Lesion.jpg"
        )

        classification_paths.append(
            str(image_path)
        )

    dataframe[
        "classification_image_path"
    ] = classification_paths

    return dataframe


train_df = add_classification_paths(
    train_df,
    "train"
)

val_df = add_classification_paths(
    val_df,
    "val"
)

test_df = add_classification_paths(
    test_df,
    "test"
)


# ============================================================
# VERIFY ALL CLASSIFICATION IMAGES EXIST
# ============================================================

print("=" * 70)
print("VERIFYING LESION-FOCUSED IMAGES")
print("=" * 70)


def verify_images(dataframe, split_name):

    missing = []

    for path in dataframe[
        "classification_image_path"
    ]:

        if not Path(path).exists():

            missing.append(path)

    if missing:

        print(
            f"Missing {split_name} images: "
            f"{len(missing)}"
        )

        for path in missing[:10]:
            print(path)

        raise FileNotFoundError(
            f"Missing lesion-focused images "
            f"in {split_name} split."
        )

    print(
        f"[OK] {split_name}: "
        f"{len(dataframe)} images found."
    )


verify_images(
    train_df,
    "train"
)

verify_images(
    val_df,
    "validation"
)

verify_images(
    test_df,
    "test"
)

print()


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

train_melanoma = int(
    (train_df["melanoma"] == 1).sum()
)

train_non_melanoma = int(
    (train_df["melanoma"] == 0).sum()
)

print("=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(
    f"Melanoma     : "
    f"{train_melanoma}"
)

print(
    f"Non-melanoma : "
    f"{train_non_melanoma}"
)

print()


# ============================================================
# CALCULATE CLASS WEIGHT
# ============================================================

if train_melanoma == 0:

    raise ValueError(
        "No melanoma samples in training set."
    )

if train_non_melanoma == 0:

    raise ValueError(
        "No non-melanoma samples in training set."
    )


POS_WEIGHT = (
    train_non_melanoma
    / train_melanoma
)

print(
    f"Melanoma positive-class weight: "
    f"{POS_WEIGHT:.4f}"
)

print()


# ============================================================
# TRANSFORMS
# ============================================================

# ImageNet normalization for pretrained EfficientNetV2-S.
IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.10,
        hue=0.02,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# DATASET
# ============================================================

class MelanomaDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(
            row["classification_image_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(
            image
        )

        label = torch.tensor(
            float(row["melanoma"]),
            dtype=torch.float32
        )

        return (
            image,
            label,
            row["image_id"],
        )


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = MelanomaDataset(
    train_df,
    train_transform,
)

val_dataset = MelanomaDataset(
    val_df,
    eval_transform,
)

test_dataset = MelanomaDataset(
    test_df,
    eval_transform,
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=False,
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=False,
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=False,
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNETV2-S
# ============================================================

print("=" * 70)
print("LOADING PRETRAINED EFFICIENTNETV2-S")
print("=" * 70)

weights = (
    EfficientNet_V2_S_Weights.DEFAULT
)

model = efficientnet_v2_s(
    weights=weights
)


# ============================================================
# REPLACE CLASSIFICATION HEAD
# ============================================================

input_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    input_features,
    1
)


model = model.to(
    DEVICE
)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    f"Total parameters     : "
    f"{parameter_count:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_parameter_count:,}"
)

print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [POS_WEIGHT],
        dtype=torch.float32,
        device=DEVICE
    )
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_classification_metrics(
    labels,
    probabilities,
    threshold=0.5,
):

    labels = np.asarray(
        labels
    ).astype(int)

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions
        )
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )


    sensitivity = recall


    # --------------------------------------------------------
    # AUROC / AUPRC
    # --------------------------------------------------------

    try:

        auroc = roc_auc_score(
            labels,
            probabilities
        )

    except ValueError:

        auroc = float("nan")


    try:

        auprc = average_precision_score(
            labels,
            probabilities
        )

    except ValueError:

        auprc = float("nan")


    return {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "sensitivity": float(
            sensitivity
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1
        ),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "auroc": float(
            auroc
        ),

        "auprc": float(
            auprc
        ),

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn),

        "tp": int(tp),
    }


# ============================================================
# RUN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0

    total_samples = 0

    total_batches = len(
        loader
    )


    epoch_start = time.time()


    for batch_index, (
        images,
        labels,
        _
    ) in enumerate(
        loader,
        start=1
    ):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            images
        )


        logits = logits.squeeze(
            1
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_VALUE
        )


        optimizer.step()


        batch_size_current = (
            images.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size_current
        )

        total_samples += (
            batch_size_current
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index == 1
            or batch_index % PRINT_EVERY == 0
            or batch_index == total_batches
        ):

            current_loss = (
                running_loss
                / total_samples
            )

            elapsed = (
                time.time()
                - epoch_start
            )

            if elapsed > 0:

                speed = (
                    batch_index
                    / elapsed
                )

            else:

                speed = 0


            remaining_batches = (
                total_batches
                - batch_index
            )


            if speed > 0:

                eta = (
                    remaining_batches
                    / speed
                )

            else:

                eta = 0


            print(
                f"\r"
                f"Batch "
                f"{batch_index:03d}/"
                f"{total_batches} | "
                f"Loss "
                f"{current_loss:.4f} | "
                f"ETA "
                f"{eta / 60:.1f} min",
                end="",
                flush=True
            )


    print()


    return (
        running_loss
        / total_samples
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    all_labels = []

    all_probabilities = []

    all_image_ids = []


    with torch.no_grad():

        for images, labels, image_ids in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            logits = model(
                images
            )

            logits = logits.squeeze(
                1
            )


            loss = criterion(
                logits,
                labels
            )


            probabilities = torch.sigmoid(
                logits
            )


            current_batch_size = (
                images.size(0)
            )


            total_loss += (
                loss.item()
                * current_batch_size
            )


            total_samples += (
                current_batch_size
            )


            all_labels.extend(
                labels.cpu().numpy().tolist()
            )


            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )


            all_image_ids.extend(
                list(image_ids)
            )


    average_loss = (
        total_loss
        / total_samples
    )


    labels_array = np.asarray(
        all_labels
    ).astype(int)


    probabilities_array = np.asarray(
        all_probabilities
    )


    metrics = calculate_classification_metrics(
        labels_array,
        probabilities_array,
        threshold=0.5
    )


    return (
        average_loss,
        metrics,
        labels_array,
        probabilities_array,
        all_image_ids,
    )


# ============================================================
# SET STAGE 1: FREEZE BACKBONE
# ============================================================

def freeze_backbone(
    model
):

    for parameter in (
        model.features.parameters()
    ):

        parameter.requires_grad = False


    for parameter in (
        model.classifier.parameters()
    ):

        parameter.requires_grad = True


# ============================================================
# SET STAGE 2: UNFREEZE LAST FEATURE BLOCKS
# ============================================================

def prepare_fine_tuning(
    model,
):

    # Freeze everything first.
    for parameter in model.parameters():

        parameter.requires_grad = False


    # Unfreeze last N feature modules.
    total_features = len(
        model.features
    )

    start_index = max(
        0,
        total_features
        - UNFREEZE_LAST_FEATURE_BLOCKS
    )


    for module_index in range(
        start_index,
        total_features
    ):

        for parameter in (
            model.features[
                module_index
            ].parameters()
        ):

            parameter.requires_grad = True


    # Classification head always trainable.
    for parameter in (
        model.classifier.parameters()
    ):

        parameter.requires_grad = True


# ============================================================
# OPTIMIZER CREATION
# ============================================================

def create_stage1_optimizer(
    model
):

    return torch.optim.AdamW(

        model.classifier.parameters(),

        lr=HEAD_LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )


def create_stage2_optimizer(
    model
):

    feature_parameters = []

    classifier_parameters = []


    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue


        if name.startswith(
            "features."
        ):

            feature_parameters.append(
                parameter
            )

        else:

            classifier_parameters.append(
                parameter
            )


    return torch.optim.AdamW(

        [
            {
                "params":
                    feature_parameters,
                "lr":
                    BACKBONE_LEARNING_RATE,
            },

            {
                "params":
                    classifier_parameters,
                "lr":
                    CLASSIFIER_LEARNING_RATE,
            },
        ],

        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    stage,
    epoch,
    best_val_auroc,
    epochs_without_improvement,
):

    torch.save(

        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "stage":
                stage,

            "epoch":
                epoch,

            "best_val_auroc":
                best_val_auroc,

            "epochs_without_improvement":
                epochs_without_improvement,

            "config":
                {
                    "image_size":
                        IMAGE_SIZE,

                    "batch_size":
                        BATCH_SIZE,

                    "head_epochs":
                        HEAD_EPOCHS,

                    "finetune_epochs":
                        FINETUNE_EPOCHS,

                    "head_learning_rate":
                        HEAD_LEARNING_RATE,

                    "backbone_learning_rate":
                        BACKBONE_LEARNING_RATE,

                    "classifier_learning_rate":
                        CLASSIFIER_LEARNING_RATE,

                    "pos_weight":
                        POS_WEIGHT,

                    "seed":
                        RANDOM_SEED,
                },
        },

        LAST_CHECKPOINT_FILE
    )


# ============================================================
# HISTORY
# ============================================================

history = []


# ============================================================
# TRAINING STATE
# ============================================================

start_stage = 1
start_epoch = 1

best_val_auroc = -math.inf

epochs_without_improvement = 0

optimizer = None


# ============================================================
# RESUME FROM CHECKPOINT
# ============================================================

if (
    RESUME
    and LAST_CHECKPOINT_FILE.exists()
):

    print("=" * 70)
    print("CHECKPOINT FOUND")
    print("=" * 70)

    checkpoint = torch.load(
        LAST_CHECKPOINT_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    start_stage = int(
        checkpoint["stage"]
    )

    start_epoch = int(
        checkpoint["epoch"]
    ) + 1

    best_val_auroc = float(
        checkpoint["best_val_auroc"]
    )

    epochs_without_improvement = int(
        checkpoint[
            "epochs_without_improvement"
        ]
    )


    if start_stage == 1:

        freeze_backbone(
            model
        )

        optimizer = (
            create_stage1_optimizer(
                model
            )
        )

    else:

        prepare_fine_tuning(
            model
        )

        optimizer = (
            create_stage2_optimizer(
                model
            )
        )


    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )


    print(
        f"Resuming Stage {start_stage}, "
        f"Epoch {start_epoch}"
    )

    print()


# ============================================================
# STAGE 1
# ============================================================

if start_stage <= 1:

    print("=" * 70)
    print("STAGE 1 — CLASSIFICATION HEAD TRAINING")
    print("=" * 70)

    freeze_backbone(
        model
    )

    optimizer = (
        create_stage1_optimizer(
            model
        )
    )


    for epoch in range(
        start_epoch,
        HEAD_EPOCHS + 1
    ):

        print()
        print(
            f"Stage 1 | "
            f"Epoch "
            f"{epoch}/{HEAD_EPOCHS}"
        )


        epoch_start = time.time()


        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )


        (
            val_loss,
            val_metrics,
            val_labels,
            val_probabilities,
            val_image_ids,
        ) = evaluate_model(
            model,
            val_loader,
            criterion,
        )


        val_auroc = val_metrics[
            "auroc"
        ]


        epoch_time = (
            time.time()
            - epoch_start
        )


        history_row = {

            "stage": 1,

            "epoch": epoch,

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_precision":
                val_metrics[
                    "precision"
                ],

            "val_recall":
                val_metrics[
                    "recall"
                ],

            "val_specificity":
                val_metrics[
                    "specificity"
                ],

            "val_f1":
                val_metrics[
                    "f1"
                ],

            "val_auroc":
                val_metrics[
                    "auroc"
                ],

            "val_auprc":
                val_metrics[
                    "auprc"
                ],

            "learning_rate":
                optimizer.param_groups[
                    0
                ]["lr"],

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


        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val F1: "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"Val AUROC: "
            f"{val_metrics['auroc']:.4f}"
        )

        print(
            f"Val AUPRC: "
            f"{val_metrics['auprc']:.4f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time / 60:.1f} min"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        improved = (
            not np.isnan(val_auroc)
            and val_auroc > best_val_auroc
        )


        if improved:

            best_val_auroc = (
                val_auroc
            )

            epochs_without_improvement = 0


            torch.save(

                {
                    "model_state_dict":
                        model.state_dict(),

                    "stage":
                        1,

                    "epoch":
                        epoch,

                    "val_auroc":
                        val_auroc,

                    "val_metrics":
                        val_metrics,

                    "config":
                        {
                            "image_size":
                                IMAGE_SIZE,

                            "seed":
                                RANDOM_SEED,
                        },
                },

                BEST_MODEL_FILE
            )


            print(
                "[BEST MODEL SAVED]"
            )


        else:

            epochs_without_improvement += 1


        save_checkpoint(

            model,
            optimizer,
            1,
            epoch,
            best_val_auroc,
            epochs_without_improvement,
        )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "Stage 1 early stopping."
            )

            break


    # Move to Stage 2
    start_stage = 2
    start_epoch = 1
    epochs_without_improvement = 0


# ============================================================
# STAGE 2
# ============================================================

print()
print("=" * 70)
print("STAGE 2 — EFFICIENTNET FINE-TUNING")
print("=" * 70)

prepare_fine_tuning(
    model
)

optimizer = (
    create_stage2_optimizer(
        model
    )
)


# When starting Stage 2, use the best
# Stage 1 model if available.
if BEST_MODEL_FILE.exists():

    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


best_stage2_auroc = best_val_auroc

epochs_without_improvement = 0


for epoch in range(
    start_epoch,
    FINETUNE_EPOCHS + 1
):

    print()
    print(
        f"Stage 2 | "
        f"Epoch "
        f"{epoch}/{FINETUNE_EPOCHS}"
    )


    epoch_start = time.time()


    train_loss = (
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
    )


    (
        val_loss,
        val_metrics,
        val_labels,
        val_probabilities,
        val_image_ids,
    ) = evaluate_model(
        model,
        val_loader,
        criterion,
    )


    val_auroc = val_metrics[
        "auroc"
    ]


    epoch_time = (
        time.time()
        - epoch_start
    )


    history_row = {

        "stage": 2,

        "epoch": epoch,

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_metrics[
                "accuracy"
            ],

        "val_precision":
            val_metrics[
                "precision"
            ],

        "val_recall":
            val_metrics[
                "recall"
            ],

        "val_specificity":
            val_metrics[
                "specificity"
            ],

        "val_f1":
            val_metrics[
                "f1"
            ],

        "val_auroc":
            val_metrics[
                "auroc"
            ],

        "val_auprc":
            val_metrics[
                "auprc"
            ],

        "learning_rate":
            optimizer.param_groups[
                0
            ]["lr"],

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


    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Val Precision: "
        f"{val_metrics['precision']:.4f}"
    )

    print(
        f"Val Recall: "
        f"{val_metrics['recall']:.4f}"
    )

    print(
        f"Val Specificity: "
        f"{val_metrics['specificity']:.4f}"
    )

    print(
        f"Val F1: "
        f"{val_metrics['f1']:.4f}"
    )

    print(
        f"Val AUROC: "
        f"{val_metrics['auroc']:.4f}"
    )

    print(
        f"Val AUPRC: "
        f"{val_metrics['auprc']:.4f}"
    )

    print(
        f"Epoch time: "
        f"{epoch_time / 60:.1f} min"
    )


    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    improved = (
        not np.isnan(val_auroc)
        and val_auroc > best_stage2_auroc
    )


    if improved:

        best_stage2_auroc = (
            val_auroc
        )

        best_val_auroc = (
            val_auroc
        )

        epochs_without_improvement = 0


        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "stage":
                    2,

                "epoch":
                    epoch,

                "val_auroc":
                    val_auroc,

                "val_metrics":
                    val_metrics,

                "config":
                    {
                        "image_size":
                            IMAGE_SIZE,

                        "batch_size":
                            BATCH_SIZE,

                        "seed":
                            RANDOM_SEED,

                        "pos_weight":
                            POS_WEIGHT,
                    },
            },

            BEST_MODEL_FILE
        )


        print(
            "[BEST MODEL SAVED]"
        )


    else:

        epochs_without_improvement += 1


    save_checkpoint(

        model,
        optimizer,
        2,
        epoch,
        best_val_auroc,
        epochs_without_improvement,
    )


    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "Stage 2 early stopping."
        )

        break


# ============================================================
# LOAD BEST MODEL FOR FINAL EVALUATION
# ============================================================

print()
print("=" * 70)
print("LOADING BEST EFFICIENTNETV2-S")
print("=" * 70)

best_checkpoint = torch.load(
    BEST_MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    f"Best checkpoint stage: "
    f"{best_checkpoint['stage']}"
)

print(
    f"Best checkpoint epoch: "
    f"{best_checkpoint['epoch']}"
)

print(
    f"Best validation AUROC: "
    f"{best_checkpoint['val_auroc']:.4f}"
)

print()


# ============================================================
# FINAL VALIDATION PREDICTIONS
# ============================================================

print("=" * 70)
print("GENERATING FINAL VALIDATION PREDICTIONS")
print("=" * 70)

(
    val_loss,
    val_metrics,
    val_labels,
    val_probabilities,
    val_image_ids,
) = evaluate_model(
    model,
    val_loader,
    criterion
)


val_prediction_df = pd.DataFrame({

    "image_id":
        val_image_ids,

    "label":
        val_labels,

    "melanoma_probability":
        val_probabilities,
})


val_prediction_df.to_csv(
    VAL_PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SELECT OPTIMAL CLASSIFICATION THRESHOLD
# ============================================================

print()
print("=" * 70)
print("SELECTING VALIDATION THRESHOLD")
print("=" * 70)


thresholds = np.arange(
    0.05,
    0.951,
    0.01
)


best_threshold = 0.50

best_threshold_f1 = -1.0

best_threshold_balanced_accuracy = -1.0


for threshold in thresholds:

    metrics = calculate_classification_metrics(
        val_labels,
        val_probabilities,
        threshold=float(threshold)
    )


    current_f1 = metrics[
        "f1"
    ]

    current_balanced_accuracy = (
        metrics[
            "balanced_accuracy"
        ]
    )


    # Primary criterion:
    # F1.
    #
    # Secondary criterion:
    # Balanced accuracy.

    if (
        current_f1
        > best_threshold_f1
    ):

        best_threshold_f1 = (
            current_f1
        )

        best_threshold_balanced_accuracy = (
            current_balanced_accuracy
        )

        best_threshold = float(
            threshold
        )


    elif (
        current_f1
        == best_threshold_f1
        and
        current_balanced_accuracy
        > best_threshold_balanced_accuracy
    ):

        best_threshold_balanced_accuracy = (
            current_balanced_accuracy
        )

        best_threshold = float(
            threshold
        )


optimal_threshold_metrics = (
    calculate_classification_metrics(
        val_labels,
        val_probabilities,
        threshold=best_threshold
    )
)


print(
    f"Optimal threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1 at threshold: "
    f"{optimal_threshold_metrics['f1']:.4f}"
)


with open(
    THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        {
            "threshold":
                best_threshold,

            "validation_metrics":
                optimal_threshold_metrics,

        },

        f,

        indent=4
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

(
    test_loss,
    test_metrics_05,
    test_labels,
    test_probabilities,
    test_image_ids,
) = evaluate_model(
    model,
    test_loader,
    criterion
)


# Metrics using threshold = 0.50
test_metrics_optimal = (
    calculate_classification_metrics(
        test_labels,
        test_probabilities,
        threshold=best_threshold
    )
)


print()
print("TEST RESULTS — THRESHOLD 0.50")
print("-" * 70)

print(
    f"Accuracy          : "
    f"{test_metrics_05['accuracy']:.4f}"
)

print(
    f"Precision         : "
    f"{test_metrics_05['precision']:.4f}"
)

print(
    f"Recall/Sensitivity: "
    f"{test_metrics_05['recall']:.4f}"
)

print(
    f"Specificity       : "
    f"{test_metrics_05['specificity']:.4f}"
)

print(
    f"F1                : "
    f"{test_metrics_05['f1']:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{test_metrics_05['balanced_accuracy']:.4f}"
)

print(
    f"AUROC             : "
    f"{test_metrics_05['auroc']:.4f}"
)

print(
    f"AUPRC             : "
    f"{test_metrics_05['auprc']:.4f}"
)


print()
print(
    f"TEST RESULTS — "
    f"VALIDATION-OPTIMIZED THRESHOLD "
    f"({best_threshold:.2f})"
)

print("-" * 70)

print(
    f"Accuracy          : "
    f"{test_metrics_optimal['accuracy']:.4f}"
)

print(
    f"Precision         : "
    f"{test_metrics_optimal['precision']:.4f}"
)

print(
    f"Recall/Sensitivity: "
    f"{test_metrics_optimal['recall']:.4f}"
)

print(
    f"Specificity       : "
    f"{test_metrics_optimal['specificity']:.4f}"
)

print(
    f"F1                : "
    f"{test_metrics_optimal['f1']:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{test_metrics_optimal['balanced_accuracy']:.4f}"
)

print(
    f"AUROC             : "
    f"{test_metrics_optimal['auroc']:.4f}"
)

print(
    f"AUPRC             : "
    f"{test_metrics_optimal['auprc']:.4f}"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_prediction_df = pd.DataFrame({

    "image_id":
        test_image_ids,

    "label":
        test_labels,

    "melanoma_probability":
        test_probabilities,

    "prediction_threshold_050":
        (
            test_probabilities
            >= 0.50
        ).astype(int),

    "prediction_optimal_threshold":
        (
            test_probabilities
            >= best_threshold
        ).astype(int),
})


test_prediction_df.to_csv(
    TEST_PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE TEST METRICS
# ============================================================

test_results = {

    "threshold_0_50":
        test_metrics_05,

    "validation_optimized_threshold":
        {
            "threshold":
                best_threshold,

            "metrics":
                test_metrics_optimal,
        },

    "test_loss":
        test_loss,

    "best_checkpoint":
        {
            "stage":
                int(
                    best_checkpoint[
                        "stage"
                    ]
                ),

            "epoch":
                int(
                    best_checkpoint[
                        "epoch"
                    ]
                ),

            "validation_auroc":
                float(
                    best_checkpoint[
                        "val_auroc"
                    ]
                ),
        },

    "dataset":
        {
            "train":
                len(train_df),

            "validation":
                len(val_df),

            "test":
                len(test_df),

            "train_melanoma":
                train_melanoma,

            "train_non_melanoma":
                train_non_melanoma,
        },
}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        test_results,
        f,
        indent=4
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    test_labels,

    (
        test_probabilities
        >= best_threshold
    ).astype(int),

    labels=[0, 1]
)


plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm
)

plt.title(
    "EfficientNetV2-S Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["Non-melanoma", "Melanoma"]
)

plt.yticks(
    [0, 1],
    ["Non-melanoma", "Melanoma"]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    test_labels,
    test_probabilities
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=(
        f"EfficientNetV2-S "
        f"(AUROC = "
        f"{test_metrics_05['auroc']:.4f})"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ROC_CURVE_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        test_labels,
        test_probabilities
    )
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    label=(
        f"EfficientNetV2-S "
        f"(AUPRC = "
        f"{test_metrics_05['auprc']:.4f})"
    )
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PR_CURVE_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE CONFIGURATION
# ============================================================

configuration = {

    "device":
        str(DEVICE),

    "cpu_threads":
        CPU_THREADS,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "head_epochs":
        HEAD_EPOCHS,

    "finetune_epochs":
        FINETUNE_EPOCHS,

    "head_learning_rate":
        HEAD_LEARNING_RATE,

    "backbone_learning_rate":
        BACKBONE_LEARNING_RATE,

    "classifier_learning_rate":
        CLASSIFIER_LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "early_stopping_patience":
        EARLY_STOPPING_PATIENCE,

    "gradient_clip_value":
        GRADIENT_CLIP_VALUE,

    "unfreeze_last_feature_blocks":
        UNFREEZE_LAST_FEATURE_BLOCKS,

    "random_seed":
        RANDOM_SEED,

    "positive_class_weight":
        POS_WEIGHT,

    "classification_input":
        "U-Net-generated lesion-focused images",

    "classification_model":
        "EfficientNetV2-S",

    "classification_task":
        "Binary melanoma vs non-melanoma",

    "threshold_selection":
        "Validation-set F1 optimization",

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
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EFFICIENTNETV2-S STAGE COMPLETE")
print("=" * 70)

print()
print("Best model:")
print(BEST_MODEL_FILE)

print()
print("Validation predictions:")
print(VAL_PREDICTIONS_FILE)

print()
print("Test predictions:")
print(TEST_PREDICTIONS_FILE)

print()
print("Test metrics:")
print(METRICS_FILE)

print()
print("Optimal threshold:")
print(THRESHOLD_FILE)

print()
print("Training history:")
print(HISTORY_FILE)

print()
print("ROC curve:")
print(ROC_CURVE_FILE)

print()
print("Precision-Recall curve:")
print(PR_CURVE_FILE)

print()
print("Confusion matrix:")
print(CONFUSION_MATRIX_FILE)

print()
print("=" * 70)
print("DONE")
print("=" * 70)