import os
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"E:\SkinCancer"

SPLIT_DIR = os.path.join(ROOT, "splits")
LESION_DIR = os.path.join(ROOT, "segmentation", "lesion_focused")

OUT_DIR = os.path.join(ROOT, "classification", "dinov2")
FEATURE_DIR = os.path.join(OUT_DIR, "features")
MODEL_DIR = os.path.join(OUT_DIR, "models")
RESULT_DIR = os.path.join(OUT_DIR, "results")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 1e-3

DEVICE = torch.device("cpu")

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(min(8, os.cpu_count() or 1))

print("=" * 70)
print("DINOv2 ViT-S/14 CLASSIFICATION PIPELINE")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print()

# ============================================================
# DINOv2 PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# ============================================================
# DATASET
# ============================================================

class LesionDataset(Dataset):

    def __init__(self, dataframe, split):
        self.df = dataframe.reset_index(drop=True)
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = row["image_id"]
        label = int(row["label"])

        split_dir = os.path.join(LESION_DIR, self.split)

        possible_paths = [
            os.path.join(split_dir, f"{image_id}_Lesion.jpg"),
            os.path.join(split_dir, f"{image_id}_Lesion.png"),
            os.path.join(split_dir, f"{image_id}.jpg"),
            os.path.join(split_dir, f"{image_id}.png"),
        ]

        image_path = next(
            (path for path in possible_paths if os.path.exists(path)),
            None
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Missing lesion-focused image for {image_id}. "
                f"Checked: {possible_paths}"
            )

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        return image, label, image_id


# ============================================================
# LOAD SPLITS
# ============================================================

def load_split(name):

    path = os.path.join(SPLIT_DIR, f"{name}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)

    # The frozen split files use "melanoma" as the target column.
    # Convert it to the generic "label" name expected by this script.
    if "label" in df.columns:
        pass
    elif "melanoma" in df.columns:
        df["label"] = (
            df["melanoma"]
            .astype(float)
            .astype(int)
        )
    else:
        raise ValueError(
            f"{name}.csv must contain either "
            f"'label' or 'melanoma'. Found: {list(df.columns)}"
        )

    return df

train_df = load_split("train")
val_df = load_split("val")
test_df = load_split("test")

print("Dataset sizes:")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")
print()

print("Class distribution:")
print(
    f"Train melanoma: {train_df['label'].sum()} / "
    f"{len(train_df)}"
)
print(
    f"Val melanoma:   {val_df['label'].sum()} / "
    f"{len(val_df)}"
)
print(
    f"Test melanoma:  {test_df['label'].sum()} / "
    f"{len(test_df)}"
)
print()

# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = LesionDataset(train_df, "train")
val_dataset = LesionDataset(val_df, "val")
test_dataset = LesionDataset(test_df, "test")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# ============================================================
# LOAD DINOv2
# ============================================================

print("Loading DINOv2 ViT-S/14...")

dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

dinov2 = dinov2.to(DEVICE)
dinov2.eval()

for parameter in dinov2.parameters():
    parameter.requires_grad = False

FEATURE_DIM = 384

print("DINOv2 loaded.")
print(f"Feature dimension: {FEATURE_DIM}")
print()

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def feature_file(split):
    return os.path.join(
        FEATURE_DIR,
        f"{split}_features.pt"
    )


@torch.no_grad()
def extract_features(loader, split):

    output_path = feature_file(split)

    # --------------------------------------------------------
    # Reuse previously extracted features
    # --------------------------------------------------------
    if os.path.exists(output_path):

        print(
            f"[{split.upper()}] Cached features found. "
            f"Loading instead of recomputing..."
        )

        data = torch.load(
            output_path,
            map_location="cpu"
        )

        print(
            f"[{split.upper()}] Loaded "
            f"{len(data['labels'])} samples."
        )

        return data

    print()
    print("-" * 70)
    print(f"Extracting DINOv2 features: {split}")
    print("-" * 70)

    all_features = []
    all_labels = []
    all_ids = []

    start_time = time.time()

    dinov2.eval()

    total_batches = len(loader)

    for batch_idx, (images, labels, image_ids) in enumerate(loader):

        images = images.to(DEVICE)

        # DINOv2 normalized CLS-token embedding
        outputs = dinov2.forward_features(images)

        features = outputs["x_norm_clstoken"]

        features = features.cpu()

        all_features.append(features)
        all_labels.append(labels.cpu())
        all_ids.extend(image_ids)

        elapsed = time.time() - start_time
        done = batch_idx + 1
        rate = elapsed / done
        remaining = rate * (total_batches - done)

        print(
            f"\r[{split.upper()}] "
            f"Batch {done}/{total_batches} | "
            f"Elapsed: {elapsed/60:.1f} min | "
            f"ETA: {remaining/60:.1f} min",
            end="",
            flush=True
        )

    print()

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    data = {
        "features": features,
        "labels": labels,
        "image_ids": all_ids
    }

    torch.save(data, output_path)

    print(
        f"[{split.upper()}] Saved features to:\n"
        f"{output_path}"
    )
    print(
        f"Feature tensor shape: {features.shape}"
    )

    return data


# ============================================================
# EXTRACT / LOAD FEATURES
# ============================================================

train_features = extract_features(
    train_loader,
    "train"
)

val_features = extract_features(
    val_loader,
    "val"
)

test_features = extract_features(
    test_loader,
    "test"
)

# ============================================================
# CLASSIFIER
# ============================================================

class DINOClassifier(nn.Module):

    def __init__(self, input_dim=384):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(x).squeeze(1)


model = DINOClassifier(FEATURE_DIM).to(DEVICE)

# ============================================================
# CLASS IMBALANCE
# ============================================================

train_labels = train_features["labels"]

negative_count = int((train_labels == 0).sum())
positive_count = int((train_labels == 1).sum())

pos_weight_value = negative_count / positive_count

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

print()
print("Classifier:")
print(model)
print()
print(
    f"Negative samples: {negative_count}"
)
print(
    f"Positive samples: {positive_count}"
)
print(
    f"Positive class weight: {pos_weight_value:.4f}"
)
print()

# ============================================================
# TRAINING DATA
# ============================================================

X_train = train_features["features"].float().to(DEVICE)
y_train = train_features["labels"].float().to(DEVICE)

X_val = val_features["features"].float().to(DEVICE)
y_val = val_features["labels"].float().to(DEVICE)

X_test = test_features["features"].float().to(DEVICE)
y_test = test_features["labels"].float().to(DEVICE)

# ============================================================
# EVALUATION
# ============================================================

def get_probabilities(model, X):

    model.eval()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


def find_best_threshold(y_true, probabilities):

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(0.10, 0.91, 0.01):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

best_val_auc = -1.0
best_epoch = 0
patience_counter = 0

history = []

best_model_path = os.path.join(
    MODEL_DIR,
    "best_dinov2_classifier.pth"
)

print("=" * 70)
print("TRAINING DINOv2 CLASSIFIER")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    logits = model(X_train)

    loss = criterion(
        logits,
        y_train
    )

    loss.backward()

    optimizer.step()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_probabilities = get_probabilities(
        model,
        X_val
    )

    val_labels_np = y_val.cpu().numpy()

    val_auc = roc_auc_score(
        val_labels_np,
        val_probabilities
    )

    val_auprc = average_precision_score(
        val_labels_np,
        val_probabilities
    )

    val_threshold, val_f1 = find_best_threshold(
        val_labels_np,
        val_probabilities
    )

    history.append({
        "epoch": epoch,
        "loss": float(loss.item()),
        "val_auroc": float(val_auc),
        "val_auprc": float(val_auprc),
        "val_f1": float(val_f1),
        "val_threshold": float(val_threshold)
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {loss.item():.4f} | "
        f"Val AUROC: {val_auc:.4f} | "
        f"Val AUPRC: {val_auprc:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Threshold: {val_threshold:.2f}"
    )

    # --------------------------------------------------------
    # Save best model based on validation AUROC
    # --------------------------------------------------------

    if val_auc > best_val_auc:

        best_val_auc = val_auc
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print()
            print(
                f"Early stopping at epoch {epoch}."
            )
            break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = os.path.join(
    RESULT_DIR,
    "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST DINOv2 CLASSIFIER")
print("=" * 70)

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

model.eval()

# ============================================================
# VALIDATION THRESHOLD
# ============================================================

val_probabilities = get_probabilities(
    model,
    X_val
)

val_labels_np = y_val.cpu().numpy()

optimal_threshold, optimal_val_f1 = find_best_threshold(
    val_labels_np,
    val_probabilities
)

threshold_path = os.path.join(
    RESULT_DIR,
    "optimal_threshold.json"
)

with open(threshold_path, "w") as f:

    json.dump(
        {
            "threshold": optimal_threshold,
            "validation_f1": optimal_val_f1,
            "best_validation_auroc": best_val_auc,
            "best_epoch": best_epoch
        },
        f,
        indent=4
    )

print(
    f"Best validation AUROC: {best_val_auc:.4f}"
)

print(
    f"Optimal validation threshold: "
    f"{optimal_threshold:.2f}"
)

print(
    f"Validation F1 at threshold: "
    f"{optimal_val_f1:.4f}"
)

# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

val_predictions_df = pd.DataFrame({
    "image_id": val_features["image_ids"],
    "label": val_labels_np.astype(int),
    "probability": val_probabilities
})

val_predictions_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "validation_predictions.csv"
    ),
    index=False
)

# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_probabilities = get_probabilities(
    model,
    X_test
)

test_labels_np = y_test.cpu().numpy()

test_predictions = (
    test_probabilities >= optimal_threshold
).astype(int)

accuracy = accuracy_score(
    test_labels_np,
    test_predictions
)

precision = precision_score(
    test_labels_np,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels_np,
    test_predictions,
    zero_division=0
)

specificity = recall_score(
    test_labels_np,
    test_predictions,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    test_labels_np,
    test_predictions,
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    test_labels_np,
    test_predictions
)

auroc = roc_auc_score(
    test_labels_np,
    test_probabilities
)

auprc = average_precision_score(
    test_labels_np,
    test_probabilities
)

tn, fp, fn, tp = confusion_matrix(
    test_labels_np,
    test_predictions
).ravel()

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions_df = pd.DataFrame({
    "image_id": test_features["image_ids"],
    "label": test_labels_np.astype(int),
    "probability": test_probabilities,
    "prediction": test_predictions
})

test_predictions_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "test_predictions.csv"
    ),
    index=False
)

# ============================================================
# SAVE TEST METRICS
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall_sensitivity": float(recall),
    "specificity": float(specificity),
    "f1_score": float(f1),
    "balanced_accuracy": float(balanced_accuracy),
    "auroc": float(auroc),
    "auprc": float(auprc),
    "threshold": float(optimal_threshold),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
    "best_validation_auroc": float(best_val_auc),
    "best_epoch": int(best_epoch)
}

metrics_path = os.path.join(
    RESULT_DIR,
    "test_metrics.json"
)

with open(metrics_path, "w") as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# ============================================================
# SAVE COMPLETE MODEL INFORMATION
# ============================================================

info = {
    "model": "DINOv2 ViT-S/14",
    "parameters": 22056576,
    "feature_dimension": FEATURE_DIM,
    "pretrained": True,
    "backbone_frozen": True,
    "classifier": "128-unit MLP",
    "dataset": "ISIC 2017",
    "input": "U-Net lesion-focused images",
    "seed": SEED,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "positive_class_weight": float(pos_weight_value)
}

with open(
    os.path.join(
        RESULT_DIR,
        "experiment_info.json"
    ),
    "w"
) as f:

    json.dump(
        info,
        f,
        indent=4
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("DINOv2 RESULTS")
print("=" * 70)

print(f"Best validation epoch : {best_epoch}")
print(f"Validation AUROC      : {best_val_auc:.4f}")
print(f"Validation threshold  : {optimal_threshold:.2f}")
print()

print(f"Test Accuracy         : {accuracy:.4f}")
print(f"Test Precision        : {precision:.4f}")
print(f"Test Recall           : {recall:.4f}")
print(f"Test Specificity      : {specificity:.4f}")
print(f"Test F1               : {f1:.4f}")
print(f"Test Balanced Acc.    : {balanced_accuracy:.4f}")
print(f"Test AUROC            : {auroc:.4f}")
print(f"Test AUPRC            : {auprc:.4f}")

print()
print("Confusion Matrix:")
print(
    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
)

print()
print("Saved:")
print(f"Features : {FEATURE_DIR}")
print(f"Model    : {best_model_path}")
print(f"Results  : {RESULT_DIR}")

print()
print("=" * 70)
print("DINOv2 PIPELINE COMPLETE")
print("=" * 70)