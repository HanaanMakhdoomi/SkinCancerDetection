import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

ROOT = r"E:\SkinCancer"

EFF_DIR = os.path.join(
    ROOT,
    "classification",
    "efficientnetv2",
    "results",
    "predictions"
)

DINO_DIR = os.path.join(
    ROOT,
    "classification",
    "dinov2",
    "results"
)

OUT_DIR = os.path.join(
    ROOT,
    "classification",
    "fusion"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file:\n{path}")

    df = pd.read_csv(path)

    # EfficientNetV2-S saves the probability as
    # "melanoma_probability", while DINOv2 uses "probability".
    if "melanoma_probability" in df.columns:
        probability_column = "melanoma_probability"
    elif "probability" in df.columns:
        probability_column = "probability"
    else:
        raise ValueError(
            f"{path} does not contain a probability column. "
            f"Found columns: {list(df.columns)}"
        )

    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{path} must contain 'image_id' and 'label'. "
            f"Found columns: {list(df.columns)}"
        )

    result = df[
        ["image_id", "label", probability_column]
    ].copy()

    result = result.rename(
        columns={probability_column: "probability"}
    )

    return result
eff_val = load_predictions(
    os.path.join(
        EFF_DIR,
        "validation_predictions.csv"
    )
)

eff_test = load_predictions(
    os.path.join(
        EFF_DIR,
        "test_predictions.csv"
    )
)

dino_val = load_predictions(
    os.path.join(
        DINO_DIR,
        "validation_predictions.csv"
    )
)

dino_test = load_predictions(
    os.path.join(
        DINO_DIR,
        "test_predictions.csv"
    )
)

# ============================================================
# MERGE BY IMAGE ID
# ============================================================

def merge_predictions(eff, dino, split_name):

    merged = eff.merge(
        dino,
        on="image_id",
        how="inner",
        suffixes=("_eff", "_dino")
    )

    if len(merged) != len(eff) or len(merged) != len(dino):
        raise ValueError(
            f"{split_name}: prediction files do not contain "
            f"the exact same image IDs."
        )

    if not np.array_equal(
        merged["label_eff"].values,
        merged["label_dino"].values
    ):
        raise ValueError(
            f"{split_name}: EfficientNet and DINO labels do not match."
        )

    merged["label"] = merged["label_eff"].astype(int)

    merged["eff_probability"] = merged[
        "probability_eff"
    ].astype(float)

    merged["dino_probability"] = merged[
        "probability_dino"
    ].astype(float)

    return merged[
        [
            "image_id",
            "label",
            "eff_probability",
            "dino_probability"
        ]
    ]


val = merge_predictions(
    eff_val,
    dino_val,
    "validation"
)

test = merge_predictions(
    eff_test,
    dino_test,
    "test"
)

print("=" * 70)
print("MODEL PROBABILITY FUSION")
print("=" * 70)

print(f"Validation samples: {len(val)}")
print(f"Test samples:       {len(test)}")
print()

# ============================================================
# VALIDATION FUSION WEIGHT SEARCH
# ============================================================

y_val = val["label"].values

best_weight = None
best_val_auc = -1.0
best_val_auprc = -1.0

weight_results = []

# EfficientNet weight from 0.00 to 1.00
# DINO weight is automatically (1 - alpha)

for alpha in np.arange(0.00, 1.001, 0.01):

    fused_probability = (
        alpha * val["eff_probability"].values
        +
        (1.0 - alpha) * val["dino_probability"].values
    )

    auc = roc_auc_score(
        y_val,
        fused_probability
    )

    auprc = average_precision_score(
        y_val,
        fused_probability
    )

    weight_results.append({
        "efficientnet_weight": round(float(alpha), 2),
        "dinov2_weight": round(float(1.0 - alpha), 2),
        "validation_auroc": float(auc),
        "validation_auprc": float(auprc)
    })

    # Primary criterion: validation AUROC
    if auc > best_val_auc:
        best_val_auc = auc
        best_val_auprc = auprc
        best_weight = float(alpha)

weight_df = pd.DataFrame(weight_results)

weight_df.to_csv(
    os.path.join(
        OUT_DIR,
        "fusion_weight_search.csv"
    ),
    index=False
)

best_dino_weight = 1.0 - best_weight

print("Best validation fusion:")
print(
    f"EfficientNetV2-S weight: {best_weight:.2f}"
)
print(
    f"DINOv2 weight:           {best_dino_weight:.2f}"
)
print(
    f"Validation AUROC:        {best_val_auc:.4f}"
)
print(
    f"Validation AUPRC:        {best_val_auprc:.4f}"
)
print()

# ============================================================
# VALIDATION THRESHOLD SEARCH
# ============================================================

val_fused_probability = (
    best_weight * val["eff_probability"].values
    +
    best_dino_weight * val["dino_probability"].values
)

best_threshold = 0.50
best_f1 = -1.0

threshold_results = []

for threshold in np.arange(0.10, 0.91, 0.01):

    predictions = (
        val_fused_probability >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "validation_f1": float(f1)
    })

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(threshold)

pd.DataFrame(
    threshold_results
).to_csv(
    os.path.join(
        OUT_DIR,
        "fusion_threshold_search.csv"
    ),
    index=False
)

print(
    f"Optimal validation threshold: {best_threshold:.2f}"
)
print(
    f"Validation F1:                {best_f1:.4f}"
)
print()

# ============================================================
# FINAL TEST FUSION
# ============================================================

y_test = test["label"].values

test_fused_probability = (
    best_weight * test["eff_probability"].values
    +
    best_dino_weight * test["dino_probability"].values
)

test_predictions = (
    test_fused_probability >= best_threshold
).astype(int)

# ============================================================
# TEST METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

specificity = recall_score(
    y_test,
    test_predictions,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_predictions
)

auroc = roc_auc_score(
    y_test,
    test_fused_probability
)

auprc = average_precision_score(
    y_test,
    test_fused_probability
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions
).ravel()

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_output = pd.DataFrame({
    "image_id": test["image_id"],
    "label": y_test,
    "efficientnet_probability":
        test["eff_probability"],
    "dinov2_probability":
        test["dino_probability"],
    "fused_probability":
        test_fused_probability,
    "prediction":
        test_predictions
})

test_output.to_csv(
    os.path.join(
        OUT_DIR,
        "test_fusion_predictions.csv"
    ),
    index=False
)

# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "fusion_method": "weighted probability averaging",

    "efficientnetv2_weight": best_weight,
    "dinov2_weight": best_dino_weight,

    "validation_auroc": best_val_auc,
    "validation_auprc": best_val_auprc,
    "validation_f1": best_f1,
    "validation_threshold": best_threshold,

    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall_sensitivity": float(recall),
    "test_specificity": float(specificity),
    "test_f1": float(f1),
    "test_balanced_accuracy": float(balanced_accuracy),
    "test_auroc": float(auroc),
    "test_auprc": float(auprc),

    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp)
}

with open(
    os.path.join(
        OUT_DIR,
        "fusion_metrics.json"
    ),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# ============================================================
# SAVE HUMAN-READABLE SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "Model": "EfficientNetV2-S",
        "Test_AUROC": 0.8235,
        "Test_AUPRC": 0.5053
    },
    {
        "Model": "DINOv2 ViT-S/14",
        "Test_AUROC": 0.7965,
        "Test_AUPRC": 0.5182
    },
    {
        "Model": "Fused",
        "Test_AUROC": auroc,
        "Test_AUPRC": auprc
    }
])

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)

# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("=" * 70)
print("FUSION RESULTS")
print("=" * 70)

print(
    f"EfficientNetV2-S weight : {best_weight:.2f}"
)
print(
    f"DINOv2 weight           : {best_dino_weight:.2f}"
)
print(
    f"Classification threshold: {best_threshold:.2f}"
)

print()
print(f"Test Accuracy           : {accuracy:.4f}")
print(f"Test Precision          : {precision:.4f}")
print(f"Test Recall             : {recall:.4f}")
print(f"Test Specificity        : {specificity:.4f}")
print(f"Test F1                 : {f1:.4f}")
print(f"Test Balanced Accuracy  : {balanced_accuracy:.4f}")
print(f"Test AUROC              : {auroc:.4f}")
print(f"Test AUPRC              : {auprc:.4f}")

print()
print("Confusion Matrix:")
print(
    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
)

print()
print(f"Saved results to:")
print(OUT_DIR)

print()
print("=" * 70)
print("FUSION EXPERIMENT COMPLETE")
print("=" * 70)