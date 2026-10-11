from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(r"E:\SkinCancer")
OUTPUT_DIR = BASE_DIR / "classification" / "confusion_matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Non-melanoma", "Melanoma"]
COLOR_MIN, COLOR_MAX = 0, 244

EXPERIMENTS = [
    {
        "key": "efficientnetv2_unet",
        "title": "EfficientNetV2-S + U-Net",
        "csv": BASE_DIR / "classification" / "efficientnetv2" / "results" / "predictions" / "test_predictions.csv",
        "prob_cols": ["melanoma_probability", "probability"],
        "threshold": 0.37,
        "expected": [[227, 17], [34, 22]],
    },
    {
        "key": "dinov2",
        "title": "DINOv2 ViT-S/14",
        "csv": BASE_DIR / "classification" / "dinov2" / "results" / "test_predictions.csv",
        "prob_cols": ["probability", "melanoma_probability"],
        "threshold": 0.69,
        "expected": [[210, 34], [26, 30]],
    },
    {
        "key": "fusion",
        "title": "Probability Fusion",
        "csv": BASE_DIR / "classification" / "fusion" / "test_fusion_predictions.csv",
        "prob_cols": ["fused_probability", "probability"],
        "threshold": 0.40,
        "expected": [[227, 17], [34, 22]],
    },
    {
        "key": "efficientnetv2_without_unet",
        "title": "EfficientNetV2-S without U-Net",
        "csv": BASE_DIR / "classification" / "efficientnetv2_ablation" / "results" / "predictions" / "test_predictions.csv",
        "prob_cols": ["melanoma_probability", "probability"],
        "threshold": 0.22,
        "expected": [[220, 24], [30, 26]],
    },
]


def load_one(exp):
    path = exp["csv"]
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found:\n{path}")
    df = pd.read_csv(path)
    for col in ["image_id", "label"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing '{col}'. Columns: {list(df.columns)}")
    prob_col = next((c for c in exp["prob_cols"] if c in df.columns), None)
    if prob_col is None:
        raise ValueError(f"No probability column in {path}. Columns: {list(df.columns)}")

    y_true = df["label"].astype(int).to_numpy()
    prob = df[prob_col].astype(float).to_numpy()
    if not np.isfinite(prob).all():
        raise ValueError(f"Invalid probabilities in {path}")

    y_pred = (prob >= exp["threshold"]).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if not np.array_equal(cm, np.asarray(exp["expected"])):
        warnings.warn(
            f"{exp['title']}: computed matrix differs from expected counts.\n"
            f"Computed:\n{cm}\nExpected:\n{np.asarray(exp['expected'])}"
        )

    print(f"\n{exp['title']} | threshold={exp['threshold']:.2f} | source column={prob_col}")
    print(cm)
    print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    return {**exp, "cm": cm}


def draw_matrix(ax, cm, title):
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues,
                   vmin=COLOR_MIN, vmax=COLOR_MAX)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=9)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, rotation=12, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    midpoint = (COLOR_MIN + COLOR_MAX) / 2
    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            ax.text(j, i, f"{value}", ha="center", va="center",
                    color="white" if value > midpoint else "black",
                    fontsize=14, fontweight="bold")
    return im


results = [load_one(exp) for exp in EXPERIMENTS]

# Individual matrix figures in a blue heatmap style.
for exp in results:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = draw_matrix(ax, exp["cm"],
                     f"{exp['title']}\nThreshold = {exp['threshold']:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Number of images")
    fig.tight_layout()
    dest = OUTPUT_DIR / f"confusion_matrix_{exp['key']}.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {dest}")

# Combined 2-by-2 four-panel figure.
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, exp in zip(axes.flat, results):
    draw_matrix(ax, exp["cm"],
                f"{exp['title']}\nThreshold = {exp['threshold']:.2f}")

sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
sm.set_clim(COLOR_MIN, COLOR_MAX)
fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.82, pad=0.025,
             label="Number of images")
fig.suptitle("Confusion Matrices for Extended ISIC 2017 Classification Experiments",
             fontsize=15, fontweight="bold", y=0.99)
fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.91,
                    wspace=0.35, hspace=0.40)

combined = OUTPUT_DIR / "confusion_matrices_all_models.png"
fig.savefig(combined, dpi=300, bbox_inches="tight")
plt.close(fig)

summary = []
for exp in results:
    cm = exp["cm"]
    summary.append({
        "model": exp["title"],
        "threshold": exp["threshold"],
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    })
counts_file = OUTPUT_DIR / "confusion_matrix_counts.csv"
pd.DataFrame(summary).to_csv(counts_file, index=False)

print("\n" + "=" * 68)
print("CONFUSION MATRICES CREATED")
print("=" * 68)
print(f"Output folder : {OUTPUT_DIR}")
print(f"Combined image: {combined}")
print(f"Counts CSV   : {counts_file}")
