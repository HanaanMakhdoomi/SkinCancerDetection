from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(r"E:\SkinCancer")

SEGMENTED_FILE = BASE_DIR / "classification" / "efficientnetv2" / "results" / "predictions" / "test_predictions.csv"
ABLATION_FILE = BASE_DIR / "classification" / "efficientnetv2_ablation" / "results" / "predictions" / "test_predictions.csv"

OUTPUT_DIR = BASE_DIR / "classification" / "statistical_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_ITERATIONS = 10000
PERMUTATION_ITERATIONS = 10000
RANDOM_SEED = 42

print("=" * 70)
print("STATISTICAL COMPARISON OF EFFICIENTNETV2-S")
print("=" * 70)
print()
print("Proposed : U-Net lesion segmentation + EfficientNetV2-S")
print("Ablation : Original image + EfficientNetV2-S")
print(f"Bootstrap iterations   : {BOOTSTRAP_ITERATIONS:,}")
print(f"Permutation iterations : {PERMUTATION_ITERATIONS:,}")
print(f"Random seed            : {RANDOM_SEED}")
print()

def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} prediction file not found:\n{path}")

    df = pd.read_csv(path)
    required = ["image_id", "label", "melanoma_probability"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{model_name} prediction file is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    result = df[required].copy()
    result["image_id"] = result["image_id"].astype(str)
    result["label"] = result["label"].astype(int)
    result["melanoma_probability"] = result["melanoma_probability"].astype(float)

    if result["image_id"].duplicated().any():
        raise ValueError(f"{model_name} contains duplicate image IDs.")

    if not result["label"].isin([0, 1]).all():
        raise ValueError(f"{model_name} labels must be binary 0/1.")

    if result["melanoma_probability"].isna().any():
        raise ValueError(f"{model_name} contains missing probabilities.")

    if not np.isfinite(result["melanoma_probability"].to_numpy()).all():
        raise ValueError(f"{model_name} contains non-finite probabilities.")

    return result

segmented = load_predictions(SEGMENTED_FILE, "U-Net + EfficientNetV2-S")
ablation = load_predictions(ABLATION_FILE, "Ablation EfficientNetV2-S")

merged = segmented.merge(
    ablation,
    on="image_id",
    how="inner",
    suffixes=("_segmented", "_ablation")
)

if len(merged) != len(segmented) or len(merged) != len(ablation):
    raise ValueError("The two test prediction files do not contain exactly the same image IDs.")

if not np.array_equal(
    merged["label_segmented"].to_numpy(),
    merged["label_ablation"].to_numpy()
):
    raise ValueError("Labels differ between the two prediction files.")

merged = merged.sort_values("image_id").reset_index(drop=True)

y = merged["label_segmented"].to_numpy(dtype=int)
p_segmented = merged["melanoma_probability_segmented"].to_numpy(dtype=float)
p_ablation = merged["melanoma_probability_ablation"].to_numpy(dtype=float)

n_samples = len(merged)
n_positive = int((y == 1).sum())
n_negative = int((y == 0).sum())

print("Paired test set verified.")
print(f"Samples       : {n_samples}")
print(f"Melanoma      : {n_positive}")
print(f"Non-melanoma  : {n_negative}")
print()

auc_segmented = roc_auc_score(y, p_segmented)
auc_ablation = roc_auc_score(y, p_ablation)
auprc_segmented = average_precision_score(y, p_segmented)
auprc_ablation = average_precision_score(y, p_ablation)
observed_difference = auc_segmented - auc_ablation

print("=" * 70)
print("POINT ESTIMATES")
print("=" * 70)
print(f"U-Net + EfficientNetV2-S AUROC : {auc_segmented:.4f}")
print(f"Ablation AUROC                 : {auc_ablation:.4f}")
print(f"AUROC difference               : {observed_difference:.4f}")
print(f"U-Net + EfficientNetV2-S AUPRC : {auprc_segmented:.4f}")
print(f"Ablation AUPRC                 : {auprc_ablation:.4f}")
print()

print("=" * 70)
print("PAIRED STRATIFIED BOOTSTRAP")
print("=" * 70)

rng = np.random.default_rng(RANDOM_SEED)
positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]
bootstrap_differences = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)

for i in range(BOOTSTRAP_ITERATIONS):
    sampled_positive = rng.choice(positive_indices, size=len(positive_indices), replace=True)
    sampled_negative = rng.choice(negative_indices, size=len(negative_indices), replace=True)
    indices = np.concatenate([sampled_positive, sampled_negative])

    auc_seg = roc_auc_score(y[indices], p_segmented[indices])
    auc_abl = roc_auc_score(y[indices], p_ablation[indices])
    bootstrap_differences[i] = auc_seg - auc_abl

    if (i + 1) % 1000 == 0 or i == 0:
        print(
            f"\rBootstrap progress: {i + 1:,}/{BOOTSTRAP_ITERATIONS:,}",
            end="",
            flush=True
        )

print()
ci_low, ci_high = np.percentile(bootstrap_differences, [2.5, 97.5])

left_tail = np.mean(bootstrap_differences <= 0.0)
right_tail = np.mean(bootstrap_differences >= 0.0)
bootstrap_p_value = min(1.0, 2.0 * min(left_tail, right_tail))

print(f"AUROC difference (segmented - ablation): {observed_difference:.4f}")
print(f"95% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Bootstrap two-sided p approximation: {bootstrap_p_value:.4f}")
print()

print("=" * 70)
print("PAIRED PERMUTATION TEST")
print("=" * 70)
print("Scores are randomly swapped within each test image under the null hypothesis.")

perm_rng = np.random.default_rng(RANDOM_SEED + 1)
permutation_differences = np.empty(PERMUTATION_ITERATIONS, dtype=float)

for i in range(PERMUTATION_ITERATIONS):
    swap_mask = perm_rng.integers(0, 2, size=n_samples).astype(bool)

    perm_left = np.where(swap_mask, p_ablation, p_segmented)
    perm_right = np.where(swap_mask, p_segmented, p_ablation)

    perm_auc_left = roc_auc_score(y, perm_left)
    perm_auc_right = roc_auc_score(y, perm_right)
    permutation_differences[i] = perm_auc_left - perm_auc_right

    if (i + 1) % 1000 == 0 or i == 0:
        print(
            f"\rPermutation progress: {i + 1:,}/{PERMUTATION_ITERATIONS:,}",
            end="",
            flush=True
        )

print()

permutation_p_value = (
    np.sum(np.abs(permutation_differences) >= abs(observed_difference)) + 1
) / (PERMUTATION_ITERATIONS + 1)

print(f"Observed AUROC difference: {observed_difference:.4f}")
print(f"Permutation two-sided p-value: {permutation_p_value:.4f}")
print()

ci_interpretation = (
    "The 95% bootstrap confidence interval excludes zero."
    if (ci_low > 0 or ci_high < 0)
    else
    "The 95% bootstrap confidence interval includes zero."
)

p_interpretation = (
    "The paired permutation test is statistically significant at the 0.05 level."
    if permutation_p_value < 0.05
    else
    "The paired permutation test is not statistically significant at the 0.05 level."
)

print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print(ci_interpretation)
print(p_interpretation)
print()

paired_file = OUTPUT_DIR / "paired_test_predictions.csv"
bootstrap_file = OUTPUT_DIR / "bootstrap_auroc_differences.csv"
results_file = OUTPUT_DIR / "statistical_comparison.json"

merged.to_csv(paired_file, index=False)

pd.DataFrame({
    "auroc_difference": bootstrap_differences
}).to_csv(bootstrap_file, index=False)

results = {
    "comparison": {
        "proposed_model": "U-Net lesion segmentation + EfficientNetV2-S",
        "ablation_model": "Original image + EfficientNetV2-S",
    },
    "test_set": {
        "samples": n_samples,
        "melanoma": n_positive,
        "non_melanoma": n_negative,
    },
    "point_estimates": {
        "segmented_auroc": float(auc_segmented),
        "ablation_auroc": float(auc_ablation),
        "auroc_difference_segmented_minus_ablation": float(observed_difference),
        "segmented_auprc": float(auprc_segmented),
        "ablation_auprc": float(auprc_ablation),
    },
    "bootstrap": {
        "iterations": BOOTSTRAP_ITERATIONS,
        "random_seed": RANDOM_SEED,
        "difference_ci_95_percent": [float(ci_low), float(ci_high)],
        "two_sided_p_approximation": float(bootstrap_p_value),
    },
    "paired_permutation_test": {
        "iterations": PERMUTATION_ITERATIONS,
        "random_seed": RANDOM_SEED + 1,
        "two_sided_p_value": float(permutation_p_value),
    },
    "interpretation": {
        "confidence_interval": ci_interpretation,
        "permutation_test": p_interpretation,
    },
}

with open(results_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("=" * 70)
print("FINAL STATISTICAL SUMMARY")
print("=" * 70)
print(f"U-Net + EfficientNetV2-S AUROC : {auc_segmented:.4f}")
print(f"Without U-Net AUROC            : {auc_ablation:.4f}")
print(f"Absolute AUROC improvement     : {observed_difference:+.4f}")
print(f"95% Bootstrap CI               : [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Paired permutation p-value     : {permutation_p_value:.4f}")
print()
print("Saved:")
print(f"  {results_file}")
print(f"  {paired_file}")
print(f"  {bootstrap_file}")
print()
print("=" * 70)
print("STATISTICAL COMPARISON COMPLETE")
print("=" * 70)
