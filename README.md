# Skin Cancer Detection & Classification Pipeline

An end-to-end deep learning framework for dermoscopic skin lesion analysis using the ISIC-2017 dataset. The pipeline integrates lesion segmentation via U-Net, classification with EfficientNetV2 and DINOv2 foundation models, decision-level fusion, and rigorous statistical evaluation.

---

## 🌟 Key Features

- **Lesion Segmentation (U-Net)**: Automated lesion boundary delineation to filter confounding background artifacts (hair, markers, vignetting).
- **Lesion-Focused Classification (EfficientNetV2)**: Fine-tuned EfficientNetV2-S on segmented/cropped lesion regions.
- **Ablation Study**: Direct comparison of classification performance with vs. without U-Net segmentation masking.
- **Foundation Model Feature Extraction (DINOv2)**: Leverages self-supervised DINOv2 vision representations for dermoscopic lesion classification.
- **Model Fusion**: Weighted probability ensemble combining convolutional (EfficientNetV2) and vision transformer (DINOv2) architectures.
- **Statistical Significance Testing**: Bootstrap AUROC difference analysis with 95% confidence intervals and p-values.

---

## 📁 Repository Structure

```text
├── classification/
│   ├── confusion_matrices/         # Confusion matrix plots and counts
│   ├── dinov2/                     # DINOv2 training metrics and predictions
│   ├── efficientnetv2/             # EfficientNetV2 (with U-Net) results & plots
│   ├── efficientnetv2_ablation/    # EfficientNetV2 (without U-Net) results & plots
│   ├── fusion/                     # Model ensemble metrics and thresholds
│   └── statistical_comparison/     # Paired test bootstrap AUROC comparisons
├── segmentation/
│   ├── results/                    # U-Net configuration, metrics, and history
│   └── visualizations/             # Qualitative segmentation and publication plots
├── splits/                         # Stratified train, val, test CSV splits
├── visual_checks/                  # Dataset and mask alignment verification
├── .gitignore                      # Excludes large raw data, model weights, and crops
├── compare_auroc.py                # Statistical AUROC bootstrap evaluation
├── create_splits.py                # Stratified train/val/test split generator
├── dataset_metadata.csv            # Processed metadata mapping
├── fuse_models.py                  # Multi-model prediction fusion
├── generate_confusion_matrices.py  # Publication-ready confusion matrices
├── make_publication_segmentation_figure.py # Multi-panel segmentation figures
├── prepare_dataset.py              # Dataset verification and pairing
├── train_dinov2.py                 # DINOv2 feature extraction & classification
├── train_efficientnet.py           # EfficientNetV2 training with lesion-focused inputs
├── train_efficientnet_ablation.py  # EfficientNetV2 baseline training (no segmentation)
├── train_unet.py                   # U-Net segmentation training & crop generation
└── visualize_masks.py              # Mask inspection utilities
```

---

## 🚀 Pipeline Workflow

1. **Dataset Preparation & Splits**
   - Verify image-mask alignment and generate stratified splits:
     ```bash
     python prepare_dataset.py
     python create_splits.py
     ```

2. **Lesion Segmentation**
   - Train U-Net on ISIC-2017 masks and generate focused lesion crops:
     ```bash
     python train_unet.py
     ```

3. **Classification**
   - Train EfficientNetV2 on segmented lesions:
     ```bash
     python train_efficientnet.py
     ```
   - Train EfficientNetV2 baseline (ablation without segmentation):
     ```bash
     python train_efficientnet_ablation.py
     ```
   - Extract DINOv2 representations and train classification head:
     ```bash
     python train_dinov2.py
     ```

4. **Fusion & Evaluation**
   - Search optimal weights and ensemble predictions:
     ```bash
     python fuse_models.py
     ```
   - Run bootstrap statistical hypothesis testing:
     ```bash
     python compare_auroc.py
     ```
   - Generate summary figures and confusion matrices:
     ```bash
     python generate_confusion_matrices.py
     python make_publication_segmentation_figure.py
     ```

---

## 📊 Results Summary

Performance metrics, ROC/PR curves, confusion matrices, and detailed threshold evaluations are archived in the `classification/` directory.

---

## 🔒 Data & Weights Notice

In accordance with best practices for version control:
- Raw image data (`ISIC-2017_Training_Data/`, `ISIC-2017_Training_Part1_GroundTruth/`) and generated cropped images (`segmentation/lesion_focused/`, `segmentation/predicted_masks/`) are excluded via `.gitignore`.
- Model checkpoints (`*.pth`, `*.pt`) are omitted from this repository.
