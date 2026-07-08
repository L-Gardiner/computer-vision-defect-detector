# Model Card: Computer Vision Defect Detector

## Overview
- **Task:** Multi-class image classification
- **Model type:** ResNet18 (pretrained on ImageNet) with a custom 2-layer classification head
- **Version:** 0.1.0
- **Date:** 2026-07-08

## Intended use
- **Primary use case:** Demonstrate transfer learning for industrial steel surface defect detection (educational / portfolio project).
- **Out-of-scope / not intended for:** Production quality control without retraining on the target production line, and any safety-critical application.

## Data
- **Training data:** NEU Surface Defect Database train split (1,260 images, 210 per class), downloaded from Kaggle.
- **Evaluation data:** Held-out test split (270 images, 45 per class) from the same dataset.
- **Known biases / gaps:**
  - All classes are balanced in the dataset; real-world defect frequencies are usually imbalanced.
  - Images are 200×200 grayscale converted to RGB; the model may not generalize to different resolutions, lighting, or camera angles.
  - The dataset contains only hot-rolled steel strip defects; other materials or defect types are not represented.

## Metrics

| Metric | Value | Notes |
|---|---|---|
| Accuracy | 98.52% | On held-out test set (n=270) |
| Precision (macro) | 98.64% | Averaged across 6 classes |
| Recall (macro) | 98.56% | Averaged across 6 classes |
| F1 (macro) | 98.59% | Averaged across 6 classes |

### Per-class test metrics

| Class | Accuracy | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| crazing | 100.00% | 100.00% | 100.00% | 100.00% | 51 |
| inclusion | 97.56% | 100.00% | 97.56% | 98.77% | 41 |
| patches | 100.00% | 100.00% | 100.00% | 100.00% | 42 |
| pitted_surface | 96.15% | 100.00% | 96.15% | 98.04% | 52 |
| rolled-in_scale | 100.00% | 100.00% | 100.00% | 100.00% | 41 |
| scratches | 97.67% | 100.00% | 97.67% | 98.82% | 43 |

## Evaluation method
- **Split strategy:** 70/15/15 train/validation/test split with a fixed random seed (42) using `torch.utils.data.random_split`.
- **Validation approach:** Early stopping based on validation loss with patience=5; the best validation checkpoint is evaluated on the test set exactly once.
- **Baseline compared against:** Random guessing (16.7% accuracy for 6 classes) and an untrained ResNet18 head.

## Limitations & ethical considerations
- **Failure modes:** May fail under novel defect types, different steel grades, poor lighting, rotation beyond 15°, or image resolutions far from 200×200.
- **Fairness notes:** Not applicable — this is a materials-defect classification task with no human subjects.
- **Risks:** A false negative could send defective material downstream; a false positive could waste material. The default 0.5 threshold should be tuned for the specific production cost function.

## Caveats
- This is a portfolio / educational implementation, not a certified production system.
- The reported metrics are on a single public benchmark; performance should be re-validated on the target production line before deployment.
- Inference has not been benchmarked on GPU or at high throughput / large batch sizes.
- Grad-CAM provides a coarse attention map and should not be treated as a precise defect localization or segmentation mask.
