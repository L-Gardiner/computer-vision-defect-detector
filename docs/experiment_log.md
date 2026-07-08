# Experiment Log: Computer Vision Defect Detector

A lightweight running log of experiments for the NEU Surface Defect classification project.

| Date | Exp ID | Change / hypothesis | Key params | Metric(s) | Result vs baseline | Notes |
|---|---|---|---|---|---|---|
| 2026-07-08 | exp-001 | Baseline: ResNet18 transfer learning | backbone=resnet18, lr=1e-3, batch=32, epochs=20, patience=5 | Accuracy: 98.52%<br>Precision: 98.64%<br>Recall: 98.56%<br>F1: 98.59% | — | Reference point. Strong baseline. |
| 2026-07-08 | exp-002 | Data loader extended to JPG/PNG/BMP | image glob patterns updated | Same train/test split | 0% change | Required because Kaggle NEU-DET release uses `.jpg` instead of `.bmp`. |
| 2026-07-08 | exp-003 | Evaluate with real test labels and fixed per-class metrics | precision_recall_fscore_support with labels=[i] | Per-class precision fixed | Resolves ValueError on singleton test classes | Committed alongside trained artifacts. |

## Decisions

- **Chosen approach:** exp-001. ResNet18 with a two-layer custom head, frozen early layers, Adam optimizer, and early stopping achieved 98.5%+ on the held-out test set with no overfitting.
- **Discarded:** Larger backbones (ResNet50) and alternative optimizers were not explored because the small 1,800-image dataset already saturated ResNet18's capacity. Future work could test test-time augmentation and ensembling.
