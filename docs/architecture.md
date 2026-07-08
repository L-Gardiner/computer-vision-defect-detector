# Architecture: Computer Vision Defect Detector

## System overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interfaces                             │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit App (app.py)      │      FastAPI (api.py)           │
│  - Image upload              │      - POST /predict             │
│  - Real-time prediction      │      - JSON response             │
│  - Grad-CAM visualization    │      - Batch inference ready     │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  predict.py: Predictor class                                    │
│  - Load model from disk                                         │
│  - Preprocess image (resize, normalize)                         │
│  - Forward pass → logits                                        │
│  - Softmax → probabilities                                      │
│  - Return top-3 predictions                                     │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Explainability Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  explain.py: Grad-CAM                                           │
│  - Register forward/backward hooks on layer4                    │
│  - Compute gradients w.r.t. target class                        │
│  - Weight activations by gradients                              │
│  - Generate heatmap (H, W)                                      │
│  - Overlay on original image                                    │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Model Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│  ResNet18 Backbone (Pretrained on ImageNet)                     │
│  ├── Conv1 (7×7, stride 2)                                      │
│  ├── Layer1 (2 residual blocks, 64 channels)                    │
│  ├── Layer2 (2 residual blocks, 128 channels)                   │
│  ├── Layer3 (2 residual blocks, 256 channels)                   │
│  └── Layer4 (2 residual blocks, 512 channels) ← Grad-CAM hook  │
│                                                                  │
│  Custom Classification Head                                     │
│  ├── Global Average Pooling (512-dim)                           │
│  ├── Dropout(0.5)                                               │
│  ├── Linear(512 → 256)                                          │
│  ├── ReLU                                                        │
│  ├── Dropout(0.3)                                               │
│  └── Linear(256 → 6) [logits for 6 classes]                     │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Data Pipeline                                │
├─────────────────────────────────────────────────────────────────┤
│  data.py: NEUDefectDataset                                      │
│  ├── Load images from data/raw/{class_name}/*.{bmp,jpg,png}               │
│  ├── Apply transforms (resize, normalize)                       │
│  └── Return (image_tensor, label)                               │
│                                                                  │
│  Dataloaders (train/val/test)                                   │
│  ├── Train: shuffle=True, augment=True, batch_size=32           │
│  ├── Val:   shuffle=False, augment=False, batch_size=32         │
│  └── Test:  shuffle=False, augment=False, batch_size=32         │
└─────────────────────────────────────────────────────────────────┘
```

## Training pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                   train.py: Training Loop                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  For each epoch:                                                 │
│  ├─ Train phase:                                                 │
│  │  ├─ Load batch from train_loader                              │
│  │  ├─ Forward pass: logits = model(images)                      │
│  │  ├─ Compute loss: CrossEntropyLoss(logits, labels)            │
│  │  ├─ Backward pass: loss.backward()                            │
│  │  ├─ Optimizer step: Adam update                               │
│  │  └─ Accumulate train_loss                                     │
│  │                                                                │
│  ├─ Validation phase:                                            │
│  │  ├─ Load batch from val_loader                                │
│  │  ├─ Forward pass (no_grad)                                    │
│  │  ├─ Compute val_loss & val_accuracy                           │
│  │  └─ Check early stopping criterion                            │
│  │      ├─ If val_loss improved: save best_model.pt              │
│  │      └─ Else: increment patience counter                      │
│  │                                                                │
│  └─ If patience >= 5: break (early stopping)                     │
│                                                                   │
│  Output: training_history.json                                   │
│  ├─ train_loss: [epoch0, epoch1, ...]                            │
│  ├─ val_loss: [epoch0, epoch1, ...]                              │
│  └─ val_accuracy: [epoch0, epoch1, ...]                          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Evaluation pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                 evaluate.py: Test Evaluation                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Load best_model.pt                                              │
│  ├─ Set to eval mode                                             │
│  └─ Disable gradients                                            │
│                                                                   │
│  For each batch in test_loader:                                  │
│  ├─ Forward pass: logits = model(images)                         │
│  ├─ Get predictions: _, preds = torch.max(logits, 1)             │
│  └─ Accumulate predictions & labels                              │
│                                                                   │
│  Compute metrics:                                                │
│  ├─ Accuracy: (preds == labels).sum() / total                    │
│  ├─ Precision (macro): sklearn.metrics.precision_score           │
│  ├─ Recall (macro): sklearn.metrics.recall_score                 │
│  ├─ F1 (macro): sklearn.metrics.f1_score                         │
│  ├─ Per-class metrics: for each class i                          │
│  │  └─ accuracy, precision, recall, f1, support                  │
│  └─ Confusion matrix: sklearn.metrics.confusion_matrix           │
│                                                                   │
│  Visualizations:                                                 │
│  ├─ confusion_matrix.png: heatmap of predictions                 │
│  └─ test_metrics.json: all metrics as JSON                       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Configuration (config.py)

All hyperparameters and paths are centralized in `Settings`:

```python
class Settings:
    # Model
    backbone: str = "resnet18"
    num_classes: int = 6
    image_size: int = 224
    pretrained: bool = True

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 20
    early_stopping_patience: int = 5
    random_seed: int = 42

    # Paths
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    models_dir: str = "./models"

    # Device
    device: str = "mps"  # or "cuda", "cpu"
```

## Key design decisions

### 1. Transfer Learning
- **Why ResNet18?** Small, fast, good accuracy. Larger models (ResNet50) would be overkill for this dataset.
- **Why freeze early layers?** Early layers learn generic features (edges, textures); only the head needs to adapt to defect classification.
- **Why custom head?** Allows the model to learn task-specific features while leveraging pretrained weights.

### 2. Grad-CAM for explainability
- **Why Grad-CAM?** Lightweight, doesn't require retraining, produces human-interpretable heatmaps.
- **Why layer4?** High-level features are most relevant to the classification decision.
- **How it works:**
  1. Register hooks on layer4 to capture activations and gradients
  2. Forward pass with target class
  3. Backward pass to compute gradients
  4. Weight activations by gradients: `cam = sum(weights * activations)`
  5. ReLU to keep only positive contributions
  6. Normalize to [0, 1] and overlay on original image

### 3. Data discipline
- **Fixed seed (42)**: Ensures reproducible train/val/test splits across runs.
- **No data leakage**: Augmentation only on training set; validation/test use original images.
- **Proper split**: 70/15/15 is standard; prevents overfitting and gives honest test metrics.

### 4. Early stopping
- **Patience = 5**: Stop if validation loss doesn't improve for 5 epochs.
- **Prevents overfitting**: Especially important for small datasets.
- **Saves best model**: Only the checkpoint with lowest validation loss is kept.

## Inference flow (single image)

```
Image (JPG/PNG/BMP)
    ↓
Load & convert to RGB
    ↓
Resize to 224×224
    ↓
Normalize (ImageNet stats)
    ↓
Add batch dimension: (1, 3, 224, 224)
    ↓
Forward pass through model
    ↓
Logits (1, 6)
    ↓
Softmax → Probabilities (1, 6)
    ↓
Top-3 predictions
    ↓
Grad-CAM (optional)
    ├─ Register hooks
    ├─ Forward pass with target class
    ├─ Backward pass
    └─ Generate heatmap
    ↓
Return: {
    "predicted_class": "crazing",
    "confidence": 0.95,
    "top_3": [
        {"class": "crazing", "confidence": 0.95},
        {"class": "patches", "confidence": 0.04},
        {"class": "scratches", "confidence": 0.01}
    ]
}
```

## Testing strategy

- **Unit tests** (test_model.py): Model initialization, forward pass, save/load
- **Integration tests** (test_data.py): Dataset loading, transforms, dataloaders
- **Inference tests** (test_predict.py): Predictor class, prediction output format
- **API tests** (test_api.py): FastAPI endpoints, response schema
- **Smoke tests** (test_smoke.py): Package imports, config loading

**Coverage target**: Core logic (model, data, predict) at 90%+; app/api/train/evaluate at 0% (integration tested manually).

---

For more details, see the main `README.md` and individual module docstrings.
