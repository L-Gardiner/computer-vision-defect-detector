# Dataset: NEU Surface Defect Database

## Overview

The **NEU Surface Defect Database** is a public dataset from Northeast University (China) containing images of six types of surface defects on steel plates. This project uses it to demonstrate transfer learning for industrial quality control.

## Source

- **Kaggle dataset**: [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- **IEEE DataPort**: [NEU-DET](https://ieee-dataport.org/documents/neu-det)
- **License**: Public research dataset (cite if publishing)
- **Citation**: Luo et al., "Automated Visual Surface Inspection Using Deep Learning", 2016

## Classes (6 types)

1. **Crazing** - Fine cracks on the surface
2. **Inclusion** - Foreign particles embedded in the surface
3. **Patches** - Discolored or worn patches
4. **Pitted Surface** - Small pits or holes
5. **Rolled-in Scale** - Oxidized scale rolled into the surface
6. **Scratches** - Linear scratches or marks

## Data organization

```
data/
├── raw/
│   ├── crazing/           # ~300 images
│   ├── inclusion/         # ~300 images
│   ├── patches/           # ~300 images
│   ├── pitted_surface/    # ~300 images
│   ├── rolled-in_scale/   # ~300 images
│   └── scratches/         # ~300 images
└── processed/             # (generated during training)
```

## Download instructions

### Option 1: Download from Kaggle (recommended)

1. **Create a Kaggle account** (free): https://www.kaggle.com/
2. **Download the dataset**:
   - Go to: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
   - Click "Download" button
   - Extract the ZIP file

3. **Organize files**:
```bash
# After downloading and extracting, organize into class folders:
cd /path/to/computer-vision-defect-detector-2/data/raw

# Create class directories
mkdir -p crazing inclusion patches pitted_surface rolled-in_scale scratches

# Move images to respective folders (adjust paths based on extracted structure)
# The extracted folder should have images organized by class
```

### Option 2: Download from IEEE DataPort

1. Go to: https://ieee-dataport.org/documents/neu-det
2. Click "Download" and follow IEEE DataPort instructions
3. Extract and organize as above

### Option 3: Use Kaggle API (for automation)

```bash
# Install kaggle CLI
pip install kaggle

# Set up credentials (~/.kaggle/kaggle.json)
# Then download:
kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database

# Extract
unzip neu-surface-defect-database.zip -d data/raw/
```

## Image specifications

- **Format**: BMP (200×200 pixels)
- **Color**: Grayscale (converted to RGB in preprocessing)
- **Total images**: ~1,800 (300 per class)
- **Size on disk**: ~50 MB

## Data split

The project uses a **fixed-seed random split**:
- **Training**: 70% (~1,260 images)
- **Validation**: 15% (~270 images)
- **Test**: 15% (~270 images)

Seed is set to `42` for reproducibility.

## Preprocessing

Images are:
1. Loaded as RGB (converted from grayscale)
2. Resized to 224×224 (ResNet18 standard)
3. Normalized using ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

**Training augmentation** (applied only to training set):
- Random horizontal flip (50%)
- Random vertical flip (50%)
- Random rotation (±15°)
- Color jitter (brightness/contrast ±20%)

## License & attribution

This dataset is provided for research and educational purposes. If you publish results using this data, please cite:

```bibtex
@article{luo2016automated,
  title={Automated Visual Surface Inspection Using Deep Learning},
  author={Luo, Q. and Fang, X. and Liu, L. and Yang, C. and Sun, Y.},
  journal={Neurocomputing},
  year={2016}
}
```

## Notes

- **Images not committed**: To keep the repo small, raw images are gitignored. Download them separately from Kaggle or IEEE DataPort.
- **Reproducibility**: Fixed seed ensures consistent train/val/test splits across runs.
- **Class balance**: All classes have equal representation (no imbalance handling needed for this dataset).

---

For more information:
- **Kaggle dataset**: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
- **IEEE DataPort**: https://ieee-dataport.org/documents/neu-det
