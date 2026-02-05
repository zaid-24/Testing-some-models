# RIVA Cell Detection - YOLOv11 Pipeline

Automated detection and classification of cells in Pap smear images using YOLOv11, classifying into 8 Bethesda categories.

## 🎯 Task Overview

- **Task**: Detection + Classification
- **Classes**: 8 Bethesda categories (NILM, ENDO, INFL, ASCUS, LSIL, HSIL, ASCH, SCC)
- **Metric**: Mean Average Precision (mAP@[50:95])
- **Model**: YOLOv11-Large (Ultralytics)

## 📁 Project Structure

```
ISBI_A/
├── dataset/
│   ├── annotations/annotations/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── sample_submission.csv
│   └── images/images/
│       ├── train/   (828 images)
│       ├── val/     (131 images)
│       └── test/    (81 images)
├── data/
│   ├── labels/      (YOLO format - generated)
│   │   ├── train/
│   │   └── val/
│   └── riva.yaml    (Dataset config)
├── scripts/
│   ├── convert_annotations.py
│   ├── analyze_dataset.py
│   ├── train.py
│   ├── inference.py
│   └── visualize_predictions.py
├── runs/            (Training outputs)
├── outputs/         (Analysis & inference results)
├── run.py           (Main pipeline runner)
├── requirements.txt
└── README.md
```
### ✅ Project Files
- [ ] All scripts in `scripts/` folder
- [ ] Dataset in `dataset/` folder
- [ ] `data/riva.yaml` configured
- [ ] `requirements.txt` present
- [ ] No test outputs (trained_models/, runs/, outputs/)


## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Convert CSV annotations to YOLO format
python run.py convert

# (Optional) Analyze dataset statistics
python run.py analyze
```

### 3. Train Model

```bash
# Option A: Test pipeline on laptop (4GB GPU)
# Uses: yolo11n, 640px, 3 epochs, batch=4
python run.py train --mode test

# Option B: Multi-Scale Training (⭐ RECOMMENDED)
# Progressive resolution: 640px → 896px → 1024px
# Stage 1: 640px (100 epochs) - Learn coarse features fast
# Stage 2: 896px (150 epochs) - Refine with medium resolution
# Stage 3: 1024px (150 epochs) - Fine details for ASCUS/ASCH
# Expected: +2-4% mAP improvement over single-resolution
python run.py train --mode multiscale

# Option C: Focal Loss Training (for Class Imbalance)
# Uses: yolo11l, 1024px, 400 epochs, batch=6
# Higher cls weight handles severe class imbalance
python run.py train --mode focal

# Option D: ADH Training (Attention Decoupled Head)
# Uses: yolo11l, 1024px, 400 epochs, batch=6
# Higher box weight (10.0) for better localization
python run.py train --mode adh

# Option E: Full training (extreme augmentation baseline)
# Uses: yolo11l, 1024px, 500 epochs, batch=6
python run.py train --mode full

# Resume interrupted training
python run.py train --mode focal --resume
```

**Training Mode Summary**:
| Mode | Focus | Best For |
|------|-------|----------|
| `multiscale` | ⭐ Progressive resolution training | Coarse-to-fine learning, better final accuracy |
| `focal` | Class imbalance (higher cls weight) | Minority class detection (ASCUS, ASCH) |
| `adh` | Localization precision | Better bounding boxes, mAP@75 |
| `full` | Extreme augmentation | Baseline comparison |

### 4. Generate Submission

```bash
# Run inference on test set
python run.py infer

# With custom settings
python run.py infer --conf 0.25 --iou 0.45
```

### 5. Visualize Results

```bash
# Visualize ground truth
python run.py visualize --split val --source csv

# Visualize YOLO format labels
python run.py visualize --split train --source yolo

# Visualize predictions
python run.py visualize --source predictions
```

## ⚙️ Configuration

### Training Modes

| Setting | Test Mode (Laptop) | Full Mode (Workstation) |
|---------|-------------------|------------------------|
| Model | yolo11n (Nano) | yolo11l (Large) |
| Image Size | 640×640 | 1024×1024 |
| Batch Size | 4 | 8 |
| Epochs | 3 | 300 |
| Early Stop | 2 epochs | 50 epochs |
| GPU Memory | ~4GB | ~10GB |

### Data Augmentation

Full training uses augmentation inspired by Trans-YOLOv5 paper:
- Mosaic: 1.0
- Mixup: 0.5
- HSV: h=0.5, s=0.7, v=0.4
- Rotation: ±5°
- Scale: 0.3
- Shear: ±5°
- Flip: Horizontal & Vertical (0.5)
- Label Smoothing: 0.1

### Class Mapping

| ID | Class | Description |
|----|-------|-------------|
| 0 | NILM | Negative for Intraepithelial Lesion or Malignancy |
| 1 | ENDO | Endocervical cells |
| 2 | INFL | Inflammatory cells |
| 3 | ASCUS | Atypical Squamous Cells of Undetermined Significance |
| 4 | LSIL | Low-grade Squamous Intraepithelial Lesion |
| 5 | HSIL | High-grade Squamous Intraepithelial Lesion |
| 6 | ASCH | Atypical Squamous Cells, cannot exclude HSIL |
| 7 | SCC | Squamous Cell Carcinoma |

## 📊 Expected Outputs

### Training Outputs (`runs/detect/<run_name>/`)
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Latest checkpoint
- `results.csv` - Training metrics
- `confusion_matrix.png` - Class confusion matrix
- `PR_curve.png` - Precision-Recall curve
- `results.png` - Training curves

### Submission (`outputs/inference/`)
- `submission_YYYYMMDD_HHMMSS.csv` - Competition submission file
- `predictions/images/` - Annotated test images

## 🔧 Troubleshooting

### CUDA Out of Memory
Reduce batch size in `scripts/train.py`:
```python
# In get_full_config():
'batch': 4,  # Reduce from 8 to 4
```

### Labels Not Found
Ensure you run the convert step first:
```bash
python run.py convert
```

### Symlink Issues on Windows
If symlinks fail (requires admin), the script will copy files instead.

## 📈 Baseline Results

After training with vanilla YOLOv11-Large:
- mAP@50: ~XX.X%
- mAP@50-95: ~XX.X%

*(Results will be populated after training)*

## 🔬 References

- [Trans-YOLOv5 Paper](paper/Trans-YOLOv5_paper.md) - Inspiration for data augmentation and training strategies
- [Ultralytics YOLOv11](https://docs.ultralytics.com/) - Model documentation
- [RIVA Dataset](https://www.example.com) - Competition dataset

## 📝 License

This project is for academic/competition use only.
