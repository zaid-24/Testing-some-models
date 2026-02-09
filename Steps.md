## 🚀 Quick Start — Two-Stage Cell Detection Pipeline

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  INPUT IMAGE (1024×1024)                                     │
│                                                              │
│  ┌─── Stage 1: Binary Detector (CenterNet) ───────────────┐ │
│  │  ResNet-50 backbone (pretrained, stride 32)             │ │
│  │  → 3× Deconv Neck (stride 32 → 4)                      │ │
│  │  → 1-channel heatmap (binary: cell/no-cell)             │ │
│  │  → 2-channel offset (sub-pixel refinement)              │ │
│  │  Output: Cell center coordinates (x, y) + confidence    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                         ↓                                    │
│              Detected centers (x, y)                         │
│                         ↓                                    │
│  ┌─── Stage 2: Cell Classifier (EfficientNet-B2) ─────────┐ │
│  │  Crop 100×100 patch around each detected center         │ │
│  │  Resize to 224×224                                      │ │
│  │  EfficientNet-B2 (pretrained on ImageNet)               │ │
│  │  → 8-class softmax (Bethesda categories)                │ │
│  │  Output: Cell class + classification confidence         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                         ↓                                    │
│  Final: (x, y, 100, 100, det_conf × cls_conf, class)        │
└──────────────────────────────────────────────────────────────┘
```

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# IMPORTANT: Install PyTorch with CUDA FIRST
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Then install other dependencies (includes timm for EfficientNet)
pip install -r requirements.txt
```

**Verify Installation:**
```bash
python verify_installation.py

# Should show:
# [OK] PyTorch, torchvision, timm installed
# [OK] CUDA available
# [OK] Stage 1 Detector test passed
# [OK] Stage 2 Classifier test passed
```

### 2. Analyze Dataset (Optional)

```bash
python run.py analyze
```

**Key Dataset Insights:**
- **828 train images** / 131 val / 81 test
- **~13,267 total annotations** (cell instances)
- **Severe class imbalance**: 21:1 ratio (INFL vs ASCUS)
- **ALL boxes are 100×100 pixels** — point detection problem!

### 3. Train Models

#### Quick Test (Both Stages)
```bash
# Verify pipeline works (5 epochs each stage)
python run.py train --mode test --stage both
```

#### Full Training (Recommended)
```bash
# Train both stages sequentially
python run.py train --mode full --stage both

# Or train stages separately:
python run.py train --mode full --stage detect           # Stage 1 only
python run.py train --mode full --stage classify         # Stage 2 only

# Use different backbones:
python run.py train --mode full --stage detect --backbone resnet101
python run.py train --mode full --stage classify --cls-backbone efficientnet_b0

# Resume interrupted training
python run.py train --mode full --stage both --resume
```

**Training Options:**

| Stage | Mode | Epochs | Input | Backbone | Best For |
|-------|------|--------|-------|----------|----------|
| Detect | `test` | 5 | 512×512 | resnet50 | Pipeline test |
| Detect | `full` | 100 | 1024×1024 | resnet50 | **Competition** |
| Classify | `test` | 5 | 224×224 | efficientnet_b2 | Pipeline test |
| Classify | `full` | 50 | 224×224 | efficientnet_b2 | **Competition** |

**Available Backbones:**

*Detector (Stage 1):*
- `resnet34` — Fastest, 21M params
- `resnet50` — **Recommended**, 25M params
- `resnet101` — Highest capacity, 44M params

*Classifier (Stage 2):*
- `efficientnet_b0` — Fastest, ~5M params
- `efficientnet_b2` — **Recommended**, ~9M params
- `convnext_tiny` — Highest capacity, ~28M params

**Model Saving:**
- Detector: `trained_models/best_detector_latest.pth`
- Classifier: `trained_models/best_classifier_latest.pth`
- Run checkpoints: `runs/detect/` and `runs/classify/`

### 4. Generate Submission

```bash
# Standard inference (auto-detects both models)
python run.py infer --conf 0.3

# Use specific models
python run.py infer --det-model trained_models/best_detector_latest.pth \
                    --cls-model trained_models/best_classifier_latest.pth

# Lower confidence for higher recall
python run.py infer --conf 0.2
```

**Confidence Threshold Tuning:**
- `--conf 0.3` (default) — Balanced precision/recall
- `--conf 0.2` — Higher recall, more detections
- `--conf 0.4` — Higher precision, fewer false positives

**Output Format:**
```csv
image_filename,x,y,width,height,conf,class
ASCUS_2_patch02.png,512.3,256.7,100,100,0.7234,3
```
- All boxes are **100×100 pixels** (fixed)
- Confidence = detection_conf × classification_conf
- Submission saved to: `outputs/inference/submission_YYYYMMDD_HHMMSS.csv`

### 5. Visualize Results

```bash
# Visualize ground truth annotations (validation set)
python run.py visualize --split val --source csv

# Visualize predictions on test set
python run.py visualize --source predictions

# Visualize training set
python run.py visualize --split train --source csv --num-samples 20
```

---

## 🧠 Why Two-Stage?

### Problem Analysis
All bounding boxes in this dataset are **exactly 100×100 pixels**. This means:
1. **Detection = Point Detection** — We only need to find cell centers, not predict box sizes.
2. **Classification is separate** — With ~13K cell annotations, we have enough data for a strong classifier.
3. **Class imbalance hurts detection** — Trying to detect and classify simultaneously (CenterNet 8-class) spreads the model thin.

### Two-Stage Advantages
| Aspect | Single-Stage (Old) | Two-Stage (New) |
|--------|-------------------|-----------------|
| Detection task | 8-class heatmap (complex) | Binary heatmap (simple) |
| Classification data | 828 images | **13,267 patches** (16× more) |
| Class imbalance | Focal loss only | **Weighted sampling + weighted CE** |
| Failure modes | Detection + classification coupled | Stages fail independently |
| Debugging | Hard to tell what's wrong | Easy to diagnose each stage |
| VRAM usage | Large model, one pass | Two smaller models |

---

## 📊 Training Details

### Stage 1: Detector
- **Task**: Binary cell detection (is there a cell here? Yes/No)
- **Architecture**: CenterNet (ResNet-50 + 3× Deconv + 1-ch heatmap + 2-ch offset)
- **Loss**: Focal loss (heatmap) + L1 loss (offset)
- **Training**: 100 epochs, warmup 5 → cosine decay
- **AMP**: Mixed precision for faster GPU training
- **Differential LR**: backbone 0.1× head LR
- **Key insight**: Binary detection is MUCH easier than 8-class — focal loss can focus purely on finding cells

### Stage 2: Classifier
- **Task**: 8-class cell type classification (Bethesda categories)
- **Architecture**: EfficientNet-B2 (pretrained on ImageNet)
- **Input**: 100×100 crop → resized to 224×224
- **Loss**: CrossEntropyLoss with class weights
- **Sampling**: WeightedRandomSampler ensures all classes seen equally
- **Training**: 50 epochs, warmup 3 → cosine decay
- **Key insight**: ~13,267 training patches with class-balanced sampling gives every class fair representation

### Class Distribution & Weights
| Class | Count (train) | Sampling Weight | Notes |
|-------|---------------|-----------------|-------|
| NILM | 3821 (29%) | 1.14× | Majority |
| ENDO | 668 (5%) | 6.53× | Minority |
| INFL | 4360 (33%) | 1.00× | Majority |
| ASCUS | 207 (2%) | **21.06×** | Critical minority |
| LSIL | 1581 (12%) | 2.76× | Moderate |
| HSIL | 1232 (9%) | 3.54× | Moderate |
| ASCH | 316 (2%) | **13.80×** | Critical minority |
| SCC | 1082 (8%) | 4.03× | Moderate |

---

## 🔧 Troubleshooting

### CUDA Out of Memory (Stage 1 — Detector)
- Reduce batch size: edit `scripts/train.py` → `get_detect_config()` → `'batch_size': 2`
- Use smaller input: `'input_size': 768`
- Use ResNet-34: `--backbone resnet34`

### CUDA Out of Memory (Stage 2 — Classifier)
- Reduce batch size: edit `scripts/train.py` → `get_classify_config()` → `'batch_size': 32`
- Use EfficientNet-B0: `--cls-backbone efficientnet_b0`

### Detector Not Finding Cells
- Lower confidence threshold: `python run.py infer --conf 0.1`
- Train longer (increase epochs to 150)
- Try lower learning rate: `'lr': 5e-5`

### Classifier Low Accuracy on Minority Classes
- Train longer (increase epochs to 80)
- Check class-wise accuracy in training logs
- Try ConvNeXt-Tiny for more capacity: `--cls-backbone convnext_tiny`

### Training Interrupted
```bash
python run.py train --mode full --stage both --resume
```

---

## 🎯 Competition Tips

1. **Start with `--mode test`** to verify the pipeline works end-to-end
2. **Train Stage 1 first**, verify it detects cells, then train Stage 2
3. **Monitor per-class accuracy** — ASCUS and ASCH are hardest
4. **Tune confidence threshold** on validation predictions
5. **Full training takes ~6-8 hours** (detector) + **~1-2 hours** (classifier)
6. **Ensemble tip**: Train multiple classifiers (B0 + B2) and average predictions

**Good luck! 🍀**
