## 🚀 Quick Start — CenterNet Pipeline

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# IMPORTANT: Install PyTorch with CUDA FIRST
# For CUDA 12.1 (RTX A2000):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Then install other dependencies
pip install -r requirements.txt
```

**Verify Installation:**
```bash
# Quick verification
python verify_installation.py

# Should show:
# [OK] PyTorch installed
# [OK] CUDA available: True
# [OK] GPU device: NVIDIA RTX A2000
# [OK] CenterNet model test passed
```

### 2. Analyze Dataset (Optional)

```bash
# Analyze dataset statistics & class distribution
python run.py analyze

# Generates:
# - Class distribution plots
# - Annotations per image histogram
# - Saves to: outputs/analysis/
```

**Key Dataset Insights:**
- **828 train images** / 131 val / 81 test
- **Severe class imbalance**: 21:1 ratio (INFL vs ASCUS)
- **ALL boxes are 100×100 pixels** — This is a **point detection** problem!

### 3. Train CenterNet Model

**🎯 CRITICAL INSIGHT**: All bounding boxes are exactly 100×100 pixels, so we use **CenterNet** (Objects as Points) which predicts cell centers as heatmap peaks — no box regression needed!

**Architecture:**
- **Backbone**: ResNet-50 (pretrained on ImageNet)
- **Neck**: 3× Deconvolution layers (stride 32 → 4)
- **Heads**: 8-channel heatmap + 2-channel offset
- **Loss**: Focal loss (naturally handles 21:1 class imbalance)

```bash
# Quick test (5 epochs, 512px) — verify pipeline works
python run.py train --mode test

# Full training (140 epochs, 1024px) —main
python run.py train --mode full

# Full training with ResNet-101 (more capacity, slower)
python run.py train --mode full --backbone resnet101

# Resume interrupted training
python run.py train --mode full --resume
```

**Training Options:**

| Mode | Epochs | Input Size | Backbone | Time (RTX A2000) | Best For |
|------|--------|------------|----------|------------------|----------|
| `test` | 5 | 512×512 | resnet50 | ~10-15 min | Pipeline validation |
| `full` | 140 | 1024×1024 | resnet50 | ~10-14 hours | **Competition** |
| `full` + `resnet101` | 140 | 1024×1024 | resnet101 | ~14-18 hours | Maximum accuracy |

**Available Backbones:**
- `resnet34` — Fastest, 21M params
- `resnet50` — **Recommended**, 25M params, best speed/accuracy
- `resnet101` — Highest capacity, 44M params, slower

**Model Saving:**
- Checkpoint: `runs/centernet/<mode>_<backbone>/best.pth`
- Trained models: `trained_models/best_centernet_<backbone>_YYYYMMDD_HHMMSS.pth`
- Latest: `trained_models/best_latest.pth` (always points to most recent)

**Training Details:**
- Mixed precision (AMP) for 2× speedup on GPU
- Differential LR: backbone 0.1×, heads 1× base LR
- Linear warmup (5 epochs) → cosine annealing
- Early stopping (patience=30 epochs)
- Gradient clipping (max_norm=35, following CenterNet paper)
- Augmentation: flips, rotation, shift-scale-rotate, color jitter, CLAHE, blur

**Loss Function:**
- **Heatmap loss**: Modified focal loss (α=2, β=4)
  - Down-weights easy negatives (background)
  - Focuses on hard examples (minority classes)
  - Naturally handles 21:1 class imbalance
- **Offset loss**: L1 loss for sub-pixel refinement

### 4. Generate Submission

```bash
# Standard inference (auto-detects best model)
python run.py infer --conf 0.3

# Use specific model
python run.py infer --model trained_models/best_latest.pth --conf 0.3

# Lower confidence for higher recall (more detections)
python run.py infer --conf 0.2

# Specify input size (should match training)
python run.py infer --conf 0.3 --imgsz 1024
```

**Confidence Threshold Tuning:**
- `--conf 0.3` (default) — Balanced precision/recall
- `--conf 0.2` — Higher recall, more detections (good for minorities)
- `--conf 0.4` — Higher precision, fewer false positives

**Output Format:**
```csv
image_filename,x,y,width,height,conf,class
ASCUS_2_patch02.png,512.3,256.7,100,100,0.8523,3
```
- All boxes are **100×100 pixels** (matches ground truth)
- Coordinates are in **original image space** (before any resizing)
- Submission saved to: `outputs/inference/submission_YYYYMMDD_HHMMSS.csv`

**📦 Transferring Model Between Systems:**
1. After training on RTX A2000, copy:
   ```
   trained_models/best_centernet_resnet50_YYYYMMDD_HHMMSS.pth
   ```
2. On the other system, place in `trained_models/` folder
3. Rename to `best_latest.pth` or specify with `--model` flag
4. Run inference: `python run.py infer --conf 0.3`

### 5. Visualize Results

```bash
# Visualize ground truth annotations (validation set)
python run.py visualize --split val --source csv

# Visualize predictions on test set
python run.py visualize --source predictions

# Visualize training set (check for annotation errors)
python run.py visualize --split train --source csv --num-samples 20
```

**Output:**
- Color-coded bounding boxes (one color per class)
- Class label + confidence score on each box
- Saved to: `outputs/visualizations/`
- Legend: `outputs/visualizations/class_legend.png`

---

## 📊 Expected Performance

### Baseline (ResNet-50, 140 epochs, 1024px)
- **Training**: ~10-14 hours on RTX A2000
- **Expected mAP@50**: 65-75% (depending on validation/test split similarity)
- **Minority class detection**: Focal loss ensures ASCUS/ASCH are learned

### Class-Specific Challenges
| Class | Count (train) | Difficulty | Notes |
|-------|---------------|------------|-------|
| INFL | 4360 (33%) | Easy | Majority class |
| NILM | 3821 (29%) | Easy | Majority class |
| LSIL | 1581 (12%) | Moderate | Good representation |
| HSIL | 1232 (9%) | Moderate | Good representation |
| SCC | 1082 (8%) | Moderate | Good representation |
| ENDO | 668 (5%) | Hard | Minority class |
| ASCH | 316 (2%) | Very Hard | Critical minority |
| ASCUS | 207 (2%) | Very Hard | Critical minority |

**Key Advantage of CenterNet + Focal Loss:**
The focal loss automatically down-weights easy negatives (background + majority classes once learned) and focuses learning on hard examples (minority classes). This is **much more effective** than manual class weighting or data augmentation alone.

---

## 🔧 Troubleshooting

### CUDA Out of Memory
Reduce batch size in training config:
- Edit `scripts/train.py` → `get_config()` function
- Change `'batch_size': 4` to `'batch_size': 2`
- Or use smaller input size: `'input_size': 768` instead of 1024

### Model Not Converging
- Check dataset: `python run.py analyze`
- Verify CUDA: `python verify_installation.py`
- Try lower learning rate: edit `scripts/train.py` → `'lr': 1.25e-4` to `5e-5`
- Increase training time: `'epochs': 140` to `200`

### Low Recall on Minority Classes
- Lower confidence threshold: `python run.py infer --conf 0.2`
- Train longer (focal loss needs time to focus on minorities)
- Try ResNet-101 for more model capacity

### Training Interrupted
Resume seamlessly:
```bash
python run.py train --mode full --resume
# Loads last checkpoint with optimizer/scheduler state
```

---

## 📝 Key Differences from YOLO

| Aspect | YOLO (Old) | CenterNet (New) |
|--------|-----------|----------------|
| **Architecture** | Anchor-based object detection | Point detection (heatmaps) |
| **Box regression** | Yes (wasted for 100×100 boxes) | No (only center + offset) |
| **NMS** | Required (IoU threshold) | Not needed (max pooling on heatmaps) |
| **Class imbalance** | Manual weighting/augmentation | Focal loss (automatic) |
| **Data format** | Requires YOLO .txt conversion | Reads CSV directly |
| **Output** | (x,y,w,h,class) | Heatmap peaks → (x,y,class) |
| **Fixed boxes** | Ignores this insight | Optimized for it |

**Why CenterNet is Better for This Dataset:**
1. No wasted capacity on box size prediction (all boxes are 100×100)
2. Focal loss naturally handles severe 21:1 class imbalance
3. Simpler pipeline (no CSV→YOLO conversion, no NMS tuning)
4. Faster inference (no anchor generation, no complex post-processing)
5. More accurate centers (sub-pixel offset refinement)

---

## 🎯 Competition Tips

1. **Start with test mode** to verify everything works
2. **Use ResNet-50** for best speed/accuracy trade-off
3. **Full training takes ~12 hours** — start before bed
4. **Tune confidence threshold** on validation set predictions
5. **Ensemble multiple models** if time permits (ResNet-50 + ResNet-101)
6. **Monitor validation loss** — early stopping prevents overfitting
7. **Check minority class AP** after training — should be >10% each

**Good luck! 🍀**
