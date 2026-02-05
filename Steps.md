## 🚀 Quick Start

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

**⚠️ Important:** See `INSTALL.md` for detailed installation instructions, especially for ensuring CUDA-enabled PyTorch is installed correctly.

**Verify Installation:**
```bash
# Quick verification
python verify_installation.py

# Should show:
# [OK] PyTorch installed
# [OK] CUDA available: True
# [OK] GPU device: NVIDIA RTX A2000
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
# Test pipeline first on laptop (recommended)
python run.py train --mode test

# ⭐ RECOMMENDED: Multi-Scale Progressive Training
# Stage 1: 640px (100 epochs) - Fast coarse learning
# Stage 2: 896px (150 epochs) - Refine features
# Stage 3: 1024px (150 epochs) - Fine details for ASCUS/ASCH
# Total: 400 epochs, automatically progresses through stages
python run.py train --mode multiscale

# Alternative training modes:
python run.py train --mode focal      # Class imbalance focus (higher cls weight)
python run.py train --mode adh        # Localization precision (higher box weight)
python run.py train --mode full       # Extreme augmentation baseline

# Resume interrupted training
python run.py train --mode multiscale --resume
```

**⚠️ Important Notes:**

**Configuration:**
- **Model**: yolo11l (large) - **Best proven performance**
- **Image Size**: Progressive 640→896→1024px (multi-scale) or fixed 1024px (other modes)

**Training Options:**

| Mode | Description | Time | Best For |
|------|-------------|------|----------|
| `multiscale` | ⭐ Progressive resolution | ~12-16 hours | Best overall accuracy |
| `focal` | Higher cls weight (4.0) | ~14-18 hours | Class imbalance |
| `adh` | Higher box weight (10.0) | ~14-18 hours | Localization precision |
| `full` | Extreme augmentation | ~16-20 hours | Baseline comparison |

**Model Saving:**
- **Multi-scale**: `trained_models/best_multiscale_YYYYMMDD_HHMMSS.pt`
- **Focal**: `trained_models/best_focal_loss_YYYYMMDD_HHMMSS.pt`
- **ADH**: `trained_models/best_adh_YYYYMMDD_HHMMSS.pt`
- **Full**: `trained_models/best_full_extreme_YYYYMMDD_HHMMSS.pt`
- **Latest**: `trained_models/best_latest.pt` (always points to most recent)

### 4. Generate Submission

```bash
# Run inference with Test-Time Augmentation (recommended)
python run.py infer --tta --conf 0.15 --iou 0.5

# Without TTA (faster)
python run.py infer --conf 0.15 --iou 0.5

# Use specific model
python run.py infer --tta --model best.pt --conf 0.15
```

**📦 Transferring Model Between Systems:**
1. After training on RTX A2000, copy the model:
   ```
   trained_models/best_yolo11l_YYYYMMDD_HHMMSS.pt
   ```
2. On the other system, place it in `trained_models/` folder
3. Rename to `best_latest.pt` or specify with `--model` flag

### 5. Visualize Results

```bash
# Visualize ground truth
python run.py visualize --split val --source csv

# Visualize YOLO format labels
python run.py visualize --split train --source yolo

# Visualize predictions
python run.py visualize --source predictions