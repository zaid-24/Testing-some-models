"""
Two--Stage Training Pipeline for Cervical Cell Detection.

Stage 1 -- Detector Training:
    Binary CenterNet (ResNet--50 + Deconv) that finds ALL cell centers.
    Single-channel heatmap with focal loss + L1 offset loss.
    The detector is class--agnostic -- it only says "there's a cell here."

Stage 2 -- Classifier Training:
    EfficientNet--B2 that classifies 100x100 patches into 8 Bethesda classes.
    Cross--entropy loss with class--weighted sampling for imbalance handling.
    Trained on GT annotations (crops from original images).

Usage:
    # Train both stages (recommended)
    python scripts/train.py --mode full --stage both

    # Train detector only
    python scripts/train.py --mode full --stage detect

    # Train classifier only
    python scripts/train.py --mode full --stage classify

    # Quick test (both stages, fewer epochs)
    python scripts/train.py --mode test --stage both

    # Resume interrupted training
    python scripts/train.py --mode full --stage both --resume
"""

import os
import sys
import argparse
import platform
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.centernet import CellCenterNet
from models.classifier import CellClassifier
from models.losses import CenterNetLoss
from scripts.dataset import (
    DetectionDataset, PatchClassificationDataset,
    CLASS_NAMES, NUM_CLASSES
)


# =====================================================================
# Configuration
# =====================================================================

def get_detect_config(mode, backbone='resnet50'):
    """Configuration for Stage 1 (detector) training."""
    num_workers = 0 if platform.system() == 'Windows' else 4

    configs = {
        'test': {
            'backbone': backbone,
            'input_size': 512,
            'batch_size': 8,
            'epochs': 5,
            'lr': 2.5e-4,
            'weight_decay': 1e-4,
            'backbone_lr_factor': 0.1,
            'hm_weight': 1.0,
            'off_weight': 1.0,
            'patience': 5,
            'save_period': 1,
            'num_workers': num_workers,
            'warmup_epochs': 1,
        },
        'full': {
            'backbone': backbone,
            'input_size': 1024,
            'batch_size': 4,
            'epochs': 100,
            'lr': 1.25e-4,
            'weight_decay': 1e-4,
            'backbone_lr_factor': 0.1,
            'hm_weight': 1.0,
            'off_weight': 1.0,
            'patience': 25,
            'save_period': 5,
            'num_workers': num_workers,
            'warmup_epochs': 5,
        },
    }

    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Choose from: {list(configs.keys())}")

    config = configs[mode]
    config['backbone'] = backbone
    config['mode'] = mode
    return config


def get_classify_config(mode, backbone='efficientnet_b2'):
    """Configuration for Stage 2 (classifier) training."""
    num_workers = 0 if platform.system() == 'Windows' else 4

    configs = {
        'test': {
            'backbone': backbone,
            'input_size': 224,
            'patch_size': 100,
            'batch_size': 64,
            'epochs': 5,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'patience': 5,
            'save_period': 1,
            'num_workers': num_workers,
            'warmup_epochs': 1,
            'drop_rate': 0.3,
        },
        'full': {
            'backbone': backbone,
            'input_size': 224,
            'patch_size': 100,
            'batch_size': 64,
            'epochs': 50,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'patience': 15,
            'save_period': 5,
            'num_workers': num_workers,
            'warmup_epochs': 3,
            'drop_rate': 0.3,
        },
    }

    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Choose from: {list(configs.keys())}")

    config = configs[mode]
    config['backbone'] = backbone
    config['mode'] = mode
    return config


# =====================================================================
# Stage 1: Detector Training
# =====================================================================

def validate_detector(model, val_loader, criterion, device, use_amp):
    """Run validation and return loss metrics."""
    model.eval()
    total_loss = total_hm = total_off = 0
    n = 0

    with torch.no_grad():
        for images, gt_hm, gt_off, gt_mask in val_loader:
            images = images.to(device)
            gt_hm = gt_hm.to(device)
            gt_off = gt_off.to(device)
            gt_mask = gt_mask.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                pred_hm, pred_off = model(images)
                loss_dict = criterion(pred_hm, pred_off, gt_hm, gt_off, gt_mask)

            total_loss += loss_dict['total'].item()
            total_hm += loss_dict['hm_loss'].item()
            total_off += loss_dict['off_loss'].item()
            n += 1

    n = max(n, 1)
    return {
        'val_loss': total_loss / n,
        'val_hm_loss': total_hm / n,
        'val_off_loss': total_off / n,
    }


def train_detector(config, base_dir, resume=False):
    """Train Stage 1: Binary cell detector (CenterNet with 1--channel heatmap)."""
    base_dir = Path(base_dir).resolve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print(f"\n{'='*70}")
    print("  STAGE 1: TRAINING BINARY CELL DETECTOR (CenterNet)")
    print(f"{'='*70}")
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # === PATHS ===
    ann_dir = base_dir / 'dataset' / 'annotations' / 'annotations'
    img_dir = base_dir / 'dataset' / 'images' / 'images'
    output_dir = base_dir / 'runs' / 'detect' / f"{config['mode']}_{config['backbone']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # === DATASETS ===
    print("\n[1/4] Loading detection datasets...")
    train_ds = DetectionDataset(
        csv_path=str(ann_dir / 'train.csv'),
        images_dir=str(img_dir / 'train'),
        input_size=config['input_size'],
        augment=True,
    )
    val_ds = DetectionDataset(
        csv_path=str(ann_dir / 'val.csv'),
        images_dir=str(img_dir / 'val'),
        input_size=config['input_size'],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )

    print(f"  Train: {len(train_ds)} images, {len(train_loader)} batches/epoch")
    print(f"  Val:   {len(val_ds)} images, {len(val_loader)} batches")

    # === MODEL (binary: num_classes=1) ===
    print(f"\n[2/4] Creating binary CenterNet detector...")
    print(f"  Backbone: {config['backbone']} (pretrained on ImageNet)")
    print(f"  Heatmap channels: 1 (binary cell/no-cell)")
    model = CellCenterNet(
        num_classes=1,
        backbone=config['backbone'],
        pretrained=True,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # === LOSS ===
    criterion = CenterNetLoss(
        hm_weight=config['hm_weight'],
        off_weight=config['off_weight'],
    )

    # === OPTIMIZER (differential LR) ===
    backbone_params = [p for n, p in model.named_parameters()
                       if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if 'backbone' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['lr'] * config['backbone_lr_factor']},
        {'params': head_params, 'lr': config['lr']},
    ], weight_decay=config['weight_decay'])

    # === LR SCHEDULER: warmup -> cosine ===
    warmup_epochs = config['warmup_epochs']
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(config['epochs'] -- warmup_epochs, 1),
        eta_min=config['lr'] * 0.01
    )
    scheduler = SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    # === AMP SCALER ===
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # === RESUME ===
    start_epoch = 0
    best_val_loss = float('inf')
    if resume:
        ckpt_path = output_dir / 'last.pth'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            print(f"  [OK] Resumed from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")
        else:
            print(f"  [WARNING] No checkpoint at {ckpt_path}, starting fresh")

    # === TRAINING LOOP ===
    print(f"\n[3/4] Training detector...")
    print(f"  Epochs:     {config['epochs']} (warmup: {warmup_epochs})")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  LR:         {config['lr']} (backbone: x{config['backbone_lr_factor']})")
    print(f"  Input size: {config['input_size']}x{config['input_size']}")
    print(f"  Heatmap:    {config['input_size']//4}x{config['input_size']//4} (stride=4)")
    print(f"  AMP:        {use_amp}")
    print(f"  Patience:   {config['patience']} epochs")
    print("--" * 70)

    patience_counter = 0

    for epoch in range(start_epoch, config['epochs']):
        # ---- Train one epoch ----
        model.train()
        epoch_loss = epoch_hm = epoch_off = 0

        pbar = tqdm(train_loader, desc=f"Detect Ep {epoch+1}/{config['epochs']}")
        for images, gt_hm, gt_off, gt_mask in pbar:
            images = images.to(device, non_blocking=True)
            gt_hm = gt_hm.to(device, non_blocking=True)
            gt_off = gt_off.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                pred_hm, pred_off = model(images)
                loss_dict = criterion(pred_hm, pred_off, gt_hm, gt_off, gt_mask)
                loss = loss_dict['total']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_hm += loss_dict['hm_loss'].item()
            epoch_off += loss_dict['off_loss'].item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'hm': f"{loss_dict['hm_loss'].item():.4f}",
                'off': f"{loss_dict['off_loss'].item():.4f}",
            })

        scheduler.step()

        # ---- Epoch averages ----
        n = len(train_loader)
        avg_loss = epoch_loss / n
        avg_hm = epoch_hm / n
        avg_off = epoch_off / n

        # ---- Validate ----
        val = validate_detector(model, val_loader, criterion, device, use_amp)

        # ---- Log ----
        lr = optimizer.param_groups[--1]['lr']
        print(f"  Train -- loss: {avg_loss:.4f}  hm: {avg_hm:.4f}  off: {avg_off:.4f}")
        print(f"  Val   -- loss: {val['val_loss']:.4f}  "
              f"hm: {val['val_hm_loss']:.4f}  off: {val['val_off_loss']:.4f}")
        print(f"  LR: {lr:.6f}  Patience: {patience_counter}/{config['patience']}")

        # ---- Save best ----
        if val['val_loss'] < best_val_loss:
            best_val_loss = val['val_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'stage': 'detect',
            }, output_dir / 'best.pth')
            print(f"  [OK] New best detector! val_loss={best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # ---- Checkpoint ----
        if (epoch + 1) % config['save_period'] == 0 or epoch == config['epochs'] -- 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }, output_dir / 'last.pth')

        # ---- Early stopping ----
        if patience_counter >= config['patience']:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {config['patience']} epochs)")
            break

    # === SAVE FINAL MODEL ===
    print(f"\n[4/4] Saving detector model...")
    trained_dir = base_dir / 'trained_models'
    trained_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_src = output_dir / 'best.pth'

    if best_src.exists():
        dest = trained_dir / f"best_detector_{config['backbone']}_{timestamp}.pth"
        shutil.copy2(best_src, dest)

        latest = trained_dir / 'best_detector_latest.pth'
        if latest.exists():
            latest.unlink()
        shutil.copy2(best_src, latest)

        print(f"\n{'='*70}")
        print("[SUCCESS] DETECTOR TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"  Best model:    {dest}")
        print(f"  Latest model:  {latest}")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Run dir:       {output_dir}")
    else:
        print(f"\n[WARNING] Best detector model not found at {best_src}")

    return model


# =====================================================================
# Stage 2: Classifier Training
# =====================================================================

def validate_classifier(model, val_loader, criterion, device, use_amp):
    """Run validation and return loss, accuracy, and per--class accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = torch.zeros(NUM_CLASSES)
    class_total = torch.zeros(NUM_CLASSES)
    n = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for c in range(NUM_CLASSES):
                mask = labels == c
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c] += mask.sum().item()

            n += 1

    acc = correct / max(total, 1)
    class_acc = {}
    for c in range(NUM_CLASSES):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c].item() / class_total[c].item()

    return {
        'val_loss': total_loss / max(n, 1),
        'val_acc': acc,
        'class_acc': class_acc,
    }


def train_classifier(config, base_dir, resume=False):
    """Train Stage 2: Cell type classifier (EfficientNet on 100x100 patches)."""
    base_dir = Path(base_dir).resolve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print(f"\n{'='*70}")
    print("  STAGE 2: TRAINING CELL TYPE CLASSIFIER (EfficientNet)")
    print(f"{'='*70}")
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # === PATHS ===
    ann_dir = base_dir / 'dataset' / 'annotations' / 'annotations'
    img_dir = base_dir / 'dataset' / 'images' / 'images'
    output_dir = base_dir / 'runs' / 'classify' / f"{config['mode']}_{config['backbone']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # === DATASETS ===
    print("\n[1/4] Loading patch classification datasets...")
    train_ds = PatchClassificationDataset(
        csv_path=str(ann_dir / 'train.csv'),
        images_dir=str(img_dir / 'train'),
        patch_size=config['patch_size'],
        classifier_input_size=config['input_size'],
        augment=True,
    )
    val_ds = PatchClassificationDataset(
        csv_path=str(ann_dir / 'val.csv'),
        images_dir=str(img_dir / 'val'),
        patch_size=config['patch_size'],
        classifier_input_size=config['input_size'],
        augment=False,
    )

    print(f"  Train patches: {len(train_ds)} (from GT annotations)")
    print(f"  Val patches:   {len(val_ds)}")
    print(f"  Class distribution (train):")
    for c in sorted(train_ds.class_counts.keys()):
        name = CLASS_NAMES.get(c, '?')
        count = train_ds.class_counts[c]
        weight = train_ds.class_weights_dict[c]
        print(f"    {c} ({name:5s}): {count:5d}  (sampling weight: {weight:.2f})")

    # Weighted sampler for class balance
    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )

    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # === MODEL ===
    print(f"\n[2/4] Creating cell classifier...")
    print(f"  Backbone: {config['backbone']} (pretrained on ImageNet)")
    print(f"  Input size: {config['input_size']}x{config['input_size']}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Dropout: {config['drop_rate']}")

    model = CellClassifier(
        num_classes=NUM_CLASSES,
        backbone=config['backbone'],
        pretrained=True,
        drop_rate=config['drop_rate'],
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # === LOSS (with class weights) ===
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Loss: CrossEntropy with class weights")

    # === OPTIMIZER ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # === LR SCHEDULER ===
    warmup_epochs = config['warmup_epochs']
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(config['epochs'] -- warmup_epochs, 1),
        eta_min=config['lr'] * 0.01,
    )
    scheduler = SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    # === AMP SCALER ===
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # === RESUME ===
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    if resume:
        ckpt_path = output_dir / 'last.pth'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            best_val_acc = ckpt.get('best_val_acc', 0.0)
            print(f"  [OK] Resumed from epoch {start_epoch} (best_val_acc={best_val_acc:.4f})")
    else:
            print(f"  [WARNING] No checkpoint at {ckpt_path}, starting fresh")

    # === TRAINING LOOP ===
    print(f"\n[3/4] Training classifier...")
    print(f"  Epochs:     {config['epochs']} (warmup: {warmup_epochs})")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  LR:         {config['lr']}")
    print(f"  Patch size: {config['patch_size']}x{config['patch_size']} -> "
          f"{config['input_size']}x{config['input_size']}")
    print(f"  AMP:        {use_amp}")
    print(f"  Patience:   {config['patience']} epochs")
    print("--" * 70)

    patience_counter = 0

    for epoch in range(start_epoch, config['epochs']):
        # ---- Train one epoch ----
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Classify Ep {epoch+1}/{config['epochs']}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{epoch_correct / max(epoch_total, 1):.3f}",
            })

        scheduler.step()

        # ---- Epoch averages ----
        n = len(train_loader)
        train_acc = epoch_correct / max(epoch_total, 1)

        # ---- Validate ----
        val = validate_classifier(model, val_loader, criterion, device, use_amp)

        # ---- Log ----
        lr = optimizer.param_groups[0]['lr']
        print(f"  Train -- loss: {epoch_loss/n:.4f}  acc: {train_acc:.4f}")
        print(f"  Val   -- loss: {val['val_loss']:.4f}  acc: {val['val_acc']:.4f}")

        # Per--class accuracy
        acc_strs = []
        for c in range(NUM_CLASSES):
            if c in val['class_acc']:
                acc_strs.append(f"{CLASS_NAMES[c]}:{val['class_acc'][c]:.2f}")
        print(f"  Per--class: {' | '.join(acc_strs)}")
        print(f"  LR: {lr:.6f}  Patience: {patience_counter}/{config['patience']}")

        # ---- Save best (based on val accuracy) ----
        if val['val_acc'] > best_val_acc:
            best_val_acc = val['val_acc']
            best_val_loss = val['val_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'stage': 'classify',
            }, output_dir / 'best.pth')
            print(f"  [OK] New best classifier! val_acc={best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # ---- Checkpoint ----
        if (epoch + 1) % config['save_period'] == 0 or epoch == config['epochs'] -- 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
            }, output_dir / 'last.pth')

        # ---- Early stopping ----
        if patience_counter >= config['patience']:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {config['patience']} epochs)")
            break

    # === SAVE FINAL MODEL ===
    print(f"\n[4/4] Saving classifier model...")
    trained_dir = base_dir / 'trained_models'
    trained_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_src = output_dir / 'best.pth'

    if best_src.exists():
        dest = trained_dir / f"best_classifier_{config['backbone']}_{timestamp}.pth"
        shutil.copy2(best_src, dest)

        latest = trained_dir / 'best_classifier_latest.pth'
        if latest.exists():
            latest.unlink()
        shutil.copy2(best_src, latest)

        print(f"\n{'='*70}")
        print("[SUCCESS] CLASSIFIER TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"  Best model:    {dest}")
        print(f"  Latest model:  {latest}")
        print(f"  Best val acc:  {best_val_acc:.4f}")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Run dir:       {output_dir}")
    else:
        print(f"\n[WARNING] Best classifier model not found at {best_src}")

    return model


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Two--Stage Cell Detection Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (both stages)
  python scripts/train.py --mode test --stage both

  # Full training (both stages)
  python scripts/train.py --mode full --stage both

  # Detector only
  python scripts/train.py --mode full --stage detect --backbone resnet50

  # Classifier only
  python scripts/train.py --mode full --stage classify --cls-backbone efficientnet_b2

  # Resume interrupted training
  python scripts/train.py --mode full --stage both --resume
        """
    )
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                        help='Training mode: test (quick) or full (competition)')
    parser.add_argument('--stage', choices=['detect', 'classify', 'both'], default='both',
                        help='Which stage to train')
    parser.add_argument('--backbone', choices=['resnet34', 'resnet50', 'resnet101'],
                        default='resnet50', help='Detector backbone')
    parser.add_argument('--cls-backbone',
                        choices=['efficientnet_b0', 'efficientnet_b2', 'convnext_tiny'],
                        default='efficientnet_b2', help='Classifier backbone')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base project directory')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("     RIVA Cell Detection - Two-Stage Training Pipeline")
    print("     Stage 1: Binary Detector (CenterNet + ResNet)")
    print("     Stage 2: Cell Classifier (EfficientNet)")
    print(f"{'='*70}")

    if args.stage in ('detect', 'both'):
        det_config = get_detect_config(args.mode, args.backbone)
        print(f"\n  [Stage 1] Detector: {det_config['backbone']}, "
              f"{det_config['epochs']} epochs, "
              f"{det_config['input_size']}x{det_config['input_size']}")
        train_detector(det_config, args.base_dir, args.resume)

    if args.stage in ('classify', 'both'):
        cls_config = get_classify_config(args.mode, args.cls_backbone)
        print(f"\n  [Stage 2] Classifier: {cls_config['backbone']}, "
              f"{cls_config['epochs']} epochs, "
              f"patches {cls_config['patch_size']}->{cls_config['input_size']}px")
        train_classifier(cls_config, args.base_dir, args.resume)

    print(f"\n{'='*70}")
    print("[SUCCESS] ALL TRAINING COMPLETE")
    print(f"{'='*70}")
    print("\n  To run inference:")
    print("    python run.py infer --conf 0.3")


if __name__ == '__main__':
    main()
