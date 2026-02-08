"""
Train CenterNet for Pap Smear Cell Detection.

CenterNet treats every cell as a single point (its center) and predicts
heatmaps where peaks = cell centers. No box regression needed since
all boxes are fixed 100x100 pixels.

Usage:
    # Quick test (5 epochs, 512px, ResNet-50)
    python scripts/train.py --mode test

    # Full training (140 epochs, 1024px, ResNet-50)
    python scripts/train.py --mode full

    # Full training with ResNet-101 backbone
    python scripts/train.py --mode full --backbone resnet101

    # Resume interrupted training
    python scripts/train.py --mode full --resume
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.centernet import CellCenterNet
from models.losses import CenterNetLoss
from models.decode import decode_heatmap
from scripts.dataset import RIVACenterNetDataset, CLASS_NAMES


def get_config(mode, backbone='resnet50'):
    """Get training configuration for the specified mode."""
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
            'epochs': 140,
            'lr': 1.25e-4,
            'weight_decay': 1e-4,
            'backbone_lr_factor': 0.1,
            'hm_weight': 1.0,
            'off_weight': 1.0,
            'patience': 30,
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


def validate(model, val_loader, criterion, device, use_amp):
    """Run validation and compute losses."""
    model.eval()
    total_loss = 0
    total_hm_loss = 0
    total_off_loss = 0
    num_batches = 0

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
            total_hm_loss += loss_dict['hm_loss'].item()
            total_off_loss += loss_dict['off_loss'].item()
            num_batches += 1

    n = max(num_batches, 1)
    return {
        'val_loss': total_loss / n,
        'val_hm_loss': total_hm_loss / n,
        'val_off_loss': total_off_loss / n,
    }


def train(config, base_dir, resume=False):
    """Main training function."""
    base_dir = Path(base_dir).resolve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # === PATHS ===
    annotations_dir = base_dir / 'dataset' / 'annotations' / 'annotations'
    images_dir = base_dir / 'dataset' / 'images' / 'images'
    output_dir = base_dir / 'runs' / 'centernet' / f"{config['mode']}_{config['backbone']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # === DATASETS ===
    print("\n[1/4] Loading datasets...")
    train_dataset = RIVACenterNetDataset(
        csv_path=str(annotations_dir / 'train.csv'),
        images_dir=str(images_dir / 'train'),
        input_size=config['input_size'],
        augment=True,
    )
    val_dataset = RIVACenterNetDataset(
        csv_path=str(annotations_dir / 'val.csv'),
        images_dir=str(images_dir / 'val'),
        input_size=config['input_size'],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )

    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches/epoch")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")

    # === MODEL ===
    print(f"\n[2/4] Creating CenterNet model...")
    print(f"  Backbone: {config['backbone']} (pretrained on ImageNet)")
    model = CellCenterNet(
        num_classes=8,
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

    # === OPTIMIZER (differential LR: backbone slower, heads faster) ===
    backbone_params = [p for n, p in model.named_parameters()
                       if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if 'backbone' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['lr'] * config['backbone_lr_factor']},
        {'params': head_params, 'lr': config['lr']},
    ], weight_decay=config['weight_decay'])

    # === LR SCHEDULER: linear warmup → cosine decay ===
    warmup_epochs = config['warmup_epochs']
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(config['epochs'] - warmup_epochs, 1),
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
            print(f"  ✓ Resumed from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")
        else:
            print(f"  [WARNING] No checkpoint at {ckpt_path}, starting fresh")

    # === TRAINING LOOP ===
    print(f"\n[3/4] Training...")
    print(f"  Epochs:     {config['epochs']} (warmup: {warmup_epochs})")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  LR:         {config['lr']} (backbone: ×{config['backbone_lr_factor']})")
    print(f"  Input size: {config['input_size']}×{config['input_size']}")
    print(f"  Heatmap:    {config['input_size']//4}×{config['input_size']//4} (stride=4)")
    print(f"  AMP:        {use_amp}")
    print(f"  Patience:   {config['patience']} epochs")
    print("-" * 70)

    patience_counter = 0

    for epoch in range(start_epoch, config['epochs']):
        # --- Train one epoch ---
        model.train()
        epoch_loss = 0
        epoch_hm_loss = 0
        epoch_off_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
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
            # Gradient clipping (following CenterNet, max_norm=35)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_hm_loss += loss_dict['hm_loss'].item()
            epoch_off_loss += loss_dict['off_loss'].item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'hm': f"{loss_dict['hm_loss'].item():.4f}",
                'off': f"{loss_dict['off_loss'].item():.4f}",
            })

        scheduler.step()

        # --- Epoch averages ---
        n = len(train_loader)
        avg_loss = epoch_loss / n
        avg_hm = epoch_hm_loss / n
        avg_off = epoch_off_loss / n

        # --- Validate ---
        val_metrics = validate(model, val_loader, criterion, device, use_amp)

        # --- Log ---
        lr = optimizer.param_groups[-1]['lr']
        print(f"  Train — loss: {avg_loss:.4f}  hm: {avg_hm:.4f}  off: {avg_off:.4f}")
        print(f"  Val   — loss: {val_metrics['val_loss']:.4f}  "
              f"hm: {val_metrics['val_hm_loss']:.4f}  off: {val_metrics['val_off_loss']:.4f}")
        print(f"  LR: {lr:.6f}  Patience: {patience_counter}/{config['patience']}")

        # --- Save best model ---
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }, output_dir / 'best.pth')
            print(f"  ✓ New best model! val_loss={best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # --- Save checkpoint periodically ---
        if (epoch + 1) % config['save_period'] == 0 or epoch == config['epochs'] - 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }, output_dir / 'last.pth')

        # --- Early stopping ---
        if patience_counter >= config['patience']:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {config['patience']} epochs)")
            break

    # === SAVE FINAL MODEL ===
    print(f"\n[4/4] Saving final model...")
    trained_dir = base_dir / 'trained_models'
    trained_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_src = output_dir / 'best.pth'

    if best_src.exists():
        dest = trained_dir / f"best_centernet_{config['backbone']}_{timestamp}.pth"
        shutil.copy2(best_src, dest)

        latest = trained_dir / 'best_latest.pth'
        if latest.exists():
            latest.unlink()
        shutil.copy2(best_src, latest)

        print(f"\n{'='*70}")
        print("[SUCCESS] TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"  Best model:    {dest}")
        print(f"  Latest model:  {latest}")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Run dir:       {output_dir}")
        print(f"\n  To run inference:")
        print(f"    python run.py infer --conf 0.3")
    else:
        print(f"\n[WARNING] Best model not found at {best_src}")
        print("  Training may have failed early. Check logs above.")

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train CenterNet for Pap Smear Cell Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 epochs, 512px)
  python scripts/train.py --mode test

  # Full training (140 epochs, 1024px)
  python scripts/train.py --mode full

  # Use ResNet-101 backbone
  python scripts/train.py --mode full --backbone resnet101

  # Resume interrupted training
  python scripts/train.py --mode full --resume
        """
    )
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                        help='Training mode: test (quick validation) or full (competition)')
    parser.add_argument('--backbone', choices=['resnet34', 'resnet50', 'resnet101'],
                        default='resnet50', help='Backbone architecture')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base project directory')
    args = parser.parse_args()

    config = get_config(args.mode, args.backbone)

    print(f"\n{'='*70}")
    print("     RIVA Cell Detection — CenterNet Training Pipeline")
    print("     Pap Smear Bethesda Classification (8 classes)")
    print(f"{'='*70}")
    print(f"\n  Mode:       {args.mode}")
    print(f"  Backbone:   {config['backbone']}")
    print(f"  Input size: {config['input_size']}×{config['input_size']}")
    print(f"  Epochs:     {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")

    train(config, args.base_dir, args.resume)


if __name__ == '__main__':
    main()
