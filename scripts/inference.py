"""
Run CenterNet inference on test images and generate submission CSV.

Process:
    1. Load trained CenterNet model
    2. For each test image: forward pass → decode heatmap peaks → detections
    3. Scale predictions to original image coordinates
    4. Attach fixed 100×100 bounding boxes
    5. Generate submission CSV

Usage:
    python scripts/inference.py
    python scripts/inference.py --model trained_models/best_latest.pth --conf 0.3
    python scripts/inference.py --conf 0.2 --imgsz 1024
"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.centernet import CellCenterNet
from models.decode import decode_heatmap

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = {
    0: 'NILM', 1: 'ENDO', 2: 'INFL', 3: 'ASCUS',
    4: 'LSIL', 5: 'HSIL', 6: 'ASCH', 7: 'SCC'
}

FIXED_BOX_SIZE = 100


def get_inference_transform():
    """Preprocessing: normalize + to tensor."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def run_inference(model_path, test_images_dir, output_dir,
                  conf_thresh=0.3, input_size=1024, device_id=0):
    """
    Run inference on test images.

    Args:
        model_path:      Path to trained model checkpoint (.pth)
        test_images_dir: Directory containing test images
        output_dir:      Directory to save results
        conf_thresh:     Confidence threshold for detections
        input_size:      Model input size (overridden by checkpoint config if available)
        device_id:       GPU device ID

    Returns:
        DataFrame with all predictions
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    # === Load model ===
    print(f"\n[1/3] Loading model: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract config from checkpoint if available
    if 'config' in ckpt:
        config = ckpt['config']
        backbone = config.get('backbone', 'resnet50')
        input_size = config.get('input_size', input_size)
        print(f"  Config from checkpoint: backbone={backbone}, input_size={input_size}")
    else:
        backbone = 'resnet50'
        print(f"  No config in checkpoint, using defaults: backbone={backbone}")

    model = CellCenterNet(num_classes=8, backbone=backbone, pretrained=False)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    print(f"  ✓ Model loaded ({backbone}, device={device})")

    # === Find test images ===
    test_dir = Path(test_images_dir)
    image_files = sorted(list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg')))
    print(f"\n[2/3] Found {len(image_files)} test images")

    if len(image_files) == 0:
        print("  ⚠️ No images found in test directory!")
        return pd.DataFrame()

    # === Run inference ===
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    transform = get_inference_transform()
    stride = 4
    use_amp = device.type == 'cuda'

    all_predictions = []

    print(f"\n[3/3] Running inference...")
    print(f"  Confidence threshold: {conf_thresh}")
    print(f"  Input size: {input_size}×{input_size}")
    print(f"  Fixed box size: {FIXED_BOX_SIZE}×{FIXED_BOX_SIZE}")
    print("-" * 50)

    for img_path in tqdm(image_files, desc="Processing"):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARNING] Failed to load: {img_path.name}")
            continue

        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_size, input_size))

        # Preprocess
        transformed = transform(image=img_resized)
        img_tensor = transformed['image'].unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                heatmap, offset = model(img_tensor)

        # Decode heatmap peaks → detections
        batch_dets = decode_heatmap(
            heatmap, offset, stride=stride, conf_thresh=conf_thresh
        )
        detections = batch_dets[0]  # Single image batch

        # Scale coordinates back to original image space
        scale_x = orig_w / input_size
        scale_y = orig_h / input_size

        for det in detections:
            all_predictions.append({
                'image_filename': img_path.name,
                'x': round(det['x'] * scale_x, 1),
                'y': round(det['y'] * scale_y, 1),
                'width': FIXED_BOX_SIZE,
                'height': FIXED_BOX_SIZE,
                'conf': round(det['conf'], 4),
                'class': det['class'],
            })

    # Create DataFrame
    df = pd.DataFrame(all_predictions)
    print(f"\n✓ Generated {len(df)} predictions from {len(image_files)} images")

    return df


def generate_submission(predictions_df, output_path, sample_submission_path=None):
    """
    Generate submission CSV matching competition format.

    Format: image_filename, x, y, width, height, conf, class
    (matches sample_submission.csv)
    """
    submission_df = predictions_df.copy()

    # Ensure correct column order
    columns = ['image_filename', 'x', 'y', 'width', 'height', 'conf', 'class']
    for col in columns:
        if col not in submission_df.columns:
            raise ValueError(f"Missing required column: {col}")

    submission_df = submission_df[columns]
    submission_df.to_csv(output_path, index=False)

    print(f"\n✓ Submission saved to: {output_path}")
    print(f"  • Total predictions: {len(submission_df)}")
    print(f"  • Unique images: {submission_df['image_filename'].nunique()}")

    # Class distribution
    print("\n  Class distribution:")
    class_counts = submission_df['class'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        class_name = CLASS_NAMES.get(class_id, 'Unknown')
        print(f"    {class_id} ({class_name:5s}): {count:5d}")

    # Validate against sample submission
    if sample_submission_path and os.path.exists(sample_submission_path):
        sample_df = pd.read_csv(sample_submission_path)
        if list(sample_df.columns) == list(submission_df.columns):
            print("\n  ✓ Format validated against sample submission")
        else:
            print(f"\n  ⚠️ Column mismatch with sample submission!")
            print(f"     Expected: {list(sample_df.columns)}")
            print(f"     Got:      {list(submission_df.columns)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='CenterNet Inference & Submission Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inference.py
  python scripts/inference.py --model trained_models/best_latest.pth
  python scripts/inference.py --conf 0.2 --imgsz 1024
        """
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (default: auto-detect)')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3, try 0.2 for higher recall)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Input image size (default: 1024)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base project directory')
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()

    print("""
    ==============================================================
         RIVA Cell Detection — CenterNet Inference
    ==============================================================
    """)

    # Auto-detect model if not specified
    model_path = args.model
    if model_path is None:
        trained_dir = base_dir / 'trained_models'
        runs_dir = base_dir / 'runs' / 'centernet'

        candidates = [
            trained_dir / 'best_latest.pth',
        ]

        # Check for timestamped models
        if trained_dir.exists():
            pth_files = glob.glob(str(trained_dir / 'best_centernet_*.pth'))
            if pth_files:
                most_recent = max(pth_files, key=lambda p: Path(p).stat().st_mtime)
                candidates.insert(1, Path(most_recent))

        # Check run directories
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                best_pth = run_dir / 'best.pth'
                if best_pth.exists():
                    candidates.append(best_pth)

        for mp in candidates:
            if mp.exists():
                model_path = str(mp)
                print(f"  Auto-detected model: {model_path}")
                break

        if model_path is None:
            print("[ERROR] No trained model found!")
            print("  Train first: python run.py train --mode test")
            print("  Or specify:  --model path/to/best.pth")
            return

    # Paths
    test_dir = base_dir / 'dataset' / 'images' / 'images' / 'test'
    output_dir = base_dir / 'outputs' / 'inference'
    sample_sub = base_dir / 'dataset' / 'annotations' / 'annotations' / 'sample_submission.csv'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = output_dir / f'submission_{timestamp}.csv'

    # Run inference
    predictions_df = run_inference(
        model_path=model_path,
        test_images_dir=str(test_dir),
        output_dir=str(output_dir),
        conf_thresh=args.conf,
        input_size=args.imgsz,
        device_id=args.device,
    )

    if len(predictions_df) == 0:
        print("\n[WARNING] No predictions generated!")
        print("  Try lowering --conf threshold (e.g., --conf 0.1)")
        return

    # Generate submission
    generate_submission(
        predictions_df=predictions_df,
        output_path=str(submission_path),
        sample_submission_path=str(sample_sub),
    )

    print(f"\n{'='*60}")
    print("[SUCCESS] INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"  Submission: {submission_path}")


if __name__ == '__main__':
    main()
