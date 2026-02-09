"""
Two--Stage Inference Pipeline for Cervical Cell Detection.

Process:
    1. Stage 1 (Detector): Binary CenterNet finds all cell centers
    2. Stage 2 (Classifier): Crops 100x100 patches -> classifies into 8 classes
    3. Combines results -> generates submission CSV

Output format:
    image_filename, x, y, width, height, conf, class
    (width and height are always 100 -- fixed box size)

Usage:
    python scripts/inference.py --conf 0.3
    python scripts/inference.py --det-model path/to/detector.pth --cls-model path/to/classifier.pth
    python scripts/inference.py --conf 0.2  # higher recall
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
from models.classifier import CellClassifier
from models.decode import decode_heatmap

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = {
    0: 'NILM', 1: 'ENDO', 2: 'INFL', 3: 'ASCUS',
    4: 'LSIL', 5: 'HSIL', 6: 'ASCH', 7: 'SCC'
}

FIXED_BOX_SIZE = 100


def get_detection_transform():
    """Preprocessing for the detector."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_classification_transform(input_size=224):
    """Preprocessing for the classifier (resize + normalize)."""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_detector(model_path, device):
    """Load trained detector model from checkpoint."""
    print(f"  Loading: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    config = ckpt.get('config', {})
    backbone = config.get('backbone', 'resnet50')
    input_size = config.get('input_size', 1024)

    model = CellCenterNet(num_classes=1, backbone=backbone, pretrained=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device).eval()
    print(f"  [OK] Detector loaded ({backbone}, input={input_size})")
    return model, input_size, backbone


def load_classifier(model_path, device):
    """Load trained classifier model from checkpoint."""
    print(f"  Loading: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    config = ckpt.get('config', {})
    backbone = config.get('backbone', 'efficientnet_b2')
    input_size = config.get('input_size', 224)

    model = CellClassifier(num_classes=8, backbone=backbone, pretrained=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device).eval()
    print(f"  [OK] Classifier loaded ({backbone}, input={input_size})")
    return model, input_size, backbone


def extract_patch(image, cx, cy, patch_size=100):
    """
    Extract a patch from the image centered at (cx, cy).
    Pads with black pixels if the crop extends beyond image boundaries.
    """
    h, w = image.shape[:2]
    half = patch_size // 2
    x1 = int(round(cx -- half))
    y1 = int(round(cy -- half))
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    # Compute padding needed
    pad_left = max(0, --x1)
    pad_top = max(0, --y1)
    pad_right = max(0, x2 -- w)
    pad_bottom = max(0, y2 -- h)

    # Clamp to image bounds
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)

    crop = image[y1c:y2c, x1c:x2c]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    return crop


def run_inference(det_model_path, cls_model_path, test_images_dir, output_dir,
                  conf_thresh=0.3, det_input_size=1024, cls_input_size=224,
                  device_id=0):
    """
    Run two--stage inference on test images.

    Stage 1: Detect cell centers (binary heatmap)
    Stage 2: Classify each detected cell (100x100 patch -> 8 classes)
    
    Returns:
        DataFrame with columns: image_filename, x, y, width, height, conf, class
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    # === Load models ===
    print(f"\n[1/4] Loading Stage 1: Detector")
    det_model, det_size, det_backbone = load_detector(det_model_path, device)
    det_input_size = det_size  # Use size from checkpoint

    print(f"\n[2/4] Loading Stage 2: Classifier")
    cls_model, cls_size, cls_backbone = load_classifier(cls_model_path, device)
    cls_input_size = cls_size  # Use size from checkpoint

    # === Find test images ===
    test_dir = Path(test_images_dir)
    image_files = sorted(list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg')))
    print(f"\n[3/4] Found {len(image_files)} test images")
    
    if len(image_files) == 0:
        print("  [WARNING] No images found in test directory!")
        return pd.DataFrame()
    
    # === Transforms ===
    det_transform = get_detection_transform()
    cls_transform = get_classification_transform(cls_input_size)

    # === Run two--stage inference ===
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_predictions = []
    stride = 4

    print(f"\n[4/4] Running two--stage inference...")
    print(f"  Detection threshold:  {conf_thresh}")
    print(f"  Detector input:       {det_input_size}x{det_input_size}")
    print(f"  Classifier input:     {cls_input_size}x{cls_input_size}")
    print(f"  Fixed box size:       {FIXED_BOX_SIZE}x{FIXED_BOX_SIZE}")
    print("--" * 60)

    total_detections = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        # Load original image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [WARNING] Failed to load: {img_path.name}")
                continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # === Stage 1: Detection ===
        img_resized = cv2.resize(img_rgb, (det_input_size, det_input_size))
        transformed = det_transform(image=img_resized)
        img_tensor = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                heatmap, offset = det_model(img_tensor)

        # Decode heatmap peaks -> candidate cell centers
        batch_dets = decode_heatmap(
            heatmap, offset, stride=stride, conf_thresh=conf_thresh
        )
        detections = batch_dets[0]  # Single image batch

        if len(detections) == 0:
                        continue
                    
        # Scale detection coords to original image space
        scale_x = orig_w / det_input_size
        scale_y = orig_h / det_input_size

        # === Stage 2: Classification ===
        patches = []
        det_confs = []
        det_coords = []

        for det in detections:
            orig_cx = det['x'] * scale_x
            orig_cy = det['y'] * scale_y
            det_confs.append(det['conf'])
            det_coords.append((orig_cx, orig_cy))

            # Extract 100x100 patch from original resolution image
            patch = extract_patch(img_rgb, orig_cx, orig_cy, FIXED_BOX_SIZE)
            transformed_patch = cls_transform(image=patch)
            patches.append(transformed_patch['image'])

        # Batch classify all patches from this image
        patch_batch = torch.stack(patches).to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits = cls_model(patch_batch)
                probs = torch.softmax(logits, dim=1)

        pred_classes = probs.argmax(dim=1)
        pred_class_confs = probs.max(dim=1).values

        # Combine detection + classification results
        for i, (cx, cy) in enumerate(det_coords):
            det_conf = det_confs[i]
            cls_conf = pred_class_confs[i].item()
            # Final confidence = detection confidence x classification confidence
            combined_conf = det_conf * cls_conf
                
                all_predictions.append({
                    'image_filename': img_path.name,
                'x': round(cx, 1),
                'y': round(cy, 1),
                'width': FIXED_BOX_SIZE,
                'height': FIXED_BOX_SIZE,
                'conf': round(combined_conf, 4),
                'class': pred_classes[i].item(),
            })

        total_detections += len(detections)
    
    # Create DataFrame
    df = pd.DataFrame(all_predictions)
    print(f"\n[OK] Generated {len(df)} predictions from {len(image_files)} images")
    print(f"  Avg detections/image: {total_detections / max(len(image_files), 1):.1f}")
    
    return df


def generate_submission(predictions_df, output_path, sample_submission_path=None):
    """
    Generate submission CSV matching competition format.

    Format: image_filename, x, y, width, height, conf, class
    """
    submission_df = predictions_df.copy()
    
    # Ensure correct column order
    columns = ['image_filename', 'x', 'y', 'width', 'height', 'conf', 'class']
    for col in columns:
        if col not in submission_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    submission_df = submission_df[columns]
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Submission saved to: {output_path}")
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
            print("\n  [OK] Format validated against sample submission")
        else:
            print(f"\n  [WARNING] Column mismatch with sample submission!")
            print(f"     Expected: {list(sample_df.columns)}")
            print(f"     Got:      {list(submission_df.columns)}")
    
    return output_path


def find_model(base_dir, model_type='detector'):
    """Auto--detect the best available model."""
    base_dir = Path(base_dir)
    trained_dir = base_dir / 'trained_models'

    if model_type == 'detector':
        candidates = [trained_dir / 'best_detector_latest.pth']

        # Check for timestamped detector models
        if trained_dir.exists():
            pth_files = glob.glob(str(trained_dir / 'best_detector_*.pth'))
            if pth_files:
                most_recent = max(pth_files, key=lambda p: Path(p).stat().st_mtime)
                candidates.insert(1, Path(most_recent))

        # Check run directories
        runs_dir = base_dir / 'runs' / 'detect'
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                best = run_dir / 'best.pth'
                if best.exists():
                    candidates.append(best)

    else:  # classifier
        candidates = [trained_dir / 'best_classifier_latest.pth']

        # Check for timestamped classifier models
        if trained_dir.exists():
            pth_files = glob.glob(str(trained_dir / 'best_classifier_*.pth'))
            if pth_files:
                most_recent = max(pth_files, key=lambda p: Path(p).stat().st_mtime)
                candidates.insert(1, Path(most_recent))

        # Check run directories
        runs_dir = base_dir / 'runs' / 'classify'
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                best = run_dir / 'best.pth'
                if best.exists():
                    candidates.append(best)

    for mp in candidates:
        if mp.exists():
            return str(mp)

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Two--Stage Cell Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inference.py
  python scripts/inference.py --conf 0.3
  python scripts/inference.py --det-model trained_models/best_detector_latest.pth
  python scripts/inference.py --conf 0.2  # higher recall
        """
    )
    parser.add_argument('--det-model', type=str, default=None,
                        help='Path to detector model (default: auto--detect)')
    parser.add_argument('--cls-model', type=str, default=None,
                        help='Path to classifier model (default: auto--detect)')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Detection confidence threshold (default: 0.3)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Detector input image size (default: 1024)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base project directory')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    
    print(f"\n{'='*60}")
    print("  RIVA Cell Detection - Two-Stage Inference")
    print(f"{'='*60}")

    # Auto--detect models
    det_path = args.det_model or find_model(base_dir, 'detector')
    cls_path = args.cls_model or find_model(base_dir, 'classifier')

    if det_path is None:
        print("\n[ERROR] No detector model found!")
        print("  Train first:  python run.py train --stage detect --mode full")
        print("  Or specify:   --det-model path/to/detector.pth")
        return

    if cls_path is None:
        print("\n[ERROR] No classifier model found!")
        print("  Train first:  python run.py train --stage classify --mode full")
        print("  Or specify:   --cls-model path/to/classifier.pth")
            return
    
    print(f"  Detector:   {det_path}")
    print(f"  Classifier: {cls_path}")

    # Paths
    test_dir = base_dir / 'dataset' / 'images' / 'images' / 'test'
    output_dir = base_dir / 'outputs' / 'inference'
    sample_sub = base_dir / 'dataset' / 'annotations' / 'annotations' / 'sample_submission.csv'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = output_dir / f'submission_{timestamp}.csv'
    
    # Run two--stage inference
    predictions_df = run_inference(
        det_model_path=det_path,
        cls_model_path=cls_path,
        test_images_dir=str(test_dir),
        output_dir=str(output_dir),
        conf_thresh=args.conf,
        det_input_size=args.imgsz,
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
