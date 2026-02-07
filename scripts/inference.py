"""
Run inference on test images and generate submission CSV.

Includes Test-Time Augmentation (TTA) for improved robustness.

Usage:
    # Using best model with TTA (recommended)
    python scripts/inference.py --tta
    
    # Without TTA (faster)
    python scripts/inference.py
    
    # Specify a custom model
    python scripts/inference.py --model runs/detect/riva_yolo11m_generalized/weights/best.pt
    
    # Adjust confidence threshold
    python scripts/inference.py --conf 0.25 --iou 0.45
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2


# Class mapping (ID to name)
CLASS_NAMES = {
    0: 'NILM',
    1: 'ENDO', 
    2: 'INFL',
    3: 'ASCUS',
    4: 'LSIL',
    5: 'HSIL',
    6: 'ASCH',
    7: 'SCC'
}


def apply_tta_transforms(image_path):
    """
    Generate Test-Time Augmentation (TTA) variants of an image.
    
    Returns:
        List of (image, transform_type) tuples
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return [(image_path, 'original')]
    
    transforms = []
    
    # 1. Original
    transforms.append((img, 'original'))
    
    # 2. Horizontal flip
    transforms.append((cv2.flip(img, 1), 'hflip'))
    
    # 3. Vertical flip
    transforms.append((cv2.flip(img, 0), 'vflip'))
    
    # 4. Both flips
    transforms.append((cv2.flip(cv2.flip(img, 0), 1), 'hvflip'))
    
    # 5. Brightness increase (+10%)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    transforms.append((cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), 'bright'))
    
    # 6. Brightness decrease (-10%)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.9, 0, 255)
    transforms.append((cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), 'dark'))
    
    return transforms


def reverse_tta_transform(boxes_data, transform_type, img_width, img_height):
    """
    Reverse the TTA transformation on bounding box coordinates.
    
    Args:
        boxes_data: List of dicts with box coordinates
        transform_type: Type of transform that was applied
        img_width: Image width
        img_height: Image height
    
    Returns:
        List of dicts with transformed coordinates
    """
    if transform_type == 'original' or transform_type == 'bright' or transform_type == 'dark':
        return boxes_data
    
    reversed_boxes = []
    for box in boxes_data:
        x, y = box['x'], box['y']
        
        # Reverse horizontal flip
        if 'hflip' in transform_type:
            x = img_width - x
        
        # Reverse vertical flip
        if 'vflip' in transform_type:
            y = img_height - y
        
        reversed_boxes.append({
            'x': x,
            'y': y,
            'width': box['width'],
            'height': box['height'],
            'conf': box['conf'],
            'class': box['class']
        })
    
    return reversed_boxes


def merge_tta_predictions(all_predictions, iou_threshold=0.5):
    """
    Merge predictions from multiple TTA variants using Weighted Boxes Fusion.
    
    Simple implementation: Average coordinates of overlapping boxes, take max confidence.
    
    Args:
        all_predictions: List of prediction dicts from different TTA variants
        iou_threshold: IoU threshold for considering boxes as the same detection
    
    Returns:
        Merged list of predictions
    """
    if len(all_predictions) == 0:
        return []
    
    # Simple averaging approach
    merged = []
    used = set()
    
    for i, pred1 in enumerate(all_predictions):
        if i in used:
            continue
        
        # Find all overlapping predictions
        group = [pred1]
        used.add(i)
        
        for j, pred2 in enumerate(all_predictions):
            if j <= i or j in used:
                continue
            
            # Check if same class
            if pred1['class'] != pred2['class']:
                continue
            
            # Compute IoU
            x1_min = pred1['x'] - pred1['width'] / 2
            y1_min = pred1['y'] - pred1['height'] / 2
            x1_max = pred1['x'] + pred1['width'] / 2
            y1_max = pred1['y'] + pred1['height'] / 2
            
            x2_min = pred2['x'] - pred2['width'] / 2
            y2_min = pred2['y'] - pred2['height'] / 2
            x2_max = pred2['x'] + pred2['width'] / 2
            y2_max = pred2['y'] + pred2['height'] / 2
            
            # Intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                
                area1 = pred1['width'] * pred1['height']
                area2 = pred2['width'] * pred2['height']
                union_area = area1 + area2 - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou >= iou_threshold:
                    group.append(pred2)
                    used.add(j)
        
        # Merge group by averaging
        avg_pred = {
            'x': np.mean([p['x'] for p in group]),
            'y': np.mean([p['y'] for p in group]),
            'width': np.mean([p['width'] for p in group]),
            'height': np.mean([p['height'] for p in group]),
            'conf': np.max([p['conf'] for p in group]),  # Take max confidence
            'class': pred1['class']
        }
        merged.append(avg_pred)
    
    return merged


def run_inference(
    model_path: str,
    test_images_dir: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 1024,
    device: int = 0,
    save_visualizations: bool = True,
    use_tta: bool = False,
    fixed_anchor: bool = False,
    fixed_box_size: int = 100
) -> pd.DataFrame:
    """
    Run inference on test images and collect predictions.
    
    Args:
        model_path: Path to trained model weights
        test_images_dir: Directory containing test images
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        imgsz: Image size for inference
        device: GPU device ID
        save_visualizations: Whether to save annotated images
        use_tta: Whether to use Test-Time Augmentation
        fixed_anchor: If True, override predicted box sizes with fixed_box_size
        fixed_box_size: Fixed box size in pixels (default: 100)
    
    Returns:
        DataFrame with all predictions
    """
    from ultralytics import YOLO
    
    print(f"\n[1/3] Loading model from: {model_path}")
    model = YOLO(model_path)
    print("  ✓ Model loaded successfully")
    
    # Get test images
    test_dir = Path(test_images_dir)
    image_files = sorted(list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg')))
    print(f"\n[2/3] Found {len(image_files)} test images")
    
    if len(image_files) == 0:
        print("  ⚠️ No images found in test directory!")
        return pd.DataFrame()
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_visualizations:
        vis_dir = output_path / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    # Collect all predictions
    all_predictions = []
    prediction_id = 0
    
    print(f"\n[3/3] Running inference...")
    print(f"  - Confidence threshold: {conf_threshold}")
    print(f"  - IoU threshold: {iou_threshold}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Test-Time Augmentation: {'ENABLED (6 variants)' if use_tta else 'DISABLED'}")
    if fixed_anchor:
        print(f"  - FIXED ANCHOR MODE: Forcing all boxes to {fixed_box_size}x{fixed_box_size}px")
    print("-" * 50)
    
    for img_path in tqdm(image_files, desc="Processing"):
        if use_tta:
            # Test-Time Augmentation: Run inference on multiple variants
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            all_tta_predictions = []
            
            # Generate TTA variants
            transforms = apply_tta_transforms(img_path)
            
            for aug_img, transform_type in transforms:
                # Save temporary augmented image
                temp_path = output_path / f'temp_{img_path.stem}_{transform_type}.png'
                cv2.imwrite(str(temp_path), aug_img)
                
                # Run inference on augmented image
                results = model.predict(
                    source=str(temp_path),
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                    save=False
                )
                
                # Clean up temp file
                temp_path.unlink()
                
                # Extract predictions for this transform
                transform_predictions = []
                for result in results:
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        width = x2 - x1
                        height = y2 - y1
                        x_center = x1 + width / 2
                        y_center = y1 + height / 2
                        
                        conf = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        transform_predictions.append({
                            'x': x_center,
                            'y': y_center,
                            'width': width,
                            'height': height,
                            'conf': conf,
                            'class': class_id
                        })
                
                # Reverse transform to get coordinates in original image space
                reversed_preds = reverse_tta_transform(transform_predictions, transform_type, img_width, img_height)
                all_tta_predictions.extend(reversed_preds)
            
            # Merge overlapping predictions from different TTA variants
            merged_predictions = merge_tta_predictions(all_tta_predictions, iou_threshold=0.5)
            
            # Add merged predictions to final list
            for pred in merged_predictions:
                # Use fixed box size if enabled, otherwise use predicted size
                out_width = fixed_box_size if fixed_anchor else pred['width']
                out_height = fixed_box_size if fixed_anchor else pred['height']
                
                all_predictions.append({
                    'id': prediction_id,
                    'image_filename': img_path.name,
                    'x': round(pred['x'], 1),
                    'y': round(pred['y'], 1),
                    'width': round(out_width, 1),
                    'height': round(out_height, 1),
                    'conf': round(pred['conf'], 4),
                    'class': pred['class']
                })
                prediction_id += 1
        
        else:
            # Standard inference (no TTA)
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
                save=save_visualizations,
                project=str(output_path / 'predictions') if save_visualizations else None,
                name='images',
                exist_ok=True
            )
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                # Get image dimensions
                img_height, img_width = result.orig_shape
                
                for i in range(len(boxes)):
                    # Get box coordinates (already in xyxy format)
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Convert to center format (x, y, width, height)
                    pred_width = x2 - x1
                    pred_height = y2 - y1
                    x_center = x1 + pred_width / 2
                    y_center = y1 + pred_height / 2
                    
                    # Use fixed box size if enabled, otherwise use predicted size
                    out_width = fixed_box_size if fixed_anchor else pred_width
                    out_height = fixed_box_size if fixed_anchor else pred_height
                    
                    # Get confidence and class
                    conf = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Add prediction
                    all_predictions.append({
                        'id': prediction_id,
                        'image_filename': img_path.name,
                        'x': round(x_center, 1),
                        'y': round(y_center, 1),
                        'width': round(out_width, 1),
                        'height': round(out_height, 1),
                        'conf': round(conf, 4),
                        'class': class_id
                    })
                    prediction_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(all_predictions)
    
    print(f"\n✓ Generated {len(df)} predictions from {len(image_files)} images")
    
    return df


def generate_submission(
    predictions_df: pd.DataFrame,
    output_path: str,
    sample_submission_path: str = None
) -> str:
    """
    Generate submission CSV file with sequential 'id' column.
    
    Args:
        predictions_df: DataFrame with predictions
        output_path: Path to save submission CSV
        sample_submission_path: Optional path to sample submission for format validation
    
    Returns:
        Path to saved submission file
    """
    # Create a copy to avoid modifying original
    submission_df = predictions_df.copy()
    
    # Add sequential 'id' column starting from 1
    submission_df['id'] = range(1, len(submission_df) + 1)
    
    # Ensure correct column order: id, image_filename, x, y, width, height, conf, class
    columns = ['id', 'image_filename', 'x', 'y', 'width', 'height', 'conf', 'class']
    
    # Validate columns exist (except id which we just added)
    for col in columns[1:]:  # Skip 'id'
        if col not in submission_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Reorder columns
    submission_df = submission_df[columns]
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Submission saved to: {output_path}")
    print(f"  • Total predictions: {len(submission_df)}")
    print(f"  • Unique images: {submission_df['image_filename'].nunique()}")
    
    # Print class distribution
    print("\n  Class distribution:")
    class_counts = submission_df['class'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        class_name = CLASS_NAMES.get(class_id, 'Unknown')
        print(f"    {class_id} ({class_name:5s}): {count:5d}")
    
    # Validate format if sample submission provided
    if sample_submission_path and os.path.exists(sample_submission_path):
        sample_df = pd.read_csv(sample_submission_path)
        if list(sample_df.columns) == list(submission_df.columns):
            print("\n  ✓ Format validated against sample submission")
        else:
            print(f"\n  ⚠️ Column mismatch with sample submission")
            print(f"     Expected: {list(sample_df.columns)}")
            print(f"     Got: {list(submission_df.columns)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Run inference and generate submission for RIVA competition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with default settings
  python scripts/inference.py
  
  # Use specific model
  python scripts/inference.py --model path/to/best.pt
  
  # Adjust thresholds
  python scripts/inference.py --conf 0.3 --iou 0.5
  
  # Skip visualizations for faster inference
  python scripts/inference.py --no-vis
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default=None,
        help='Path to model weights (default: auto-detect from runs/)'
    )
    parser.add_argument(
        '--conf', 
        type=float, 
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou', 
        type=float, 
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=1024,
        help='Image size for inference (default: 1024 - matches training, preserves cell details)'
    )
    parser.add_argument(
        '--device', 
        type=int, 
        default=0,
        help='GPU device ID (default: 0)'
    )
    parser.add_argument(
        '--tta', 
        action='store_true',
        help='Enable Test-Time Augmentation (6 variants - improves robustness)'
    )
    parser.add_argument(
        '--no-vis', 
        action='store_true',
        help='Skip saving visualization images'
    )
    parser.add_argument(
        '--fixed-anchor',
        action='store_true',
        help='Force fixed box size (100x100) for all predictions. Use with models trained in fixedanchor mode.'
    )
    parser.add_argument(
        '--box-size',
        type=int,
        default=100,
        help='Fixed box size in pixels when using --fixed-anchor (default: 100)'
    )
    parser.add_argument(
        '--base-dir', 
        type=str, 
        default='.',
        help='Base project directory'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    
    print("""
    ==============================================================
         RIVA Cell Detection - Inference & Submission          
    ==============================================================
    """)
    
    # Find model if not specified
    model_path = args.model
    if model_path is None:
        # Try to find best model from training (prioritize trained_models folder)
        runs_dir = base_dir / 'runs' / 'detect'
        trained_models_dir = base_dir / 'trained_models'
        
        # Check trained_models folder first (for easy transfer between systems)
        possible_models = [
            trained_models_dir / 'best_latest.pt',  # Latest trained model
            base_dir / 'best.pt',  # Check root directory
            base_dir / 'last.pt',
            base_dir / 'bestA.pt',
            runs_dir / 'riva_yolo11l_fixed_anchor' / 'weights' / 'best.pt',  # Fixed anchor model
            runs_dir / 'riva_yolo11l_extreme' / 'weights' / 'best.pt',
            runs_dir / 'riva_yolo11x_extreme' / 'weights' / 'best.pt',
            runs_dir / 'riva_yolo11s_generalized' / 'weights' / 'best.pt',
            runs_dir / 'riva_yolo11m_generalized' / 'weights' / 'best.pt',
            runs_dir / 'riva_yolo11l_balanced' / 'weights' / 'best.pt',
            runs_dir / 'riva_yolo11l_baseline' / 'weights' / 'best.pt',
            runs_dir / 'riva_test' / 'weights' / 'best.pt',
        ]
        
        # Also check for timestamped models in trained_models
        import glob
        timestamped_models = glob.glob(str(trained_models_dir / 'best_fixed_anchor_*.pt'))  # Fixed anchor first
        timestamped_models += glob.glob(str(trained_models_dir / 'best_yolo11l_extreme_*.pt'))
        timestamped_models += glob.glob(str(trained_models_dir / 'best_yolo11x_*.pt'))
        timestamped_models += glob.glob(str(trained_models_dir / 'best_yolo11s_*.pt'))
        timestamped_models += glob.glob(str(trained_models_dir / 'best_yolo11m_*.pt'))
        timestamped_models += glob.glob(str(trained_models_dir / 'best_yolo11l_*.pt'))
        if timestamped_models:
            # Add the most recent one
            most_recent = max(timestamped_models, key=lambda p: Path(p).stat().st_mtime)
            possible_models.insert(1, Path(most_recent))
        
        for mp in possible_models:
            if mp.exists():
                model_path = str(mp)
                print(f"Auto-detected model: {model_path}")
                break
        
        if model_path is None:
            print("[ERROR] No trained model found!")
            print("   Please train a model first with: python scripts/train.py --mode test")
            print("   Or specify a model path with: --model trained_models/best_latest.pt")
            return
    
    # Set paths
    test_images_dir = base_dir / 'dataset' / 'images' / 'images' / 'test'
    output_dir = base_dir / 'outputs' / 'inference'
    sample_submission = base_dir / 'dataset' / 'annotations' / 'annotations' / 'sample_submission.csv'
    
    # Timestamp for submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = output_dir / f'submission_{timestamp}.csv'
    
    # Print fixed anchor info if enabled
    if args.fixed_anchor:
        print(f"\n*** FIXED ANCHOR MODE ***")
        print(f"All predictions will use {args.box_size}x{args.box_size}px boxes")
        print(f"(Matches ground truth: all boxes are 100x100 pixels)")
    
    # Run inference
    predictions_df = run_inference(
        model_path=model_path,
        test_images_dir=str(test_images_dir),
        output_dir=str(output_dir),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_visualizations=not args.no_vis,
        use_tta=args.tta,
        fixed_anchor=args.fixed_anchor,
        fixed_box_size=args.box_size
    )
    
    if len(predictions_df) == 0:
        print("\n[WARNING] No predictions generated!")
        print("   This could mean:")
        print("   1. No objects detected (try lowering --conf threshold)")
        print("   2. No test images found")
        print("   3. Model issue")
        return
    
    # Generate submission
    generate_submission(
        predictions_df=predictions_df,
        output_path=str(submission_path),
        sample_submission_path=str(sample_submission)
    )
    
    print("\n" + "=" * 60)
    print("[SUCCESS] INFERENCE COMPLETE")
    print("=" * 60)
    print(f"\nSubmission file: {submission_path}")
    if not args.no_vis:
        print(f"Visualizations: {output_dir / 'predictions' / 'images'}")


if __name__ == '__main__':
    main()
