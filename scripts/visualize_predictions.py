"""
Visualize predictions on images for debugging and analysis.

Usage:
    # Visualize predictions on validation set
    python scripts/visualize_predictions.py --split val
    
    # Visualize specific images
    python scripts/visualize_predictions.py --images image1.png image2.png
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random


# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),      # NILM - Green
    1: (255, 165, 0),    # ENDO - Orange
    2: (0, 255, 255),    # INFL - Yellow
    3: (255, 0, 0),      # ASCUS - Blue
    4: (0, 165, 255),    # LSIL - Orange-Red
    5: (0, 0, 255),      # HSIL - Red
    6: (255, 0, 255),    # ASCH - Magenta
    7: (128, 0, 128),    # SCC - Purple
}

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


def draw_boxes(image, boxes, box_type='ground_truth'):
    """
    Draw bounding boxes on image.
    
    Args:
        image: OpenCV image (BGR)
        boxes: List of (class_id, x_center, y_center, width, height, [conf])
        box_type: 'ground_truth' (solid) or 'prediction' (dashed)
    
    Returns:
        Image with boxes drawn
    """
    img = image.copy()
    height, width = img.shape[:2]
    
    for box in boxes:
        if len(box) >= 5:
            class_id = int(box[0])
            x_center, y_center, w, h = box[1:5]
            conf = box[5] if len(box) > 5 else None
            
            # Convert to pixel coordinates if normalized
            if x_center <= 1 and y_center <= 1:
                x_center *= width
                y_center *= height
                w *= width
                h *= height
            
            # Calculate corners
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            
            # Get color
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            
            # Draw box
            if box_type == 'ground_truth':
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            else:
                # Dashed rectangle for predictions
                for i in range(0, x2 - x1, 10):
                    cv2.line(img, (x1 + i, y1), (min(x1 + i + 5, x2), y1), color, 2)
                    cv2.line(img, (x1 + i, y2), (min(x1 + i + 5, x2), y2), color, 2)
                for i in range(0, y2 - y1, 10):
                    cv2.line(img, (x1, y1 + i), (x1, min(y1 + i + 5, y2)), color, 2)
                    cv2.line(img, (x2, y1 + i), (x2, min(y1 + i + 5, y2)), color, 2)
            
            # Draw label
            label = CLASS_NAMES.get(class_id, str(class_id))
            if conf is not None:
                label = f"{label} {conf:.2f}"
            
            # Label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img, 
                (x1, y1 - label_h - 5), 
                (x1 + label_w, y1), 
                color, 
                -1
            )
            cv2.putText(
                img, label, 
                (x1, y1 - 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1
            )
    
    return img


def visualize_from_csv(
    csv_path: str,
    images_dir: str,
    output_dir: str,
    num_samples: int = 10,
    random_sample: bool = True
):
    """Visualize annotations from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Get unique images
    unique_images = df['image_filename'].unique()
    
    if random_sample and len(unique_images) > num_samples:
        selected_images = random.sample(list(unique_images), num_samples)
    else:
        selected_images = unique_images[:num_samples]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in tqdm(selected_images, desc="Visualizing"):
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Get annotations for this image
        img_df = df[df['image_filename'] == img_name]
        
        boxes = []
        for _, row in img_df.iterrows():
            # Check if it's a prediction (has 'conf') or ground truth
            if 'conf' in row:
                boxes.append([
                    row['class'],
                    row['x'], row['y'],
                    row['width'], row['height'],
                    row['conf']
                ])
            else:
                boxes.append([
                    row['class'],
                    row['x'], row['y'],
                    row['width'], row['height']
                ])
        
        # Draw boxes
        box_type = 'prediction' if 'conf' in df.columns else 'ground_truth'
        img_with_boxes = draw_boxes(img, boxes, box_type)
        
        # Save
        output_path = os.path.join(output_dir, f"vis_{img_name}")
        cv2.imwrite(output_path, img_with_boxes)
    
    print(f"\n✓ Saved {len(selected_images)} visualizations to: {output_dir}")


def create_legend(output_path: str):
    """Create a color legend for the classes."""
    legend_height = 40 * len(CLASS_NAMES)
    legend = np.ones((legend_height, 300, 3), dtype=np.uint8) * 255
    
    for i, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
        y = i * 40 + 25
        color = CLASS_COLORS[class_id]
        
        # Draw color box
        cv2.rectangle(legend, (10, y - 15), (40, y + 5), color, -1)
        cv2.rectangle(legend, (10, y - 15), (40, y + 5), (0, 0, 0), 1)
        
        # Draw text
        cv2.putText(
            legend, 
            f"{class_id}: {class_name}", 
            (50, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (0, 0, 0), 1
        )
    
    cv2.imwrite(output_path, legend)
    print(f"✓ Legend saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize predictions and annotations')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val',
                        help='Dataset split to visualize')
    parser.add_argument('--source', type=str, choices=['csv', 'predictions'],
                        default='csv', help='Annotation source format')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of images to visualize')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base project directory')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    output_base = base_dir / 'outputs' / 'visualizations'
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           RIVA Dataset Visualization                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Create legend
    output_base.mkdir(parents=True, exist_ok=True)
    create_legend(str(output_base / 'class_legend.png'))
    
    if args.source == 'csv':
        csv_path = base_dir / 'dataset' / 'annotations' / 'annotations' / f'{args.split}.csv'
        images_dir = base_dir / 'dataset' / 'images' / 'images' / args.split
        output_dir = output_base / f'{args.split}_csv'
        
        visualize_from_csv(
            csv_path=str(csv_path),
            images_dir=str(images_dir),
            output_dir=str(output_dir),
            num_samples=args.num_samples
        )
    
    elif args.source == 'predictions':
        # Find latest submission
        predictions_dir = base_dir / 'outputs' / 'inference'
        submissions = sorted(predictions_dir.glob('submission_*.csv'))
        
        if not submissions:
            print("No submission files found!")
            return
        
        latest_submission = submissions[-1]
        print(f"Using: {latest_submission}")
        
        images_dir = base_dir / 'dataset' / 'images' / 'images' / 'test'
        output_dir = output_base / 'predictions'
        
        visualize_from_csv(
            csv_path=str(latest_submission),
            images_dir=str(images_dir),
            output_dir=str(output_dir),
            num_samples=args.num_samples
        )


if __name__ == '__main__':
    main()
