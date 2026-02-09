"""
Analyze RIVA dataset statistics and visualize class distribution.

Usage:
    python scripts/analyze_dataset.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import argparse


# Class mapping
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


def analyze_csv(csv_path: str, split_name: str) -> dict:
    """Analyze annotations from CSV file."""
    df = pd.read_csv(csv_path)
    
    stats = {
        'split': split_name,
        'total_annotations': len(df),
        'unique_images': df['image_filename'].nunique(),
        'class_distribution': df['class_name'].value_counts().to_dict(),
        'annotations_per_image': df.groupby('image_filename').size().describe().to_dict(),
        'bbox_stats': {
            'width': df['width'].describe().to_dict(),
            'height': df['height'].describe().to_dict(),
        }
    }
    
    return stats, df


def analyze_images(images_dir: str, sample_size: int = 50) -> dict:
    """Analyze image dimensions and properties."""
    image_files = list(Path(images_dir).glob('*.png'))
    
    if len(image_files) == 0:
        return {'error': 'No images found'}
    
    # Sample images for faster analysis
    sample_files = image_files[:min(sample_size, len(image_files))]
    
    widths = []
    heights = []
    
    for img_path in tqdm(sample_files, desc="Analyzing images"):
        try:
            with Image.open(img_path) as img:
                widths.append(img.size[0])
                heights.append(img.size[1])
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    return {
        'total_images': len(image_files),
        'sampled': len(sample_files),
        'width_stats': {
            'min': min(widths),
            'max': max(widths),
            'mean': np.mean(widths),
            'std': np.std(widths)
        },
        'height_stats': {
            'min': min(heights),
            'max': max(heights),
            'mean': np.mean(heights),
            'std': np.std(heights)
        }
    }


def plot_class_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, save_path: str):
    """Plot class distribution for train and validation sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color palette
    colors = sns.color_palette("husl", 8)
    
    # Training set
    train_counts = train_df['class_name'].value_counts()
    axes[0].bar(train_counts.index, train_counts.values, color=colors)
    axes[0].set_title('Training Set Class Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add count labels
    for i, (idx, val) in enumerate(train_counts.items()):
        axes[0].text(i, val + 50, str(val), ha='center', fontsize=9)
    
    # Validation set
    val_counts = val_df['class_name'].value_counts()
    axes[1].bar(val_counts.index, val_counts.values, color=colors)
    axes[1].set_title('Validation Set Class Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    for i, (idx, val) in enumerate(val_counts.items()):
        axes[1].text(i, val + 10, str(val), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved class distribution plot: {save_path}")


def plot_annotations_per_image(train_df: pd.DataFrame, val_df: pd.DataFrame, save_path: str):
    """Plot distribution of annotations per image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training
    train_per_img = train_df.groupby('image_filename').size()
    axes[0].hist(train_per_img, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(train_per_img.mean(), color='red', linestyle='--', label=f'Mean: {train_per_img.mean():.1f}')
    axes[0].set_title('Training: Annotations per Image', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Number of Annotations')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Validation
    val_per_img = val_df.groupby('image_filename').size()
    axes[1].hist(val_per_img, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(val_per_img.mean(), color='red', linestyle='--', label=f'Mean: {val_per_img.mean():.1f}')
    axes[1].set_title('Validation: Annotations per Image', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Number of Annotations')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved annotations distribution plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze RIVA dataset')
    parser.add_argument('--base-dir', type=str, default='.', help='Base project directory')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Paths
    annotations_dir = base_dir / 'dataset' / 'annotations' / 'annotations'
    images_base_dir = base_dir / 'dataset' / 'images' / 'images'
    output_dir = base_dir / 'outputs' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("RIVA Dataset Analysis")
    print("=" * 60)
    
    # Analyze annotations
    print("\n[1/4] Analyzing Training Annotations...")
    train_stats, train_df = analyze_csv(
        str(annotations_dir / 'train.csv'), 
        'train'
    )
    
    print(f"  • Total annotations: {train_stats['total_annotations']}")
    print(f"  • Unique images: {train_stats['unique_images']}")
    
    print("\n[2/4] Analyzing Validation Annotations...")
    val_stats, val_df = analyze_csv(
        str(annotations_dir / 'val.csv'), 
        'val'
    )
    
    print(f"  • Total annotations: {val_stats['total_annotations']}")
    print(f"  • Unique images: {val_stats['unique_images']}")
    
    # Analyze images
    print("\n[3/4] Analyzing Image Dimensions...")
    train_img_stats = analyze_images(str(images_base_dir / 'train'))
    print(f"  • Train images: {train_img_stats.get('total_images', 'N/A')}")
    print(f"  • Typical size: {train_img_stats['width_stats']['mean']:.0f} x {train_img_stats['height_stats']['mean']:.0f}")
    
    test_images = list((images_base_dir / 'test').glob('*.png'))
    print(f"  • Test images: {len(test_images)}")
    
    # Generate plots
    print("\n[4/4] Generating Visualizations...")
    plot_class_distribution(
        train_df, val_df, 
        str(output_dir / 'class_distribution.png')
    )
    
    plot_annotations_per_image(
        train_df, val_df,
        str(output_dir / 'annotations_per_image.png')
    )
    
    # Print detailed summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    print("\n📊 Class Distribution (Training):")
    print("-" * 40)
    for class_name in CLASS_NAMES.values():
        count = train_stats['class_distribution'].get(class_name, 0)
        pct = count / train_stats['total_annotations'] * 100
        bar = '█' * int(pct / 2)
        print(f"  {class_name:8s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    print("\n📊 Class Distribution (Validation):")
    print("-" * 40)
    for class_name in CLASS_NAMES.values():
        count = val_stats['class_distribution'].get(class_name, 0)
        pct = count / val_stats['total_annotations'] * 100
        bar = '█' * int(pct / 2)
        print(f"  {class_name:8s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    print("\n📈 Annotations per Image:")
    print("-" * 40)
    train_per_img = train_df.groupby('image_filename').size()
    val_per_img = val_df.groupby('image_filename').size()
    print(f"  Train - Mean: {train_per_img.mean():.1f}, Max: {train_per_img.max()}, Min: {train_per_img.min()}")
    print(f"  Val   - Mean: {val_per_img.mean():.1f}, Max: {val_per_img.max()}, Min: {val_per_img.min()}")
    
    print("\n📐 Bounding Box Size:")
    print("-" * 40)
    print(f"  Width  - Mean: {train_df['width'].mean():.1f}, Std: {train_df['width'].std():.1f}")
    print(f"  Height - Mean: {train_df['height'].mean():.1f}, Std: {train_df['height'].std():.1f}")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
