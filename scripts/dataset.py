"""
PyTorch Datasets for the Two-Stage Cell Detection Pipeline.

Stage 1 — Detection Dataset (DetectionDataset):
    Reads images and CSV annotations. Generates BINARY CenterNet targets:
    - Heatmap: [1, H/4, W/4] — single channel, Gaussian blobs at ALL cell centers
    - Offset:  [2, H/4, W/4] — sub-pixel (dx, dy) at center pixels
    - Mask:    [H/4, W/4] — binary mask of GT center locations
    The detector finds cells, it does NOT classify them.

Stage 2 — Patch Classification Dataset (PatchClassificationDataset):
    Extracts 100×100 crops around cell centers (from GT annotations).
    Returns (patch_tensor, class_label) pairs for 8-class classification.
    Supports class-weighted sampling for handling severe imbalance.

Data Augmentation:
    - Detection: geometric (flips, rotation, shift-scale-rotate) + photometric
    - Classification: same + CoarseDropout for regularization
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.decode import gaussian_radius, draw_gaussian


# ImageNet normalization (for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class mapping (8 Bethesda categories)
CLASS_NAMES = {
    0: 'NILM', 1: 'ENDO', 2: 'INFL', 3: 'ASCUS',
    4: 'LSIL', 5: 'HSIL', 6: 'ASCH', 7: 'SCC'
}
NUM_CLASSES = 8


# =====================================================================
# Stage 1: Detection Dataset (Binary — cell vs. background)
# =====================================================================

def get_detection_train_transforms(input_size):
    """Training augmentations for the detector with keypoint support."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.15, rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 5)),
        ], p=0.2),
        A.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(
        format='xy',
        label_fields=['class_labels'],
        remove_invisible=True,
    ))


def get_detection_val_transforms():
    """Validation transforms for detector (normalize only)."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class DetectionDataset(Dataset):
    """
    Stage 1: Binary cell detection dataset.

    Generates a SINGLE-channel heatmap where ALL cell centers
    (regardless of class) are marked with Gaussian blobs.
    The detector only needs to find cells, not classify them.
    """

    def __init__(self, csv_path, images_dir, input_size=1024, stride=4,
                 augment=True, fixed_box_size=100, min_overlap=0.7):
        """
        Args:
            csv_path:       Path to annotations CSV (train.csv or val.csv)
            images_dir:     Path to image directory
            input_size:     Model input size (images resized to this)
            stride:         Output stride (default: 4, heatmap = input_size/4)
            augment:        Whether to apply data augmentation
            fixed_box_size: Known fixed box size in pixels (100)
            min_overlap:    Min IoU for Gaussian radius computation
        """
        self.images_dir = images_dir
        self.input_size = input_size
        self.stride = stride
        self.output_size = input_size // stride
        self.augment = augment
        self.fixed_box_size = fixed_box_size
        self.min_overlap = min_overlap

        # Load annotations
        self.df = pd.read_csv(csv_path)
        self.image_filenames = self.df['image_filename'].unique().tolist()

        # Pre-group annotations by image (only need x, y — class-agnostic)
        self.annotations = {}
        for filename in self.image_filenames:
            mask = self.df['image_filename'] == filename
            self.annotations[filename] = self.df[mask][['x', 'y']].values

        # Setup transforms
        if augment:
            self.transform = get_detection_train_transforms(input_size)
        else:
            self.transform = get_detection_val_transforms()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, filename)

        # 1. Load image
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # 2. Resize to model input
        img = cv2.resize(img, (self.input_size, self.input_size))
        scale_x = self.input_size / orig_w
        scale_y = self.input_size / orig_h

        # 3. Scale keypoints
        anns = self.annotations[filename]
        keypoints = []
        class_labels = []  # Dummy labels (all 0 for binary)

        for x, y in anns:
            kp_x = np.clip(float(x * scale_x), 0, self.input_size - 1)
            kp_y = np.clip(float(y * scale_y), 0, self.input_size - 1)
            keypoints.append((kp_x, kp_y))
            class_labels.append(0)  # All cells → channel 0

        # 4. Apply augmentation
        if self.augment:
            transformed = self.transform(
                image=img, keypoints=keypoints, class_labels=class_labels
            )
            img_tensor = transformed['image']
            keypoints = transformed['keypoints']
            class_labels = transformed['class_labels']
        else:
            transformed = self.transform(image=img)
            img_tensor = transformed['image']

        # 5. Generate BINARY heatmap target (1 channel)
        heatmap = torch.zeros(1, self.output_size, self.output_size)
        offset = torch.zeros(2, self.output_size, self.output_size)
        offset_mask = torch.zeros(self.output_size, self.output_size)

        # Gaussian radius from box size
        box_w_hm = self.fixed_box_size * scale_x / self.stride
        box_h_hm = self.fixed_box_size * scale_y / self.stride
        radius = max(2, int(gaussian_radius((box_h_hm, box_w_hm), self.min_overlap)))

        for (kp_x, kp_y), _ in zip(keypoints, class_labels):
            hm_x = kp_x / self.stride
            hm_y = kp_y / self.stride

            cx_int = int(min(hm_x, self.output_size - 1))
            cy_int = int(min(hm_y, self.output_size - 1))

            if 0 <= cx_int < self.output_size and 0 <= cy_int < self.output_size:
                # All cells go to the single heatmap channel [0]
                draw_gaussian(heatmap[0], (cx_int, cy_int), radius)

                # Sub-pixel offset
                offset[0, cy_int, cx_int] = hm_x - cx_int
                offset[1, cy_int, cx_int] = hm_y - cy_int
                offset_mask[cy_int, cx_int] = 1

        return img_tensor, heatmap, offset, offset_mask


# =====================================================================
# Stage 2: Patch Classification Dataset
# =====================================================================

def get_patch_train_transforms(input_size=224):
    """Training augmentations for the patch classifier."""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 5)),
        ], p=0.2),
        A.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_patch_val_transforms(input_size=224):
    """Validation transforms for patch classifier (resize + normalize)."""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class PatchClassificationDataset(Dataset):
    """
    Stage 2: Extracts 100×100 patches around cell centers for classification.

    Each annotation (x, y, class) becomes one sample:
        - Crop 100×100 pixels around (x, y) from the original image
        - Resize to classifier input size (224×224)
        - Apply augmentation
        - Return (patch_tensor, class_label)

    Provides sample_weights for WeightedRandomSampler to handle
    severe class imbalance (21:1 ratio between majority and minority).
    """

    def __init__(self, csv_path, images_dir, patch_size=100,
                 classifier_input_size=224, augment=True):
        """
        Args:
            csv_path:              Path to annotations CSV
            images_dir:            Path to image directory
            patch_size:            Crop size around cell center (100)
            classifier_input_size: Resize patches to this size (224)
            augment:               Whether to apply augmentation
        """
        self.images_dir = images_dir
        self.patch_size = patch_size
        self.classifier_input_size = classifier_input_size

        # Load annotations — each row is one cell
        df = pd.read_csv(csv_path)
        self.samples = []
        for _, row in df.iterrows():
            self.samples.append({
                'image_filename': row['image_filename'],
                'x': float(row['x']),
                'y': float(row['y']),
                'class': int(row['class']),
            })

        # Compute class distribution for weighted sampling
        self.class_counts = {}
        for s in self.samples:
            c = s['class']
            self.class_counts[c] = self.class_counts.get(c, 0) + 1

        max_count = max(self.class_counts.values())
        self.class_weights_dict = {
            c: max_count / count for c, count in self.class_counts.items()
        }

        # Per-sample weights for WeightedRandomSampler
        self.sample_weights = torch.DoubleTensor(
            [self.class_weights_dict[s['class']] for s in self.samples]
        )

        # Transforms
        if augment:
            self.transform = get_patch_train_transforms(classifier_input_size)
        else:
            self.transform = get_patch_val_transforms(classifier_input_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img_path = os.path.join(self.images_dir, sample['image_filename'])
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Crop 100×100 around cell center
        cx, cy = sample['x'], sample['y']
        half = self.patch_size // 2
        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = x1 + self.patch_size
        y2 = y1 + self.patch_size

        # Handle edge cases: pad with black if crop extends beyond image
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w, x2)
        y2c = min(h, y2)

        crop = img[y1c:y2c, x1c:x2c]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        # Apply transforms (resize + augment + normalize)
        transformed = self.transform(image=crop)
        img_tensor = transformed['image']

        return img_tensor, sample['class']

    def get_class_weights(self):
        """Return tensor of class weights for CrossEntropyLoss."""
        weights = torch.zeros(NUM_CLASSES)
        for c, w in self.class_weights_dict.items():
            if c < NUM_CLASSES:
                weights[c] = w
        return weights
