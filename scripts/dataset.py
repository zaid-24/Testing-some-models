"""
PyTorch Dataset for CenterNet-based cervical cell detection.

Reads images and CSV annotations directly (no YOLO format conversion needed).
Generates CenterNet training targets:
    - Heatmaps: [num_classes, H/4, W/4] with Gaussian blobs at cell centers
    - Offsets:  [2, H/4, W/4] sub-pixel (dx, dy) at center pixels
    - Mask:     [H/4, W/4] binary mask of GT center locations

Data Augmentation (via albumentations):
    - Geometric: flips, rotation, shift-scale-rotate
    - Photometric: color jitter, CLAHE (staining variations), blur
    - All transforms properly handle keypoint coordinates
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


# ImageNet normalization constants (for pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class mapping
CLASS_NAMES = {
    0: 'NILM', 1: 'ENDO', 2: 'INFL', 3: 'ASCUS',
    4: 'LSIL', 5: 'HSIL', 6: 'ASCH', 7: 'SCC'
}
NUM_CLASSES = 8


def get_train_transforms(input_size):
    """
    Training augmentation pipeline with keypoint support.
    
    Includes:
    - Geometric: flips, rotation, shift-scale-rotate
    - Photometric: color jitter, CLAHE, Gaussian blur
    - Normalization + tensor conversion
    
    Keypoints are automatically transformed alongside the image.
    """
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
        # CLAHE is excellent for Pap smear staining variations
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(
        format='xy',
        label_fields=['class_labels'],
        remove_invisible=True,
    ))


def get_val_transforms():
    """Validation/inference transforms: just normalize + to tensor."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class RIVACenterNetDataset(Dataset):
    """
    PyTorch Dataset for CenterNet training on RIVA Pap smear data.
    
    Reads CSV annotations directly and generates heatmap targets on-the-fly.
    No pre-conversion to YOLO format needed.
    """

    def __init__(self, csv_path, images_dir, input_size=1024, stride=4,
                 num_classes=8, augment=True, fixed_box_size=100, min_overlap=0.7):
        """
        Args:
            csv_path:       Path to annotations CSV (train.csv or val.csv)
            images_dir:     Path to image directory
            input_size:     Model input size (images resized to this)
            stride:         Output stride (default: 4 for CenterNet)
            num_classes:    Number of cell classes (8)
            augment:        Whether to apply data augmentation
            fixed_box_size: Known fixed box size in pixels (100)
            min_overlap:    Min IoU for Gaussian radius computation (0.7)
        """
        self.images_dir = images_dir
        self.input_size = input_size
        self.stride = stride
        self.output_size = input_size // stride
        self.num_classes = num_classes
        self.augment = augment
        self.fixed_box_size = fixed_box_size
        self.min_overlap = min_overlap

        # Load annotations
        self.df = pd.read_csv(csv_path)
        self.image_filenames = self.df['image_filename'].unique().tolist()

        # Pre-group annotations by image for fast lookup
        self.annotations = {}
        for filename in self.image_filenames:
            mask = self.df['image_filename'] == filename
            self.annotations[filename] = self.df[mask][['x', 'y', 'class']].values

        # Setup transforms
        if augment:
            self.transform = get_train_transforms(input_size)
        else:
            self.transform = get_val_transforms()

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

        # 2. Resize to model input size
        img = cv2.resize(img, (self.input_size, self.input_size))

        # 3. Scale annotation coordinates to resized image
        scale_x = self.input_size / orig_w
        scale_y = self.input_size / orig_h

        anns = self.annotations[filename]
        keypoints = []
        class_labels = []

        for x, y, cls in anns:
            kp_x = float(x * scale_x)
            kp_y = float(y * scale_y)
            # Clamp to valid image bounds
            kp_x = np.clip(kp_x, 0, self.input_size - 1)
            kp_y = np.clip(kp_y, 0, self.input_size - 1)
            keypoints.append((kp_x, kp_y))
            class_labels.append(int(cls))

        # 4. Apply augmentation (transforms both image and keypoints)
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

        # 5. Generate CenterNet targets
        heatmap = torch.zeros(self.num_classes, self.output_size, self.output_size)
        offset = torch.zeros(2, self.output_size, self.output_size)
        offset_mask = torch.zeros(self.output_size, self.output_size)

        # Compute Gaussian radius from box size in heatmap space
        box_w_hm = self.fixed_box_size * scale_x / self.stride
        box_h_hm = self.fixed_box_size * scale_y / self.stride
        radius = max(0, int(gaussian_radius((box_h_hm, box_w_hm), self.min_overlap)))
        radius = max(radius, 2)  # Enforce minimum radius

        for (kp_x, kp_y), cls in zip(keypoints, class_labels):
            cls = int(cls)  # Ensure class index is integer
            # Convert to heatmap coordinates
            hm_x = kp_x / self.stride
            hm_y = kp_y / self.stride

            # Quantized center (integer pixel in heatmap)
            cx_int = int(min(hm_x, self.output_size - 1))
            cy_int = int(min(hm_y, self.output_size - 1))

            if 0 <= cx_int < self.output_size and 0 <= cy_int < self.output_size:
                # Place Gaussian blob on the class-specific heatmap channel
                draw_gaussian(heatmap[cls], (cx_int, cy_int), radius)

                # Store sub-pixel offset (fractional part of hm coordinates)
                offset[0, cy_int, cx_int] = hm_x - cx_int
                offset[1, cy_int, cx_int] = hm_y - cy_int

                # Mark this pixel as a positive location for offset loss
                offset_mask[cy_int, cx_int] = 1

        return img_tensor, heatmap, offset, offset_mask
