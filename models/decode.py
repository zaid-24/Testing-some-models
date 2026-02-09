"""
Heatmap target generation and decoding utilities for CenterNet.

Target Generation (Training):
    For each GT cell at (cx, cy) with class k:
    1. Compute Gaussian radius from box size (100x100) in heatmap space
    2. Place Gaussian blob on heatmap channel k centered at (cx/stride, cy/stride)
    3. Store sub-pixel offset at the quantized center pixel

Decoding (Inference):
    1. Apply NMS via 3x3 max pooling on heatmaps
    2. Find local maxima (peaks) above confidence threshold
    3. Add sub-pixel offset refinement
    4. Scale coordinates back to original image space
    5. Attach fixed 100x100 box -- done! No traditional NMS needed.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Compute Gaussian radius for a bounding box to ensure that a shifted
    box still has IoU >= min_overlap with the original.

    From CenterNet / CornerNet.

    For RIVA's 100x100 boxes with stride 4 (25x25 in heatmap space):
        radius ≈ 6, sigma ≈ 2.17

    Args:
        det_size: (height, width) of the box in heatmap space
        min_overlap: Minimum IoU overlap threshold (default: 0.7)

    Returns:
        Gaussian radius (float)
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0, b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0, b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0, b3 ** 2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2d(shape, sigma=1.0):
    """
    Generate a 2D Gaussian kernel.

    Args:
        shape: (height, width) of the kernel
        sigma: Standard deviation of the Gaussian

    Returns:
        numpy array of shape (height, width)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draw a Gaussian blob on the heatmap at the given center.
    Uses element-wise max to handle overlapping cells
    (multiple cells near each other don't cancel out).

    Args:
        heatmap: [H, W] single-channel heatmap tensor
        center: (x, y) center coordinates in heatmap space (integer)
        radius: Gaussian radius in pixels
        k: Peak value multiplier (default: 1)

    Returns:
        Modified heatmap (in-place)
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)
    gaussian = torch.from_numpy(gaussian).float()

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape

    # Clip the Gaussian to fit within heatmap boundaries
    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def decode_heatmap(heatmap, offset, stride=4, conf_thresh=0.3, nms_kernel=3):
    """
    Extract cell detections from predicted heatmaps.
    No traditional NMS needed — peaks in heatmaps are naturally separated.

    Process:
    1. Apply max pooling for local NMS (keep only local maxima)
    2. Threshold to remove low-confidence detections
    3. Add sub-pixel offset refinement
    4. Scale to input image coordinates

    Args:
        heatmap: [B, C, H, W] sigmoid-activated class heatmaps
        offset:  [B, 2, H, W] sub-pixel offsets (dx, dy)
        stride:  Output stride (default: 4)
        conf_thresh: Minimum confidence to keep a detection
        nms_kernel:  Max pooling kernel for local NMS (default: 3)

    Returns:
        List of lists (one per batch item), each containing dicts:
        {'x': float, 'y': float, 'class': int, 'conf': float}
        Coordinates are in the INPUT IMAGE space (before any resize).
    """
    # NMS via max pooling: only keep local maxima
    pad = nms_kernel // 2
    hmax = F.max_pool2d(heatmap, nms_kernel, stride=1, padding=pad)
    peaks = (heatmap == hmax) & (heatmap > conf_thresh)

    batch_size = heatmap.shape[0]
    num_classes = heatmap.shape[1]
    detections = []

    for b in range(batch_size):
        batch_dets = []
        for cls in range(num_classes):
            ys, xs = torch.where(peaks[b, cls])
            for y_idx, x_idx in zip(ys, xs):
                conf = heatmap[b, cls, y_idx, x_idx].item()

                # Sub-pixel refinement from offset head
                dx = offset[b, 0, y_idx, x_idx].item()
                dy = offset[b, 1, y_idx, x_idx].item()

                # Scale to input image coordinates (multiply by stride)
                real_x = (x_idx.item() + dx) * stride
                real_y = (y_idx.item() + dy) * stride

                batch_dets.append({
                    'x': real_x,
                    'y': real_y,
                    'class': cls,
                    'conf': conf,
                })
        detections.append(batch_dets)

    return detections
