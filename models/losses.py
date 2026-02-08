"""
Loss functions for CenterNet-based cell detection.

Two losses:
1. Modified Focal Loss for heatmaps (from CornerNet/CenterNet)
   - Naturally handles severe class imbalance (21:1 ratio in RIVA)
   - Down-weights easy negatives (background pixels)
   - Penalty-reduced near cell centers (Gaussian weighting)

2. L1 Loss for sub-pixel offsets
   - Only supervised at GT center locations
   - Provides sub-pixel refinement for accurate center prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def centernet_focal_loss(pred_hm, gt_hm, alpha=2, beta=4):
    """
    Modified focal loss for heatmaps (CenterNet/CornerNet).
    Naturally handles class imbalance — no extra weighting needed!

    How it works:
    - Positive pixels (gt == 1, exact center): standard focal loss
    - Negative pixels (gt < 1, includes Gaussian tails + background):
      * Weighted by (1 - gt)^beta — pixels near centers get LESS penalty
      * Weighted by pred^alpha — easy negatives get LESS penalty
      * This focuses learning on hard cases: minority class centers

    Args:
        pred_hm: Predicted heatmaps [B, C, H, W], sigmoid-activated
        gt_hm:   Ground truth heatmaps [B, C, H, W], values in [0, 1]
                 Peak = 1.0 at GT centers, Gaussian falloff around them
        alpha:   Focal exponent for hard example mining (default: 2)
        beta:    Penalty reduction exponent near centers (default: 4)

    Returns:
        Scalar loss value, normalized by number of positive pixels
    """
    pos_mask = gt_hm.eq(1).float()
    neg_mask = gt_hm.lt(1).float()

    # Clamp predictions for numerical stability
    pred_hm = torch.clamp(pred_hm, min=1e-4, max=1 - 1e-4)

    # Positive loss: focal loss at exact center pixels
    # Higher loss when prediction is low (hard positive)
    pos_loss = -((1 - pred_hm) ** alpha) * torch.log(pred_hm) * pos_mask

    # Negative loss: penalty-reduced focal loss
    # (1 - gt)^beta: reduce penalty near centers (Gaussian tail)
    # pred^alpha: reduce penalty for easy negatives (low prediction)
    neg_loss = (
        -((1 - gt_hm) ** beta)
        * (pred_hm ** alpha)
        * torch.log(1 - pred_hm)
        * neg_mask
    )

    num_pos = pos_mask.sum()
    loss = (pos_loss.sum() + neg_loss.sum()) / (num_pos + 1e-4)

    return loss


def offset_l1_loss(pred_off, gt_off, mask):
    """
    L1 loss for sub-pixel offset prediction.
    Only computed at GT center locations (everywhere else is irrelevant).

    Args:
        pred_off: Predicted offsets [B, 2, H, W]
        gt_off:   Ground truth offsets [B, 2, H, W]
        mask:     Binary mask [B, H, W] indicating GT center pixels

    Returns:
        Scalar L1 loss, normalized by number of positive pixels
    """
    # Expand mask to match offset channels: [B, H, W] -> [B, 2, H, W]
    mask = mask.unsqueeze(1).expand_as(pred_off)
    num_pos = mask.sum() + 1e-4

    loss = F.l1_loss(pred_off * mask, gt_off * mask, reduction='sum') / num_pos
    return loss


class CenterNetLoss(nn.Module):
    """
    Combined CenterNet loss: focal heatmap loss + L1 offset loss.

    Total loss = hm_weight * focal_loss + off_weight * offset_loss

    Default weights follow CenterNet paper:
    - hm_weight = 1.0 (heatmap focal loss)
    - off_weight = 1.0 (offset L1 loss)
    """

    def __init__(self, hm_weight=1.0, off_weight=1.0, focal_alpha=2, focal_beta=4):
        super().__init__()
        self.hm_weight = hm_weight
        self.off_weight = off_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta

    def forward(self, pred_hm, pred_off, gt_hm, gt_off, gt_mask):
        """
        Compute combined loss.

        Args:
            pred_hm:  Predicted heatmaps [B, C, H, W]
            pred_off: Predicted offsets [B, 2, H, W]
            gt_hm:    Ground truth heatmaps [B, C, H, W]
            gt_off:   Ground truth offsets [B, 2, H, W]
            gt_mask:  Binary mask of GT centers [B, H, W]

        Returns:
            Dict with 'total', 'hm_loss', 'off_loss' keys
        """
        hm_loss = centernet_focal_loss(
            pred_hm, gt_hm, self.focal_alpha, self.focal_beta
        )
        off_loss = offset_l1_loss(pred_off, gt_off, gt_mask)

        total_loss = self.hm_weight * hm_loss + self.off_weight * off_loss

        return {
            'total': total_loss,
            'hm_loss': hm_loss.detach(),
            'off_loss': off_loss.detach(),
        }
