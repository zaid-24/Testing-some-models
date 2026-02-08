"""
CellCenterNet: CenterNet-based Point Detection for Cervical Cells.

Based on "Objects as Points" (Zhou et al., 2019).
Optimized for the RIVA dataset where ALL bounding boxes are 100x100 pixels,
making this effectively a point detection + classification problem.

Architecture:
    Input (1024x1024)
        -> ResNet Backbone (stride 32)  -> [B, 2048, 32, 32]
        -> 3x Deconv Neck (stride 4)    -> [B, 64, 256, 256]
        -> Heatmap Head (8 classes)     -> [B, 8, 256, 256]
        -> Offset Head (2 channels)     -> [B, 2, 256, 256]

Key Design:
    - No box size regression (all boxes are fixed 100x100)
    - Focal loss on heatmaps naturally handles 21:1 class imbalance
    - Sub-pixel offset refinement for accurate center localization
    - Output stride = 4 (input/4 heatmap resolution)
"""

import math
import torch
import torch.nn as nn
import torchvision.models as models


class CellCenterNet(nn.Module):
    """
    CenterNet-style point detector for cervical cells.
    Predicts class heatmaps + sub-pixel offsets.
    No box regression needed (all boxes are 100x100).
    """

    SUPPORTED_BACKBONES = {
        'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
    }

    def __init__(self, num_classes=8, backbone='resnet50', pretrained=True,
                 head_channels=64, neck_channels=(256, 128, 64)):
        """
        Args:
            num_classes: Number of cell classes (8 Bethesda categories)
            backbone: Backbone architecture ('resnet34', 'resnet50', 'resnet101')
            pretrained: Whether to use ImageNet-pretrained backbone
            head_channels: Channels in the final feature map (default: 64)
            neck_channels: Channel progression for 3 deconv layers
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from: {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        # === BACKBONE (pretrained on ImageNet) ===
        factory_fn, weights, backbone_out_ch = self.SUPPORTED_BACKBONES[backbone]
        base_model = factory_fn(weights=weights if pretrained else None)
        # Remove avgpool and fc layers — keep only feature extractor
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        # Output: [B, backbone_out_ch, H/32, W/32]

        # === UPSAMPLE NECK (3x deconv: stride 32 -> stride 4) ===
        assert len(neck_channels) == 3, "Neck must have exactly 3 deconv stages"
        self.neck = nn.Sequential(
            self._make_deconv_block(backbone_out_ch, neck_channels[0], kernel_size=4),
            self._make_deconv_block(neck_channels[0], neck_channels[1], kernel_size=4),
            self._make_deconv_block(neck_channels[1], neck_channels[2], kernel_size=4),
        )
        # Output: [B, head_channels, H/4, W/4]

        # === HEATMAP HEAD (num_classes channels — one per cell type) ===
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(neck_channels[2], neck_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_channels[2], num_classes, 1),
        )

        # === OFFSET HEAD (2 channels: dx, dy sub-pixel refinement) ===
        self.offset_head = nn.Sequential(
            nn.Conv2d(neck_channels[2], neck_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_channels[2], 2, 1),
        )

        # Weight initialization
        self._init_weights()

    @staticmethod
    def _make_deconv_block(in_ch, out_ch, kernel_size=4):
        """
        Single deconvolution block: DeConv -> BN -> ReLU.
        With kernel_size=4 and stride=2, this exactly doubles spatial dims.
        """
        # For k=4, stride=2: padding=1 gives exact 2x upsample
        padding = kernel_size // 2 - 1
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2,
                               padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        """
        Initialize weights following CenterNet conventions:
        - Deconv layers: normal(0, 0.001)
        - Conv heads: normal(0, 0.001)
        - Heatmap final bias: -2.19 (focal loss prior, p≈0.1)
        - BatchNorm: weight=1, bias=0
        """
        for m in self.neck.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for head in [self.heatmap_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Focal loss prior for heatmap head final layer
        # bias = log(prior / (1 - prior)) with prior = 0.1 → -2.19
        self.heatmap_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            heatmap: [B, num_classes, H/4, W/4] — sigmoid activated class heatmaps
            offset:  [B, 2, H/4, W/4] — sub-pixel (dx, dy) offsets
        """
        features = self.backbone(x)           # [B, C, H/32, W/32]
        features = self.neck(features)         # [B, 64, H/4, W/4]

        heatmap = torch.sigmoid(self.heatmap_head(features))   # [B, 8, H/4, W/4]
        offset = self.offset_head(features)                     # [B, 2, H/4, W/4]

        return heatmap, offset

    def freeze_backbone(self):
        """Freeze backbone parameters (for initial training with pretrained weights)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Quick test
    print("Testing CellCenterNet...")
    model = CellCenterNet(num_classes=8, backbone='resnet50', pretrained=False)
    x = torch.randn(2, 3, 1024, 1024)
    hm, off = model(x)
    print(f"Input:   {x.shape}")
    print(f"Heatmap: {hm.shape}  (expected [2, 8, 256, 256])")
    print(f"Offset:  {off.shape}  (expected [2, 2, 256, 256])")
    print(f"Params:  {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Test passed!")
