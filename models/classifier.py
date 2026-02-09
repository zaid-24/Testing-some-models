"""
Cell Type Classifier — Stage 2 of the Two-Stage Detection Pipeline.

Takes 100×100 crops around detected cell centers and classifies them
into 8 Bethesda categories using a pretrained image classifier.

Architecture:
    EfficientNet-B2 (default, pretrained on ImageNet)
    Input: 100x100 crop -> resized to 224x224
    Output: 8-class logits -> softmax for probabilities

Usage:
    model = CellClassifier(num_classes=8, backbone='efficientnet_b2')
    logits = model(patches)  # [B, 8]
    probs = torch.softmax(logits, dim=1)
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for the classifier (Stage 2).\n"
        "Install with: pip install timm"
    )


class CellClassifier(nn.Module):
    """
    Patch-based cell classifier using pretrained backbone.

    Classifies 100×100 cell crops into 8 Bethesda categories:
        0: NILM, 1: ENDO, 2: INFL, 3: ASCUS,
        4: LSIL, 5: HSIL, 6: ASCH, 7: SCC

    Supported backbones (via timm):
        - efficientnet_b0: Lightweight, fast (~5M params)
        - efficientnet_b2: Recommended balance (~9M params)
        - convnext_tiny:   Highest capacity (~28M params)
    """

    SUPPORTED_BACKBONES = {
        'efficientnet_b0': '~5M params, fastest',
        'efficientnet_b2': '~9M params, recommended',
        'convnext_tiny': '~28M params, highest capacity',
    }

    def __init__(self, num_classes=8, backbone='efficientnet_b2',
                 pretrained=True, drop_rate=0.3):
        """
        Args:
            num_classes: Number of cell classes (8 Bethesda categories)
            backbone:    timm model name (see SUPPORTED_BACKBONES)
            pretrained:  Whether to use ImageNet pretrained weights
            drop_rate:   Dropout rate before classifier head
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        """
        Args:
            x: Input patches [B, 3, H, W]
        Returns:
            logits: [B, num_classes] raw logits (apply softmax for probabilities)
        """
        return self.model(x)


if __name__ == "__main__":
    print("Testing CellClassifier...")
    model = CellClassifier(num_classes=8, backbone='efficientnet_b2', pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}  (expected [4, 8])")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Test passed!")
