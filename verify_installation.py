"""
Quick script to verify PyTorch and CUDA installation for CenterNet.

Usage:
    python verify_installation.py
"""

import sys


def main():
    print("=" * 70)
    print("INSTALLATION VERIFICATION (CenterNet Pipeline)")
    print("=" * 70)

    print(f"\nPython version: {sys.version}")

    # Check PyTorch
    try:
        import torch
        print(f"\n[OK] PyTorch installed: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"[OK] CUDA available: True")
            print(f"[OK] CUDA version: {torch.version.cuda}")
            print(f"[OK] GPU device: {torch.cuda.get_device_name(0)}")
            print(f"[OK] GPU count: {torch.cuda.device_count()}")
        else:
            print(f"[WARNING] CUDA available: False")
            print(f"\n    You have CPU-only PyTorch installed!")
            print(f"    Training will be VERY slow without GPU.")
            print(f"\n    To fix:")
            print(f"    1. Uninstall: pip uninstall torch torchvision torchaudio -y")
            print(f"    2. Reinstall with CUDA:")
            print(f"       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("[ERROR] PyTorch not installed!")
        print("\n    Install with:")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    # Check torchvision
    try:
        import torchvision
        print(f"[OK] torchvision installed: {torchvision.__version__}")
    except ImportError:
        print("[ERROR] torchvision not installed!")
        print("\n    Install with PyTorch:")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    # Check other dependencies
    try:
        import pandas
        import numpy
        import cv2
        import albumentations
        print(f"[OK] pandas: {pandas.__version__}")
        print(f"[OK] numpy: {numpy.__version__}")
        print(f"[OK] opencv: {cv2.__version__}")
        print(f"[OK] albumentations: {albumentations.__version__}")
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\n    Install with:")
        print("    pip install -r requirements.txt")
        return False

    # Quick model test
    print(f"\n{'='*70}")
    print("Quick Model Test...")
    try:
        from models.centernet import CellCenterNet
        model = CellCenterNet(num_classes=8, backbone='resnet50', pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        hm, off = model(x)
        print(f"[OK] CenterNet model: input {x.shape} → heatmap {hm.shape}, offset {off.shape}")
    except Exception as e:
        print(f"[WARNING] Model test failed: {e}")
        print("    This might be OK if you haven't set up the project structure yet.")

    # Final status
    print(f"\n{'='*70}")
    if cuda_available:
        print("[SUCCESS] All checks passed! Ready for GPU training.")
        print("\nNext steps:")
        print("  1. python run.py analyze        # Check dataset")
        print("  2. python run.py train --mode test   # Quick test")
        print("  3. python run.py train --mode full   # Full training")
    else:
        print("[WARNING] PyTorch installed but CUDA not available.")
        print("You can train on CPU, but it will be VERY slow.")
    print("=" * 70)

    return cuda_available


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
