"""
Main runner script for RIVA Cell Detection Pipeline.

Two-Stage Architecture:
    Stage 1: Binary CenterNet detector — finds all cell centers (class-agnostic)
    Stage 2: EfficientNet classifier — classifies cell patches into 8 classes

Usage:
    # Analyze dataset statistics
    python run.py analyze

    # Train both stages
    python run.py train --mode full --stage both

    # Train detector only
    python run.py train --mode full --stage detect --backbone resnet50

    # Train classifier only
    python run.py train --mode full --stage classify --cls-backbone efficientnet_b2

    # Run two-stage inference
    python run.py infer --conf 0.3

    # Visualize results
    python run.py visualize --split val --source csv
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"[Running] {description}")
    print('=' * 60)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n[OK] {description} -- Complete!")


def main():
    parser = argparse.ArgumentParser(
        description='RIVA Cell Detection -- Two-Stage Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. analyze    - Analyze dataset statistics & class distribution
  2. train      - Train detector and/or classifier (two stages)
  3. infer      - Two-stage inference -> submission CSV
  4. visualize  - Visualize annotations or predictions

Examples:
  python run.py analyze
  python run.py train --mode test --stage both
  python run.py train --mode full --stage detect --backbone resnet50
  python run.py train --mode full --stage classify --cls-backbone efficientnet_b2
  python run.py train --mode full --stage both --resume
  python run.py infer --conf 0.3
  python run.py visualize --split val --source csv
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    subparsers.add_parser('analyze', help='Analyze dataset statistics')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train detector and/or classifier')
    train_parser.add_argument(
        '--mode', choices=['test', 'full'], default='test',
        help='Training mode: test (quick 5 epochs) or full (competition)'
    )
    train_parser.add_argument(
        '--stage', choices=['detect', 'classify', 'both'], default='both',
        help='Which stage to train (default: both)'
    )
    train_parser.add_argument(
        '--backbone', choices=['resnet34', 'resnet50', 'resnet101'],
        default='resnet50', help='Detector backbone (default: resnet50)'
    )
    train_parser.add_argument(
        '--cls-backbone',
        choices=['efficientnet_b0', 'efficientnet_b2', 'convnext_tiny'],
        default='efficientnet_b2', help='Classifier backbone (default: efficientnet_b2)'
    )
    train_parser.add_argument(
        '--resume', action='store_true', help='Resume from last checkpoint'
    )

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run two-stage inference on test set')
    infer_parser.add_argument(
        '--det-model', type=str, default=None,
        help='Path to detector model (default: auto-detect)'
    )
    infer_parser.add_argument(
        '--cls-model', type=str, default=None,
        help='Path to classifier model (default: auto-detect)'
    )
    infer_parser.add_argument(
        '--conf', type=float, default=0.3,
        help='Detection confidence threshold (default: 0.3)'
    )
    infer_parser.add_argument(
        '--imgsz', type=int, default=1024,
        help='Detector input image size (default: 1024)'
    )

    # Visualize command
    vis_parser = subparsers.add_parser('visualize', help='Visualize annotations/predictions')
    vis_parser.add_argument(
        '--split', choices=['train', 'val'], default='val'
    )
    vis_parser.add_argument(
        '--source', choices=['csv', 'predictions'], default='csv',
        help='Annotation source: csv (ground truth) or predictions'
    )
    vis_parser.add_argument(
        '--num-samples', type=int, default=10
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Banner
    print("""
    ===================================================================

         RIVA DET -- Pap Smear Cell Detection -- Two-Stage Pipeline
         Stage 1: Binary CenterNet Detector (ResNet-50)
         Stage 2: Cell Type Classifier (EfficientNet-B2)

    ===================================================================
    """)

    python_cmd = sys.executable

    if args.command == 'analyze':
        run_command(
            [python_cmd, 'scripts/analyze_dataset.py'],
            'Analyzing dataset statistics'
        )

    elif args.command == 'train':
        cmd = [
            python_cmd, 'scripts/train.py',
            '--mode', args.mode,
            '--stage', args.stage,
            '--backbone', args.backbone,
            '--cls-backbone', args.cls_backbone,
        ]
        if args.resume:
            cmd.append('--resume')

        stage_desc = {
            'detect': f'Stage 1: Detector ({args.backbone})',
            'classify': f'Stage 2: Classifier ({args.cls_backbone})',
            'both': f'Both Stages ({args.backbone} + {args.cls_backbone})',
        }
        run_command(cmd, f'Training {stage_desc[args.stage]} [{args.mode} mode]')

    elif args.command == 'infer':
        cmd = [
            python_cmd, 'scripts/inference.py',
            '--conf', str(args.conf),
            '--imgsz', str(args.imgsz),
        ]
        if args.det_model:
            cmd.extend(['--det-model', args.det_model])
        if args.cls_model:
            cmd.extend(['--cls-model', args.cls_model])
        run_command(cmd, 'Running two-stage inference')

    elif args.command == 'visualize':
        run_command(
            [python_cmd, 'scripts/visualize_predictions.py',
             '--split', args.split,
             '--source', args.source,
             '--num-samples', str(args.num_samples)],
            'Generating visualizations'
        )


if __name__ == '__main__':
    main()
