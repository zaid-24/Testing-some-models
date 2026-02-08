"""
Main runner script for RIVA Cell Detection Pipeline (CenterNet).

This is the single entry point for all pipeline operations.

Usage:
    # Analyze dataset statistics
    python run.py analyze

    # Train CenterNet model
    python run.py train --mode test           # Quick test (5 epochs, 512px)
    python run.py train --mode full           # Full training (140 epochs, 1024px)
    python run.py train --mode full --backbone resnet101

    # Run inference on test set
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

    print(f"\n[OK] {description} — Complete!")


def main():
    parser = argparse.ArgumentParser(
        description='RIVA Cell Detection Pipeline (CenterNet)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. analyze    - Analyze dataset statistics & class distribution
  2. train      - Train CenterNet model
  3. infer      - Run inference on test set & generate submission
  4. visualize  - Visualize annotations or predictions

Examples:
  python run.py analyze
  python run.py train --mode test
  python run.py train --mode full --backbone resnet50
  python run.py infer --conf 0.3
  python run.py visualize --split val --source csv
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    subparsers.add_parser('analyze', help='Analyze dataset statistics')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train CenterNet model')
    train_parser.add_argument(
        '--mode', choices=['test', 'full'], default='test',
        help='Training mode: test (quick 5 epochs) or full (140 epochs)'
    )
    train_parser.add_argument(
        '--backbone', choices=['resnet34', 'resnet50', 'resnet101'],
        default='resnet50', help='Backbone architecture (default: resnet50)'
    )
    train_parser.add_argument(
        '--resume', action='store_true', help='Resume from last checkpoint'
    )

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on test set')
    infer_parser.add_argument(
        '--model', type=str, default=None, help='Path to model weights'
    )
    infer_parser.add_argument(
        '--conf', type=float, default=0.3,
        help='Confidence threshold (default: 0.3)'
    )
    infer_parser.add_argument(
        '--imgsz', type=int, default=1024, help='Image size for inference'
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

         RIVA DET — Pap Smear Cell Detection — CenterNet Pipeline

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
            '--backbone', args.backbone,
        ]
        if args.resume:
            cmd.append('--resume')
        run_command(
            cmd,
            f'Training CenterNet ({args.mode} mode, {args.backbone} backbone)'
        )

    elif args.command == 'infer':
        cmd = [python_cmd, 'scripts/inference.py']
        if args.model:
            cmd.extend(['--model', args.model])
        cmd.extend(['--conf', str(args.conf), '--imgsz', str(args.imgsz)])
        run_command(cmd, 'Running inference on test set')

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
