"""
Main runner script for RIVA Cell Detection Pipeline.

This is the single entry point for all pipeline operations.

Usage:
    # Step 1: Convert annotations to YOLO format
    python run.py convert
    
    # Step 2: Analyze dataset
    python run.py analyze
    
    # Step 3a: Test training pipeline (laptop)
    python run.py train --mode test
    
    # Step 3b: Full training (workstation)
    python run.py train --mode full
    
    # Step 4: Run inference and generate submission
    python run.py infer
    
    # Optional: Visualize results
    python run.py visualize --split val
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
    
    print(f"\n[OK] {description} - Complete!")


def main():
    parser = argparse.ArgumentParser(
        description='RIVA Cell Detection Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. convert    - Convert CSV annotations to YOLO format
  2. analyze    - Analyze dataset statistics
  3. train      - Train YOLOv11 model
  4. infer      - Run inference on test set
  5. visualize  - Visualize predictions

Examples:
  python run.py convert
  python run.py train --mode test
  python run.py train --mode full --resume
  python run.py infer --conf 0.25
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    subparsers.add_parser('convert', help='Convert CSV annotations to YOLO format')
    
    # Analyze command
    subparsers.add_parser('analyze', help='Analyze dataset statistics')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train YOLOv11 model')
    train_parser.add_argument('--mode', choices=['test', 'full', 'focal', 'adh', 'multiscale'], default='test',
                              help='Training mode (test=laptop, full=extreme aug, focal=focal loss, adh=attention decoupled head, multiscale=progressive resolution)')
    train_parser.add_argument('--resume', action='store_true',
                              help='Resume from last checkpoint')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on test set')
    infer_parser.add_argument('--model', type=str, default=None,
                              help='Path to model weights')
    infer_parser.add_argument('--conf', type=float, default=0.25,
                              help='Confidence threshold')
    infer_parser.add_argument('--iou', type=float, default=0.45,
                              help='IoU threshold for NMS')
    infer_parser.add_argument('--tta', action='store_true',
                              help='Enable Test-Time Augmentation')
    
    # Visualize command
    vis_parser = subparsers.add_parser('visualize', help='Visualize annotations/predictions')
    vis_parser.add_argument('--split', choices=['train', 'val'], default='val')
    vis_parser.add_argument('--source', choices=['csv', 'yolo', 'predictions'], default='csv')
    vis_parser.add_argument('--num-samples', type=int, default=10)
    
    # Full pipeline
    subparsers.add_parser('all', help='Run full pipeline (convert -> analyze -> train test)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Print banner
    print("""
    ===================================================================
                                                                       
         RIVA DET - Pap Smear Cell Detection - YOLOv11 Pipeline       
                                                                       
    ===================================================================
    """)
    
    python_cmd = sys.executable
    
    if args.command == 'convert':
        run_command(
            [python_cmd, 'scripts/convert_annotations.py'],
            'Converting CSV annotations to YOLO format'
        )
    
    elif args.command == 'analyze':
        run_command(
            [python_cmd, 'scripts/analyze_dataset.py'],
            'Analyzing dataset statistics'
        )
    
    elif args.command == 'train':
        cmd = [python_cmd, 'scripts/train.py', '--mode', args.mode]
        if args.resume:
            cmd.append('--resume')
        run_command(cmd, f'Training YOLOv11 ({args.mode} mode)')
    
    elif args.command == 'infer':
        cmd = [python_cmd, 'scripts/inference.py']
        if args.model:
            cmd.extend(['--model', args.model])
        cmd.extend(['--conf', str(args.conf), '--iou', str(args.iou)])
        if hasattr(args, 'tta') and args.tta:
            cmd.append('--tta')
        run_command(cmd, 'Running inference on test set')
    
    elif args.command == 'visualize':
        run_command(
            [python_cmd, 'scripts/visualize_predictions.py',
             '--split', args.split,
             '--source', args.source,
             '--num-samples', str(args.num_samples)],
            'Generating visualizations'
        )
    
    elif args.command == 'all':
        # Run full pipeline
        run_command(
            [python_cmd, 'scripts/convert_annotations.py'],
            'Step 1/3: Converting annotations'
        )
        run_command(
            [python_cmd, 'scripts/analyze_dataset.py'],
            'Step 2/3: Analyzing dataset'
        )
        run_command(
            [python_cmd, 'scripts/train.py', '--mode', 'test'],
            'Step 3/3: Testing training pipeline'
        )
        
        print("\n" + "=" * 60)
        print("[SUCCESS] PIPELINE TEST COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. If test succeeded, run full training:")
        print("     python run.py train --mode full")
        print("\n  2. After training, generate submission:")
        print("     python run.py infer")


if __name__ == '__main__':
    main()
