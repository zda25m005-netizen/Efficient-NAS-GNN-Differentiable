"""
Week 1 Main Entry Point: Train ResNet-20 on CIFAR-10
Target: ~94% test accuracy

Usage:
    python train_resnet20.py
    python train_resnet20.py --epochs 200 --lr 0.1 --device cuda
    python train_resnet20.py --quick_test   # 5-epoch smoke test
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-20 on CIFAR-10")
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--momentum",    type=float, default=0.9)
    parser.add_argument("--cutout",      action="store_true", default=True)
    parser.add_argument("--cutout_len",  type=int,   default=16)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--data_dir",    type=str,   default="./data")
    parser.add_argument("--results_dir", type=str,   default="./results")
    parser.add_argument("--save_dir",    type=str,   default="./results/checkpoints")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--quick_test",  action="store_true",
                        help="Run 5 epochs only for smoke testing")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.quick_test:
        args.epochs = 5
        print("[QUICK TEST MODE] Running 5 epochs only.")

    set_seed(args.seed)
    print(f"\n{'='*60}")
    print(f"  ResNet-20 CIFAR-10 Training")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  LR:      {args.lr}")
    print(f"  Seed:    {args.seed}")
    print(f"{'='*60}\n")

    # Imports (local to week1/)
    from models.resnet import resnet20
    from data.cifar10_loader import get_cifar10_loaders
    from utils.trainer import Trainer

    # Data
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cutout=args.cutout,
        cutout_length=args.cutout_len,
    )
    print(f"Dataset: CIFAR-10  |  Train: {len(train_loader.dataset):,}  "
          f"Test: {len(test_loader.dataset):,}")

    # Model
    model = resnet20(num_classes=10)
    print(f"Model: ResNet-20  |  Parameters: {model.count_parameters():,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        experiment_name="resnet20_cifar10",
        device=device,
        results_dir=args.results_dir,
    )
    history = trainer.fit()

    # Final test evaluation
    from utils.trainer import evaluate
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    test_stats = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"  Final Test Accuracy: {test_stats['acc']:.2f}%")
    print(f"  Best Val  Accuracy:  {trainer.best_acc:.2f}%")
    print(f"  Target:              94.00%")
    print(f"  Status: {'✓ TARGET MET' if test_stats['acc'] >= 94.0 else '⚠ Below target'}")
    print(f"{'='*60}\n")

    return history, test_stats


if __name__ == "__main__":
    main()
