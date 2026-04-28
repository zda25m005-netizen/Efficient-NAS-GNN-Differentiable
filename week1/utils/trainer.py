"""
CIFAR-10 Training Pipeline
Full training loop with LR scheduling, logging, and checkpointing.
Target: ResNet-20 → 94% test accuracy on CIFAR-10.
"""
import os
import time
import json
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(format=fmt, level=level)
    logger = logging.getLogger(name)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Metrics tracking
# ─────────────────────────────────────────────────────────────────────────────
class AverageMeter:
    """Computes and stores running average."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / bs).item())
        return res


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation steps
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    model.train()
    losses  = AverageMeter()
    top1    = AverageMeter()
    t_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping (optional, helps stability)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        acc1, = accuracy(outputs, targets, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1, inputs.size(0))

        if logger and (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"Epoch [{epoch}] Step [{batch_idx+1}/{len(loader)}] "
                f"Loss: {losses.avg:.4f}  Acc@1: {top1.avg:.2f}%  "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}  "
                f"Time: {elapsed:.1f}s"
            )

    return {"loss": losses.avg, "acc": top1.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses = AverageMeter()
    top1   = AverageMeter()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        acc1,   = accuracy(outputs, targets, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1, inputs.size(0))

    return {"loss": losses.avg, "acc": top1.avg}


# ─────────────────────────────────────────────────────────────────────────────
# Full training pipeline
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    """
    Complete training pipeline for CIFAR-10 classification.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, config)
        history = trainer.fit()
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        nesterov: bool = True,
        lr_milestones=(100, 150),
        lr_gamma: float = 0.1,
        label_smoothing: float = 0.0,
        save_dir: str = "./results/checkpoints",
        experiment_name: str = "resnet20_cifar10",
        device: Optional[torch.device] = None,
        log_interval: int = 50,
        results_dir: str = "./results",
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.epochs         = epochs
        self.save_dir       = save_dir
        self.experiment_name = experiment_name
        self.log_interval   = log_interval
        self.results_dir    = results_dir

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
        )

        # LR scheduler
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=list(lr_milestones), gamma=lr_gamma
        )

        # Logger & history
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        log_path = os.path.join(results_dir, f"{experiment_name}.log")
        self.logger = setup_logger(experiment_name, log_file=log_path)
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "lr":         [],
        }
        self.best_acc  = 0.0
        self.best_epoch = 0

    def fit(self) -> Dict[str, list]:
        self.logger.info(
            f"Starting training: {self.experiment_name} | "
            f"device={self.device} | epochs={self.epochs} | "
            f"params={sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )
        t_total = time.time()

        for epoch in range(1, self.epochs + 1):
            # Train
            train_stats = train_one_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, epoch,
                self.log_interval, self.logger,
            )
            # Evaluate
            val_stats = evaluate(
                self.model, self.val_loader, self.criterion, self.device
            )
            self.scheduler.step()

            # Record
            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_stats["loss"])
            self.history["train_acc"].append(train_stats["acc"])
            self.history["val_loss"].append(val_stats["loss"])
            self.history["val_acc"].append(val_stats["acc"])
            self.history["lr"].append(lr)

            # Logging
            self.logger.info(
                f"[Epoch {epoch:3d}/{self.epochs}] "
                f"Train Loss: {train_stats['loss']:.4f}  Train Acc: {train_stats['acc']:.2f}%  |  "
                f"Val Loss: {val_stats['loss']:.4f}  Val Acc: {val_stats['acc']:.2f}%  |  "
                f"LR: {lr:.6f}"
            )

            # Save best
            if val_stats["acc"] > self.best_acc:
                self.best_acc   = val_stats["acc"]
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_stats["acc"], is_best=True)

            # Save every 50 epochs
            if epoch % 50 == 0:
                self._save_checkpoint(epoch, val_stats["acc"], is_best=False)

        total_time = time.time() - t_total
        self.logger.info(
            f"Training complete in {total_time/60:.1f} min | "
            f"Best Val Acc: {self.best_acc:.2f}% @ epoch {self.best_epoch}"
        )
        self._save_history()
        return self.history

    def _save_checkpoint(self, epoch: int, acc: float, is_best: bool):
        state = {
            "epoch":      epoch,
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "acc":        acc,
            "best_acc":   self.best_acc,
        }
        tag  = "best" if is_best else f"epoch{epoch:03d}"
        path = os.path.join(self.save_dir, f"{self.experiment_name}_{tag}.pth")
        torch.save(state, path)
        if is_best:
            self.logger.info(f"  ✓ New best checkpoint: {acc:.2f}% → {path}")

    def _save_history(self):
        path = os.path.join(self.results_dir, f"{self.experiment_name}_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved → {path}")
