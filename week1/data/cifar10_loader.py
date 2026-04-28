"""
CIFAR-10 Data Pipeline
Handles loading, augmentation, and DataLoader creation.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


# ─────────────────────────────────────────────────────────────────────────────
# Cutout augmentation (DeVries & Taylor, 2017)
# ─────────────────────────────────────────────────────────────────────────────
class Cutout:
    """Randomly mask out a square patch from the image."""
    def __init__(self, length: int):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=torch.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
        x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0.0
        mask = mask.expand_as(img)
        return img * mask


# ─────────────────────────────────────────────────────────────────────────────
# Transform factories
# ─────────────────────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_train_transform(cutout: bool = True, cutout_length: int = 16):
    ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if cutout:
        ops.append(Cutout(cutout_length))
    return transforms.Compose(ops)


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────
def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    cutout: bool = True,
    cutout_length: int = 16,
    val_split: float = 0.0,          # 0 → use full test set as validation
    seed: int = 42,
):
    """
    Returns (train_loader, val_loader, test_loader).

    If val_split > 0, a fraction of the training set is carved out as a
    validation set (useful for NAS, where the test set must not be seen).
    """
    train_transform = get_train_transform(cutout, cutout_length)
    val_transform   = get_val_transform()

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    val_loader = None
    if val_split > 0.0:
        rng = np.random.RandomState(seed)
        n = len(train_dataset)
        indices = rng.permutation(n).tolist()
        split = int(n * val_split)
        val_indices   = indices[:split]
        train_indices = indices[split:]

        # Val uses the clean (no-augment) transform
        val_dataset_clean = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=val_transform
        )
        val_loader = DataLoader(
            Subset(val_dataset_clean, val_indices),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        train_dataset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    if val_loader is None:
        val_loader = test_loader

    return train_loader, val_loader, test_loader


def get_proxy_batch(data_dir: str = "./data", batch_size: int = 32, seed: int = 42):
    """Return a single (inputs, targets) batch for zero-cost proxy evaluation."""
    transform = get_val_transform()
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng)[:batch_size].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    inputs, targets = next(iter(loader))
    return inputs, targets
