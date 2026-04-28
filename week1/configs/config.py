"""
Week 1 Configuration
Central config for all Week 1 experiments.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    data_dir: str = "./data"
    num_workers: int = 4
    pin_memory: bool = True
    # CIFAR-10 standard normalization
    mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: List[float] = field(default_factory=lambda: [0.2470, 0.2435, 0.2616])


@dataclass
class TrainConfig:
    # Training hyperparameters (He et al. 2016 ResNet paper settings)
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True
    # LR schedule: divide by 10 at epochs 100 and 150
    lr_milestones: List[int] = field(default_factory=lambda: [100, 150])
    lr_gamma: float = 0.1
    # Regularization
    cutout: bool = True
    cutout_length: int = 16
    label_smoothing: float = 0.0
    # Logging
    log_interval: int = 50
    save_dir: str = "./results/checkpoints"
    seed: int = 42


@dataclass
class NASBench201Config:
    # Path to NAS-Bench-201 benchmark file (download separately)
    benchmark_path: str = "./data/NAS-Bench-201-v1_1-096897.pth"
    dataset: str = "cifar10"
    # Number of architectures to sample for correlation study
    num_sample_archs: int = 100
    api_available: bool = False  # Set True when benchmark file is present


@dataclass
class ProxyConfig:
    # Zero-cost proxy settings
    batch_size: int = 32           # Mini-batch for proxy computation
    num_batches: int = 1           # How many batches to average over
    device: str = "cuda"           # "cuda" or "cpu"
    proxies: List[str] = field(default_factory=lambda: [
        "snip", "grasp", "grad_norm", "synflow", "naswot", "jacob_cov", "l2_norm"
    ])


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    nas_bench: NASBench201Config = field(default_factory=NASBench201Config)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    experiment_name: str = "week1_baseline"
    results_dir: str = "./results"
    device: str = "cuda"


def get_default_config() -> Config:
    return Config()
