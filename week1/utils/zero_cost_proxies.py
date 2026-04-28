"""
Zero-Cost Neural Architecture Search Proxies
Implementations of gradient-based and gradient-free proxies that estimate
architecture quality without any training.

Implemented proxies:
  1. SNIP         – Single-shot Network Pruning (Lee et al., 2019)
  2. GraSP        – Gradient Signal Preservation (Wang et al., 2020)
  3. GradNorm     – L2 norm of gradients at init
  4. Synflow      – Synaptic Flow (Tanaka et al., 2020) — path-wise
  5. NASWOT       – Neural Architecture Search Without Training (Mellor et al., 2021)
  6. Jacob_cov    – Jacobian covariance rank (Mellor et al., 2021)
  7. L2Norm       – L2 norm of all parameters (simple baseline)

References:
  - SNIP:    https://arxiv.org/abs/1810.02734
  - GraSP:   https://arxiv.org/abs/2002.07376
  - Synflow: https://arxiv.org/abs/2006.05467
  - NASWOT:  https://arxiv.org/abs/2006.04647
"""
import copy
import math
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def _temp_model(model: nn.Module):
    """Deep-copy model, yield the copy, then discard it."""
    m = copy.deepcopy(model)
    try:
        yield m
    finally:
        del m


def _get_grad_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _zero_grads(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# 1. SNIP — connection sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def compute_snip(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Optional[Callable] = None,
) -> float:
    """
    SNIP score = sum |g_i * w_i| over all weights.
    Measures how much each connection contributes to the loss gradient.
    Higher → more useful, well-connected architecture.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    with _temp_model(model) as m:
        m.train()
        _zero_grads(m)
        device = next(m.parameters()).device
        x, y = inputs.to(device), targets.to(device)
        loss = loss_fn(m(x), y)
        loss.backward()

        score = 0.0
        for p in _get_grad_params(m):
            if p.grad is not None:
                score += (p.grad * p.data).abs().sum().item()
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 2. GraSP — gradient signal preservation
# ─────────────────────────────────────────────────────────────────────────────
def compute_grasp(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Optional[Callable] = None,
    T: float = 200.0,
) -> float:
    """
    GraSP score measures preservation of gradient flow through the network.
    Uses a Hessian-gradient product approximation.
    Score = -sum H_{ii} * g_i  (higher → better gradient flow preserved)
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    with _temp_model(model) as m:
        m.train()
        device = next(m.parameters()).device
        x, y = inputs.to(device), targets.to(device)

        # First-order gradient
        _zero_grads(m)
        loss1 = loss_fn(m(x) / T, y)
        grads = torch.autograd.grad(loss1, _get_grad_params(m),
                                    create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, _get_grad_params(m))]

        # Hessian-gradient product via second backward
        gnorm  = sum(g.pow(2).sum() for g in grads)
        _zero_grads(m)
        gnorm.backward()

        score = 0.0
        for g, p in zip(grads, _get_grad_params(m)):
            if p.grad is not None:
                score -= (p.grad * g).sum().item()  # negative Hessian curvature
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 3. GradNorm
# ─────────────────────────────────────────────────────────────────────────────
def compute_grad_norm(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Optional[Callable] = None,
) -> float:
    """
    GradNorm = L2 norm of all parameter gradients.
    Large gradient norms at init → network is responsive → likely trainable.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    with _temp_model(model) as m:
        m.train()
        _zero_grads(m)
        device = next(m.parameters()).device
        x, y = inputs.to(device), targets.to(device)
        loss = loss_fn(m(x), y)
        loss.backward()

        score = 0.0
        for p in _get_grad_params(m):
            if p.grad is not None:
                score += p.grad.pow(2).sum().item()
    return math.sqrt(score)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Synflow — synaptic flow (iterative, data-free)
# ─────────────────────────────────────────────────────────────────────────────
def compute_synflow(
    model: nn.Module,
    input_shape: tuple = (1, 3, 32, 32),
    num_iters: int = 100,
) -> float:
    """
    Synflow: data-free path-norm proxy.
    Uses all-ones inputs to avoid layer collapse.
    Score = sum |w_i * g_i| where g_i is gradient of sum(output) w.r.t. w_i.
    Iteratively applies pruning masks (here we just compute once at init).
    """
    with _temp_model(model) as m:
        m.train()
        device = next(m.parameters()).device

        # Replace BN with identity for data-free computation
        def _linearize(mod):
            for name, child in mod.named_children():
                if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    # Freeze BN to prevent NaN with all-ones input
                    child.eval()
                    child.track_running_stats = False
                else:
                    _linearize(child)
        _linearize(m)

        # All-positive input (sign-preserving)
        dummy = torch.ones(input_shape, device=device)

        # Make all params positive (data-free trick)
        signs = {}
        for name, p in m.named_parameters():
            signs[name] = p.sign()
            p.data.abs_()

        _zero_grads(m)
        out  = m(dummy)
        loss = out.sum()
        loss.backward()

        score = 0.0
        for name, p in m.named_parameters():
            if p.grad is not None:
                score += (p.grad * p.data).abs().sum().item()

        # Restore signs
        for name, p in m.named_parameters():
            if name in signs:
                p.data.mul_(signs[name])

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. NASWOT + Jacobian covariance (Mellor et al., 2021)
# ─────────────────────────────────────────────────────────────────────────────
def _get_relu_activation_pattern(
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Collect the binary ReLU activation pattern (K matrix) across a batch.
    Returns a (batch_size, num_relu_activations) binary tensor.
    """
    activations = []

    def hook_fn(module, inp, out):
        # Record sign of ReLU output (1 if active, 0 if not)
        activations.append((out > 0).float().view(out.size(0), -1))

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device
        model(inputs.to(device))

    for h in hooks:
        h.remove()

    if not activations:
        return torch.zeros(inputs.size(0), 1)
    return torch.cat(activations, dim=1)   # (B, total_relu_units)


def compute_naswot(
    model: nn.Module,
    inputs: torch.Tensor,
) -> float:
    """
    NASWOT = log|det(K)|  where K is the activation kernel matrix.
    K[i,j] = <binary_pattern(x_i), binary_pattern(x_j)>
    High rank → diverse activations → architectures that distinguish inputs.
    Uses log determinant for numerical stability.
    """
    with _temp_model(model) as m:
        patterns = _get_relu_activation_pattern(m, inputs)  # (B, D)
        K = patterns @ patterns.T                            # (B, B)
        # Numerically stable log-det
        try:
            sign, logdet = torch.linalg.slogdet(K + 1e-5 * torch.eye(K.size(0)))
            score = (sign.item() * logdet.item()) if sign.item() > 0 else -1e6
        except Exception:
            score = -1e6
    return score


def compute_jacob_cov(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Jacobian covariance rank.
    Computes df/dx (Jacobian w.r.t. inputs), then measures the rank via
    the log determinant of J @ J^T.
    Higher rank → more expressive → better architecture.
    """
    with _temp_model(model) as m:
        m.eval()
        device = next(m.parameters()).device
        x = inputs.to(device).requires_grad_(True)

        output = m(x)
        # Compute Jacobian row-by-row (one per class)
        num_out = output.size(1)
        batch   = x.size(0)
        jacobians = []

        for i in range(num_out):
            _zero_grads(m)
            if x.grad is not None:
                x.grad.zero_()
            grad = torch.autograd.grad(
                output[:, i].sum(), x,
                retain_graph=(i < num_out - 1),
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                jacobians.append(torch.zeros(batch, x[0].numel()))
            else:
                jacobians.append(grad.view(batch, -1).detach().cpu())

        J = torch.stack(jacobians, dim=1)         # (B, num_out, input_dim)
        J = J.view(batch, -1)                     # (B, num_out * input_dim)
        cov = J @ J.T + 1e-5 * torch.eye(batch)  # (B, B)

        try:
            sign, logdet = torch.linalg.slogdet(cov)
            score = (sign.item() * logdet.item()) if sign.item() > 0 else -1e6
        except Exception:
            score = -1e6
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 7. L2Norm — parameter magnitude (baseline)
# ─────────────────────────────────────────────────────────────────────────────
def compute_l2_norm(model: nn.Module) -> float:
    """
    L2 norm of all parameters at initialization.
    Simple baseline to check if magnitude alone correlates with performance.
    """
    total = 0.0
    for p in model.parameters():
        total += p.data.pow(2).sum().item()
    return math.sqrt(total)


# ─────────────────────────────────────────────────────────────────────────────
# Unified proxy evaluator
# ─────────────────────────────────────────────────────────────────────────────
class ZeroCostEvaluator:
    """
    Compute all zero-cost proxies for a given model and data batch.

    Usage:
        evaluator = ZeroCostEvaluator(device='cuda')
        scores = evaluator.evaluate(model, inputs, targets)
    """

    AVAILABLE_PROXIES = [
        "snip", "grasp", "grad_norm", "synflow", "naswot", "jacob_cov", "l2_norm"
    ]

    def __init__(
        self,
        device: str = "cuda",
        proxies: Optional[list] = None,
    ):
        self.device  = torch.device(device if torch.cuda.is_available() else "cpu")
        self.proxies = proxies or self.AVAILABLE_PROXIES

    def evaluate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all requested proxies. Returns {proxy_name: score}.
        Each proxy uses a fresh deep-copy of the model.
        """
        model = model.to(self.device)
        inputs  = inputs.to(self.device)
        targets = targets.to(self.device)

        scores: Dict[str, float] = {}

        for proxy in self.proxies:
            try:
                if proxy == "snip":
                    scores["snip"] = compute_snip(model, inputs, targets)
                elif proxy == "grasp":
                    scores["grasp"] = compute_grasp(model, inputs, targets)
                elif proxy == "grad_norm":
                    scores["grad_norm"] = compute_grad_norm(model, inputs, targets)
                elif proxy == "synflow":
                    in_shape = (1,) + tuple(inputs.shape[1:])
                    scores["synflow"] = compute_synflow(model, input_shape=in_shape)
                elif proxy == "naswot":
                    scores["naswot"] = compute_naswot(model, inputs)
                elif proxy == "jacob_cov":
                    # Limit batch size for Jacobian (expensive)
                    scores["jacob_cov"] = compute_jacob_cov(
                        model, inputs[:min(16, inputs.size(0))],
                        targets[:min(16, targets.size(0))],
                    )
                elif proxy == "l2_norm":
                    scores["l2_norm"] = compute_l2_norm(model)
                else:
                    logger.warning(f"Unknown proxy: {proxy}")
            except Exception as e:
                logger.warning(f"Proxy '{proxy}' failed: {e}")
                scores[proxy] = float("nan")

        return scores

    def evaluate_batch(
        self,
        models_and_archs: list,   # list of (arch_index, nn.Module)
        inputs: torch.Tensor,
        targets: torch.Tensor,
        verbose: bool = True,
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate proxies for a list of (arch_idx, model) pairs.
        Returns {arch_idx: {proxy: score}}.
        """
        results = {}
        n = len(models_and_archs)
        for i, (arch_idx, model) in enumerate(models_and_archs):
            if verbose and (i % 10 == 0 or i == n - 1):
                logger.info(f"  Evaluating proxies: [{i+1}/{n}]")
            results[arch_idx] = self.evaluate(model, inputs, targets)
        return results


if __name__ == "__main__":
    from models.resnet import resnet20
    import torch

    model   = resnet20()
    inputs  = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))

    evaluator = ZeroCostEvaluator(device="cpu")
    scores    = evaluator.evaluate(model, inputs, targets)

    print("Zero-cost proxy scores:")
    for k, v in scores.items():
        print(f"  {k:12s}: {v:.4f}")
    print("Zero-cost proxy evaluation OK ✓")
