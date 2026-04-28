"""
Week 1 Deliverable: Correlation Analysis
Zero-cost proxies vs. NAS-Bench-201 ground-truth accuracy.

Produces:
  - results/proxy_scores.json       — raw proxy scores per architecture
  - results/correlation_report.json — Spearman & Kendall τ per proxy
  - results/correlation_plots/      — scatter plots + heatmap

Usage:
    python experiments/correlation_analysis.py [--n_archs 100] [--device cpu]
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

# Ensure week1/ is on path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.nas_bench_201 import NASBench201API, Architecture
from utils.zero_cost_proxies import ZeroCostEvaluator
from data.cifar10_loader import get_proxy_batch

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("correlation_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation ρ ∈ [-1, 1]."""
    def _rankdata(a):
        n   = len(a)
        idx = np.argsort(a)
        r   = np.empty(n)
        r[idx] = np.arange(1, n + 1, dtype=float)
        return r
    rx, ry = _rankdata(x), _rankdata(y)
    d  = rx - ry
    return float(1.0 - 6.0 * np.sum(d**2) / (len(x) * (len(x)**2 - 1)))


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's τ_b ∈ [-1, 1]."""
    n          = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = x[i] - x[j], y[i] - y[j]
            if xi * xj > 0:
                concordant += 1
            elif xi * xj < 0:
                discordant += 1
    total = n * (n - 1) // 2
    return float((concordant - discordant) / total) if total > 0 else 0.0


def top_k_overlap(proxy_scores: np.ndarray, gt_scores: np.ndarray, k: int) -> float:
    """Fraction of true top-k architectures found in proxy top-k."""
    top_k_proxy = set(np.argsort(proxy_scores)[-k:])
    top_k_gt    = set(np.argsort(gt_scores)[-k:])
    return len(top_k_proxy & top_k_gt) / k


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (matplotlib, graceful fallback if unavailable)
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(
    proxy_data: dict,        # {proxy_name: np.array of scores}
    gt_accs: np.ndarray,     # ground-truth accuracies (same order)
    correlations: dict,      # {proxy_name: {spearman, kendall}}
    output_dir: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots. "
                       "Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    proxies = list(proxy_data.keys())
    n       = len(proxies)

    # ── 1. Individual scatter plots ──────────────────────────────────────────
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, proxy in enumerate(proxies):
        scores = proxy_data[proxy]
        valid  = ~np.isnan(scores)
        ax     = axes[i]
        spear  = correlations[proxy]["spearman"]
        kend   = correlations[proxy]["kendall"]

        ax.scatter(scores[valid], gt_accs[valid], alpha=0.5, s=15, c="#4C72B0")
        # Trend line
        if valid.sum() > 2:
            z = np.polyfit(scores[valid], gt_accs[valid], 1)
            p = np.poly1d(z)
            xs = np.linspace(scores[valid].min(), scores[valid].max(), 100)
            ax.plot(xs, p(xs), "r--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel(proxy, fontsize=10)
        ax.set_ylabel("GT Accuracy (%)", fontsize=10)
        ax.set_title(f"{proxy}\nρ={spear:.3f}  τ={kend:.3f}", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Zero-Cost Proxies vs. NAS-Bench-201 Accuracy (CIFAR-10)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "proxy_scatter_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Scatter plots saved → {path}")

    # ── 2. Correlation bar chart ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2196F3" if correlations[p]["spearman"] > 0 else "#F44336"
              for p in proxies]

    spear_vals = [correlations[p]["spearman"] for p in proxies]
    kend_vals  = [correlations[p]["kendall"]  for p in proxies]

    bars1 = ax1.barh(proxies, spear_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Spearman ρ", fontsize=11)
    ax1.set_title("Spearman Rank Correlation", fontsize=12, fontweight="bold")
    ax1.set_xlim(-1, 1)
    ax1.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars1, spear_vals):
        ax1.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

    colors2 = ["#2196F3" if v > 0 else "#F44336" for v in kend_vals]
    bars2 = ax2.barh(proxies, kend_vals, color=colors2, edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Kendall τ", fontsize=11)
    ax2.set_title("Kendall Rank Correlation", fontsize=12, fontweight="bold")
    ax2.set_xlim(-1, 1)
    ax2.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars2, kend_vals):
        ax2.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

    fig.suptitle("Zero-Cost Proxy Correlation with True Accuracy",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path2 = os.path.join(output_dir, "correlation_bars.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Correlation bar chart saved → {path2}")

    # ── 3. Top-k overlap chart ───────────────────────────────────────────────
    n_archs = len(gt_accs)
    k_vals  = [5, 10, 20]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos   = np.arange(len(proxies))
    width   = 0.25

    for ki, k in enumerate(k_vals):
        overlaps = []
        for proxy in proxies:
            s = proxy_data[proxy]
            valid = ~np.isnan(s)
            if valid.sum() < k:
                overlaps.append(0.0)
            else:
                ovlp = top_k_overlap(s, gt_accs, min(k, valid.sum()))
                overlaps.append(ovlp)
        ax.bar(x_pos + ki * width, overlaps, width, label=f"Top-{k}", alpha=0.85)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(proxies, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of Top-k Overlap", fontsize=11)
    ax.set_title("Top-k Architecture Recovery by Proxy", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(output_dir, "topk_overlap.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Top-k overlap chart saved → {path3}")


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_correlation_analysis(
    n_archs: int = 100,
    device: str = "cpu",
    data_dir: str = "./data",
    results_dir: str = "./results",
    nas_bench_path: str = None,
    seed: int = 42,
    batch_size: int = 32,
):
    """
    Full correlation analysis pipeline.
    Returns the correlation report dict.
    """
    logger.info(f"=== Correlation Analysis: {n_archs} architectures ===")
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Load NAS-Bench-201 ────────────────────────────────────────────────
    api = NASBench201API(benchmark_path=nas_bench_path, dataset="cifar10")
    logger.info(f"API: {api}")

    # ── 2. Sample architectures ──────────────────────────────────────────────
    architectures = api.sample_architectures(n_archs, seed=seed, strategy="diverse")
    logger.info(f"Sampled {len(architectures)} architectures.")

    # ── 3. Get ground-truth accuracies ───────────────────────────────────────
    gt_accuracies = []
    for arch in architectures:
        acc = api.query_accuracy(arch)
        gt_accuracies.append(acc)
    gt_accuracies = np.array(gt_accuracies)
    logger.info(f"GT accuracies: min={gt_accuracies.min():.2f}  "
                f"max={gt_accuracies.max():.2f}  mean={gt_accuracies.mean():.2f}")

    # ── 4. Load proxy data batch ─────────────────────────────────────────────
    logger.info("Loading CIFAR-10 data for proxy evaluation …")
    try:
        inputs, targets = get_proxy_batch(
            data_dir=data_dir, batch_size=batch_size, seed=seed
        )
    except Exception as e:
        logger.warning(f"CIFAR-10 download failed ({e}), using random data.")
        inputs  = torch.randn(batch_size, 3, 32, 32)
        targets = torch.randint(0, 10, (batch_size,))

    # ── 5. Compute proxies ───────────────────────────────────────────────────
    evaluator = ZeroCostEvaluator(device=device)
    proxy_scores_by_arch: list = []   # list of {proxy: score} per arch

    logger.info(f"Computing zero-cost proxies on {device} …")
    t0 = time.time()

    for i, arch in enumerate(architectures):
        model  = api.build_network(arch, C=16)
        scores = evaluator.evaluate(model, inputs, targets)
        proxy_scores_by_arch.append(scores)
        if (i + 1) % 10 == 0 or i == n_archs - 1:
            elapsed = time.time() - t0
            logger.info(f"  [{i+1}/{n_archs}] elapsed: {elapsed:.1f}s")

    logger.info(f"Proxy computation done in {time.time()-t0:.1f}s")

    # ── 6. Restructure: proxy → array of scores ───────────────────────────────
    proxy_names = list(proxy_scores_by_arch[0].keys())
    proxy_data  = {}
    for proxy in proxy_names:
        proxy_data[proxy] = np.array([d.get(proxy, np.nan)
                                      for d in proxy_scores_by_arch])

    # ── 7. Correlation statistics ────────────────────────────────────────────
    correlations = {}
    logger.info("\n" + "─" * 60)
    logger.info(f"{'Proxy':15s}  {'Spearman ρ':>12}  {'Kendall τ':>12}  {'Top-10%':>10}")
    logger.info("─" * 60)

    for proxy in proxy_names:
        s = proxy_data[proxy]
        valid = ~np.isnan(s)
        if valid.sum() < 5:
            logger.warning(f"  {proxy}: too many NaNs, skipping.")
            correlations[proxy] = {"spearman": float("nan"), "kendall": float("nan"),
                                   "top10_overlap": float("nan"), "n_valid": int(valid.sum())}
            continue

        sv, gv = s[valid], gt_accuracies[valid]
        spear  = spearman_correlation(sv, gv)
        kend   = kendall_tau(sv, gv)
        k10    = max(1, int(0.1 * valid.sum()))
        ovlp   = top_k_overlap(sv, gv, k10)

        correlations[proxy] = {
            "spearman":      round(spear, 4),
            "kendall":       round(kend, 4),
            "top10_overlap": round(ovlp, 4),
            "n_valid":       int(valid.sum()),
        }
        logger.info(f"  {proxy:15s}  {spear:+12.4f}  {kend:+12.4f}  {ovlp:10.3f}")

    logger.info("─" * 60)

    # Rank proxies by |Spearman|
    ranked = sorted(
        [(p, abs(correlations[p]["spearman"])) for p in proxy_names
         if not np.isnan(correlations[p]["spearman"])],
        key=lambda x: x[1], reverse=True,
    )
    logger.info("Proxies ranked by |Spearman ρ|:")
    for rank, (p, s) in enumerate(ranked, 1):
        logger.info(f"  #{rank}: {p:15s}  ρ={correlations[p]['spearman']:+.4f}")

    # ── 8. Save results ───────────────────────────────────────────────────────
    raw_results = {
        "n_archs":     n_archs,
        "device":      device,
        "gt_accuracies": gt_accuracies.tolist(),
        "arch_indices":  [arch.to_index() for arch in architectures],
        "proxy_scores":  {p: proxy_data[p].tolist() for p in proxy_names},
    }
    raw_path = os.path.join(results_dir, "proxy_scores.json")
    with open(raw_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    logger.info(f"Raw scores saved → {raw_path}")

    report = {
        "summary": {
            "n_archs":    n_archs,
            "best_proxy_by_spearman": ranked[0][0] if ranked else "N/A",
            "best_spearman": ranked[0][1] if ranked else 0.0,
            "ranked_proxies": [p for p, _ in ranked],
        },
        "per_proxy": correlations,
        "gt_stats": {
            "min":  float(gt_accuracies.min()),
            "max":  float(gt_accuracies.max()),
            "mean": float(gt_accuracies.mean()),
            "std":  float(gt_accuracies.std()),
        },
    }
    report_path = os.path.join(results_dir, "correlation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Correlation report saved → {report_path}")

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    plots_dir = os.path.join(results_dir, "correlation_plots")
    make_plots(proxy_data, gt_accuracies, correlations, plots_dir)

    logger.info("=== Correlation analysis complete ✓ ===")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Zero-cost proxy correlation analysis")
    p.add_argument("--n_archs",    type=int,   default=100,    help="Architectures to sample")
    p.add_argument("--device",     type=str,   default="cpu",  help="cuda or cpu")
    p.add_argument("--data_dir",   type=str,   default="./data")
    p.add_argument("--results_dir",type=str,   default="./results")
    p.add_argument("--nas_bench",  type=str,   default=None,   help="Path to NAS-Bench-201 .pth file")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--batch_size", type=int,   default=32,     help="Proxy eval batch size")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_correlation_analysis(
        n_archs=args.n_archs,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        nas_bench_path=args.nas_bench,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print("\nTop proxy by Spearman correlation:",
          report["summary"]["best_proxy_by_spearman"],
          f"ρ = {report['summary']['best_spearman']:.4f}")
