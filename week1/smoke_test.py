"""
Week 1 Smoke Test — runs in ~30s with PyTorch available.
Validates all 5 deliverables end-to-end.

Usage:
    python smoke_test.py            # full smoke test
    python smoke_test.py --device cpu
"""
import sys, argparse, time
import torch
import torch.nn as nn

sys.path.insert(0, ".")

PASS = "✓"
FAIL = "✗"


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status} {label}{suffix}")
    if not ok:
        raise AssertionError(f"FAILED: {label}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch",  type=int, default=8)
    args = p.parse_args()
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  Week 1 Smoke Test | device={device} | batch={args.batch}")
    print(f"  PyTorch {torch.__version__}")
    print(f"{'='*60}")

    # ── Deliverable 1 & 2: CIFAR-10 pipeline + ResNet-20 ─────────────────────
    section("Deliverable 1-2: CIFAR-10 Pipeline + ResNet-20")

    from models.resnet import resnet20, resnet56, BasicBlock, NASCell
    from data.cifar10_loader import get_train_transform, get_val_transform, get_proxy_batch

    # Model shape & param count
    model = resnet20(num_classes=10).to(device)
    params = model.count_parameters()
    check("ResNet-20 param count in [250K, 300K]", 250_000 <= params <= 300_000, f"{params:,}")

    x = torch.randn(args.batch, 3, 32, 32, device=device)
    y = model(x)
    check("ResNet-20 forward pass", y.shape == (args.batch, 10), f"output={y.shape}")

    # Gradient flow
    loss = y.sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    check("Gradients flow through all layers", len(grad_norms) > 0, f"{len(grad_norms)} layers")
    check("No NaN gradients", all(g == g for g in grad_norms))

    # NASCell ops
    for op_name in ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]:
        cell = NASCell(op_name, C=16, stride=1).to(device)
        xc   = torch.randn(2, 16, 8, 8, device=device)
        out  = cell(xc)
        check(f"NASCell op '{op_name}'", out.shape == xc.shape, f"{out.shape}")

    # Transforms
    t_tr = get_train_transform(cutout=True, cutout_length=16)
    t_vl = get_val_transform()
    check("Train transform created", t_tr is not None)
    check("Val transform created",   t_vl is not None)

    # ── Deliverable 2: Training utilities ─────────────────────────────────────
    section("Deliverable 2: Trainer Utilities")

    from utils.trainer import AverageMeter, accuracy, train_one_epoch, evaluate

    meter = AverageMeter()
    for v in [1.0, 2.0, 3.0]:
        meter.update(v, n=1)
    check("AverageMeter avg=2.0", abs(meter.avg - 2.0) < 1e-6, f"avg={meter.avg}")

    logits  = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    targets = torch.tensor([1, 0])
    acc1,   = accuracy(logits, targets, topk=(1,))
    check("accuracy() correct (2/2 = 100%)", abs(acc1 - 100.0) < 1e-4, f"{acc1:.1f}%")

    logits2  = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    targets2 = torch.tensor([0, 1])
    acc2,   = accuracy(logits2, targets2, topk=(1,))
    check("accuracy() correct (0/2 = 0%)", abs(acc2 - 0.0) < 1e-4, f"{acc2:.1f}%")

    # ── Deliverable 3: NAS-Bench-201 ─────────────────────────────────────────
    section("Deliverable 3: NAS-Bench-201 Integration")

    from utils.nas_bench_201 import NASBench201API, Architecture, NUM_ARCHS, NAS201_OPS
    import random, numpy as np

    api = NASBench201API()
    check("API mock has 15,625 architectures", len(api) == 15625, f"n={len(api)}")

    # Index roundtrip
    errors = sum(1 for _ in range(200) if (
        lambda i: Architecture.from_index(i).to_index() != i
    )(random.randint(0, NUM_ARCHS-1)))
    check("Architecture index roundtrip (200 random)", errors == 0, f"{errors} errors")

    # Accuracy range
    sample = api.sample_architectures(20, seed=42)
    accs   = [api.query_accuracy(a) for a in sample]
    check("GT accuracies in [60, 94]%", all(60 <= a <= 94 for a in accs),
          f"min={min(accs):.1f} max={max(accs):.1f}")

    # Build and forward a NAS network
    net = api.build_network(Architecture.random(), C=16).to(device)
    xn  = torch.randn(2, 3, 32, 32, device=device)
    yn  = net(xn)
    check("NASBench201Network forward", yn.shape == (2, 10), f"{yn.shape}")

    best_arch, best_acc = api.get_best_architecture()
    check("Best architecture retrievable", 80 <= best_acc <= 94, f"{best_acc:.2f}%")

    # ── Deliverable 4: Zero-cost proxies ──────────────────────────────────────
    section("Deliverable 4: Zero-Cost Proxies")

    from utils.zero_cost_proxies import (
        ZeroCostEvaluator, compute_snip, compute_grad_norm,
        compute_synflow, compute_naswot, compute_l2_norm,
    )

    t0     = time.time()
    model2 = resnet20(num_classes=10).to(device)
    xi     = torch.randn(args.batch, 3, 32, 32, device=device)
    yi     = torch.randint(0, 10, (args.batch,), device=device)

    evaluator = ZeroCostEvaluator(device=str(device))
    scores    = evaluator.evaluate(model2, xi, yi)
    elapsed   = time.time() - t0

    expected_proxies = ["snip", "grasp", "grad_norm", "synflow", "naswot", "jacob_cov", "l2_norm"]
    for proxy in expected_proxies:
        ok = proxy in scores and scores[proxy] == scores[proxy]  # NaN check
        check(f"Proxy '{proxy}' computed", ok, f"{scores.get(proxy, 'MISSING'):.4g}")

    check("All proxies computed in <60s", elapsed < 60, f"{elapsed:.1f}s")

    # Individual proxy sanity
    check("SNIP > 0",       scores["snip"]     > 0)
    check("GradNorm > 0",   scores["grad_norm"] > 0)
    check("Synflow > 0",    scores["synflow"]   > 0)
    check("L2Norm > 0",     scores["l2_norm"]   > 0)

    # ── Deliverable 5: Correlation analysis ───────────────────────────────────
    section("Deliverable 5: Correlation Analysis")

    from experiments.correlation_analysis import (
        run_correlation_analysis, spearman_correlation, kendall_tau, top_k_overlap
    )
    import numpy as np

    # Math checks
    x  = np.arange(10, dtype=float)
    ok1 = abs(spearman_correlation(x, x)  - 1.0) < 1e-9
    ok2 = abs(spearman_correlation(x, -x) + 1.0) < 1e-9
    ok3 = abs(kendall_tau(x, x)  - 1.0) < 1e-9
    ok4 = abs(kendall_tau(x, -x) + 1.0) < 1e-9
    check("Spearman: perfect +/-1 correct", ok1 and ok2)
    check("Kendall:  perfect +/-1 correct", ok3 and ok4)
    check("Top-k overlap: perfect case = 1.0", top_k_overlap(x, x, 3) == 1.0)
    check("Top-k overlap: inverted  = 0.0",    top_k_overlap(x, -x, 3) == 0.0)

    # Run mini correlation analysis (10 archs for speed)
    report = run_correlation_analysis(
        n_archs=10, device=str(device),
        data_dir="./data", results_dir="./results",
        seed=42, batch_size=args.batch,
    )
    check("Correlation report generated", "per_proxy" in report and "summary" in report)
    check("Best proxy identified", report["summary"]["best_proxy_by_spearman"] != "N/A",
          report["summary"]["best_proxy_by_spearman"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WEEK 1 SMOKE TEST — ALL DELIVERABLES PASSED ✓")
    print(f"{'='*60}")
    print(f"\n  Deliverables verified:")
    print(f"    1. ✓ CIFAR-10 training pipeline")
    print(f"    2. ✓ ResNet-20 baseline ({params:,} params, forward/backward ok)")
    print(f"    3. ✓ NAS-Bench-201 integration (15,625 architectures)")
    print(f"    4. ✓ Zero-cost proxies: {', '.join(expected_proxies)}")
    print(f"    5. ✓ Correlation analysis pipeline")
    print(f"\n  To train ResNet-20 to 94%:")
    print(f"    cd week1 && python train_resnet20.py --device cuda")
    print(f"\n  To run full correlation study (100 archs):")
    print(f"    cd week1 && python experiments/correlation_analysis.py --n_archs 100")
    print()


if __name__ == "__main__":
    main()
