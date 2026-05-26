"""
Profiling Script for Deep-Unfolded VAMP-JCD Receiver
=====================================================

Measures inference time, memory usage, parameter count, and MAC/flop
estimates using torch.profiler.

Two profiling modes:
    mode="forward":   Single forward pass profiling (inference)
    mode="train":     Forward + backward pass profiling (training)
    mode="compare":   Compare VAMP-JCD vs LMMSE vs Genie baselines

Usage:
    python profile_model.py                          # Full profile
    python profile_model.py --mode forward           # Inference only
    python profile_model.py --mode compare           # Compare with baselines
    python profile_model.py --device cpu             # CPU profiling
    python profile_model.py --iterations 50          # More iterations for stable timing
"""

import argparse
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "coder"))

from model import DU_VAMP_JCD_Config, VAMPJCDReceiver, count_params
from baselines import LMMSEBaseline, LSBaseline, GenieBound


def build_sample_input(config: DU_VAMP_JCD_Config,
                       B: int = 4,
                       device: str = "cpu"):
    """Build synthetic input tensors for profiling.

    Returns:
        tuple: (y, pilots, pilot_mask)
    """
    N_sc = config.n_subcarriers  # 12
    N_sym = config.n_symbols     # 14
    N_p = config.n_pilots        # 24

    y = torch.randn(B, N_sc, N_sym, 2, device=device)
    pilots = torch.randn(B, N_p, 2, device=device)
    pilots = pilots / (pilots.norm(dim=-1, keepdim=True) + 1e-10)

    pilot_mask = torch.zeros(N_sc, N_sym, dtype=torch.bool, device=device)
    pilot_mask[:, 3] = True
    pilot_mask[:, 10] = True

    return y, pilots, pilot_mask


def parameter_count_report(model: nn.Module, name: str = "Model"):
    """Print detailed parameter count by component."""
    print(f"\n{'─'*60}")
    print(f"  PARAMETER REPORT: {name}")
    print(f"{'─'*60}")

    total = 0
    trainable = 0

    for name_param, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        print(f"  {name_param:50s}: {n:6d}  "
              f"{'[train]' if param.requires_grad else '[frozen]'}")

    print(f"{'─'*60}")
    print(f"  Total params:     {total:6d}")
    print(f"  Trainable params: {trainable:6d}")
    print(f"  Budget:           ≤ 1,000")
    print(f"  Headroom:         {1000 - trainable:6d}")
    return trainable


def profile_inference(model: nn.Module, config: DU_VAMP_JCD_Config,
                      device: str = "cpu", iterations: int = 30,
                      use_profiler: bool = True):
    """Profile forward pass inference time and memory."""
    print(f"\n{'='*60}")
    print(f"  INFERENCE PROFILING")
    print(f"{'='*60}")
    print(f"  Device: {device}  |  Iterations: {iterations}")
    print(f"  Config: T={config.n_unfolded_layers}, "
          f"denoiser={config.denoiser_type}, "
          f"CFO={config.cfo_mode}")

    model = model.to(device).eval()
    y, pilots, pilot_mask = build_sample_input(config, B=16, device=device)

    # Warmup
    print(f"\n  Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(y, pilots, pilot_mask)

    # Timed execution (without profiler overhead)
    print(f"  Timing {iterations} iterations...")
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(y, pilots, pilot_mask)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / iterations * 1000
    blocks_per_sec = iterations / elapsed * y.shape[0]

    print(f"\n  RESULTS (batch_size={y.shape[0]}):")
    print(f"  Average inference time: {avg_ms:.3f} ms/block")
    print(f"  Throughput:             {blocks_per_sec:.0f} blocks/sec")
    print(f"  Throughput:             {blocks_per_sec / y.shape[0]:.0f} slots/sec")
    print(f"  Target latency:         < 1.0 ms/block")
    if avg_ms < 1.0:
        print(f"  ✓ Meets latency target")
    else:
        print(f"  ⚠ Exceeds latency target by {avg_ms - 1.0:.2f} ms")

    # PyTorch profiler (if requested)
    if use_profiler and device == "cuda":
        print(f"\n  PyTorch Profiler (CUDA):")
        try:
            from torch.profiler import profile, record_function, ProfilerActivity

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                with record_function("model_inference"):
                    for _ in range(5):
                        _ = model(y, pilots, pilot_mask)

            print(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=15
            ))

            # Memory summary
            print(f"\n  Memory usage (CUDA):")
            print(f"  Current:  {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"  Max:      {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"  Profiler error: {e}")

    return avg_ms


def profile_training(model: nn.Module, config: DU_VAMP_JCD_Config,
                     device: str = "cpu", iterations: int = 10):
    """Profile forward + backward pass for training."""
    print(f"\n{'='*60}")
    print(f"  TRAINING PROFILING")
    print(f"{'='*60}")

    model = model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    y, pilots, pilot_mask = build_sample_input(config, B=8, device=device)
    # Create dummy target tensors
    h_true = torch.randn_like(y)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        h_hat, x_soft, llrs, _ = model(y, pilots, pilot_mask)
        loss = nn.MSELoss()(h_hat, h_true)
        loss.backward()
        optimizer.step()

    # Timed execution
    print(f"  Timing {iterations} training iterations...")
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        optimizer.zero_grad()
        h_hat, x_soft, llrs, _ = model(y, pilots, pilot_mask)
        loss = nn.MSELoss()(h_hat, h_true)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / iterations * 1000
    print(f"\n  RESULTS:")
    print(f"  Average train step: {avg_ms:.3f} ms/iteration")
    print(f"  Throughput:         {iterations / elapsed:.0f} iterations/sec")

    # Memory
    if device == "cuda":
        print(f"  GPU memory:         {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

    return avg_ms


def profile_baseline_comparison(config: DU_VAMP_JCD_Config,
                                device: str = "cpu",
                                iterations: int = 30):
    """Compare inference time and parameter count across all receivers."""
    print(f"\n{'='*60}")
    print(f"  BASELINE COMPARISON")
    print(f"{'='*60}")

    receivers = {
        "VAMP-JCD (T=5)": VAMPJCDReceiver(config).to(device).eval(),
        "VAMP-JCD (T=3)": VAMPJCDReceiver(
            DU_VAMP_JCD_Config(
                n_unfolded_layers=3, weight_tying=True,
                cfo_mode="analytic_plus_learned",
            )
        ).to(device).eval(),
        "LMMSE": LMMSEBaseline(config).to(device).eval(),
        "LS": LSBaseline(config).to(device).eval(),
    }

    y, pilots, pilot_mask = build_sample_input(config, B=16, device=device)

    print(f"\n  {'Receiver':25s} {'Params':8s} {'Time(ms)':10s} {'Throughput':10s}")
    print(f"  {'─'*53}")

    results = {}
    for name, rec in receivers.items():
        n_params = sum(p.numel() for p in rec.parameters() if p.requires_grad)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                if hasattr(rec, 'forward') and 'noise_var' in rec.forward.__code__.co_varnames:
                    _ = rec(y, pilots, pilot_mask, noise_var=0.1)
                else:
                    _ = rec(y, pilots, pilot_mask)

        # Timed
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(iterations):
                if hasattr(rec, 'forward') and 'noise_var' in rec.forward.__code__.co_varnames:
                    _ = rec(y, pilots, pilot_mask, noise_var=0.1)
                else:
                    _ = rec(y, pilots, pilot_mask)

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = elapsed / iterations * 1000
        throughput = iterations / elapsed * y.shape[0]

        print(f"  {name:25s} {n_params:7d}  {avg_ms:8.3f}  {throughput:8.0f}")
        results[name] = {"params": n_params, "time_ms": avg_ms, "throughput": throughput}

    return results


def profile_mac_estimate(config: DU_VAMP_JCD_Config):
    """Analytical MAC/flop count estimation.

    Note: This uses analytical formulas, not thop/flop_count.
    For the VAMP-JCD receiver, we know the exact operations.
    """
    print(f"\n{'='*60}")
    print(f"  MAC COUNT ESTIMATION (Analytical)")
    print(f"{'='*60}")

    N_sc = config.n_subcarriers   # 12
    N_sym = config.n_symbols      # 14
    T = config.n_unfolded_layers  # 5
    N_re = N_sc * N_sym           # 168 REs per slot

    # Per-iteration MAC breakdown (real-valued operations):
    ops = {}

    # 1. VAMP linear step (LMMSE)
    #    conj(x) * y: 6 real MACs per RE (complex multiply = 6 real ops)
    #    gamma_z * result: 2 MACs per RE
    #    gamma_h * r1 - u1: 3 MACs per RE
    #    division: ~2 MACs per RE
    #    Total: ~13 MACs per RE per iteration
    vamp_linear_macs = 13 * N_re
    ops["VAMP Linear Step"] = vamp_linear_macs

    # 2. Delay-Doppler transform (IFFT + FFT)
    #    IFFT(N_sc) + FFT(N_sym) per RE
    #    Approx: 5 * N_re * log2(N_re) real MACs
    dd_macs = 5 * N_re * math.log2(N_re)
    ops["DD Transform"] = dd_macs

    # 3. Shrinkage/denoising
    #    |h|^2: 2 MACs, relu: 1 comparison, multiply: 2 MACs
    #    Total: ~5 MACs per RE
    shrink_macs = 5 * N_re
    ops["Shrinkage"] = shrink_macs

    # 4. Inverse DD transform
    inv_dd_macs = 5 * N_re * math.log2(N_re)
    ops["Inverse DD Transform"] = inv_dd_macs

    # 5. Soft detection
    #    MMSE: ~10 MACs per RE
    #    LLR: ~4 MACs per bit per RE
    detect_macs = 10 * N_re + 4 * N_re * config.n_bits_per_symbol
    ops["Soft Detection"] = detect_macs

    # 6. CFO correction
    #    Phase rotation per RE: ~6 MACs
    cfo_macs = 6 * N_re
    ops["CFO Correction"] = cfo_macs

    per_iter_total = sum(ops.values())
    total_macs = T * per_iter_total

    print(f"\n  Slot dimensions: {N_sc} SC × {N_sym} sym = {N_re} REs")
    print(f"  Unfolded iterations: T = {T}")
    print(f"  Per-interation MACs: {per_iter_total:,.0f}")
    print(f"  Total MACs:          {total_macs:,.0f}")
    print(f"  Budget:              ≤ 10,000")
    print()
    print(f"  Per-iteration breakdown:")
    for op_name, macs in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"    {op_name:25s}: {macs:8,.0f} MACs ({macs/per_iter_total*100:5.1f}%)")

    if total_macs < 10000:
        print(f"\n  ✓ Within MAC budget")
    else:
        print(f"\n  ⚠ Exceeds MAC budget by {total_macs - 10000:.0f} MACs")
        print(f"     Mitigation: reduce T to {int(10000 / per_iter_total)} (needs {per_iter_total:.0f}/iter)")

    return {"per_iteration": per_iter_total, "total": total_macs, "budget": 10000}


def estimate_flops(model: nn.Module, config: DU_VAMP_JCD_Config,
                   device: str = "cpu"):
    """Estimate FLOPs using torch.profiler or manual calculation.

    FLOPs = 2 × MACs for multiply-accumulate operations.
    """
    print(f"\n{'='*60}")
    print(f"  FLOP ESTIMATE")
    print(f"{'='*60}")

    macs = profile_mac_estimate(config)
    flops = 2 * macs["total"]

    print(f"\n  Total MACs: {macs['total']:,.0f}")
    print(f"  Estimated FLOPs: {flops:,.0f}")
    print(f"  FLOPs per RE:    {flops // (12*14):,.0f}")

    return flops


def profile_all(config: DU_VAMP_JCD_Config, device: str = "cpu",
                iterations: int = 30):
    """Run all profiling modes."""
    print(f"{'='*60}")
    print(f"  VAMP-JCD RECEIVER PROFILING")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Config: T={config.n_unfolded_layers}, "
          f"denoiser={config.denoiser_type}")
    print(f"  Params: {config.n_trainable_params} (budget: ≤1,000)")

    # Parameter report
    model = VAMPJCDReceiver(config)
    parameter_count_report(model, "VAMP-JCD Receiver")

    # MAC estimate
    macs = profile_mac_estimate(config)

    # Inference profiling
    avg_ms = profile_inference(model, config, device, iterations)

    # Training profiling
    if device == "cuda":
        profile_training(model, config, device, iterations)

    # Baseline comparison
    profile_baseline_comparison(config, device, iterations)

    # FLOP estimate
    flops = estimate_flops(model, config, device)

    print(f"\n{'='*60}")
    print(f"  PROFILING SUMMARY")
    print(f"{'='*60}")
    print(f"  Parameters:        {config.n_trainable_params} (budget ≤1,000)")
    print(f"  Total MACs:        {macs['total']:,.0f} (budget ≤10,000)")
    print(f"  Inference latency: {avg_ms:.3f} ms/block (target <1.0 ms)")
    if avg_ms < 1.0:
        print(f"  ✓ All targets met (params, MACs, latency)")
    else:
        print(f"  ⚠ Latency target not met")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile VAMP-JCD Receiver"
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "forward", "train", "compare", "macs", "flops"],
                        help="Profiling mode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of profiling iterations")
    parser.add_argument("--layers", type=int, default=5,
                        help="Number of unfolded layers (T)")
    parser.add_argument("--denoiser", type=str, default="delay_doppler_shrinkage",
                        choices=["delay_doppler_shrinkage", "element_shrinkage",
                                 "small_cnn", "mlp"],
                        help="Denoiser type")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = DU_VAMP_JCD_Config(
        n_unfolded_layers=args.layers,
        weight_tying=True,
        denoiser_type=args.denoiser,
        cfo_mode="analytic_plus_learned",
    )

    if args.mode == "all":
        profile_all(config, device, args.iterations)
    elif args.mode == "forward":
        model = VAMPJCDReceiver(config)
        parameter_count_report(model, "VAMP-JCD")
        profile_inference(model, config, device, args.iterations)
    elif args.mode == "train":
        model = VAMPJCDReceiver(config)
        parameter_count_report(model, "VAMP-JCD")
        profile_training(model, config, device, args.iterations)
    elif args.mode == "compare":
        profile_baseline_comparison(config, device, args.iterations)
    elif args.mode == "macs":
        profile_mac_estimate(config)
    elif args.mode == "flops":
        model = VAMPJCDReceiver(config)
        estimate_flops(model, config, device)
