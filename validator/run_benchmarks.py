"""
Comprehensive Benchmark Suite for VAMP-JCD Receiver
====================================================

Runs all domain-specific benchmarks defined for the LEO NTN IoT physical layer
receiver problem. Produces BLER vs SNR curves, NMSE vs SNR curves, baseline
comparisons, and research-quality metrics.

This script is designed to be run standalone (not as a pytest suite) for
generating benchmark results that can be saved, compared, and plotted.

Usage:
    python run_benchmarks.py                              # Full benchmark suite
    python run_benchmarks.py --fast                       # Quick sanity check
    python run_benchmarks.py --output ./benchmark_results # Custom output dir
    python run_benchmarks.py --doppler 600 1200 2400     # Multiple Doppler values
    python run_benchmarks.py --compare-vs-lmmse           # Focus on baseline comparison
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "coder"))

from model import DU_VAMP_JCD_Config, VAMPJCDReceiver, count_params
from channel import LEONTNChannelSimulator
from baselines import LMMSEBaseline, LSBaseline, GenieBound
from train import channel_nmse_loss, compute_bler, compute_ber


# ─── Configuration ─────────────────────────────────────────────────────────

def get_default_config() -> DU_VAMP_JCD_Config:
    return DU_VAMP_JCD_Config(
        n_unfolded_layers=5,
        weight_tying=True,
        denoiser_type="delay_doppler_shrinkage",
        cfo_mode="analytic_plus_learned",
        use_soft_feedback=True,
    )


# ─── Core Benchmark Functions ──────────────────────────────────────────────

@torch.no_grad()
def benchmark_bler_vs_snr(model: torch.nn.Module,
                           config: DU_VAMP_JCD_Config,
                           device: str,
                           snr_db_list: List[float],
                           doppler_hz: float = 1200.0,
                           elevation_deg: float = 45.0,
                           batch_size: int = 256,
                           n_batches: int = 5,
                           label: str = "VAMP-JCD") -> Dict:
    """Evaluate BLER, BER, and NMSE across SNR values.

    Returns:
        dict: {snr: {bler, ber, nmse}}
    """
    sim = LEONTNChannelSimulator(config)
    model.eval()
    results = {}

    print(f"\n  Benchmark: {label}")
    print(f"  Doppler={doppler_hz:.0f} Hz, Elevation={elevation_deg}°")
    print(f"  {'SNR (dB)':>10s} {'BLER':>8s} {'BER':>8s} {'NMSE':>8s}")
    print(f"  {'─'*38}")

    for snr in snr_db_list:
        blers, bers, nmses = [], [], []
        for _ in range(n_batches):
            batch = sim.generate_batch(
                batch_size=batch_size, elevation_deg=elevation_deg,
                snr_db=snr, doppler_hz=doppler_hz,
            )
            y = batch["y"].to(device)
            h_true = batch["h_true"].to(device)
            pilots = batch["pilots"].to(device)
            pilot_mask = batch["pilot_mask"].to(device)
            bits = batch["bits"].to(device)

            h_hat, x_soft, llrs, _ = model(y, pilots, pilot_mask)
            blers.append(compute_bler(llrs, bits).item())
            bers.append(compute_ber(llrs, bits).item())
            nmses.append(channel_nmse_loss(h_hat, h_true).item())

        bler = sum(blers) / len(blers)
        ber = sum(bers) / len(bers)
        nmse = sum(nmses) / len(nmses)
        results[str(snr)] = {"bler": bler, "ber": ber, "nmse": nmse}
        print(f"  {snr:>+10.1f} {bler:>8.4f} {ber:>8.6f} {nmse:>8.4f}")

    return results


@torch.no_grad()
def benchmark_bler_vs_doppler(model: torch.nn.Module,
                               config: DU_VAMP_JCD_Config,
                               device: str,
                               doppler_list: List[float],
                               snr_db: float = -3.0,
                               elevation_deg: float = 45.0,
                               batch_size: int = 256,
                               n_batches: int = 3) -> Dict:
    """Evaluate BLER vs Doppler spread at fixed SNR."""
    sim = LEONTNChannelSimulator(config)
    model.eval()
    results = {}

    print(f"\n  BLER vs Doppler @ SNR={snr_db:+.0f} dB")
    print(f"  {'Doppler (Hz)':>12s} {'BLER':>8s}")
    print(f"  {'─'*22}")

    for doppler in doppler_list:
        blers = []
        for _ in range(n_batches):
            batch = sim.generate_batch(
                batch_size=batch_size, elevation_deg=elevation_deg,
                snr_db=snr_db, doppler_hz=doppler,
            )
            y = batch["y"].to(device)
            pilots = batch["pilots"].to(device)
            pilot_mask = batch["pilot_mask"].to(device)
            bits = batch["bits"].to(device)

            _, _, llrs, _ = model(y, pilots, pilot_mask)
            blers.append(compute_bler(llrs, bits).item())

        bler = sum(blers) / len(blers)
        results[str(int(doppler))] = {"bler": bler}
        print(f"  {doppler:>10.0f}   {bler:>8.4f}")

    return results


@torch.no_grad()
def benchmark_cfo_accuracy(config: DU_VAMP_JCD_Config,
                            device: str,
                            snr_db_list: List[float] = None,
                            batch_size: int = 256,
                            n_batches: int = 3) -> Dict:
    """Benchmark CFO estimation accuracy vs true Doppler.

    Measures the mean error and standard deviation of the analytic
    CFO estimate across SNR values.
    """
    from model import CFOEstimator
    if snr_db_list is None:
        snr_db_list = [-5.0, -3.0, 0.0, 5.0, 10.0]

    cfo_est = CFOEstimator(config).to(device)
    sim = LEONTNChannelSimulator(config)
    results = {}

    print(f"\n  CFO Estimation Accuracy")
    print(f"  {'SNR (dB)':>10s} {'Mean Err':>10s} {'Std Err':>10s}")
    print(f"  {'─'*32}")

    for snr in snr_db_list:
        errs = []
        for _ in range(n_batches):
            batch = sim.generate_batch(
                batch_size=batch_size, elevation_deg=45.0,
                snr_db=snr, doppler_hz=1200.0,
            )
            y = batch["y"].to(device)
            pilots = batch["pilots"].to(device)
            pilot_mask = batch["pilot_mask"].to(device)

            delta_f_est = cfo_est.analytic_estimate(y, pilots, pilot_mask)
            # True CFO (normalized to cycles/symbol)
            delta_f_true = 1200.0 / config.subcarrier_spacing_hz / config.n_symbols
            err = (delta_f_est - delta_f_true).abs()
            errs.append(err.mean().item())

        mean_err = sum(errs) / len(errs)
        results[str(snr)] = {"mean_abs_error": mean_err}
        print(f"  {snr:>+10.1f} {mean_err:>10.6f}")

    return results


# ─── Full Comparison ───────────────────────────────────────────────────────

def benchmark_full_comparison(config: DU_VAMP_JCD_Config,
                               device: str,
                               snr_db_list: List[float],
                               output_dir: str = None) -> Dict:
    """Compare VAMP-JCD against LMMSE, LS, and Genie baselines."""
    print(f"\n{'='*60}")
    print(f"  FULL BASELINE COMPARISON")
    print(f"{'='*60}")

    sim = LEONTNChannelSimulator(config)

    vamp_model = VAMPJCDReceiver(config).to(device).eval()
    lmmse = LMMSEBaseline(config).to(device).eval()
    ls = LSBaseline(config).to(device).eval()
    genie = GenieBound(config).to(device).eval()

    receivers = {
        "VAMP-JCD": vamp_model,
        "LMMSE": lmmse,
        "LS": ls,
        "Genie": genie,
    }

    results = {}

    print(f"\n  {'SNR':>6s}", end="")
    for name in receivers:
        print(f"  {name:>10s}", end="")
    print()
    print(f"  {'─'*50}")

    for snr in snr_db_list:
        batch = sim.generate_batch(
            batch_size=256, elevation_deg=45.0,
            snr_db=snr, doppler_hz=1200.0,
        )
        y = batch["y"].to(device)
        h_true = batch["h_true"].to(device)
        pilots = batch["pilots"].to(device)
        pilot_mask = batch["pilot_mask"].to(device)
        bits = batch["bits"].to(device)

        print(f"  {snr:>+6.1f}", end="")
        snr_results = {}

        for name, rec in receivers.items():
            with torch.no_grad():
                if name == "Genie":
                    _, _, llrs, _ = rec(y, h_true, noise_var=10**(-snr/10))
                elif name in ["LMMSE"]:
                    _, _, llrs, _ = rec(y, pilots, pilot_mask, noise_var=10**(-snr/10))
                elif name == "LS":
                    _, _, llrs, _ = rec(y, pilots, pilot_mask)
                else:
                    _, _, llrs, _ = rec(y, pilots, pilot_mask)

                bler = compute_bler(llrs, bits).item()
                ber = compute_ber(llrs, bits).item()
                snr_results[name.lower().replace("-", "_")] = {"bler": bler, "ber": ber}
                print(f"  {bler:>10.4f}", end="")

        print()
        results[str(snr)] = snr_results

    return results


# ─── Channel Sparsity Probe ────────────────────────────────────────────────

@torch.no_grad()
def benchmark_channel_sparsity(config: DU_VAMP_JCD_Config,
                                device: str,
                                batch_size: int = 128) -> Dict:
    """Probe how sparse the channel is in the delay-Doppler domain.

    Measures energy concentration (fraction in top-k bins) and
    the effectiveness of soft thresholding at different levels.
    """
    from model import DelayDopplerTransform
    sim = LEONTNChannelSimulator(config)
    transform = DelayDopplerTransform().to(device)

    batch = sim.generate_batch(
        batch_size=batch_size, elevation_deg=45.0,
        snr_db=0.0, doppler_hz=1200.0,
    )
    h_true = batch["h_true"].to(device)
    h_dd = transform(h_true)

    # Energy per bin
    h_tf_energy = h_true[..., 0]**2 + h_true[..., 1]**2
    h_dd_energy = h_dd[..., 0]**2 + h_dd[..., 1]**2

    # Flatten per sample
    h_tf_flat = h_tf_energy.reshape(batch_size, -1)
    h_dd_flat = h_dd_energy.reshape(batch_size, -1)
    n_bins = h_tf_flat.shape[1]

    # Energy concentration metrics
    results = {}
    for frac in [0.1, 0.25, 0.5]:
        k = max(1, int(n_bins * frac))
        tf_top = h_tf_flat.topk(k, dim=1).values.sum(dim=1).mean().item()
        dd_top = h_dd_flat.topk(k, dim=1).values.sum(dim=1).mean().item()
        tf_total = h_tf_flat.sum(dim=1).mean().item()
        dd_total = h_dd_flat.sum(dim=1).mean().item()

        results[f"top_{frac:.0%}"] = {
            "tf_concentration": tf_top / tf_total,
            "dd_concentration": dd_top / dd_total,
        }

    print(f"\n  Channel Sparsity Probe (delay-Doppler domain)")
    print(f"  {'Top frac':>10s} {'TF conc.':>10s} {'DD conc.':>10s} {'Sparsity gain':>14s}")
    print(f"  {'─'*46}")
    for frac_key in sorted(results.keys()):
        r = results[frac_key]
        gain = r["dd_concentration"] / r["tf_concentration"]
        print(f"  {frac_key:>10s} {r['tf_concentration']:>10.4f} "
              f"{r['dd_concentration']:>10.4f} {gain:>13.2f}x")

    return results


# ─── Main Benchmark Runner ─────────────────────────────────────────────────

def run_all_benchmarks(config: DU_VAMP_JCD_Config,
                       device: str,
                       output_dir: str = "benchmark_results",
                       fast: bool = False,
                       doppler_values: List[float] = None):
    """Run the full benchmark suite."""
    print(f"{'='*60}")
    print(f"  VAMP-JCD RECEIVER — COMPREHENSIVE BENCHMARK SUITE")
    print(f"{'='*60}")
    print(f"  Config: T={config.n_unfolded_layers}, "
          f"denoiser={config.denoiser_type}")
    print(f"  Device: {device}")
    print(f"  Fast mode: {fast}")
    print(f"  Doppler values: {doppler_values}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    if doppler_values is None:
        doppler_values = [600.0, 1200.0, 2400.0]
    snr_db_list = [-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 10.0]

    all_results = {
        "metadata": {
            "config": {
                k: getattr(config, k)
                for k in DU_VAMP_JCD_Config.__dataclass_fields__
            },
            "device": device,
            "fast_mode": fast,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "benchmarks": {},
    }

    # 1. Parameter count
    model = VAMPJCDReceiver(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_results["benchmarks"]["parameter_count"] = {
        "trainable": n_params,
        "budget": 1000,
        "within_budget": n_params <= 1000,
    }
    print(f"  Parameters: {n_params} (budget: ≤1000)")

    # 2. BLER vs SNR at multiple Doppler spreads
    print(f"\n{'─'*60}")
    print("  BENCHMARK: BLER vs SNR")
    print(f"{'─'*60}")
    for doppler in doppler_values:
        n_batches = 1 if fast else 3
        batch_size = 128 if fast else 256
        bler_results = benchmark_bler_vs_snr(
            model, config, device,
            snr_db_list=snr_db_list,
            doppler_hz=doppler,
            batch_size=batch_size,
            n_batches=n_batches,
            label=f"VAMP-JCD @ {doppler:.0f} Hz",
        )
        all_results["benchmarks"][f"bler_vs_snr_doppler_{int(doppler)}hz"] = bler_results

    # 3. BLER vs Doppler spread at fixed SNR
    n_batches = 1 if fast else 3
    doppler_results = benchmark_bler_vs_doppler(
        model, config, device,
        doppler_list=[300, 600, 900, 1200, 1500, 1800, 2100, 2400],
        snr_db=-3.0,
        batch_size=128,
        n_batches=n_batches,
    )
    all_results["benchmarks"]["bler_vs_doppler_snr_-3db"] = doppler_results

    # 4. Full baseline comparison (VAMP-JCD vs LMMSE vs LS vs Genie)
    comparison = benchmark_full_comparison(config, device, snr_db_list)
    all_results["benchmarks"]["baseline_comparison"] = comparison

    # 5. CFO estimation accuracy
    cfo_results = benchmark_cfo_accuracy(config, device)
    all_results["benchmarks"]["cfo_accuracy"] = cfo_results

    # 6. Channel sparsity probe
    sparsity = benchmark_channel_sparsity(config, device)
    all_results["benchmarks"]["channel_sparsity"] = sparsity

    # 7. NMSE for each baseline receiver
    print(f"\n  NMSE comparison:")
    sim = LEONTNChannelSimulator(config)
    batch = sim.generate_batch(
        batch_size=128, elevation_deg=45.0,
        snr_db=0.0, doppler_hz=1200.0,
    )
    y = batch["y"].to(device)
    h_true = batch["h_true"].to(device)
    pilots = batch["pilots"].to(device)
    pilot_mask = batch["pilot_mask"].to(device)

    lmmse = LMMSEBaseline(config).to(device)
    ls = LSBaseline(config).to(device)
    with torch.no_grad():
        h_vamp, _, _, _ = model(y, pilots, pilot_mask)
        h_lm, _, _, _ = lmmse(y, pilots, pilot_mask)
        h_ls, _, _, _ = ls(y, pilots, pilot_mask)

    nmses = {
        "vamp_jcd": channel_nmse_loss(h_vamp, h_true).item(),
        "lmmse": channel_nmse_loss(h_lm, h_true).item(),
        "ls": channel_nmse_loss(h_ls, h_true).item(),
    }
    all_results["benchmarks"]["nmse_comparison"] = nmses
    for name, nmse in nmses.items():
        print(f"    {name:15s}: NMSE = {nmse:.4f}")

    # Save all results
    output_path = os.path.join(output_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Benchmark results saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Parameters: {n_params} / 1000")
    n_bler = len([k for k in all_results["benchmarks"] if "bler" in k])
    print(f"  BLER curves evaluated: {n_bler}")
    print(f"  Baselines compared: LMMSE, LS, Genie")
    print(f"  Doppler values tested: {doppler_values}")
    print(f"  SNR range: [{snr_db_list[0]}, {snr_db_list[-1]}] dB")
    print()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VAMP-JCD Receiver Benchmark Suite"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Quick sanity check (fewer batches)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda)")
    parser.add_argument("--output", type=str, default="benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--doppler", type=float, nargs="+",
                        default=[600.0, 1200.0, 2400.0],
                        help="Doppler frequencies to test (Hz)")
    parser.add_argument("--layers", type=int, default=5,
                        help="Number of unfolded layers")
    parser.add_argument("--denoiser", type=str, default="delay_doppler_shrinkage",
                        help="Denoiser type")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = DU_VAMP_JCD_Config(
        n_unfolded_layers=args.layers,
        weight_tying=True,
        denoiser_type=args.denoiser,
        cfo_mode="analytic_plus_learned",
    )

    run_all_benchmarks(
        config, device,
        output_dir=args.output,
        fast=args.fast,
        doppler_values=args.doppler,
    )
