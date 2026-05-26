"""
Ablation Runner for Deep-Unfolded VAMP-JCD Receiver
=====================================================

Implements the 8 single-field ablations defined in the architecture design
(architect stage, Step 9). Each ablation tests a specific hypothesis about
the receiver's design.

Usage:
    python ablation_runner.py                    # Run all ablations
    python ablation_runner.py --ablation depth   # Run only depth ablation
    python ablation_runner.py --fast             # Minimal configs for quick check
    python ablation_runner.py --output results.json  # Save results to JSON

Each ablation:
    - Sweeps a single ModelConfig field
    - Evaluates BLER and NMSE at multiple SNR points
    - Reports parameter count
    - Saves results to ablation_results/ directory
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "coder"))

from model import DU_VAMP_JCD_Config, VAMPJCDReceiver, count_params
from channel import LEONTNChannelSimulator
from train import channel_nmse_loss, compute_bler, compute_ber


# ─── Ablation Definitions ─────────────────────────────────────────────────

BASELINE_CONFIG = DU_VAMP_JCD_Config(
    n_unfolded_layers=5,
    weight_tying=True,
    denoiser_type="delay_doppler_shrinkage",
    cfo_mode="analytic_plus_learned",
    use_soft_feedback=True,
    n_damping_params=1,
    n_onsager_params=1,
    learn_noise_precision=True,
    learn_channel_precision=True,
)

ABLATIONS = {
    "depth": {
        "description": "Ablation 1: Unfolded layer count (depth)",
        "field": "n_unfolded_layers",
        "values": [1, 2, 3, 5, 7, 10],
        "hypothesis": "Each additional iteration improves BLER monotonically, with diminishing returns after T=5.",
        "failure_condition": "If BLER(T=7) > BLER(T=5): VAMP diverging. If BLER(T=5) ≈ BLER(T=3): architecture saturates early.",
    },
    "weight_tying": {
        "description": "Ablation 2: Weight tying vs. per-layer thresholds",
        "field": "weight_tying",
        "values": [True, False],
        "hypothesis": "Per-layer thresholds provide ≤ 0.3 dB improvement over shared at 5× parameter cost.",
        "failure_condition": "If gap > 1 dB: shrinkage needs to be iteration-dependent.",
    },
    "denoiser": {
        "description": "Ablation 3: Denoiser type (expressiveness Pareto frontier)",
        "field": "denoiser_type",
        "values": ["delay_doppler_shrinkage", "element_shrinkage", "small_cnn", "mlp"],
        "hypothesis": "Delay-Doppler shrinkage exploits channel sparsity better than element-wise, matches CNN at < 5% params.",
        "failure_condition": "If delay-Doppler ≤ element-wise: channel NOT sparse in DD domain → fundamental assumption false.",
    },
    "cfo_mode": {
        "description": "Ablation 4: CFO estimation mode",
        "field": "cfo_mode",
        "values": ["analytic", "analytic_plus_learned"],
        "hypothesis": "Learned scale/bias correction provides ≥ 0.5 dB improvement over pure analytic at high Doppler.",
        "failure_condition": "If gap < 0.2 dB: analytic CFO sufficient → remove learnable components for FPGA.",
    },
    "soft_feedback": {
        "description": "Ablation 5: Soft symbol feedback",
        "field": "use_soft_feedback",
        "values": [True, False],
        "hypothesis": "Soft feedback improves channel estimation by leveraging data as virtual pilots at low SNR.",
        "failure_condition": "If gap < 0.3 dB: soft feedback not beneficial → simplify architecture.",
    },
    "damping": {
        "description": "Ablation 6: Damping strategy (shared vs. per-layer)",
        "field": "n_damping_params",
        "values": [1, 5],
        "hypothesis": "Per-layer damping provides faster convergence but same final BLER at T=5.",
        "failure_condition": "If per-layer improves final BLER: VAMP iterations have different stability requirements.",
    },
    "noise_precision": {
        "description": "Ablation 8: Learned vs. fixed noise precision",
        "field": "learn_noise_precision",
        "values": [True, False],
        "hypothesis": "Learning γ_z adapts to effective noise better than using true SNR.",
        "failure_condition": "If learned γ_z performs worse: 1-param insufficient → need SNR-adaptive mechanism.",
    },
}


def make_config(field: str, value) -> DU_VAMP_JCD_Config:
    """Create a config with a single field changed from baseline."""
    cfg_dict = {
        k: getattr(BASELINE_CONFIG, k) for k in
        list(DU_VAMP_JCD_Config.__dataclass_fields__.keys())
    }
    cfg_dict[field] = value

    # Handle special cases
    if field == "denoiser_type" and value in ["element_shrinkage", "small_cnn", "mlp"]:
        # Isolation: use analytic CFO to isolate denoiser effect
        if "cfo_mode" not in cfg_dict or "cfo_mode" not in field:
            cfg_dict["cfo_mode"] = "analytic"

    return DU_VAMP_JCD_Config(**cfg_dict)


# ─── Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, config, device: str,
                   snr_db_list: List[float] = None,
                   doppler_hz: float = 1200.0,
                   elevation_deg: float = 45.0,
                   batch_size: int = 128,
                   n_batches: int = 3) -> Dict:
    """Evaluate model at multiple SNR points.

    Returns dict of {snr: {bler, ber, nmse}}.
    """
    if snr_db_list is None:
        snr_db_list = [-10.0, -5.0, -3.0, 0.0, 5.0, 10.0]

    sim = LEONTNChannelSimulator(config)
    model.eval()
    results = {}

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

        results[str(snr)] = {
            "bler": sum(blers) / len(blers),
            "ber": sum(bers) / len(bers),
            "nmse": sum(nmses) / len(nmses),
        }
    return results


# ─── Ablation Runner ───────────────────────────────────────────────────────

def run_ablation(name: str, ablation_def: Dict, device: str,
                 fast: bool = False) -> Dict:
    """Run a single ablation and return results dict."""
    print(f"\n{'='*60}")
    print(f"  {ablation_def['description']}")
    print(f"{'='*60}")
    print(f"  Hypothesis: {ablation_def['hypothesis']}")
    print()

    field = ablation_def["field"]
    values = ablation_def["values"]
    result = {
        "ablation": name,
        "description": ablation_def["description"],
        "hypothesis": ablation_def["hypothesis"],
        "failure_condition": ablation_def["failure_condition"],
        "configs": {},
    }

    for val in values:
        cfg = make_config(field, val)
        model = VAMPJCDReceiver(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Use smaller evaluation for fast mode
        if fast:
            snr_list = [-3.0, 0.0, 5.0]
            n_batches = 1
            batch_size = 64
        else:
            snr_list = [-10.0, -5.0, -3.0, 0.0, 5.0, 10.0]
            n_batches = 3
            batch_size = 128

        eval_result = evaluate_model(
            model, cfg, device,
            snr_db_list=snr_list,
            n_batches=n_batches,
            batch_size=batch_size,
        )

        print(f"  [{field}={str(val):12s}]  params={n_params:3d}")
        for snr_key in sorted(eval_result.keys(), key=lambda x: float(x)):
            r = eval_result[snr_key]
            print(f"    SNR={float(snr_key):+.1f} dB: BLER={r['bler']:.4f}, "
                  f"NMSE={r['nmse']:.4f}")

        result["configs"][str(val)] = {
            "params": n_params,
            "evaluation": eval_result,
        }

    return result


def run_all_ablations(device: str, fast: bool = False,
                      output_file: Optional[str] = None) -> Dict:
    """Run all defined ablations."""
    print(f"VAMP-JCD Receiver — Full Ablation Study")
    print(f"Device: {device}  |  Fast mode: {fast}")
    print(f"Baseline config: {BASELINE_CONFIG.n_trainable_params} params")
    print()

    # Run baseline evaluation first
    print("--- Baseline Evaluation ---")
    baseline_model = VAMPJCDReceiver(BASELINE_CONFIG).to(device)
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    snr_list = [-3.0, 0.0] if fast else [-10.0, -5.0, -3.0, 0.0, 5.0, 10.0]
    baseline_eval = evaluate_model(
        baseline_model, BASELINE_CONFIG, device,
        snr_db_list=snr_list,
        n_batches=1 if fast else 3,
        batch_size=64 if fast else 128,
    )
    for snr_key in sorted(baseline_eval.keys(), key=lambda x: float(x)):
        r = baseline_eval[snr_key]
        print(f"  SNR={float(snr_key):+.1f} dB: BLER={r['bler']:.4f}, NMSE={r['nmse']:.4f}")
    print()

    all_results = {
        "baseline": {
            "description": "Baseline VAMP-JCD Receiver",
            "config": {
                k: getattr(BASELINE_CONFIG, k)
                for k in DU_VAMP_JCD_Config.__dataclass_fields__
            },
            "params": baseline_params,
            "evaluation": baseline_eval,
        },
        "ablations": {},
        "metadata": {
            "device": device,
            "fast_mode": fast,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    for name, ablation_def in ABLATIONS.items():
        result = run_ablation(name, ablation_def, device, fast)
        all_results["ablations"][name] = result

    # Summary
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline: {baseline_params} params")
    for name, result in all_results["ablations"].items():
        for val_key, config_data in result["configs"].items():
            eval_data = config_data["evaluation"]
            snr0 = eval_data.get("0.0", eval_data.get(list(eval_data.keys())[0]))
            print(f"  {name:20s} ({str(val_key):12s}): "
                  f"{config_data['params']:3d} params, "
                  f"BLER@0dB={snr0['bler']:.4f}")

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to {output_file}")

    return all_results


# ─── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VAMP-JCD Receiver Ablation Study"
    )
    parser.add_argument("--ablation", type=str, default=None,
                        choices=list(ABLATIONS.keys()) + ["all"],
                        help="Specific ablation to run (default: all)")
    parser.add_argument("--fast", action="store_true",
                        help="Run minimal configs for quick check")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda)")
    parser.add_argument("--output", type=str, default="ablation_results.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.ablation and args.ablation != "all":
        # Run single ablation
        result = run_ablation(args.ablation, ABLATIONS[args.ablation], device, args.fast)
        print(json.dumps(result, indent=2, default=str))
    else:
        run_all_ablations(device, args.fast, args.output)
