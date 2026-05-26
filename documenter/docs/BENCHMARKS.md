# Benchmarks

All numbers are reproducible with the commands shown. Numbers marked `TODO` have not been measured — do not cite them as validated results.

> **Important:** All current benchmark measurements are from **randomly initialized** model weights. BLER ≈ 1.0 (near random guessing for QPSK) and NMSE > 1.0 are expected for an untrained model. These are not scientifically meaningful for comparing architecture quality. The central comparison (VAMP-JCD vs LMMSE at low SNR) requires trained model weights. See [TRAINING.md](TRAINING.md) for the training recipe.

## Baseline comparison infrastructure

Four receiver implementations share a common interface (`forward(y, pilots, pilot_mask) -> (h_hat, x_soft, llrs, state)`):

| Baseline | File | Status |
|---|---|---|
| **VAMP-JCD** (proposed) | `coder/model.py::VAMPJCDReceiver` | 24 params, 5 unfolded iterations |
| **LMMSE** (conventional) | `coder/baselines.py::LMMSEBaseline` | LS estimation + linear interpolation + MMSE detection |
| **LS** (least-squares) | `coder/baselines.py::LSBaseline` | LS estimation + ZF detection |
| **Genie** (upper bound) | `coder/baselines.py::GenieBound` | Perfect channel knowledge + MMSE detection |

## Channel estimation NMSE vs SNR

| SNR (dB) | VAMP-JCD NMSE | LMMSE NMSE | LS NMSE | Genie NMSE | Command |
|---|---|---|---|---|---|
| –10 | > TODO: unverified | > TODO: unverified | > TODO: unverified | 0.0 | `pytest validator/test_model.py -k "test_channel_nmse_vs_snr"` |
| –5 | > TODO | > TODO | > TODO | 0.0 | |
| –3 | > TODO | > TODO | > TODO | 0.0 | |
| 0 | > TODO | > TODO | > TODO | 0.0 | |
| 5 | > TODO | > TODO | > TODO | 0.0 | |
| 10 | > TODO | > TODO | > TODO | 0.0 | |

> **Note:** NMSE = 0.0 for Genie (perfect channel knowledge). All other cells require trained model weights. At random init, NMSE values are ≥1.0 across all SNRs.

## BLER vs Eb/N0 (at fixed Doppler spreads)

### Doppler = 600 Hz
> TODO: unverified — requires trained model. Expected: VAMP-JCD achieves < 0.1 BLER at Eb/N0 = –3 dB.

### Doppler = 1.2 kHz (primary operating point)
> TODO: unverified — requires trained model. Expected: VAMP-JCD outperforms LMMSE by ≥2 dB at Eb/N0 = –3 dB.

### Doppler = 2.4 kHz
> TODO: unverified — requires trained model. ICI may dominate; expected gap vs 1.2 kHz is >2 dB.

**Reproduce:** `python validator/run_benchmarks.py --snr-range -10 10 --doppler 600 1200 2400`

## BLER vs Doppler spread (at Eb/N0 = –3 dB)

| Doppler (Hz) | VAMP-JCD BLER | LMMSE BLER | LS BLER | Genie BLER | Command |
|---|---|---|---|---|---|
| 600 | > TODO | > TODO | > TODO | > TODO | `python validator/run_benchmarks.py --doppler-range 300 2400 --snr -3` |
| 1200 | > TODO | > TODO | > TODO | > TODO | |
| 2400 | > TODO | > TODO | > TODO | > TODO | |

## CFO estimation accuracy

| SNR (dB) | Analytic CFO error | Learned CFO error | Command |
|---|---|---|---|
| –10 | > TODO | > TODO | `pytest validator/test_model.py -k "test_cfo_analytic_estimate_vs_true"` |
| 0 | > TODO | > TODO | |
| 10 | > TODO | > TODO | |

> The analytic estimator is parameter-free; the learned mode adds 2 scale/bias parameters. At high SNR (> 0 dB), both should converge to small error (< 0.01 cycles/symbol).

## Synthetic tasks

| Task | Metric | Value | Command | Notes |
|---|---|---|---|---|
| Parameter count budget | trainable params | 24 | `pytest -k "test_parameter_count"` | Verified ≤1,000 budget |
| MAC count estimation | MACs/block | ~15k (T=5) | `python validator/profile_model.py --mode macs` | At T=5 exceeds 10k budget; T=3 gives ~9k |
| Delay-Doppler sparsity | energy concentration | > TODO | `pytest -k "test_delay_doppler_sparsity_preservation"` | Measures energy in top DD bins vs time-freq |
| CFO estimate at random init | abs error | < 0.5 cycles/sym | `pytest -k "test_cfo_analytic_estimate_vs_true"` | Analytic method works at random init |
| Model determinism | max diff | < 1e-6 | `pytest -k "test_model_is_reproducible"` | Verified: seed-controlled determinism |
| VAMP-JCD vs LMMSE (random init) | BLER gap | ~0 dB | `pytest -k "test_vamp_jcd_outperforms_lmmse_at_low_snr"` | At random init, no architecture outperforms |
| bf16 compatibility | loss diff | < 1e-3 | `pytest -k "test_bf16"` | Verified bf16 forward pass |
| Numerical stability | NaN count | 0 | `pytest -k "TestNumerics"` | Stable at extreme SNR/CFO/zero input |
| Gradient flow (1 step) | grad norms | nonzero | `pytest -k "test_gradient"` | All 24 params receive nonzero gradients |

## Ablation study

All ablations run with randomly initialized weights. The values below are for infrastructure verification only — scientifically meaningful comparisons require trained models.

### Ablation 1: Unfolded layer count (depth)

| T | Params | BLER @ –3 dB | NMSE @ –3 dB | BLER @ 0 dB | Command |
|---|---|---|---|---|---|
| 1 | 24 | 1.0 | 0.98 | 1.0 | `python validator/ablation_runner.py --ablation depth` |
| 2 | 24 | 1.0 | 1.06 | 1.0 | |
| 3 | 24 | 1.0 | 1.30 | 1.0 | |
| 5 | 24 | 1.0 | 2.35 | 1.0 | |
| 7 | 24 | 1.0 | 3.37 | 1.0 | |
| 10 | 24 | 1.0 | 3.83 | 1.0 | |

**Observation (random init):** NMSE increases with T because each additional VAMP iteration at random initialization compounds estimation errors. After training, NMSE should decrease with T.

### Ablation 2: Weight tying

| Weight tying | Params | BLER @ –3 dB | NMSE @ –3 dB | Command |
|---|---|---|---|---|
| True | 24 | 1.0 | 2.37 | `python validator/ablation_runner.py --ablation weight_tying` |
| False | 96 | 1.0 | 2.35 | |

### Ablation 3: Denoiser type

| Denoiser | Params | BLER @ –3 dB | NMSE @ –3 dB | Command |
|---|---|---|---|---|
| delay_doppler_shrinkage | 24 | 1.0 | 2.38 | `python validator/ablation_runner.py --ablation denoiser` |
| element_shrinkage | 6 | 1.0 | 2.34 | |
| small_cnn | 66 | 1.0 | 1.03 | |
| mlp | 36 | 1.0 | 1.12 | |

**Note (random init):** The small_cnn and mlp denoisers produce lower NMSE at random init because their learnable weights are randomly initialized to produce near-zero output, which has lower NMSE than the shrinkage-based denoisers' positive-threshold output.

### Ablation 4: CFO mode

| CFO mode | Params | BLER @ –3 dB | NMSE @ –3 dB | Command |
|---|---|---|---|---|
| analytic | 22 | 1.0 | 2.36 | `python validator/ablation_runner.py --ablation cfo_mode` |
| analytic_plus_learned | 24 | 1.0 | 2.37 | |

### Ablation 5: Soft symbol feedback

| Soft feedback | Params | BLER @ –3 dB | NMSE @ –3 dB | Command |
|---|---|---|---|---|
| True | 24 | 1.0 | 2.37 | `python validator/ablation_runner.py --ablation soft_feedback` |
| False | 24 | 1.0 | 1.19 | |

### Ablation 6: Damping strategy

| Damping params | Params | BLER @ –3 dB | NMSE @ –3 dB | Command |
|---|---|---|---|---|
| 1 (shared) | 24 | 1.0 | 2.38 | `python validator/ablation_runner.py --ablation damping` |
| 5 (per-layer) | 28 | 1.0 | 2.31 | |

### Ablation 7: Quantization sensitivity
> TODO: unverified — requires FINN/Vitis AI toolchain or simulated INT8 quantization in PyTorch.

### Ablation 8: Noise precision

| Learn gamma_z | Params | BLER @ –3 dB | NMSE @ –3 dB | Command |
|---|---|---|---|---|
| True | 24 | 1.0 | 2.37 | `python validator/ablation_runner.py --ablation noise_precision` |
| False | 23 | 1.0 | 2.29 | |

**Reproduce all ablations:** `python validator/ablation_runner.py`

## Profiling

GPU: not applicable (benchmarked on CPU)

| Phase | Time (ms) | Peak mem (MB) | Notes |
|---|---|---|---|
| Forward pass (B=1, T=5) | ~0.5–2.0 | < 10 | CPU-only, no GPU required |
| Forward pass (B=256, T=5) | ~10–50 | < 100 | Batch inference |
| Full training step (B=256) | ~20–100 | < 200 | Forward + backward + optimizer |
| Gradient checkpointing | N/A | N/A | Negligible benefit for 24-param model |

Estimated FLOPs: ~30k per forward pass (T=5) = ~15k MACs × 2 FLOPs/MAC = ~30k FLOPs. Training: ~6 × params × tokens ≈ negligible on modern hardware.

**Reproduce:** `python validator/profile_model.py`

## Research-quality evaluation

| Dimension | Score/status | Evidence | Gaps |
|---|---|---|---|
| Novelty | 4/5 | Comprehensive literature review (7+ references), novel joint integration of CFO+channel+detection for LEO NTN IoT, 24-param FPGA deployment path addresses 2025 survey's #1 gap | No experimental validation yet; novelty is in application not in new algorithmic primitives |
| Experimental comprehensiveness | 3/5 | BLER vs SNR at 3 Doppler spreads, BLER vs Doppler, NMSE, CFO accuracy, 8 single-field ablations, numerical stability | No trained model results; no channel-coded BLER; no held-out elevation evaluation; no FPGA measurements |
| Theoretical foundation | 4/5 | 12 grounded design decisions with literature citations; VAMP state evolution; delay-Doppler sparsity; LMMSE estimation theory; turbo receiver theory | Small-system behavior (N=168) uncharacterized; no formal convergence proof for joint CFO+channel+detection |
| Result analysis | 1/5 | Benchmark infrastructure generates all required curves; numerical safety validated | All results from random init — not scientifically meaningful; no error bars; no statistical significance |
| Implementation reproducibility | 4/5 | Complete implementation (24 files), seeded RNG (seed=42), deterministic inference, cross-platform tests, JSON output | No Docker container; no pinned dependencies; no single-command end-to-end reproduce pipeline |
| Writing readiness | 3/5 | Docstrings on every class/method; ASCII architecture diagram; traceability table; consistent naming | No paper draft; no figures; no ablation interpretation |

### Blocking gaps (from validator scorecard)

| Gap | Severity | Affected claims |
|---|---|---|
| No trained model → benchmarks produce random-init values | HIGH | All novelty and performance claims |
| FPGA resource/power/latency not measured | HIGH | SWaP claims (research hypothesis) |
| Held-out elevation angle not tested | MEDIUM | Generalization claim |
| MAC budget exceeded at T=5 (~15k vs 10k budget) | MEDIUM | FPGA deployment feasibility |
| Quantization sensitivity not measured | LOW | FPGA deployment study |

### Recommended next experiments

1. **P0:** Train VAMP-JCD receiver on synthetic LEO NTN data and measure BLER vs SNR curves against LMMSE, LS, and Genie baselines. Success criterion: ≥2 dB improvement over LMMSE at SNR = –3 dB with 1.2 kHz Doppler.
2. **P0:** Sweep unfolded iterations (T=1..10) to find optimal depth for FPGA deployment. Identify knee point where marginal BLER improvement per iteration < 0.2 dB.
3. **P1:** Evaluate VAMP-JCD at held-out elevation angles (15°, 75°) after training at 45°. Success criterion: BLER gap < 3 dB.
4. **P1:** Run denoiser ablation study comparing delay-Doppler shrinkage vs element shrinkage vs small CNN vs MLP. Validate the core DD sparsity assumption.
5. **P2:** Profile INT8 quantization sensitivity using simulated quantization in PyTorch before FPGA deployment.
6. **P2:** Conduct systematic training hyperparameter study (SNR range, batch size, learning rate).
