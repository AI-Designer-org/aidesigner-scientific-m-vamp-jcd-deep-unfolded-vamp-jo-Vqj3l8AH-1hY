# Training & Reproduction

## Environment

- Python: 3.10+
- PyTorch: 2.0+ (tested with 2.1+)
- CUDA: 11.8+ (GPU recommended for training; CPU sufficient for small-batch inference)
- Other: no special kernels required (triton / flash-attn / mamba-ssm not needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pytest
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Default hyperparameters

Derived from `DU_VAMP_JCD_Config` in `coder/model.py`.

| Field | Default | Rationale |
|---|---|---|
| `n_unfolded_layers` | 5 | T=5 VAMP iterations; tradeoff between BLER and MAC budget |
| `weight_tying` | True | Share denoiser thresholds across iterations (24 params vs 104) |
| `denoiser_type` | `delay_doppler_shrinkage` | Exploit channel sparsity in delay-Doppler domain |
| `n_delay_groups` | 3 | Delay bins: short / medium / long |
| `n_doppler_groups` | 3 | Doppler bins: low / medium / high |
| `cfo_mode` | `analytic_plus_learned` | Hybrid: analytic anchor + learned scale/bias (2 params) |
| `use_soft_feedback` | True | Feed soft symbols to next iteration (virtual pilots) |
| `learn_noise_precision` | True | Learn gamma_z from data (adapts to effective noise) |
| `learn_channel_precision` | True | Learn gamma_h from data (channel prior precision) |
| `n_damping_params` | 1 | Shared damping factor across iterations |
| `n_onsager_params` | 1 | Shared Onsager coefficient |
| `training_snr_range_db` | (-10.0, 10.0) | SNR sampled uniformly per batch |
| `modulation` | `qpsk` | NB-IoT NPUSCH Format 1 |

## Recommended training recipe

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | beta1=0.9, beta2=0.999 |
| Peak LR | 1e-3 | fixed (no warmup needed for 24-param model) |
| Batch size | 256 | gradient accumulation not needed |
| Weight decay | 1e-4 | mild regularization |
| Grad clip | 1.0 | global norm; prevents early-iteration spikes |
| Precision | float32 | bf16 compatible but float32 recommended for LLR computation |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=5, min_lr=1e-6 |
| Training samples | 100,000+ channel realizations | generated on-the-fly by `LEONTNChannelSimulator` |
| Validation samples | 10,000 held-out realizations | separate seed from training |

### Loss function

The composite loss function (defined in `coder/train.py::total_loss`):

```
L = lambda_1 * NMSE(h_pred, h_true)    # Channel estimation loss
  + lambda_2 * BCE(llrs, bits)          # Detection loss (binary cross-entropy)
  + lambda_3 * ||theta||_1             # Sparsity regularization on shrinkage thresholds
```

Default weights: lambda_1 = 1.0, lambda_2 = 1.0, lambda_3 = 0.01.

### Data generation strategy

The `LEONTNChannelSimulator` generates batches on-the-fly with randomized parameters:

- **SNR**: uniform over training_snr_range_db per batch (default: –10 to +10 dB)
- **Elevation angle**: uniform over [10°, 90°] per batch
- **Doppler spread**: configurable (default 1.2 kHz); can be fixed or randomized
- **Delay spread**: 0.3 us (suburban) by default
- **Seed**: set for reproducibility (`seed=42` in all benchmark code)

This on-the-fly generation means training data is effectively infinite and never repeated.

## Expected behavior

> TODO: unverified — no reference training run has been completed. Below are the expected behaviors based on the architecture design:

- **Loss convergence:** Total loss should decrease monotonically from ~2.0–3.0 (random init) to ~0.1–0.5 (trained) within 50 epochs.
- **Channel NMSE:** Should decrease from >1.0 (random init) to <0.1 at high SNR (10 dB) and <0.5 at low SNR (–3 dB).
- **BLER at low SNR:** Should drop from ~0.75 (random init, QPSK random guessing) to ≤0.1 at Eb/N0 = –3 dB after training (with 1.2 kHz Doppler).
- **VAMP iteration utility:** BLER should improve with each unfolded iteration; after training, BLER(T=5) < BLER(T=1) by ≥2 dB at low SNR.

**To run a quick training sanity check:**

```bash
python coder/train.py
```

This runs a 50-epoch training loop with default parameters (a full training run on CPU takes ~30–60 minutes; on GPU ~5–10 minutes).

**To run evaluation after training:**

```bash
python validator/run_benchmarks.py
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss NaN in first step | Division by zero in VAMP linear step denominator | Ensure epsilon (1e-10) is added; check that gamma_z and gamma_h exp() don't underflow |
| BLER remains ~0.75 after training | Learning rate too low or training SNR range too narrow | Verify gradient flow (`pytest -k test_gradient`); try LR=1e-2 for first 10 epochs then reduce |
| Channel NMSE > 1.0 throughout | Shrinkage thresholds too large (over-denoising to zero) | Initialize shrinkage thresholds to 0.01 instead of 0.1; reduce lambda_3 |
| BLER at high SNR (10 dB) worse than at low SNR | gamma_z learned a very low precision, over-amplifying noise | Clamp gamma_z to [0.01, 100.0]; check that `learn_noise_precision` interacts correctly with `soft_mmse_detector` |
| BLER degrades with more unfolded iterations (T=7 vs T=5) | VAMP state evolution diverging in small-system regime | Reduce T to 5 or 3; increase damping (learned alpha closer to 0.8); check `test_gradient_flow_through_unfolded_iterations` |
| CFO estimate diverging across iterations | ICI at high Doppler corrupts pilot-based estimate | Disable CFO residual update at iterations > 3; or switch to pure analytic mode |
| bf16 produces NaN in LLR computation | bf16 insufficient precision for log/exp/tanh in LLR path | Use float32 for detector; cast only through VAMP linear step in bf16 |
| Gradient norm spikes at iteration T | Onsager correction accumulating correlation error | Reduce beta initialization to 0.0; use per-layer Onsager with beta closer to 0 at early iterations |
| Training loss oscillates without converging | ReduceLROnPlateau not reducing LR fast enough | Increase patience to 10; or switch to cosine annealing schedule |

### Quick diagnostic commands

```bash
# Verify model param count
python -c "from model import DU_VAMP_JCD_Config, VAMPJCDReceiver; cfg = DU_VAMP_JCD_Config(); m = VAMPJCDReceiver(cfg); print(sum(p.numel() for p in m.parameters() if p.requires_grad))"

# Check gradient flow
python -c "
import torch; from model import DU_VAMP_JCD_Config, VAMPJCDReceiver
cfg = DU_VAMP_JCD_Config(); m = VAMPJCDReceiver(cfg)
y = torch.randn(2,12,14,2); p = torch.randn(2,24,2); pm = torch.zeros(12,14,dtype=torch.bool); pm[:,3]=True; pm[:,10]=True
loss = m(y,p,pm)[2].sum(); loss.backward()
for n,p in m.named_parameters(): print(f'{n}: grad={p.grad.norm().item():.6f}' if p.grad is not None else f'{n}: NO GRAD')
"

# Verify determinism
python -c "
import torch; from model import DU_VAMP_JCD_Config, VAMPJCDReceiver
torch.manual_seed(42); cfg = DU_VAMP_JCD_Config(); m1 = VAMPJCDReceiver(cfg)
torch.manual_seed(42); cfg2 = DU_VAMP_JCD_Config(); m2 = VAMPJCDReceiver(cfg2)
y = torch.randn(2,12,14,2); p = torch.randn(2,24,2); pm = torch.zeros(12,14,dtype=torch.bool); pm[:,3]=True; pm[:,10]=True
o1 = m1(y,p,pm)[2]; o2 = m2(y,p,pm)[2]; print(f'Deterministic: {(o1-o2).abs().max().item() < 1e-6}')
"
```
