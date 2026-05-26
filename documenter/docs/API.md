# API Reference

## coder/model.py — VAMP-JCD Receiver

### `class DU_VAMP_JCD_Config`

Configuration dataclass for the VAMP-JCD receiver and channel simulator. All fields have defaults matching NB-IoT NPUSCH Format 1.

**Fields:**

| Field | Type | Default | Rationale |
|---|---|---|---|
| `domain` | str | `"scientific_ml"` | Primary ML domain |
| `subdomain` | str | `"leo_ntn_iot_receiver"` | ML subdomain |
| `n_subcarriers` | int | 12 | NB-IoT: 12 SCs × 15 kHz = 180 kHz |
| `n_symbols` | int | 14 | OFDM symbols per slot (normal CP) |
| `subcarrier_spacing_hz` | float | 15000.0 | NB-IoT subcarrier spacing |
| `n_pilots` | int | 24 | DMRS: 2 symbols × 12 SCs (NPUSCH Format 1) |
| `pilot_symbol_indices` | tuple | (3, 10) | DMRS symbol positions (0-indexed) |
| `modulation` | str | `"qpsk"` | pi/2-BPSK or QPSK |
| `n_bits_per_symbol` | int | 2 | For QPSK |
| `n_unfolded_layers` | int | 5 | T = 5 VAMP iterations |
| `weight_tying` | bool | True | Share denoiser params across layers |
| `denoiser_type` | str | `"delay_doppler_shrinkage"` | Options: `delay_doppler_shrinkage`, `element_shrinkage`, `small_cnn`, `mlp` |
| `n_delay_groups` | int | 3 | Delay bins grouped |
| `n_doppler_groups` | int | 3 | Doppler bins grouped |
| `cfo_mode` | str | `"analytic_plus_learned"` | Options: `analytic`, `analytic_plus_learned`, `mlp` |
| `cfo_learned_scale` | bool | True | Learnable scalar multiplier |
| `cfo_learned_bias` | bool | True | Learnable scalar additive bias |
| `cfo_mlp_hidden_dim` | int | 0 | 0 = analytic only; >0 enables small MLP |
| `detector_type` | str | `"mmse_soft"` | Options: `mmse_soft`, `zf`, `learned` |
| `use_soft_feedback` | bool | True | Feed soft symbols to next iteration |
| `learn_noise_precision` | bool | True | gamma_z: 1 learnable scalar |
| `learn_channel_precision` | bool | True | gamma_h: 1 learnable scalar |
| `n_damping_params` | int | 1 | 1 = shared, n_layers = per-layer |
| `n_onsager_params` | int | 1 | 1 = shared, n_layers = per-layer |
| `dropout` | float | 0.0 | No dropout in unfolded architecture |
| `use_bias` | bool | False | Minimal bias in unfolded layers |
| `dtype` | str | `"float32"` | Training precision |
| `quantization_target` | str | `"int8"` | For FPGA deployment |
| `training_snr_range_db` | tuple | (-10.0, 10.0) | SNR range for training data |
| `target_fpga` | str | `"xilinx_zynq_zu3eg"` | Target FPGA platform |
| `max_dsp_slices` | int | 600 | Zynq ZU3EG DSP budget |
| `max_power_watts` | float | 5.0 | CubeSat SWaP constraint |
| `target_latency_ms` | float | 1.0 | Real-time receiver requirement |
| `n_trainable_params` | int | 0 | Computed at init |
| `n_macs_per_block` | int | 0 | Computed at init |

**Methods:**
- `__post_init__()` — computes parameter budget after initialization
- `_compute_parameter_budget()` — compute total trainable parameters from config
- `get_layer_config(layer_idx: int) -> Dict[str, Any]` — get config overrides for a specific unfolded layer

---

### `class BaseDenoiser(ABC, nn.Module)`

Abstract base class for the VAMP denoiser operator. All denoiser variants operate in the delay-Doppler domain and preserve input shape.

**Methods:**
- `forward(h_dd: Tensor, layer_idx: int) -> Tensor` — denoise channel estimate in delay-Doppler domain
  - `h_dd`: (B, N_sc, N_sym, 2) — channel in delay-Doppler domain, last dim = (real, imag)
  - Returns: (B, N_sc, N_sym, 2) — denoised channel

---

### `class DelayDopplerShrinkageDenoiser(BaseDenoiser)`

Learnable soft-thresholding denoiser in the delay-Doppler domain. 18 learnable parameters with weight tying: 3 delay groups × 3 Doppler groups × 2 (real/imag).

**Constructor:** `DelayDopplerShrinkageDenoiser(config: DU_VAMP_JCD_Config)`

**Methods:**
- `forward(h_dd: Tensor, layer_idx: int) -> Tensor` — apply learnable soft thresholding
  - Shape invariant: (B, N_sc, N_sym, 2) → (B, N_sc, N_sym, 2)

**Internal buffers:**
- `delay_bin_map`: (N_sc,) — maps each subcarrier to a delay bin index
- `doppler_bin_map`: (N_sym,) — maps each OFDM symbol to a Doppler bin index

---

### `class ElementShrinkageDenoiser(BaseDenoiser)`

Simple element-wise shrinkage denoiser with 2 learnable thresholds (one per real/imag component). Ablation baseline — ignores delay-Doppler structure.

**Constructor:** `ElementShrinkageDenoiser(config: DU_VAMP_JCD_Config)`

---

### `class TinyCNNDenoiser(BaseDenoiser)`

Lightweight 2D CNN denoiser (62 params). Architecture: 1×1 conv (2→4) → 3×3 depthwise conv (4→4) → 1×1 conv (4→2). Operates on (B, C, H, W) internally; input/output format is (B, N_sc, N_sym, 2).

**Constructor:** `TinyCNNDenoiser(config: DU_VAMP_JCD_Config)`

---

### `class MLPDenoiser(BaseDenoiser)`

Per-element 2-layer MLP denoiser (~20 params). Applies the same MLP to each (real, imag) pair independently.

**Constructor:** `MLPDenoiser(config: DU_VAMP_JCD_Config)`

---

### `def build_denoiser(config: DU_VAMP_JCD_Config) -> BaseDenoiser`

Factory function creating the appropriate denoiser from `config.denoiser_type`.

---

### `class CFOEstimator(nn.Module)`

Carrier Frequency Offset estimator with analytic, analytic+learned, and MLP modes.

**Constructor:** `CFOEstimator(config: DU_VAMP_JCD_Config)`

**Methods:**
- `analytic_estimate(y: Tensor, pilots: Tensor, pilot_mask: Tensor) -> Tensor` — coarse CFO from pilot autocorrelation
  - `y`: (B, N_sc, N_sym, 2)
  - `pilots`: (B, N_p, 2)
  - `pilot_mask`: (N_sc, N_sym)
  - Returns: (B,) — CFO in cycles/symbol
- `mlp_estimate(y: Tensor, pilots: Tensor, pilot_mask: Tensor) -> Tensor` — MLP-based CFO estimate
- `forward(y: Tensor, pilots: Tensor, pilot_mask: Tensor) -> Tensor` — estimate CFO, dispatches by `cfo_mode`

**Learnable parameters (when `cfo_mode = "analytic_plus_learned"`):**
- `cfo_scale`: scalar multiplier
- `cfo_bias`: scalar additive bias

---

### `class CFOCompensator(nn.Module)`

Apply CFO correction via phase rotation across OFDM symbols.

**Constructor:** `CFOCompensator(n_symbols: int)`

**Methods:**
- `forward(y: Tensor, delta_f: Tensor) -> Tensor` — apply phase rotation
  - `y`: (B, N_sc, N_sym, 2)
  - `delta_f`: (B,) — CFO in cycles/symbol
  - Returns: (B, N_sc, N_sym, 2) — phase-corrected IQ

---

### `class DelayDopplerTransform(nn.Module)`

2D unitary transform between time-frequency and delay-Doppler domains. Forward: IFFT over subcarriers (delay) → FFT over symbols (Doppler). Inverse: IFFT over symbols → FFT over subcarriers.

**Methods:**
- `forward(h: Tensor) -> Tensor` — time-frequency → delay-Doppler
  - `h`: (B, N_sc, N_sym, 2) → returns: (B, N_sc, N_sym, 2)
- `inverse(h_dd: Tensor) -> Tensor` — delay-Doppler → time-frequency
  - `h_dd`: (B, N_sc, N_sym, 2) → returns: (B, N_sc, N_sym, 2)

---

### `class VAMPLinearStep(nn.Module)`

VAMP linear (LMMSE) channel estimation step. Per-RE independent operation for narrowband NB-IoT.

**Methods:**
- `forward(y, x_known, h_prev, r1, u1, gamma_z, gamma_h) -> Tensor`
  - All tensors: (B, N_sc, N_sym, 2) except `gamma_z`, `gamma_h` (scalars)
  - Returns: (B, N_sc, N_sym, 2) — LMMSE channel estimate

---

### `class SoftMMSEDetector(nn.Module)`

Soft MMSE data detector with max-log LLR computation and soft symbol reconstruction.

**Constructor:** `SoftMMSEDetector(config: DU_VAMP_JCD_Config)`

**Methods:**
- `forward(y: Tensor, h_hat: Tensor, gamma_z: Tensor) -> Tuple[Tensor, Tensor]`
  - `y`: (B, N_sc, N_sym, 2) — received signal (CFO-corrected)
  - `h_hat`: (B, N_sc, N_sym, 2) — channel estimate
  - `gamma_z`: scalar — noise precision
  - Returns: `(x_soft, llrs)` where `x_soft`: (B, N_sc, N_sym, 2), `llrs`: (B, N_sc, N_sym, n_bits)

---

### `class VAMPJCDReceiver(nn.Module)`

**Main model class.** Deep-unfolded VAMP Joint Channel Estimation and Data Detection receiver.

**Constructor:** `VAMPJCDReceiver(config: DU_VAMP_JCD_Config)`

**Methods:**
- `forward(y, pilots, pilot_mask, use_checkpoint=False, return_full_state=False) -> Tuple[Tensor, Tensor, Tensor, Dict]`
  - `y`: (B, N_sc, N_sym, 2) — received IQ samples. dtype: float32 or bfloat16
  - `pilots`: (B, N_p, 2) — known pilot symbols (normalized to unit power)
  - `pilot_mask`: (N_sc, N_sym) — boolean pilot locations (True at pilot REs)
  - `use_checkpoint`: if True, use gradient checkpointing (negligible benefit for 24-param model)
  - `return_full_state`: if True, return per-iteration states
  - Returns: `(h_hat, x_soft, llrs, state)`
    - `h_hat`: (B, N_sc, N_sym, 2) — final channel estimate
    - `x_soft`: (B, N_sc, N_sym, 2) — final soft symbol estimates
    - `llrs`: (B, N_sc, N_sym, n_bits) — final per-bit LLRs
    - `state`: dict with keys `h_per_iter`, `cfo_per_iter`, `x_soft_per_iter`, `gamma_z`, `gamma_h`
  - Shape invariants:
    - Batch B ≥ 1; N_sc ≤ config.n_subcarriers; N_sym ≤ config.n_symbols
    - dtype in {float32, bfloat16}; float16 not recommended for LLR computation
    - Residual NOT yet added (no skip connection in this architecture)

**Learnable parameters (24 total, default):**
- `log_noise_precision`: gamma_z (inverse noise variance), log-parametrized for positivity
- `log_channel_precision`: gamma_h (channel prior precision)
- `logit_damping`: sigmoid-parametrized damping factor alpha ∈ (0, 1)
- `onsager_coeff`: Onsager correction coefficient beta
- `shrinkage_thresholds`: (1 or T, 3, 3, 2) — delay-Doppler thresholds (in `DelayDopplerShrinkageDenoiser`)
- `cfo_scale`, `cfo_bias`: CFO correction parameters (in `CFOEstimator`)

---

### `def count_params(model: nn.Module) -> None`

Print total and trainable parameter counts for a model.

---

## coder/channel.py — LEO NTN Channel Simulator

### `def rician_k_factor_linear(elevation_deg: float) -> float`

Interpolate Rician K-factor from 3GPP TR 38.811 table. Returns linear-scale power ratio LOS/NLOS.

### `def doppler_spread_hz(elevation_deg, carrier_freq_hz=2.0e9, velocity_ms=7550.0) -> float`

Compute maximum Doppler spread for a LEO satellite pass.

### `def doppler_shift_hz(elevation_deg, carrier_freq_hz=2.0e9) -> float`

Compute instantaneous Doppler shift magnitude.

### `def path_loss_db(elevation_deg, distance_km=None, carrier_freq_hz=2.0e9) -> float`

Compute free-space + atmospheric path loss for LEO satellite link.

### `class LEONTNChannelSimulator`

LEO NTN channel simulator following 3GPP TR 38.811. Generates batches of channel realizations with time-varying Rician fading, Doppler shift, and AWGN.

**Constructor:** `LEONTNChannelSimulator(config)` — config must have fields: `n_subcarriers`, `n_symbols`, `subcarrier_spacing_hz`, `n_pilots`, `pilot_symbol_indices`, `n_bits_per_symbol`

**Methods:**
- `register_pilot_mask()` — create NPUSCH Format 1 pilot mask
- `generate_batch(batch_size=32, elevation_deg=45.0, snr_db=0.0, doppler_hz=None, delay_spread_s=None, seed=None) -> Dict[str, Tensor]`
  - Returns dict with keys: `y` (B, N_sc, N_sym, 2), `h_true` (B, N_sc, N_sym, 2), `x` (B, N_sc, N_sym, 2), `bits` (B, N_sc, N_sym, n_bits), `pilots` (B, N_p, 2), `pilot_mask` (N_sc, N_sym), `snr_db`, `doppler_hz`, `elevation_deg`

---

## coder/baselines.py — Baseline Receivers

### `class LMMSEBaseline(nn.Module)`

Conventional pilot-aided LMMSE receiver. Processing: LS estimation → linear interpolation → MMSE detection.

**Constructor:** `LMMSEBaseline(config)`
**Methods:** `forward(y, pilots, pilot_mask, noise_var=0.1) -> (h_hat, x_soft, llrs, state)`
— Shapes match VAMPJCDReceiver interface.

### `class LSBaseline(nn.Module)`

Least-Squares receiver with ZF detection. Simpler than LMMSE — no noise statistics used.

**Constructor:** `LSBaseline(config)`
**Methods:** `forward(y, pilots, pilot_mask, noise_var=0.1) -> (h_hat, x_soft, llrs, state)`

### `class GenieBound(nn.Module)`

Ideal genie-aided bound with perfect channel knowledge. Uses exact channel for MMSE detection with true noise variance.

**Constructor:** `GenieBound(config)`
**Methods:** `forward(y, h_true, noise_var=0.1) -> (h_hat, x_soft, llrs, state)`
— Note: takes `h_true` instead of `pilots + pilot_mask` because channel is known.

---

## coder/train.py — Training Utilities

### `def channel_nmse_loss(h_pred: Tensor, h_true: Tensor) -> Tensor`

NMSE = ||h_pred - h_true||² / ||h_true||². Averaged over batch.

### `def detection_bce_loss(llrs: Tensor, bits: Tensor) -> Tensor`

Binary cross-entropy from LLRs using numerically stable log-sum-exp formulation.

### `def sparsity_regularization(model, lambda_3=0.01) -> Tensor`

L1 regularization on shrinkage thresholds, encouraging aggressive denoising.

### `def total_loss(h_pred, h_true, llrs, bits, model=None, lambda_1=1.0, lambda_2=1.0, lambda_3=0.01) -> Dict[str, Tensor]`

Composite loss: `L = lambda_1 * NMSE + lambda_2 * BCE + lambda_3 * L1(shrinkage)`

### `def compute_bler(llrs: Tensor, bits: Tensor) -> Tensor`

Block Error Rate: fraction of blocks with any bit error. Block = (N_sc, N_sym, n_bits) per batch element.

### `def compute_ber(llrs: Tensor, bits: Tensor) -> Tensor`

Bit Error Rate: fraction of bits in error.

### `class TrainingHarness`

Full training harness with data generation, loss computation, optimizer, scheduler, and evaluation.

**Constructor:** `TrainingHarness(model, config, device="cpu")`

**Methods:**
- `train_step(batch) -> Dict[str, float]` — single training step (forward + backward + optimizer)
- `evaluate(snr_db_list=None, elevation_deg=45.0, doppler_hz=1200.0, batch_size=256, n_batches=5) -> Dict` — evaluate across SNR values; returns {snr: {bler, ber, nmse}}
- `train(num_epochs=50, batch_size=256, eval_every=10, snr_range_db=(-10.0, 10.0), doppler_hz=1200.0) -> Dict` — full training loop with ReduceLROnPlateau scheduling
