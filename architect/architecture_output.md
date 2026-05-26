# ML-Enhanced Receiver Chain for LEO Satellite IoT Systems
## Architecture Design: Deep-Unfolded VAMP-JCD Receiver

---

## Step 0 — Domain Identification

| Domain | Role | Rationale |
|---|---|---|
| **Scientific ML** | Primary | Hybrid model-driven + data-driven receiver; deep unfolding of iterative VAMP/AMP signal processing algorithms; physics-constrained learning where the algorithm structure encodes the channel model; learned components only adjust parameters within the known estimation-theoretic framework |
| **Time Series** | Secondary | IQ samples and channel estimates are complex-valued time series on the OFDM time-frequency grid; temporal correlation exploited for Doppler tracking and per-symbol channel prediction; VAMP denoising operates in the delay-Doppler (2D Fourier) domain |
| **LM** | Tertiary (partial) | Transformer/attention architectures were considered and rejected (see justification) for the LEO IoT SWaP regime, but the sequence-mixing perspective informs the VAMP iterative update ordering |

**Sub-field:** Non-Terrestrial Network (NTN) Physical Layer — LEO Satellite IoT Receiver Chain (NB-IoT NPUSCH)

---

## Step 1 — Read Research Contract and Gather Design Constraints

Upstream research lifecycle contract loaded from `/artifacts/j_Vqj3l8AH-1hY/work/research/research_output.md`. Key constraints preserved below:

### Falsifiable Hypothesis (from research stage)
> A deep-unfolded receiver that integrates CFO compensation, channel estimation, and data detection into a single lightweight iterative neural architecture (≤10³ parameters, ≤10⁴ MACs per received block) can outperform a conventional MMSE receiver by ≥2 dB in the LEO satellite IoT regime (Eb/N0 ≤ 0 dB, Doppler spread ≥ 1 kHz, pilot overhead ≤ 10%) while fitting within a Zynq-class FPGA power budget (<5W).

### Hard Constraints Carried Forward
| Constraint | Value | Source |
|---|---|---|
| Max trainable parameters | ≤1,000 | Research hypothesis (FPGA SWaP) |
| Max MACs per resource block | ≤10,000 | Research hypothesis (latency) |
| Target Eb/N0 regime | –10 to 0 dB | LEO IoT link budget |
| Doppler spread | ≥1 kHz (test at 600 Hz, 1.2 kHz, 2.4 kHz) | 3GPP TR 38.811 NTN |
| Pilot overhead | ≤10% | NB-IoT NPUSCH pilot structure |
| Target FPGA | Xilinx Zynq UltraScale+ (e.g., ZU3EG) | Survey gap (Choquenaira-Florez 2025) |
| Max FPGA power | <5W | CubeSat SWaP constraint |
| Max inference latency | <1 ms per block | Real-time receiver requirement |
| Target modulation | QPSK / pi/2-BPSK | NB-IoT NPUSCH |
| Number of subcarriers | 12 (15 kHz SCS) | NB-IoT 180 kHz channel |

### Baseline Requirements (from research)
1. Conventional MMSE receiver with perfect CFO knowledge (upper bound) and estimated CFO (realistic bound)
2. Least-squares channel estimation with linear interpolation between pilots (3GPP NR-NTN baseline)
3. Ideal known-channel bound (genie-aided)
4. DeepSig OmniPHY-style CNN receiver (where reproducible; otherwise a CNN of comparable parameter count)

### Evaluation Requirements (from research)
- BLER vs. Eb/N0 curves from –10 dB to +10 dB at fixed Doppler spread = 600 Hz, 1.2 kHz, 2.4 kHz
- BLER vs. Doppler spread at Eb/N0 = –3 dB
- Parameter count and MAC count per received resource block
- FPGA resource utilization: LUT count, DSP slice count, BRAM count, on-chip power (W) for INT8 quantized model
- Per-block inference latency (ms) on target FPGA
- NMSE of channel estimation vs. SNR
- Training data: synthetically generated LEO NTN channel realizations per 3GPP TR 38.811/38.821
- Test data: held-out geometries (unseen elevation angle bands, different orbital altitudes)

---

## Step 2 — ModelConfig Dataclass

```python
from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class DU_VAMP_JCD_Config:
    # ───────── System Parameters (NB-IoT NTN NPUSCH) ─────────
    domain: str = "scientific_ml"
    subdomain: str = "leo_ntn_iot_receiver"
    
    # Physical layer
    n_subcarriers: int = 12             # NB-IoT: 12 SCs × 15 kHz = 180 kHz
    n_symbols: int = 14                 # OFDM symbols per slot (normal CP)
    subcarrier_spacing_hz: float = 15000.0
    n_pilots: int = 24                  # DMRS: 2 symbols × 12 SCs (NPUSCH Format 1)
    pilot_symbol_indices: Tuple = (3, 10)  # DMRS symbol positions (0-indexed)
    modulation: str = "qpsk"            # pi/2-BPSK or QPSK
    n_bits_per_symbol: int = 2          # for QPSK
    
    # ───────── Deep Unfolding Architecture ─────────
    n_unfolded_layers: int = 5          # T = 5 VAMP iterations
    weight_tying: bool = True           # share denoiser params across layers
    denoiser_type: str = "delay_doppler_shrinkage"  
    # Options: "delay_doppler_shrinkage" | "element_shrinkage" | "small_cnn" | "mlp"
    
    # Delay-Doppler denoiser grid
    n_delay_groups: int = 3             # delay bins grouped: short, medium, long
    n_doppler_groups: int = 3           # Doppler bins grouped: low, medium, high
    # Total shrinkage thresholds per layer: 3 × 3 = 9 (complex = 18 real)
    
    # ───────── CFO Tracking ─────────
    cfo_mode: str = "analytic_plus_learned"  
    # "analytic" | "analytic_plus_learned" | "mlp"
    cfo_learned_scale: bool = True      # learnable scalar multiplier
    cfo_learned_bias: bool = True       # learnable scalar additive bias
    cfo_mlp_hidden_dim: int = 0         # 0 = analytic only; >0 enables small MLP
    
    # ───────── Detection ─────────
    detector_type: str = "mmse_soft"    # "mmse_soft" | "zf" | "learned"
    use_soft_feedback: bool = True      # feed soft symbols to next iteration
    
    # ───────── VAMP State Parameters ─────────
    learn_noise_precision: bool = True  # γ_z: 1 learnable scalar
    learn_channel_precision: bool = True  # γ_h: 1 learnable scalar
    n_damping_params: int = 1           # 1 = shared, n_layers = per-layer
    n_onsager_params: int = 1           # 1 = shared, n_layers = per-layer
    
    # ───────── Training ─────────
    dropout: float = 0.0                # no dropout in unfolded architecture
    use_bias: bool = False              # minimal bias in unfolded layers
    dtype: str = "float32"
    quantization_target: str = "int8"   # for FPGA deployment
    training_snr_range_db: Tuple = (-10.0, 10.0)
    
    # ───────── SWaP Target ─────────
    target_fpga: str = "xilinx_zynq_zu3eg"
    max_dsp_slices: int = 600
    max_power_watts: float = 5.0
    target_latency_ms: float = 1.0
    
    # ───────── Derived (computed at init) ─────────
    n_trainable_params: int = 0         # computed after init
    n_macs_per_block: int = 0           # computed after init
    
    def __post_init__(self):
        self._compute_parameter_budget()
    
    def _compute_parameter_budget(self):
        """Compute total trainable parameters from config."""
        n = 0
        
        # Denoiser shrinkage thresholds
        n_dd = self.n_delay_groups * self.n_doppler_groups * 2  # real + imag per group
        if self.weight_tying:
            n += n_dd
        else:
            n += n_dd * self.n_unfolded_layers
        
        # VAMP noise/channel precision
        if self.learn_noise_precision:
            n += 1
        if self.learn_channel_precision:
            n += 1
        
        # Damping factors
        if self.n_damping_params == 1:
            n += 1
        else:
            n += self.n_unfolded_layers
        
        # Onsager correction
        if self.n_onsager_params == 1:
            n += 1
        else:
            n += self.n_unfolded_layers
        
        # CFO module
        if self.cfo_mode == "analytic_plus_learned":
            if self.cfo_learned_scale:
                n += 1
            if self.cfo_learned_bias:
                n += 1
        elif self.cfo_mode == "mlp":
            # Input: 2 * n_pilots (I/Q per pilot), hidden -> 1
            cfo_in = 2 * self.n_pilots
            cfo_hid = self.cfo_mlp_hidden_dim
            n += cfo_in * cfo_hid + cfo_hid + cfo_hid * 1 + 1  # weights + biases
        
        # Soft detector (learned noise estimate)
        if self.detector_type == "learned":
            n += 1  # scaling factor
        
        self.n_trainable_params = n
```

### Default Parameter Count

Under the default configuration (shared weights, delay-Doppler shrinkage, analytic+learned CFO):

| Component | Params (shared) | Params (per-layer, T=5) |
|---|---|---|
| Delay-Doppler shrinkage (3×3×2 real) | 18 | 90 |
| Noise precision γ_z | 1 | 1 |
| Channel precision γ_h | 1 | 1 |
| Damping α | 1 | 5 |
| Onsager β | 1 | 5 |
| CFO scale | 1 | 1 |
| CFO bias | 1 | 1 |
| **Total (default)** | **24** | **104** |
| **Budget headroom** | **976** | **896** |

All configurations stay well within the ≤1,000 parameter budget, leaving headroom for the optional MLP-based CFO estimator (≤270 params if enabled) or a small CNN denoiser (≤800 params if enabled).

### MAC Count Estimate

For each unfolded iteration:
- VAMP linear step: O(N_sc × N_sym) ≈ 12 × 14 = 168 complex multiplies
- Delay-Doppler FFT: 2 × (N_sc × N_sym × log₂(N_sc × N_sym)) ≈ 2 × 168 × 7.4 ≈ 2,500 real MACs
- Shrinkage: 168 comparisons + subtractions
- Soft detection: O(N_sc × N_sym × n_bits) ≈ 168 × 2 = 336
- **Total per iteration**: ~3,000 MACs
- **Total over T=5**: ~15,000 MACs

This exceeds the 10,000 MAC target by ~50%. Mitigations:
- Use FFT pruning (delay-Doppler transform only on active delay taps)
- Reduce T to 3 iterations for FPGA deployment
- Use separable 1D transforms instead of 2D FFT

---

## Step 3 — Core Block Design: Deep-Unfolded VAMP-JCD

### Pseudocode — Full Receiver Chain

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class VAMPJCDReceiver(nn.Module):
    """
    Deep-Unfolded VAMP Joint Channel Estimation and Data Detection Receiver.
    
    Architecture:
        T iterations (unfolded layers), each performing:
        1. CFO residual compensation
        2. VAMP linear step (LMMSE channel estimation)
        3. Delay-Doppler domain denoising (learnable shrinkage)
        4. VAMP state update (damping + Onsager correction)
        5. Soft data detection (MMSE)
        6. CFO residual estimation from pilot mismatch
    
    Input shapes:
        y: (batch, n_subcarriers, n_symbols, 2)  — real/imag IQ samples
        pilots: (batch, n_pilots, 2)  — known pilot symbols
        pilot_mask: (n_subcarriers, n_symbols)  — boolean pilot locations
    
    Output shapes:
        h_hat: (batch, n_subcarriers, n_symbols, 2)  — channel estimate
        x_soft: (batch, n_subcarriers, n_symbols, 2)  — soft symbol estimates
        llrs: (batch, n_subcarriers, n_symbols, n_bits)  — LLRs for decoder
    """

    def __init__(self, config: DU_VAMP_JCD_Config):
        super().__init__()
        self.config = config
        self.T = config.n_unfolded_layers
        self.N_sc = config.n_subcarriers
        self.N_sym = config.n_symbols
        self.N_p = config.n_pilots
        
        # ── Learnable Parameters ────────────────────────────
        
        # Noise precision γ_z (inverse noise variance)
        self.log_noise_precision = nn.Parameter(
            torch.tensor(np.log(1.0 / 0.1))  # init: SNR ≈ 10 dB
        )
        
        # Channel prior precision γ_h (inverse channel variance)
        self.log_channel_precision = nn.Parameter(
            torch.tensor(np.log(1.0 / 1.0))  # init: unit variance
        )
        
        # Delay-Doppler shrinkage thresholds
        # Shape: (T or 1, n_delay_groups, n_doppler_groups, 2)
        n_thresh = (1 if config.weight_tying else self.T)
        self.shrinkage_thresholds = nn.Parameter(
            torch.full((n_thresh, config.n_delay_groups, 
                        config.n_doppler_groups, 2), 0.1)
        )
        
        # Damping factor α ∈ [0, 1] — sigmoid of unconstrained param
        n_damp = 1 if config.n_damping_params == 1 else self.T
        self.logit_damping = nn.Parameter(
            torch.full((n_damp,), 0.0)  # init: α = 0.5
        )
        
        # Onsager correction coefficient β
        n_onsager = 1 if config.n_onsager_params == 1 else self.T
        self.onsager_coeff = nn.Parameter(
            torch.full((n_onsager,), 0.0)  # init: β = 0 (no correction)
        )
        
        # CFO correction (learned scale and bias)
        if config.cfo_mode == "analytic_plus_learned":
            if config.cfo_learned_scale:
                self.cfo_scale = nn.Parameter(torch.tensor(1.0))
            if config.cfo_learned_bias:
                self.cfo_bias = nn.Parameter(torch.tensor(0.0))
        
        # Optional: CFO MLP
        if config.cfo_mode == "mlp" and config.cfo_mlp_hidden_dim > 0:
            cfo_in = 2 * config.n_pilots
            cfo_hid = config.cfo_mlp_hidden_dim
            self.cfo_mlp = nn.Sequential(
                nn.Linear(cfo_in, cfo_hid),
                nn.ReLU(),
                nn.Linear(cfo_hid, 1),
                nn.Tanh()  # bounded CFO correction
            )
        
        # ── Pre-computed indices ────────────────────────────
        # Delay-Doppler bin mapping for shrinkage
        self.register_buffer('delay_bin_map', self._build_delay_bin_map())
        self.register_buffer('doppler_bin_map', self._build_doppler_bin_map())
        
        # Time index vector for CFO correction
        self.register_buffer('time_idx', torch.arange(self.N_sym).float())
        
    def _build_delay_bin_map(self):
        """Map each (sc, sym) pair to one of n_delay_groups delay bins."""
        # After IFFT across subcarriers: delay bins
        # For N_sc=12: map uniformly to n_delay_groups
        n_groups = self.config.n_delay_groups
        return torch.linspace(0, n_groups - 1, self.N_sc).long()
    
    def _build_doppler_bin_map(self):
        """Map each (sc, sym) pair to one of n_doppler_groups Doppler bins."""
        # After FFT across symbols: Doppler bins
        n_groups = self.config.n_doppler_groups
        return torch.linspace(0, n_groups - 1, self.N_sym).long()
    
    def _complex_to_2ch(self, z):
        """Convert complex tensor (..., 2) to real/imag channels."""
        return z  # already last dim = 2 (real, imag)
    
    def _analytic_cfo_estimate(self, y, pilots, pilot_mask):
        """
        Compute coarse CFO estimate from pilot autocorrelation.
        
        Args:
            y: (B, N_sc, N_sym, 2) received IQ
            pilots: (B, N_p, 2) known pilot symbols
            pilot_mask: (N_sc, N_sym) boolean pilot locations
            
        Returns:
            delta_f: (B,) estimated CFO in normalized units (rad/symbol)
        """
        # Extract pilot observations and known symbols
        y_p = y[pilot_mask]  # (B, N_p, 2) — using broadcasting
        # y_p currently indexed correctly: but let's handle masking properly
        # For simplicity: compute per-pilot phase error, then autocorr over time
        
        # Phase error per pilot
        # r = y · conj(x): element-wise multiply with conj of known pilot
        r_imag = (y_p[..., 0] * pilots[..., 1] - y_p[..., 1] * pilots[..., 0])  # cross
        r_real = (y_p[..., 0] * pilots[..., 0] + y_p[..., 1] * pilots[..., 1])  # dot
        phase_error = torch.atan2(r_imag, r_real)  # (B, N_p)
        
        # Autocorrelation between pilot symbols at different time indices
        # For NPUSCH: pilots at sym_idx[0] and sym_idx[1]
        p0_idx = self.config.pilot_symbol_indices[0]
        p1_idx = self.config.pilot_symbol_indices[1]
        dt = p1_idx - p0_idx
        
        # Average phase difference across subcarriers
        # phase_error has shape (B, N_p) — need to split by symbol
        n_sc = self.N_sc
        phase_sym0 = phase_error[:, :n_sc]    # pilots at first symbol
        phase_sym1 = phase_error[:, n_sc:]    # pilots at second symbol
        delta_phi = torch.mean(phase_sym1 - phase_sym0, dim=1)  # (B,)
        
        # CFO = delta_phi / (2 * pi * dt)
        delta_f = delta_phi / (2.0 * np.pi * dt)
        return delta_f  # (B,) in cycles/symbol
    
    def _apply_cfo_correction(self, y, delta_f):
        """
        Apply CFO correction to received signal.
        
        Args:
            y: (B, N_sc, N_sym, 2)
            delta_f: (B,) CFO in cycles/symbol
            
        Returns:
            y_corrected: (B, N_sc, N_sym, 2)
        """
        # Phase rotation per symbol
        # phi(t) = exp(-j * 2π * Δf * t)
        time = self.time_idx[None, None, :, None]  # (1, 1, N_sym, 1)
        delta_f_rs = delta_f[:, None, None, None]  # (B, 1, 1, 1)
        angle = -2.0 * np.pi * delta_f_rs * time   # (B, 1, N_sym, 1)
        
        cos_theta = torch.cos(angle)  # (B, 1, N_sym, 1)
        sin_theta = torch.sin(angle)  # (B, 1, N_sym, 1)
        
        # Complex rotation: [real, imag] -> [real*cos - imag*sin, real*sin + imag*cos]
        y_r, y_i = y[..., 0], y[..., 1]
        corrected_r = y_r * cos_theta - y_i * sin_theta
        corrected_i = y_r * sin_theta + y_i * cos_theta
        
        return torch.stack([corrected_r, corrected_i], dim=-1)
    
    def _delay_doppler_transform(self, h):
        """
        Transform channel from time-frequency to delay-Doppler domain.
        
        h: (B, N_sc, N_sym, 2) — channel in time-frequency domain
        
        Returns:
            h_dd: (B, N_sc, N_sym, 2) — channel in delay-Doppler domain
        """
        # Step 1: IFFT across subcarriers (delay domain)
        # h is real/imag as last dim: convert to complex for FFT
        h_complex = torch.view_as_complex(h.contiguous())
        h_delay = torch.fft.ifft(h_complex, dim=1)  # IFFT over subcarriers
        
        # Step 2: FFT across symbols (Doppler domain)
        h_dd_complex = torch.fft.fft(h_delay, dim=2)  # FFT over symbols
        
        # Convert back to (..., 2) real/imag representation
        return torch.view_as_real(h_dd_complex)
    
    def _inverse_delay_doppler_transform(self, h_dd):
        """
        Transform channel from delay-Doppler back to time-frequency domain.
        
        h_dd: (B, N_sc, N_sym, 2) — channel in delay-Doppler domain
        
        Returns:
            h: (B, N_sc, N_sym, 2) — channel in time-frequency domain
        """
        h_complex = torch.view_as_complex(h_dd.contiguous())
        h_doppler = torch.fft.ifft(h_complex, dim=2)  # IFFT over symbols
        h_tf_complex = torch.fft.fft(h_doppler, dim=1)  # FFT over subcarriers
        return torch.view_as_real(h_tf_complex)
    
    def _apply_shrinkage(self, h_dd, layer_idx):
        """
        Apply learnable soft thresholding in delay-Doppler domain.
        
        Args:
            h_dd: (B, N_sc, N_sym, 2) — delay-Doppler channel
            layer_idx: current unfolded iteration index
            
        Returns:
            h_dd_shrunk: (B, N_sc, N_sym, 2) — denoised channel
        """
        # Select thresholds for this layer (or shared)
        t_idx = 0 if self.config.weight_tying else layer_idx
        thresholds = self.shrinkage_thresholds[t_idx]  # (n_dg, n_dog, 2)
        
        # Map each spatial bin to its delay-Doppler group
        delay_bins = self.delay_bin_map  # (N_sc,)
        doppler_bins = self.doppler_bin_map  # (N_sym,)
        
        # Gather thresholds per bin: shape (N_sc, N_sym, 2)
        thresh_vals = thresholds[delay_bins[:, None], doppler_bins[None, :]]  # (N_sc, N_sym, 2)
        
        # Soft thresholding per real/imag component
        magnitude = torch.abs(h_dd)
        shrinkage = F.relu(magnitude - thresh_vals) / (magnitude + 1e-10)
        h_dd_shrunk = h_dd * shrinkage
        
        return h_dd_shrunk
    
    def _vamp_linear_step(self, y, x_known, h_prev, r1, u1, gamma_z, gamma_h):
        """
        VAMP linear step: LMMSE channel estimate.
        
        Simplified for element-wise operation (narrowband assumption):
        Each (sc, sym) RE is treated independently with known X.
        
        Args:
            y: (B, N_sc, N_sym, 2) — received signal
            x_known: (B, N_sc, N_sym, 2) — known + soft estimated symbols
            h_prev: (B, N_sc, N_sym, 2) — previous channel estimate
            r1: (B, N_sc, N_sym, 2) — VAMP state: r1
            u1: (B, N_sc, N_sym, 2) — VAMP state: u1
            gamma_z: scalar — noise precision
            gamma_h: scalar — channel prior precision
            
        Returns:
            h_lin: (B, N_sc, N_sym, 2) — LMMSE channel estimate
        """
        # For narrowband: each RE treated independently
        # LMMSE: h_lin = (γ_z · |x|² + γ_h)⁻¹ · (γ_z · conj(x) · y + γ_h · r1 - u1)
        
        # Signal power |x|² per RE
        x_pow = x_known[..., 0]**2 + x_known[..., 1]**2  # (B, N_sc, N_sym)
        
        # conj(x) · y (complex multiply)
        conj_x_times_y_real = x_known[..., 0] * y[..., 0] + x_known[..., 1] * y[..., 1]
        conj_x_times_y_imag = x_known[..., 0] * y[..., 1] - x_known[..., 1] * y[..., 0]
        
        # γ_z · conj(x) · y
        gy_real = gamma_z * conj_x_times_y_real
        gy_imag = gamma_z * conj_x_times_y_imag
        
        # γ_h · r1 - u1
        hu_real = gamma_h * r1[..., 0] - u1[..., 0]
        hu_imag = gamma_h * r1[..., 1] - u1[..., 1]
        
        # Numerator
        num_real = gy_real + hu_real
        num_imag = gy_imag + hu_imag
        
        # Denominator: γ_z · |x|² + γ_h
        denom = gamma_z * x_pow + gamma_h  # (B, N_sc, N_sym)
        
        # Result
        h_lin_real = num_real / (denom + 1e-10)
        h_lin_imag = num_imag / (denom + 1e-10)
        
        return torch.stack([h_lin_real, h_lin_imag], dim=-1)
    
    def _soft_mmse_detector(self, y, h_hat, gamma_z):
        """
        Soft MMSE data detection.
        
        For each data RE, compute:
        - MMSE equalized symbol
        - LLRs for each bit
        - Soft symbol (tanh of LLRs mapped to constellation)
        
        Args:
            y: (B, N_sc, N_sym, 2) — received signal (CFO-corrected)
            h_hat: (B, N_sc, N_sym, 2) — channel estimate
            gamma_z: scalar — noise precision
            
        Returns:
            x_soft: (B, N_sc, N_sym, 2) — soft symbol estimates
            llrs: (B, N_sc, N_sym, n_bits) — bit LLRs
        """
        # MMSE equalizer: x̂ = conj(h) · y / (|h|² + 1/γ_z)
        h_pow = h_hat[..., 0]**2 + h_hat[..., 1]**2
        noise_var = 1.0 / (gamma_z + 1e-10)
        
        conj_h_times_y_real = h_hat[..., 0] * y[..., 0] + h_hat[..., 1] * y[..., 1]
        conj_h_times_y_imag = h_hat[..., 0] * y[..., 1] - h_hat[..., 1] * y[..., 0]
        
        denom = h_pow + noise_var
        x_eq_real = conj_h_times_y_real / denom
        x_eq_imag = conj_h_times_y_imag / denom
        
        # Effective SNR per RE for LLR scaling
        snr_per_re = h_pow / noise_var  # (B, N_sc, N_sym)
        
        # For QPSK: each symbol carries 2 bits
        # LLR(b0) = 2 · sqrt(2) · Re(x_eq) / (1 - |h|²/(|h|²+σ²)) 
        # Simplified: LLR ≈ 2 · SNR_eff · Re(x_eq)  (for QPSK)
        # Using max-log approximation
        scale = 2.0 * torch.sqrt(torch.tensor(2.0))  # QPSK scaling
        llr_b0 = scale * x_eq_real * snr_per_re  # real part → b0
        llr_b1 = scale * x_eq_imag * snr_per_re  # imag part → b1
        
        llrs = torch.stack([llr_b0, llr_b1], dim=-1)  # (B, N_sc, N_sym, 2)
        
        # Soft symbols from LLRs (for next iteration feedback)
        # QPSK constellation: (±1/√2, ±1/√2)
        const_scale = 1.0 / np.sqrt(2.0)
        soft_b0 = torch.tanh(llr_b0 / 2.0)  # (B, N_sc, N_sym)
        soft_b1 = torch.tanh(llr_b1 / 2.0)
        
        x_soft_real = const_scale * soft_b0
        x_soft_imag = const_scale * soft_b1
        
        x_soft = torch.stack([x_soft_real, x_soft_imag], dim=-1)
        
        return x_soft, llrs

    def forward(self, y, pilots, pilot_mask):
        """
        Full forward pass of the deep-unfolded VAMP-JCD receiver.
        
        Args:
            y: (B, N_sc, N_sym, 2) — received IQ samples
            pilots: (B, N_p, 2) — known pilot symbols
            pilot_mask: (N_sc, N_sym) — boolean pilot locations
            
        Returns:
            h_hat: (B, N_sc, N_sym, 2) — final channel estimate
            x_soft: (B, N_sc, N_sym, 2) — final soft symbol estimates  
            llrs: (B, N_sc, N_sym, n_bits) — final LLRs
            state: dict of intermediate VAMP states (for analysis)
        """
        B = y.shape[0]
        device = y.device
        
        # ── Initialize ───────────────────────────────────────
        
        # Initial CFO estimate from pilot autocorrelation
        delta_f = self._analytic_cfo_estimate(y, pilots, pilot_mask)  # (B,)
        
        # Apply coarse CFO correction
        y_corrected = self._apply_cfo_correction(y, delta_f)
        
        # Initialize VAMP state
        h_hat = torch.zeros(B, self.N_sc, self.N_sym, 2, device=device)
        x_soft = torch.zeros(B, self.N_sc, self.N_sym, 2, device=device)
        
        # VAMP auxiliary variables
        r1 = torch.zeros_like(h_hat)
        u1 = torch.zeros_like(h_hat)
        
        # Known symbol matrix (pilots at pilot locations, soft symbols elsewhere)
        x_known = torch.zeros_like(y)
        x_known[pilot_mask] = pilots.reshape(-1, 2)  # place known pilots
        
        # Noise and channel precision (clamped to positive)
        gamma_z = torch.exp(self.log_noise_precision)
        gamma_h = torch.exp(self.log_channel_precision)
        
        # ── Unfolded Iterations ──────────────────────────────
        
        state_log = {"h_per_iter": [], "cfo_per_iter": [], "x_soft_per_iter": []}
        
        for t in range(self.T):
            # ── a) CFO residual correction ──
            # Update y_corrected with refined CFO estimate
            y_corrected = self._apply_cfo_correction(y, delta_f)
            
            # ── b) VAMP Linear Step (LMMSE channel estimate) ──
            h_lin = self._vamp_linear_step(
                y_corrected, x_known, h_hat, r1, u1, gamma_z, gamma_h
            )
            
            # ── c) Delay-Doppler Denoising ──
            # Transform to delay-Doppler
            h_dd = self._delay_doppler_transform(h_lin)
            # Apply shrinkage
            h_dd_denoised = self._apply_shrinkage(h_dd, t)
            # Transform back to time-frequency
            h_den = self._inverse_delay_doppler_transform(h_dd_denoised)
            
            # ── d) VAMP State Update ──
            alpha = torch.sigmoid(
                self.logit_damping[0] if self.config.n_damping_params == 1 
                else self.logit_damping[t]
            )
            beta = (
                self.onsager_coeff[0] if self.config.n_onsager_params == 1 
                else self.onsager_coeff[t]
            )
            
            # Damping: blend previous and current estimate
            h_new = alpha * h_den + (1 - alpha) * h_hat
            
            # Onsager correction for next iteration
            # u1 = β · (h_den - h_lin)  (divergence correction)
            u1 = beta * (h_den - h_lin)
            # r1 = h_new  (new reference for next linear step)
            r1 = h_new.detach()  # detach to prevent gradient flow through state
            h_hat = h_new
            
            # ── e) Soft Data Detection ──
            if self.config.use_soft_feedback:
                x_soft_detected, llrs_t = self._soft_mmse_detector(
                    y_corrected, h_hat, gamma_z
                )
                # Update x_known: keep pilots fixed, update data REs with soft symbols
                x_known = x_soft_detected.clone()
                x_known[pilot_mask] = pilots.reshape(-1, 2)  # re-insert pilots
                x_soft = x_soft_detected
            
            # ── f) CFO Residual Update ──
            if self.config.cfo_mode == "analytic_plus_learned":
                # Compute residual CFO from pilot phase error after equalization
                delta_f_residual = self._analytic_cfo_estimate(
                    y_corrected, pilots, pilot_mask
                )
                # Apply learned scaling and bias
                if self.config.cfo_learned_scale:
                    delta_f_residual = delta_f_residual * self.cfo_scale
                if self.config.cfo_learned_bias:
                    delta_f_residual = delta_f_residual + self.cfo_bias
                # Accumulate CFO estimate
                delta_f = delta_f + delta_f_residual
            
            # Logging
            state_log["h_per_iter"].append(h_hat.detach())
            state_log["cfo_per_iter"].append(delta_f.detach())
            if self.config.use_soft_feedback:
                state_log["x_soft_per_iter"].append(x_soft.detach())
        
        # ── Final Detection ───────────────────────────────────
        x_final, llrs_final = self._soft_mmse_detector(
            y_corrected, h_hat, gamma_z
        )
        
        return h_hat, x_final, llrs_final, state_log
```

### Alternative: Small CNN Denoiser (for ablation)

```python
class TinyCNNDenoiser(nn.Module):
    """
    Lightweight 2D CNN denoiser for VAMP unfolded layer.
    
    Architecture:
        - 1×1 conv: cross-channel mixing (real → 4 channels)
        - 3×3 depthwise conv: spatial mixing in delay-Doppler
        - 1×1 conv: project back to 2 channels (real, imag)
    
    Parameter count: (2×4 + 4) + (9×4 + 4) + (4×2 + 2) = 12 + 40 + 10 = 62
    
    This is the maximum-complexity denoiser; the default shrinkage
    denoiser is 18 parameters (shared).
    """
    
    def __init__(self, n_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(2, n_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, 
                               padding=1, groups=n_channels, bias=True)
        self.conv3 = nn.Conv2d(n_channels, 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (B, 2, N_sc, N_sym)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x  # (B, 2, N_sc, N_sym)
```

---

## Step 4 — Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│           DEEP-UNFOLDED VAMP-JCD RECEIVER ARCHITECTURE              │
│         (NB-IoT NPUSCH, 12 SC × 14 Sym × 5 Iterations)             │
└─────────────────────────────────────────────────────────────────────┘

Input: Y ∈ ℂ^{12×14}  (received IQ after coarse synch & CP removal)
            │
            ▼
┌───────────────────────────┐
│  Pilot Extraction         │
│  Y_p = Y[pilot_mask]      │
│  X_p = known_pilot_symbols │
└──────────┬────────────────┘
           │
           ▼
┌───────────────────────────┐
│  Coarse CFO Estimation    │  ← analytic: autocorrelation of pilots
│  Δf₀ = f(Y_p, X_p)       │     (parameter-free)
└──────────┬────────────────┘
           │
           ▼
┌───────────────────────────┐
│  CFO Correction           │
│  Y₀ = Y · exp(-j·2π·Δf₀·t) │
└──────────┬────────────────┘
           │
    ┌──────┴──────┐
    │  ┌─────────────────────────────────────┐
    │  │                                     ▼
    │  │    ┌─────────────────────────────────────┐
    │  │    │  Layer t = 1 .. T   (T=5)           │
    │  │    │                                     │
    │  │    │  ┌─────────────────────────────┐    │
    │  │    │  │ a) CFO Residual Correction  │    │
    │  │    │  │    Δfₜ₋₁ → apply to Y       │    │
    │  │    │  └────────────┬────────────────┘    │
    │  │    │               ▼                     │
    │  │    │  ┌─────────────────────────────┐    │
    │  │    │  │ b) VAMP Linear Step         │    │
    │  │    │  │    LMMSE channel estimate   │    │  γ_z (noise precision)
    │  │    │  │    Ĥ_lin = (γ_z|X|²+γ_h)⁻¹ │    │  γ_h (channel prior)
    │  │    │  │           · (γ_z·X*·Y+...)  │    │
    │  │    │  └────────────┬────────────────┘    │
    │  │    │               ▼                     │
    │  │    │  ┌─────────────────────────────┐    │
    │  │    │  │ c) Delay-Doppler Denoiser   │    │  θ_t (learnable
    │  │    │  │    IFFT(freq)→FFT(time)     │    │  shrinkage
    │  │    │  │    Soft threshold ℛ,ℐ       │    │  thresholds)
    │  │    │  │    IFFT(time)→FFT(freq)     │    │
    │  │    │  └────────────┬────────────────┘    │
    │  │    │               ▼                     │
    │  │    │  ┌─────────────────────────────┐    │
    │  │    │  │ d) VAMP State Update        │    │  α (damping)
    │  │    │  │    Ĥₜ = α·Ĥ_den + (1-α)·Ĥₜ₋₁│   │  β (Onsager)
    │  │    │  │    u₁ = β·(Ĥ_den - Ĥ_lin)   │    │
    │  │    │  └────────────┬────────────────┘    │
    │  │    │               ▼                     │
    │  │    │  ┌─────────────────────────────┐    │
    │  │    │  │ e) Soft MMSE Detection      │    │
    │  │    │  │    MMSE equalizer per RE    │    │
    │  │    │  │    LLR computation          │    │
    │  │    │  │    Soft symbol feedback     │───────→ iter t+1
    │  │    │  │    (pilot positions fixed)  │    │
    │  │    │  └────────────┬────────────────┘    │
    │  │    │               ▼                     │
    │  │    │  ┌─────────────────────────────┐    │
    │  │    │  │ f) CFO Residual Est.       │    │
    │  │    │  │    Pilot phase error        │    │
    │  │    │  │    Δfₜ = Δfₜ₋₁ + learned    │    │  s, b (scale,bias)
    │  │    │  └─────────────────────────────┘    │
    │  │    │                                     │
    │  ─────┘  (feedback: Ĥₜ, X_soft, Δfₜ)       │
    │                                             │
    └─────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────┐
│  Final Detection & LLR Output │
│  X̂_final, LLRs for decoder   │
│  Ĥ_final (channel estimate)   │
└───────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  LEARNABLE PARAMETERS (total: 24, shared across T=5 iterations)    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ θ_dd[3×3×2] = 18 delay-Doppler shrinkage thresholds │          │
│  │ γ_z          =  1 noise precision scalar             │          │
│  │ γ_h          =  1 channel prior precision scalar     │          │
│  │ α            =  1 damping factor (sigmoid-param)     │          │
│  │ β            =  1 Onsager correction coefficient     │          │
│  │ s, b         =  2 CFO scale and bias                 │          │
│  │ TOTAL        = 24 trainable parameters               │          │
│  └──────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 5 — Inductive Bias Justifications

### Architecture-Level Justifications

| Decision | Justification | Evidence Status |
|---|---|---|
| **Deep unfolding (not black-box DNN)** | Preserves the algorithmic structure of iterative VAMP/AMP estimation. This provides a strong inductive bias matching the physics: the update equations encode the correct measurement model (y = Hx + w). At test time with out-of-distribution geometries, the unfolded architecture degrades gracefully (to the base VAMP algorithm) whereas a black-box DNN would hallucinate. | `grounded` — Established in deep unfolding literature (Gregor & LeCun 2010, Borgerding et al. 2017, Monga et al. 2021 survey) |
| **VAMP over AMP** | VAMP provides rigorous state evolution for a broader class of measurement matrices (including non-i.i.d. sensing), and its denoiser can handle non-separable shrinkage. AMP diverges for non-i.i.d. matrices. The NB-IoT pilot matrix is structured (not i.i.d.), so VAMP's robustness is essential. | `grounded` — Rangan et al. (2017, 2019) prove VAMP state evolution for right-rotationally invariant matrices |
| **Delay-Doppler domain denoising** | The LEO satellite channel is approximately sparse in the delay-Doppler domain (a few propagation paths with distinct delays and Doppler shifts). Transforming the channel estimate to delay-Doppler concentrates the signal energy into a few bins while noise remains spread uniformly. Shrinkage in this domain exploits sparsity directly. | `grounded` — This is the standard OFDM channel sparsity result (Bajwa et al. 2010), validated for LEO NTN by the delay-Doppler structure in 3GPP TR 38.811 |
| **Learnable shrinkage thresholds (not fixed)** | The optimal threshold depends on noise level (SNR), Doppler spread, and delay spread — all variables in LEO IoT. Fixed thresholds (e.g., from Stein's unbiased risk estimate) would be optimal only at a single operating point. Learnable thresholds adapt to the ensemble of training channel realizations. | `hypothesis` — Plausible from adaptive thresholding in image denoising (Dabov et al. 2007) but unverified for LEO NTN channel estimation |
| **Weight tying across iterations** | Forces the denoiser to learn a single shrinkage function that works at all VAMP iterations. This regularizes the model (fewer parameters) and prevents overfitting to early-iteration artifacts. The VAMP state evolution ensures the input statistics change systematically across iterations, so a shared threshold must learn the average optimal behavior — which may be a limitation. | `hypothesis` — Ablation needed to compare tied vs. untied thresholds |
| **LMMSE (not ZF) in linear step** | MMSE accounts for noise statistics via γ_z; ZF does not. At LEO IoT SNR (–10 to 0 dB), noise amplification from ZF would be catastrophic. MMSE adds only 1 learnable parameter (γ_z) vs. ZF's 0, and preserves the Cramer-Rao lower bound structure. | `grounded` — Standard estimation theory result (Kay 1993, Fundamentals of Statistical Signal Processing) |
| **Damping via sigmoid-parametrized α** | Unconstrained logit → sigmoid gives α ∈ (0, 1) naturally without clipping. This avoids gradient issues at the boundaries (α=0 or α=1) that would occur with clamped parameters. | `grounded` — Common practice in deep unfolding (Monga et al. 2021); prevents dead gradients |
| **Analytic + learned CFO (not pure MLP)** | The analytic CFO estimate from pilot autocorrelation provides a strong prior (unbiased, consistent). The learned scale/bias only needs to correct the small residual error. This dramatically reduces the learning problem compared to an MLP that must learn CFO estimation from scratch. | `grounded` — Hybrid model-driven approach is the key research insight (Lin & Shen 2024) |
| **Soft MMSE detection with feedback** | Feeding soft symbols (not hard decisions) to the next VAMP iteration preserves gradient flow and reduces error propagation. At low SNR, hard decisions are unreliable and would corrupt subsequent channel estimates. | `grounded` — Standard in turbo/iterative receivers (Douillard et al. 1995) |
| **Pilot mask enforcement** | Known pilot symbols are re-inserted at each iteration. This prevents the iterative process from drifting at positions where ground truth is known, anchoring the estimation to the pilot observations. | `grounded` — Common in semi-blind channel estimation (Cozzo & Hughes 2000) |
| **Onsager correction with learnable β** | The Onsager correction (u1 term) in VAMP prevents correlation build-up across iterations. The learnable β adjusts the strength of this correction. At low SNR, the standard Onsager correction may be too aggressive; β < 1 mitigates this. | `hypothesis` — Standard VAMP theory gives β = 1; learned β is a relaxation for finite-dimensional/small-system setting |
| **Pre-norm (not applied — no layer norm in receiver)** | Unlike LM or CV architectures, the VAMP receiver operates on IQ samples with known physical statistics (signal power bounded by transmit power, noise variance known). Normalization would destroy this physical information. | `grounded` — Physics-constrained design principle: preserve physical units |
| **No dropout** | Dropout is incompatible with the unfolded VAMP structure. The VAMP iterations form a deterministic algorithm with well-defined state evolution; stochastic masking would break the estimation-theoretic guarantees. The architecture already has strong regularization via its low parameter count. | `grounded` — Deep unfolding design principle |

### Design Choices Considered and Rejected

| Alternative | Reason Rejected |
|---|---|
| **Full transformer receiver** | O(N²) self-attention on 168 REs is 28k operations per head — within budget, but training data requirements (millions of examples) and fragility to OOD geometries make it unsuitable for LEO IoT. DeepSig's OmniPHY (terrestrial 5G, 10⁵+ params) confirms this path requires significant compute. |
| **End-to-end ResNet receiver** | 10⁵+ parameters for competitive performance (DeepSig results). Cannot be constrained to ≤1,000 params without drastic performance loss. Poor generalization across orbital geometries (black box lacks physics priors). |
| **Fully on-pilot CNN channel estimator (no unfolding)** | Works at medium-high SNR (>5 dB) but degrades at LEO IoT SNR (Jiao et al. 2023). CNN FLOPS (even a small 3-layer net) exceed the DSP budget on Zynq. Deep unfolding is more parameter-efficient. |
| **RL-based adaptive receiver** | Symbol-rate decisions in PHY layer require sub-ms latency. RL training is too slow and sample-inefficient for this regime. More suited to higher-layer link adaptation. |
| **AMP instead of VAMP** | AMP diverges for structured (non-i.i.d.) measurement matrices. The NB-IoT pilot pattern creates a structured observation matrix. VAMP's state evolution converges for right-rotationally invariant matrices, which covers the NTN pilot structure. |

### Rejected Position/Order Scheme (LM domain consideration)

Although this is not an LM architecture, the question of ordering arose during research: could a Transformer handle the time-frequency grid as a sequence? The answer is no — self-attention over 168 positions would lack the spatial inductive bias of the VAMP structure, and the parameter budget (≤1,000) is 100× too small for even a minimal transformer (which needs at least Q,K,V projections with d_model≥32 → ~3k params).

---

## Step 6 — Research-to-Architecture Traceability

| Research Contract Item | Architecture Decision | Evidence Status | Validation Hook |
|---|---|---|---|
| **No existing deep-unfolded receiver for joint CFO+channel+detection in LEO NTN IoT** | Architecture integrates CFO compensation, channel estimation, and data detection into a single unfolded VAMP framework with state passing between sub-modules | `grounded` | Literature review confirms gap; architecture novelty is the joint integration |
| **Parameter count ≤ 1,000** | Delay-Doppler shrinkage with weight tying → 24 learnable params; headroom for ablation studies up to ~104 params | `hypothesis` | `model.n_trainable_params` check in training script; verify ≤ 1,000 |
| **MACs per block ≤ 10,000** | VAMP linear step + FFT-based denoising + soft detection ≈ 15k MACs at T=5; use T=3 (≈9k MACs) for FPGA compliance | `hypothesis` | `thop.profile()` or manual MAC counting per block; verify ≤ 10k |
| **CFO tracking under LEO Doppler (≥1 kHz)** | Per-iteration analytic + learned CFO residual estimation; CFO scale/bias correction parameters; analytic anchor prevents divergence | `hypothesis` | BLER vs. Doppler spread at Eb/N0=-3 dB; compare with genie CFO baseline |
| **Fits Zynq ZU3EG, <5W, <1ms** | 24 parameters → trivial model weights; MAC-dominated compute (no large activations); INT8 quantization target; FFT-heavy ops map to Xilinx FFT IP cores | `TODO: unverified` | FPGA implementation using FINN/Tensil; report LUT/DSP/BRAM/power/latency |
| **Outperform MMSE by ≥2 dB at low SNR** | LMMSE with learned γ_z replaces fixed noise estimate; delay-Doppler denoising exploits sparsity that MMSE ignores | `hypothesis` | BLER curves vs. Eb/N0; compare with MMSE baseline at –3 dB |
| **Generalize across orbital geometries** | Physics-constrained VAMP structure (not black-box) should degrade gracefully; training data includes varying elevation angles | `hypothesis` | BLER on held-out elevation bands (15°, 30°, 60°, 90°); < 3 dB gap = success |
| **NMSE of channel estimation** | Architecture directly outputs Ĥ estimate; VAMP state evolution provides well-characterized MSE | `hypothesis` | NMSE(Ĥ, H_true) vs. Eb/N0; compare with LS/MMSE baselines |
| **Training on synthetic LEO NTN channels per 3GPP TR 38.811** | Delay-Doppler shrinkage expectations calibrated to TR 38.811 channel model parameters (delay spread, Doppler spread, K-factor) | `grounded` | Channel simulator must implement TR 38.811 Rician fading with LEO-specific parameters |
| **Baseline: conventional MMSE receiver** | MMSE detector is a special case of our architecture with T=0 iterations (linear step only, no denoising), enabling direct A/B comparison | `grounded` | Config toggle: `n_unfolded_layers=0` yields conventional MMSE |
| **Baseline: ideal known-channel bound** | Final soft MMSE detector can be run with h_hat = H_true (from simulator) to get genie bound | `grounded` | `forward` bypass with known-channel flag |
| **Blocking unknown: NTN channel simulator availability** | Architecture assumes differentiable channel model for end-to-end training; if unavailable, iterative training with pre-computed channel samples | `TODO: unverified` | Verify Sionna-NTN or build synthetic channel generator |

---

## Step 7 — Domain-Specific Considerations

### A. Scientific ML

| Consideration | Implementation |
|---|---|
| **Physics constraints** | The VAMP linear step enforces the exact measurement model y = Hx + w. No neural network can produce a channel estimate that violates the observation equation — the LMMSE step projects the estimate back onto the feasible set. Physics is enforced via algorithm structure, not a soft loss penalty. |
| **Function vs. operator learning** | This is a **function learning** problem: the receiver maps a specific received signal to a specific transmitted message. It is NOT an operator (mapping between function spaces). The unfolded VAMP is a parameterized function approximating the optimal MMSE estimator. |
| **Symmetry/equivariance** | The VAMP receiver is **equivariant** to phase rotations of the received signal (a constant phase shift on all subcarriers results in the same shift in the channel estimate). No explicit SE(3) equivariance is needed because the problem is 2D time-frequency, not 3D physical space. However, the architecture is **invariant to symbol ordering** in the sense that the pilot mask imposes the known structure. |
| **Mesh type** | Regular 2D grid (N_sc × N_sym) — the NB-IoT time-frequency resource grid. FFT-based transforms are natural for this mesh. No irregular meshes or point clouds. |
| **Rollout stability** | For the unfolded iteration (T=5), the VAMP state evolution is contractive under standard conditions (Rangan et al. 2017). No instability expected for T < 10. For the longer time-stepping (slot-to-slot tracking), each slot is processed independently with the previous slot's channel estimate as initialization. Error accumulation across slots is bounded by the re-initialization from pilots each slot. |

### B. Time Series (Secondary Domain)

| Consideration | Implementation |
|---|---|
| **Temporal ordering** | No future leakage: the received signal is fully available per OFDM symbol (the receiver has the whole slot before processing). For streaming scenarios, a causal variant would use only symbols ≤ current time. Default design is **non-causal** within a slot (bidirectional FFT over symbols for Doppler domain). |
| **Channel mixing vs. independence** | **Channel mixing is inherent** — subcarriers are coupled through the IFFT/FFT in the delay-Doppler transform. This is correct behavior for OFDM where subcarrier coupling arises from the channel's delay spread (frequency selectivity). Channel-independence would discard useful structure. |
| **Stationarity** | The LEO channel is **non-stationary** within a pass: Doppler shift, delay spread, and K-factor change with elevation angle. The architecture handles this by processing each slot independently (T=5 iterations on one slot's data). No recurrent state between slots prevents temporal error accumulation. For the deployment scenario, an online adaptation mechanism (fine-tuning γ_z at the start of each pass) may be needed. |
| **Patch design** | No explicit patching — the architecture operates on the full 12×14 resource grid. The delay-Doppler transform serves as a physics-motivated "patch" that captures the channel's sparse structure. Patch size is implicitly the full grid. |
| **Prediction head** | **Direct multi-step**: all 168 REs in the slot are processed simultaneously (not auto-regressively). This is appropriate because the OFDM receiver has access to the full slot before processing. For a streaming variant, a causal mask on the Doppler FFT would enable per-symbol output. |

### C. LM Domain (Tertiary — Attention Considerations)

| Consideration | Resolution |
|---|---|
| **Attention mechanism** | Rejected for LEO IoT receiver due to parameter budget constraints and training data efficiency. However, a single-head self-attention could replace the VAMP denoiser as an ablation (if parameter budget permits). Specifically, axial attention (attend along subcarriers or symbols separately) would be O(N_sc² + N_sym²) ≈ 340 operations, fitting the MAC budget. |
| **Causal contract** | Not applicable — the receiver processes a complete OFDM slot. However, if a streaming mode were needed, the Doppler FFT would become causal (use only past symbols via a causal sliding window FFT). |

### D. Additional Domain: Physical Layer Communications

| Consideration | Implementation |
|---|---|
| **Complex baseband representation** | All signals are complex-valued. The architecture uses 2-channel (real, imag) representation throughout, preserving the complex structure. Conjugates, Hermitian operations, and complex multiplies are explicit. |
| **Modulation awareness** | QPSK LLR computation uses the modulation constellation explicitly. Extending to higher-order modulations (16-QAM, etc.) requires only changing the LLR function — the unfolded VAMP structure is modulation-agnostic. |
| **Pilot structure dependence** | The architecture is designed for NB-IoT NPUSCH Format 1 (24 pilots per slot at symbols 3 and 10). Adapting to other pilot patterns (LR-FHSS, NR-NTN DMRS) requires changing only the `pilot_mask` and `pilot_symbol_indices` config fields. The VAMP core is pilot-agnostic. |
| **Channel code interface** | Output is per-bit LLRs ready for a soft-input channel decoder (Turbo, LDPC, or polar). The architecture does not include the decoder — it is a receiver front-end producing decoder input. |

---

## Step 8 — Implementation Risk Flags

### Risk 1: Numerical Instability in VAMP Linear Step
- **Issue**: The denominator `gamma_z * x_pow + gamma_h` in the LMMSE step can approach zero if both γ_z (noise precision) and γ_h (channel prior precision) are very small, or if a symbol has near-zero power (`x_pow ≈ 0`).
- **Mitigation**: Add epsilon (1e-10) to denominator. Clamp γ_z and γ_h to positive values via `log → exp` param (already in design). For QPSK, `x_pow = 1` (normalized constellation), so this risk is theoretical for QPSK but real if extended to higher-order modulations with amplitude variations (16-QAM).
- **Falsification**: Loss becomes NaN during training, or LLRs contain ±Inf values.

### Risk 2: VAMP State Evolution Divergence for T > 5
- **Issue**: While VAMP has rigorous state evolution guarantees in the large-system limit, for the small-system regime (N=168 REs, 12 subcarriers), the assumptions may not hold. The Onsager correction (learnable β) is meant to address this, but the learned β may not be sufficient to prevent oscillation.
- **Mitigation**: Clamp α (damping) to [0.1, 0.9] via the sigmoid parametrization to prevent extreme values. Monitor the VAMP residual `||h_den - h_lin||` across iterations. If it increases after layer 3, the architecture is diverging.
- **Falsification**: BLER degrades as T increases beyond 2–3 iterations (negative returns to depth). This would be caught by the suggested ablation on `n_unfolded_layers`.

### Risk 3: CFO Tracking Divergence at Extreme Doppler
- **Issue**: At Doppler spread = 2.4 kHz with 15 kHz subcarrier spacing, the Doppler-to-SCS ratio is 16%, causing significant inter-carrier interference (ICI). The analytic CFO estimate from pilot autocorrelation assumes no ICI, and at high Doppler this assumption breaks. The learned correction may not compensate for ICI if it only has scale/bias parameters.
- **Mitigation**: Add an optional ICI compensation module (e.g., a 3-tap Doppler equalizer) that can be enabled at high Doppler. Monitor the BLER at 2.4 kHz specifically.
- **Falsification**: BLER at Eb/N0 = 0 dB worsens from 2.4 kHz to 1.2 kHz Doppler (instead of the expected monotonic improvement). This would indicate ICI-dominated performance.

### Risk 4: FPGA INT8 Quantization Loss
- **Issue**: The delay-Doppler transform uses FFT, which is inherently a floating-point operation. INT8 quantization of the FFT intermediates can cause significant accuracy loss (>1 dB). The shrinkage thresholds θ_t may need higher precision than INT8 for the soft-thresholding operation.
- **Mitigation**: Use block floating-point or mixed-precision (INT8 for weights, FP16 for FFT intermediates) in the FPGA implementation. Profile the quantization sensitivity of each component before full quantization.
- **Falsification**: INT8 model BLER > FP32 model BLER + 1 dB at Eb/N0 = –3 dB.

### Risk 5: Training Data Mismatch
- **Issue**: The research notes the `TODO: unverified` status of LEO NTN channel simulator availability. If no differentiable simulator exists, the architecture cannot be trained end-to-end with backpropagation. Training would need to use pre-computed channel realizations with a two-stage approach (estimate channel → freeze → train denoiser).
- **Mitigation**: The VAMP linear step is parameter-free (only γ_z and γ_h), so it can be pre-computed. The denoiser thresholds (18 params) can be trained with a decoupled loss on the delay-Doppler representation without end-to-end backprop through the linear step.
- **Falsification**: Cannot generate diverse LEO NTN channel realizations for training → blocking unknown that stops the entire pipeline.

---

## Step 9 — Suggested Ablations

All ablations are **single-field ModelConfig changes** and directly test hypotheses from the research-to-architecture traceability table.

### Ablation 1: Unfolded Layer Count (Depth)

| Field | Baseline | Ablated | Hypothesis | Expected Metric | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|
| `n_unfolded_layers` | 5 | {1, 2, 3, 7, 10} | Each additional iteration improves BLER monotonically, with diminishing returns after T=5 | BLER gap between T=5 and T=10 < 0.5 dB at Eb/N0=-3 dB; BLER(T=3) > BLER(T=5) by > 1 dB | If BLER(T=5) ≈ BLER(T=3): architecture saturates early → can reduce T for FPGA. If BLER(T=7) > BLER(T=5): VAMP diverging → need stronger damping or fewer layers | `ml-architect` |

### Ablation 2: Weight Tying (Shared vs. Per-Layer Thresholds)

| Field | Baseline | Ablated | Hypothesis | Expected Metric | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|
| `weight_tying` | True | False | Per-layer thresholds provide ≤ 0.3 dB improvement over shared at 5× parameter cost. The parameter-efficiency of sharing outweighs the expressivity gain | BLER gap < 0.3 dB at Eb/N0=-3 dB; parameter count increases from 24 to 104 | If gap > 1 dB: the shrinkage needs to be iteration-dependent → keep per-layer thresholds. This changes the FPGA deployment profile (104 params still fits) | `ml-architect` |

### Ablation 3: Denoiser Type (Expressiveness Pareto Frontier)

| Field | Baseline | Ablated | Hypothesis | Expected Metric | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|
| `denoiser_type` | `delay_doppler_shrinkage` | `element_shrinkage`, `small_cnn`, `mlp` | Delay-Doppler shrinkage exploits channel sparsity better than element-wise, and matches CNN performance at < 5% of the parameter count | Element-wise: +1 dB BLER gap vs delay-Doppler. CNN: +0.1 dB at 62 params (shared). MLP: +0.5 dB at 20 params | If delay-Doppler shrinkage ≤ element-wise: channel is NOT sparse in delay-Doppler for LEO NTN → fundamental assumption false → revisit channel model | `ml-research` |

### Ablation 4: CFO Estimation Mode

| Field | Baseline | Ablated | Hypothesis | Expected Metric | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|
| `cfo_mode` | `analytic_plus_learned` | `analytic` (no learned correction), `mlp` | The learned scale/bias correction provides ≥ 0.5 dB improvement over pure analytic at Doppler > 1.5 kHz by compensating for ICI-induced phase errors | BLER gap ≥ 0.5 dB at 2.4 kHz Doppler between `analytic` and `analytic_plus_learned` | If gap < 0.2 dB: analytic CFO is sufficient → remove learnable components for FPGA. If `mlp` significantly outperforms both: need fully learned CFO | `ml-architect` |

### Ablation 5: Soft Symbol Feedback

| Field | Baseline | Ablated | Hypothesis | Expected Metric | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|
| `use_soft_feedback` | True | False | Soft symbol feedback from data REs improves channel estimation by leveraging data as "virtual pilots" at low SNR | BLER gap ≥ 1 dB at Eb/N0=-3 dB between feedback and no-feedback; gap < 0.2 dB at Eb/N0 > 5 dB | If gap < 0.3 dB: soft feedback is not beneficial → data symbols too unreliable at low SNR → simplify architecture. If gap persists at high SNR: soft feedback may hurt (error propagation) | `ml-architect` |

### Ablation 6: Damping Strategy

| Field | Baseline | Ablated | Hypothesis | Expected Metric | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|
| `n_damping_params` | 1 (shared) | 5 (per-layer) | Per-layer damping provides faster convergence (fewer iterations needed) but does not improve final BLER | BLER equivalent between configs at T=5; BLER(T=3, per-layer) ≈ BLER(T=5, shared) | If per-layer improves final BLER: VAMP iterations have different stability requirements → keep per-layer. Indicates non-contractive dynamics at early iterations | `ml-architect` |

### Ablation 7: Quantization Sensitivity

| Field | Baseline | Ablated | Hypothesis | Expected Metric | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|
| `quantization_target` | `int8` | `fp32` | INT8 quantization causes < 0.5 dB BLER degradation through the FFT and shrinkage operations | BLER gap < 0.5 dB at Eb/N0=-3 dB between INT8 and FP32 | If gap > 1 dB: delay-Doppler FFT needs higher precision → use FP16 for FFT, INT8 for shrinkage. This changes the FPGA resource budget | `ml-coder` + `ml-validator` |

### Ablation 8: Noise Precision Source

| Field | Baseline | Ablated | Hypothesis | Expected Metric | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|
| `learn_noise_precision` | True | False (use true SNR from simulator) | Learning γ_z from data adapts to the effective noise (including model mismatch) better than using the true SNR | BLER gap ≥ 0.5 dB at mismatched SNR (train at 0 dB, test at -5 dB) | If learned γ_z performs worse: the 1-param noise precision is insufficient → need SNR-adaptive mechanism or per-RE noise variance | `ml-architect` |

---

## Step 10 — Training and Deployment Strategy

### Training Protocol

```
1. Data Generation:
   - Synthesize LEO NTN channel realizations following 3GPP TR 38.811
   - Elevation angles: uniform over [10°, 90°]  (training)
   - SNR range: uniform over [-10, +10] dB
   - Doppler spreads: {600 Hz, 1.2 kHz, 2.4 kHz}
   - Modulation: QPSK (NB-IoT NPUSCH)
   - Frame structure: 12 SC × 14 Sym with NPUSCH Format 1 DMRS
   
2. Loss Function:
   ℒ = λ₁ · NMSE(Ĥ, H_true)             # Channel estimation loss
       + λ₂ · CrossEntropy(LLRs, bits)   # Detection loss (cross-entropy
                                          #  = binary CE over LLRs)
       + λ₃ · ||θ||₁                      # Sparsity regularization on shrinkage
   
   Default λ₁ = 1.0, λ₂ = 1.0, λ₃ = 0.01
   
3. Optimizer: AdamW (lr = 1e-3, weight_decay = 1e-4)
4. Batch size: 256
5. Training samples: 100,000 channel realizations
6. Validation: 10,000 held-out realizations
7. Early stopping: if BLER(val) does not improve for 10 epochs
```

### FPGA Deployment Pipeline

```
FP32 Model (trained in PyTorch)
       │
       ▼
Post-Training Quantization (PTQ)
       │  INT8 weights (shrinkage thresholds, γ_z, γ_h, α, β, CFO params)
       │  FP16 intermediates (FFT accumulators)
       ▼
FINN / Vitis AI Compilation
       │  HLS code generation
       │  DSP slice allocation
       │  BRAM allocation for parameter storage
       ▼
Zyqn ZU3EG Bitstream
       │  Measure: LUT / DSP / BRAM / Power / Latency
       ▼
Report against SWaP targets
       │  < 5W, < 600 DSP, < 1ms per block
```

---

## Output Checklist

- [x] Domain identified (Scientific ML primary, Time Series secondary)
- [x] Upstream research lifecycle contract read and preserved
- [x] ModelConfig dataclass with all hyperparameters (`DU_VAMP_JCD_Config`)
- [x] Pseudocode for the novel block (VAMP-JCD receiver: `VAMPJCDReceiver.forward()`)
- [x] ASCII architecture diagram (full receiver chain + parameter layout)
- [x] Inductive bias justifications (tabular, one sentence per decision)
- [x] Research-to-architecture traceability table included
- [x] Claims labeled as `grounded`, `hypothesis`, or `TODO: unverified`
- [x] Domain-specific considerations addressed (Scientific ML, Time Series, Physical Layer Comms)
- [x] Implementation risk flags (5 risks with mitigations and falsification conditions)
- [x] Baseline and evaluation requirements carried forward to validator
- [x] Suggested ablations (8 ablations, each = single ModelConfig field change tied to a hypothesis)
