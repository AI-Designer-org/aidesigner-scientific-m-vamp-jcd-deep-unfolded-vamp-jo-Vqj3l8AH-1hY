"""
Deep-Unfolded VAMP-JCD Receiver for LEO Satellite IoT Systems
==============================================================

Implements the joint channel estimation and data detection (JCD) receiver
architecture from the DLR thesis architecture design. The receiver uses
deep-unfolded Vector Approximate Message Passing (VAMP) with T=5 iterations,
each performing:

  1. CFO residual compensation
  2. VAMP linear step (LMMSE channel estimation)
  3. Delay-Doppler domain denoising with learnable shrinkage thresholds
  4. VAMP state update (damping + Onsager correction)
  5. Soft data detection (MMSE equalizer → LLRs)
  6. CFO residual re-estimation

Domain: Scientific ML (primary) — hybrid model-driven + data-driven receiver
         Time Series (secondary) — IQ samples as complex-valued time series

Total learnable parameters: ≤ 24 (default weight-tied config).

References:
  - Rangan et al. (2019), "VAMP for non-i.i.d. matrices"
  - 3GPP TR 38.811 (NTN channel model)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


# ──────────────────────────────────────────────────────────────────────────
# Config Dataclass
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class DU_VAMP_JCD_Config:
    # ── System Parameters (NB-IoT NTN NPUSCH) ──────────────────────────
    domain: str = "scientific_ml"
    subdomain: str = "leo_ntn_iot_receiver"

    # Physical layer
    n_subcarriers: int = 12                # NB-IoT: 12 SCs x 15 kHz = 180 kHz
    n_symbols: int = 14                    # OFDM symbols per slot (normal CP)
    subcarrier_spacing_hz: float = 15000.0
    n_pilots: int = 24                     # DMRS: 2 symbols x 12 SCs (NPUSCH Format 1)
    pilot_symbol_indices: Tuple = (3, 10)  # DMRS symbol positions (0-indexed)
    modulation: str = "qpsk"               # pi/2-BPSK or QPSK
    n_bits_per_symbol: int = 2             # for QPSK

    # ── Deep Unfolding Architecture ────────────────────────────────────
    n_unfolded_layers: int = 5             # T = 5 VAMP iterations
    weight_tying: bool = True              # share denoiser params across layers
    denoiser_type: str = "delay_doppler_shrinkage"
    # Options: "delay_doppler_shrinkage" | "element_shrinkage" | "small_cnn" | "mlp"

    # Delay-Doppler denoiser grid
    n_delay_groups: int = 3               # delay bins grouped: short, medium, long
    n_doppler_groups: int = 3             # Doppler bins grouped: low, medium, high
    # Total shrinkage thresholds per layer: 3 x 3 = 9 (complex = 18 real)

    # ── CFO Tracking ───────────────────────────────────────────────────
    cfo_mode: str = "analytic_plus_learned"
    # "analytic" | "analytic_plus_learned" | "mlp"
    cfo_learned_scale: bool = True        # learnable scalar multiplier
    cfo_learned_bias: bool = True         # learnable scalar additive bias
    cfo_mlp_hidden_dim: int = 0            # 0 = analytic only; >0 enables small MLP

    # ── Detection ──────────────────────────────────────────────────────
    detector_type: str = "mmse_soft"       # "mmse_soft" | "zf" | "learned"
    use_soft_feedback: bool = True         # feed soft symbols to next iteration

    # ── VAMP State Parameters ──────────────────────────────────────────
    learn_noise_precision: bool = True     # gamma_z: 1 learnable scalar
    learn_channel_precision: bool = True   # gamma_h: 1 learnable scalar
    n_damping_params: int = 1              # 1 = shared, n_layers = per-layer
    n_onsager_params: int = 1              # 1 = shared, n_layers = per-layer

    # ── Training ───────────────────────────────────────────────────────
    dropout: float = 0.0                   # no dropout in unfolded architecture
    use_bias: bool = False                 # minimal bias in unfolded layers
    dtype: str = "float32"
    quantization_target: str = "int8"      # for FPGA deployment
    training_snr_range_db: Tuple = (-10.0, 10.0)

    # ── SWaP Target ────────────────────────────────────────────────────
    target_fpga: str = "xilinx_zynq_zu3eg"
    max_dsp_slices: int = 600
    max_power_watts: float = 5.0
    target_latency_ms: float = 1.0

    # ── Derived (computed at init) ─────────────────────────────────────
    n_trainable_params: int = 0            # computed after init
    n_macs_per_block: int = 0              # computed after init

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
            cfo_in = 2 * self.n_pilots
            cfo_hid = self.cfo_mlp_hidden_dim
            n += cfo_in * cfo_hid + cfo_hid + cfo_hid * 1 + 1

        # Soft detector (learned noise estimate)
        if self.detector_type == "learned":
            n += 1

        self.n_trainable_params = n

    def get_layer_config(self, layer_idx: int) -> Dict[str, Any]:
        """Get config overrides for a specific unfolded layer."""
        return {
            "use_soft_feedback": self.use_soft_feedback and (layer_idx < self.n_unfolded_layers - 1),
            "cfo_update": True,
        }


# ──────────────────────────────────────────────────────────────────────────
# Base Operator for Denoiser
# ──────────────────────────────────────────────────────────────────────────

class BaseDenoiser(ABC, nn.Module):
    """
    Abstract base class for the VAMP denoiser operator.

    The denoiser operates in the delay-Doppler domain on the channel estimate.
    All variants must preserve the input shape.
    """

    @abstractmethod
    def forward(self, h_dd: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Denoise channel estimate in delay-Doppler domain.

        Args:
            h_dd: (B, N_sc, N_sym, 2) — channel in delay-Doppler domain,
                  last dim = (real, imag)
            layer_idx: current unfolded iteration index

        Returns:
            h_dd_denoised: (B, N_sc, N_sym, 2) — denoised channel,
                           same shape as input
        """
        pass


# ──────────────────────────────────────────────────────────────────────────
# Delay-Doppler Shrinkage Denoiser (default, 18 learnable params)
# ──────────────────────────────────────────────────────────────────────────

class DelayDopplerShrinkageDenoiser(BaseDenoiser):
    """
    Learnable soft-thresholding denoiser in the delay-Doppler domain.

    The channel is approximately sparse in the delay-Doppler domain
    (few propagation paths with distinct delays and Doppler shifts).
    Soft-thresholding with learnable per-bin thresholds exploits
    this sparsity structure.

    The delay-Doppler plane is divided into n_delay_groups x n_doppler_groups
    bins, each with separate real and imaginary shrinkage thresholds.
    With weight tying (default): 3 x 3 x 2 = 18 learnable parameters.

    Shape conventions:
        Input / Output: (B, N_sc, N_sym, 2)
        Internal: real/imag as last dim
    """

    def __init__(self, config: DU_VAMP_JCD_Config):
        super().__init__()
        self.config = config
        self.n_dg = config.n_delay_groups
        self.n_dog = config.n_doppler_groups

        # Shrinkage thresholds: shape (T or 1, n_delay_groups, n_doppler_groups, 2)
        n_thresh = 1 if config.weight_tying else config.n_unfolded_layers
        self.shrinkage_thresholds = nn.Parameter(
            torch.full((n_thresh, self.n_dg, self.n_dog, 2), 0.1)
        )

        # Pre-compute delay-Doppler bin maps (registered buffers, not params)
        self.register_buffer('delay_bin_map', self._build_delay_bin_map())
        self.register_buffer('doppler_bin_map', self._build_doppler_bin_map())

    def _build_delay_bin_map(self) -> torch.Tensor:
        """Map each subcarrier to one of n_delay_groups delay bins."""
        return torch.linspace(0, self.n_dg - 1, self.config.n_subcarriers).long()

    def _build_doppler_bin_map(self) -> torch.Tensor:
        """Map each OFDM symbol to one of n_doppler_groups Doppler bins."""
        return torch.linspace(0, self.n_dog - 1, self.config.n_symbols).long()

    def forward(self, h_dd: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Apply learnable soft thresholding in delay-Doppler domain.

        Args:
            h_dd: (B, N_sc, N_sym, 2) — delay-Doppler channel
            layer_idx: current unfolded iteration index

        Returns:
            h_dd_shrunk: (B, N_sc, N_sym, 2) — denoised channel
        """
        # Select thresholds for this layer (or shared)
        # threshold shape: (n_dg, n_dog, 2)
        t_idx = 0 if self.config.weight_tying else layer_idx
        thresholds = self.shrinkage_thresholds[t_idx]  # (n_dg, n_dog, 2)

        # Map each spatial bin to its delay-Doppler group
        delay_bins = self.delay_bin_map    # (N_sc,)
        doppler_bins = self.doppler_bin_map  # (N_sym,)

        # Gather thresholds per bin: (N_sc, N_sym, 2)
        # delay_bins[:, None] -> (N_sc, 1), doppler_bins[None, :] -> (1, N_sym)
        thresh_vals = thresholds[delay_bins[:, None], doppler_bins[None, :]]  # (N_sc, N_sym, 2)

        # Soft thresholding per real/imag component
        magnitude = torch.abs(h_dd)                              # (B, N_sc, N_sym, 2)
        shrinkage = F.relu(magnitude - thresh_vals) / (magnitude + 1e-10)  # (B, N_sc, N_sym, 2)
        h_dd_shrunk = h_dd * shrinkage                           # (B, N_sc, N_sym, 2)

        return h_dd_shrunk


# ──────────────────────────────────────────────────────────────────────────
# Element-wise Shrinkage Denoiser (ablation, 2 learnable params)
# ──────────────────────────────────────────────────────────────────────────

class ElementShrinkageDenoiser(BaseDenoiser):
    """
    Simple element-wise shrinkage denoiser with a single threshold per
    real/imag component (2 learnable params total).

    This ignores the delay-Doppler structure and serves as an ablation
    to measure how much the structural grouping helps.
    """

    def __init__(self, config: DU_VAMP_JCD_Config):
        super().__init__()
        # Single threshold per real/imag
        self.thresholds = nn.Parameter(torch.tensor([0.1, 0.1]))

    def forward(self, h_dd: torch.Tensor, layer_idx: int) -> torch.Tensor:
        magnitude = torch.abs(h_dd)                                    # (B, N_sc, N_sym, 2)
        shrinkage = F.relu(magnitude - self.thresholds) / (magnitude + 1e-10)
        return h_dd * shrinkage


# ──────────────────────────────────────────────────────────────────────────
# Tiny CNN Denoiser (ablation, 62 learnable params)
# ──────────────────────────────────────────────────────────────────────────

class TinyCNNDenoiser(BaseDenoiser):
    """
    Lightweight 2D CNN denoiser for ablation study.

    Architecture:
        - 1x1 conv: cross-channel mixing (real/imag -> 4 channels)
        - 3x3 depthwise conv: spatial mixing in delay-Doppler
        - 1x1 conv: project back to 2 channels (real, imag)

    Parameter count: (2x4 + 4) + (9x4 + 4) + (4x2 + 2) = 12 + 40 + 10 = 62

    Operates on (B, C, H, W) = (B, 2, N_sc, N_sym) internally.
    Input/output format: (B, N_sc, N_sym, 2) — last-dim = real/imag.
    """

    def __init__(self, config: DU_VAMP_JCD_Config):
        super().__init__()
        n_channels = 4
        self.conv1 = nn.Conv2d(2, n_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3,
                                padding=1, groups=n_channels, bias=True)
        self.conv3 = nn.Conv2d(n_channels, 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, h_dd: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Args:
            h_dd: (B, N_sc, N_sym, 2) — delay-Doppler channel
            layer_idx: unused (shared weights)

        Returns:
            h_dd_denoised: (B, N_sc, N_sym, 2)
        """
        # Rearrange: (B, N_sc, N_sym, 2) -> (B, 2, N_sc, N_sym) for Conv2d
        x = h_dd.permute(0, 3, 1, 2)   # (B, 2, N_sc, N_sym)
        x = self.relu(self.conv1(x))    # (B, 4, N_sc, N_sym)
        x = self.relu(self.conv2(x))    # (B, 4, N_sc, N_sym)
        x = self.conv3(x)               # (B, 2, N_sc, N_sym)
        # Rearrange back: (B, 2, N_sc, N_sym) -> (B, N_sc, N_sym, 2)
        return x.permute(0, 2, 3, 1)    # (B, N_sc, N_sym, 2)


# ──────────────────────────────────────────────────────────────────────────
# MLP Denoiser (ablation, ~20 learnable params)
# ──────────────────────────────────────────────────────────────────────────

class MLPDenoiser(BaseDenoiser):
    """
    Simple per-element MLP denoiser. Applies a 2-layer MLP to each
    (real, imag) pair independently.

    Architecture:
        - Linear: 2 -> hidden_dim
        - ReLU
        - Linear: hidden_dim -> 2

    Parameter count: 2*h + h + h*2 + 2 = 3h + 2. With h=6: 20 params.
    """

    def __init__(self, config: DU_VAMP_JCD_Config):
        super().__init__()
        hidden_dim = 6
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2, bias=True),
        )

    def forward(self, h_dd: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Args:
            h_dd: (B, N_sc, N_sym, 2)
            layer_idx: unused

        Returns:
            h_dd_denoised: (B, N_sc, N_sym, 2)
        """
        B, N_sc, N_sym, _ = h_dd.shape
        # Flatten spatial dims: (B, N_sc*N_sym, 2)
        x = h_dd.view(B, -1, 2)
        x = self.net(x)                           # (B, N_sc*N_sym, 2)
        return x.view(B, N_sc, N_sym, 2)           # (B, N_sc, N_sym, 2)


# ──────────────────────────────────────────────────────────────────────────
# Denoiser Factory
# ──────────────────────────────────────────────────────────────────────────

def build_denoiser(config: DU_VAMP_JCD_Config) -> BaseDenoiser:
    """Factory function to create the appropriate denoiser from config."""
    if config.denoiser_type == "delay_doppler_shrinkage":
        return DelayDopplerShrinkageDenoiser(config)
    elif config.denoiser_type == "element_shrinkage":
        return ElementShrinkageDenoiser(config)
    elif config.denoiser_type == "small_cnn":
        return TinyCNNDenoiser(config)
    elif config.denoiser_type == "mlp":
        return MLPDenoiser(config)
    else:
        raise ValueError(f"Unknown denoiser_type: {config.denoiser_type}")


# ──────────────────────────────────────────────────────────────────────────
# CFO Estimator Module
# ──────────────────────────────────────────────────────────────────────────

class CFOEstimator(nn.Module):
    """
    Carrier Frequency Offset (CFO) estimator with analytic and learned modes.

    For LEO satellite IoT, Doppler shifts can reach ±40 kHz (at 2 GHz).
    The analytic mode computes CFO from pilot autocorrelation (parameter-free).
    The learned mode adds learnable scale and bias to the analytic estimate.

    Shape conventions:
        y: (B, N_sc, N_sym, 2) — received IQ
        pilots: (B, N_p, 2) — known pilot symbols
        pilot_mask: (N_sc, N_sym) — boolean pilot locations
    """

    def __init__(self, config: DU_VAMP_JCD_Config):
        super().__init__()
        self.config = config
        self.N_sc = config.n_subcarriers
        self.N_sym = config.n_symbols
        self.register_buffer('time_idx', torch.arange(self.N_sym).float())

        # Learnable CFO parameters
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
                nn.Tanh(),  # bounded CFO correction: [-1, 1]
            )

    def analytic_estimate(self, y: torch.Tensor,
                          pilots: torch.Tensor,
                          pilot_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute coarse CFO estimate from pilot autocorrelation.

        Uses the phase difference between pilot symbols at different
        OFDM symbol positions. For NPUSCH Format 1, pilots are at
        symbols 3 and 10 (0-indexed).

        Args:
            y: (B, N_sc, N_sym, 2) — received IQ samples
            pilots: (B, N_p, 2) — known pilot symbols
            pilot_mask: (N_sc, N_sym) — boolean pilot locations

        Returns:
            delta_f: (B,) estimated CFO in normalized units (cycles/symbol)
        """
        B = y.shape[0]
        device = y.device

        # Extract pilot observations at pilot positions
        # pilot_mask is (N_sc, N_sym) -> broadcast over batch
        y_p = y[:, pilot_mask, :]           # (B, N_p, 2)

        # Compute phase error: r = y * conj(x)  (element-wise complex multiply)
        # r_real = y_re * x_re + y_im * x_im
        # r_imag = y_im * x_re - y_re * x_im  (note: this is cross product for conj)

        # Use float32 for numerical safety on angle computation
        y_p = y_p.float()
        pilots = pilots.float()

        r_real = y_p[..., 0] * pilots[..., 0] + y_p[..., 1] * pilots[..., 1]  # (B, N_p)
        r_imag = y_p[..., 1] * pilots[..., 0] - y_p[..., 0] * pilots[..., 1]  # (B, N_p)

        phase_error = torch.atan2(r_imag, r_real)     # (B, N_p), in radians

        # Split by pilot symbol index (first 12 = sym 3, next 12 = sym 10)
        n_sc = self.N_sc
        phase_sym0 = phase_error[:, :n_sc]             # (B, 12) pilots at first symbol
        phase_sym1 = phase_error[:, n_sc:]             # (B, 12) pilots at second symbol

        # Average phase difference across subcarriers
        delta_phi = torch.mean(phase_sym1 - phase_sym0, dim=1)  # (B,)

        # Convert phase difference to normalized frequency offset
        p0_idx = self.config.pilot_symbol_indices[0]
        p1_idx = self.config.pilot_symbol_indices[1]
        dt = float(p1_idx - p0_idx)                   # symbol spacing between pilots

        # delta_f = delta_phi / (2 * pi * dt)  in cycles/symbol
        delta_f = delta_phi / (2.0 * math.pi * dt)    # (B,)

        return delta_f.to(y.dtype)                    # (B,)

    def mlp_estimate(self, y: torch.Tensor,
                     pilots: torch.Tensor,
                     pilot_mask: torch.Tensor) -> torch.Tensor:
        """
        MLP-based CFO estimate from pilot observations.

        Args:
            y: (B, N_sc, N_sym, 2)
            pilots: (B, N_p, 2)
            pilot_mask: (N_sc, N_sym)

        Returns:
            delta_f: (B,) estimated CFO in cycles/symbol
        """
        y_p = y[:, pilot_mask, :]                     # (B, N_p, 2)
        # Flatten I/Q per pilot into a single vector per batch
        y_p_flat = y_p.reshape(y.shape[0], -1)        # (B, 2*N_p)
        delta_f = self.cfo_mlp(y_p_flat).squeeze(-1)  # (B,)
        return delta_f

    def forward(self, y: torch.Tensor,
                pilots: torch.Tensor,
                pilot_mask: torch.Tensor) -> torch.Tensor:
        """
        Estimate CFO from received signal.

        Args:
            y: (B, N_sc, N_sym, 2) — received IQ
            pilots: (B, N_p, 2) — known pilots
            pilot_mask: (N_sc, N_sym) — boolean pilot locations

        Returns:
            delta_f: (B,) estimated CFO in cycles/symbol

        Shape invariants:
            - dtype in {float32, bfloat16}; atan2 computed in float32 internally
            - Pilots should be normalized to unit power
        """
        if self.config.cfo_mode == "analytic":
            return self.analytic_estimate(y, pilots, pilot_mask)

        elif self.config.cfo_mode == "analytic_plus_learned":
            delta_f = self.analytic_estimate(y, pilots, pilot_mask)  # (B,)
            if self.config.cfo_learned_scale:
                delta_f = delta_f * self.cfo_scale
            if self.config.cfo_learned_bias:
                delta_f = delta_f + self.cfo_bias
            return delta_f

        elif self.config.cfo_mode == "mlp":
            return self.mlp_estimate(y, pilots, pilot_mask)

        else:
            raise ValueError(f"Unknown cfo_mode: {self.config.cfo_mode}")


# ──────────────────────────────────────────────────────────────────────────
# CFO Compensator Module
# ──────────────────────────────────────────────────────────────────────────

class CFOCompensator(nn.Module):
    """
    Apply CFO correction to received signal via phase rotation.

    Corrects the phase ramp across OFDM symbols caused by carrier
    frequency offset between the satellite and the IoT device.
    """

    def __init__(self, n_symbols: int):
        super().__init__()
        self.register_buffer('time_idx', torch.arange(n_symbols).float())

    def forward(self, y: torch.Tensor, delta_f: torch.Tensor) -> torch.Tensor:
        """
        Apply CFO correction.

        Args:
            y: (B, N_sc, N_sym, 2) — received IQ
            delta_f: (B,) CFO in cycles/symbol

        Returns:
            y_corrected: (B, N_sc, N_sym, 2) — phase-corrected IQ, residual NOT yet added
                         (no skip connection in CFO compensation)

        Shape invariants:
            - y.dtype in {float32, bfloat16}
            - delta_f must be on same device as y
        """
        # Phase rotation per symbol: phi(t) = exp(-j * 2*pi * delta_f * t)
        # time_idx: (N_sym,) -> broadcast to (1, 1, N_sym, 1)
        time = self.time_idx[None, None, :, None]         # (1, 1, N_sym, 1)
        delta_f_rs = delta_f[:, None, None, None]          # (B, 1, 1, 1)

        angle = -2.0 * math.pi * delta_f_rs * time         # (B, 1, N_sym, 1)

        cos_theta = torch.cos(angle)                       # (B, 1, N_sym, 1)
        sin_theta = torch.sin(angle)                       # (B, 1, N_sym, 1)

        # Complex rotation:
        # [real_out, imag_out] = [real*cos - imag*sin, real*sin + imag*cos]
        y_r, y_i = y[..., 0:1], y[..., 1:2]               # (B, N_sc, N_sym, 1)
        corrected_r = y_r * cos_theta - y_i * sin_theta    # (B, N_sc, N_sym, 1)
        corrected_i = y_r * sin_theta + y_i * cos_theta    # (B, N_sc, N_sym, 1)

        return torch.cat([corrected_r, corrected_i], dim=-1)  # (B, N_sc, N_sym, 2)


# ──────────────────────────────────────────────────────────────────────────
# Delay-Doppler Transform Module
# ──────────────────────────────────────────────────────────────────────────

class DelayDopplerTransform(nn.Module):
    """
    2D unitary transform between the time-frequency domain and the
    delay-Doppler domain.

    The LEO satellite channel is approximately sparse in the delay-Doppler
    domain (few propagation paths with distinct delays and Doppler shifts).
    This transform concentrates the channel energy into a few bins.

    Forward path:
        h (time-freq) -> IFFT over subcarriers (delay) -> FFT over symbols (Doppler) -> h_dd

    Inverse path:
        h_dd -> IFFT over symbols (Doppler) -> FFT over subcarriers (delay) -> h (time-freq)
    """

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Transform from time-frequency to delay-Doppler domain.

        Args:
            h: (B, N_sc, N_sym, 2) — channel in time-frequency domain,
               last dim = (real, imag)

        Returns:
            h_dd: (B, N_sc, N_sym, 2) — channel in delay-Doppler domain
        """
        # Convert to complex for FFT
        h_complex = torch.view_as_complex(h.contiguous())        # (B, N_sc, N_sym) complex

        # IFFT over subcarriers (freq -> delay)
        h_delay = torch.fft.ifft(h_complex, dim=1)               # (B, N_sc, N_sym) complex

        # FFT over symbols (time -> Doppler)
        h_dd_complex = torch.fft.fft(h_delay, dim=2)             # (B, N_sc, N_sym) complex

        # Convert back to (..., 2) real/imag representation
        return torch.view_as_real(h_dd_complex)                   # (B, N_sc, N_sym, 2)

    def inverse(self, h_dd: torch.Tensor) -> torch.Tensor:
        """
        Transform from delay-Doppler back to time-frequency domain.

        Args:
            h_dd: (B, N_sc, N_sym, 2) — channel in delay-Doppler domain

        Returns:
            h: (B, N_sc, N_sym, 2) — channel in time-frequency domain
        """
        h_complex = torch.view_as_complex(h_dd.contiguous())     # (B, N_sc, N_sym) complex

        # IFFT over symbols (Doppler -> time)
        h_doppler = torch.fft.ifft(h_complex, dim=2)             # (B, N_sc, N_sym) complex

        # FFT over subcarriers (delay -> freq)
        h_tf_complex = torch.fft.fft(h_doppler, dim=1)           # (B, N_sc, N_sym) complex

        return torch.view_as_real(h_tf_complex)                   # (B, N_sc, N_sym, 2)


# ──────────────────────────────────────────────────────────────────────────
# VAMP Linear Step Module
# ──────────────────────────────────────────────────────────────────────────

class VAMPLinearStep(nn.Module):
    """
    VAMP linear (LMMSE) channel estimation step.

    For the narrowband NB-IoT scenario (12 SCs), each resource element
    is treated independently. The LMMSE estimate is:

        h_lin = (gamma_z * |x|^2 + gamma_h)^{-1}
                * (gamma_z * conj(x) * y + gamma_h * r1 - u1)

    where:
        gamma_z = noise precision (inverse noise variance)
        gamma_h = channel prior precision (inverse channel variance)
        r1 = VAMP reference from denoiser branch
        u1 = Onsager correction from previous iteration

    Shape conventions:
        All tensors have last dim = 2 for (real, imag).
    """

    def forward(self,
                y: torch.Tensor,
                x_known: torch.Tensor,
                h_prev: torch.Tensor,
                r1: torch.Tensor,
                u1: torch.Tensor,
                gamma_z: torch.Tensor,
                gamma_h: torch.Tensor) -> torch.Tensor:
        """
        Compute LMMSE channel estimate.

        Args:
            y: (B, N_sc, N_sym, 2) — received signal (CFO-corrected)
            x_known: (B, N_sc, N_sym, 2) — known + soft estimated symbols
            h_prev: (B, N_sc, N_sym, 2) — previous channel estimate (unused in LMMSE
                    formula but carried for state consistency)
            r1: (B, N_sc, N_sym, 2) — VAMP reference (prior mean)
            u1: (B, N_sc, N_sym, 2) — VAMP Onsager correction
            gamma_z: scalar — noise precision
            gamma_h: scalar — channel prior precision

        Returns:
            h_lin: (B, N_sc, N_sym, 2) — LMMSE channel estimate, residual NOT yet added

        Shape invariants:
            - All input tensors must have the same shape (B, N_sc, N_sym, 2)
            - gamma_z, gamma_h must be positive scalars
            - dtype in {float32, bfloat16}; 1e-10 epsilon in denominator prevents division by zero
        """
        # Signal power |x|^2 per RE
        x_pow = x_known[..., 0]**2 + x_known[..., 1]**2       # (B, N_sc, N_sym)

        # conj(x) * y (complex multiply: a*conj(b) where we want conj(x)*y)
        # conj(x) = (x_re, -x_im), so conj(x) * y = (x_re*y_re + x_im*y_im, x_re*y_im - x_im*y_re)
        conj_x_times_y_real = x_known[..., 0] * y[..., 0] + x_known[..., 1] * y[..., 1]
        conj_x_times_y_imag = x_known[..., 0] * y[..., 1] - x_known[..., 1] * y[..., 0]

        # gamma_z * conj(x) * y
        gy_real = gamma_z * conj_x_times_y_real               # (B, N_sc, N_sym)
        gy_imag = gamma_z * conj_x_times_y_imag               # (B, N_sc, N_sym)

        # gamma_h * r1 - u1
        hu_real = gamma_h * r1[..., 0] - u1[..., 0]           # (B, N_sc, N_sym)
        hu_imag = gamma_h * r1[..., 1] - u1[..., 1]           # (B, N_sc, N_sym)

        # Numerator
        num_real = gy_real + hu_real                           # (B, N_sc, N_sym)
        num_imag = gy_imag + hu_imag                           # (B, N_sc, N_sym)

        # Denominator: gamma_z * |x|^2 + gamma_h
        denom = gamma_z * x_pow + gamma_h                      # (B, N_sc, N_sym)

        # Add epsilon for numerical safety
        denom_safe = denom + 1e-10

        # Result
        h_lin_real = num_real / denom_safe                     # (B, N_sc, N_sym)
        h_lin_imag = num_imag / denom_safe                     # (B, N_sc, N_sym)

        return torch.stack([h_lin_real, h_lin_imag], dim=-1)   # (B, N_sc, N_sym, 2)


# ──────────────────────────────────────────────────────────────────────────
# Soft MMSE Detector Module
# ──────────────────────────────────────────────────────────────────────────

class SoftMMSEDetector(nn.Module):
    """
    Soft MMSE data detector with LLR computation for QPSK.

    For each data resource element, performs:
    1. MMSE equalization
    2. Per-bit LLR computation (max-log approximation)
    3. Soft symbol reconstruction from LLRs (for iterative feedback)

    Shape conventions:
        LLRs: (B, N_sc, N_sym, n_bits) — per-bit log-likelihood ratios
        Soft symbols: (B, N_sc, N_sym, 2) — via tanh(LLR/2) mapped to QPSK
    """

    def __init__(self, config: DU_VAMP_JCD_Config):
        super().__init__()
        self.n_bits = config.n_bits_per_symbol
        # QPSK constellation scale: 1/sqrt(2) for unit average power
        self.const_scale = 1.0 / math.sqrt(2.0)

    def forward(self,
                y: torch.Tensor,
                h_hat: torch.Tensor,
                gamma_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft MMSE detection producing equalized symbols and LLRs.

        Args:
            y: (B, N_sc, N_sym, 2) — received signal (CFO-corrected)
            h_hat: (B, N_sc, N_sym, 2) — channel estimate
            gamma_z: scalar — noise precision

        Returns:
            x_soft: (B, N_sc, N_sym, 2) — soft symbol estimates, residual NOT yet added
            llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs

        Shape invariants:
            - y and h_hat must have matching shapes
            - dtype cast to float32 internally for tanh/division numerical safety
            - gamma_z must be positive; clamped via exp(log_noise_precision)
        """
        # Use float32 for numerical safety in LLR computation
        compute_dtype = torch.float32
        y = y.to(compute_dtype)
        h_hat = h_hat.to(compute_dtype)
        gamma_z = gamma_z.to(compute_dtype) if torch.is_tensor(gamma_z) else gamma_z

        # Channel power |h|^2
        h_pow = h_hat[..., 0]**2 + h_hat[..., 1]**2            # (B, N_sc, N_sym)

        # Noise variance
        noise_var = 1.0 / (gamma_z + 1e-10)

        # MMSE equalizer: x_hat = conj(h) * y / (|h|^2 + sigma^2)
        conj_h_times_y_real = h_hat[..., 0] * y[..., 0] + h_hat[..., 1] * y[..., 1]
        conj_h_times_y_imag = h_hat[..., 0] * y[..., 1] - h_hat[..., 1] * y[..., 0]

        denom = h_pow + noise_var                               # (B, N_sc, N_sym)
        x_eq_real = conj_h_times_y_real / denom                 # (B, N_sc, N_sym)
        x_eq_imag = conj_h_times_y_imag / denom                 # (B, N_sc, N_sym)

        # Effective post-equalization SNR per RE
        # SNR = |h|^2 / noise_var  (after MMSE, the signal power is reduced by |h|^2/(|h|^2+sigma^2))
        # For LLR computation using max-log: LLR = 2 * |h|^2 / noise_var * Re(x_eq) / sqrt(1 - |h|^2/(|h|^2+sigma^2))
        # Simplified for max-log: LLR = 2 * SNR_eff * d_min * Re(x_eq)
        # For QPSK with unit constellation: d_min = 2 * const_scale = sqrt(2)
        # LLR_b0 = 2 * sqrt(2) * |h|^2 / noise_var * x_eq_real  (for bit 0 = real sign)
        # LLR_b1 = 2 * sqrt(2) * |h|^2 / noise_var * x_eq_imag  (for bit 1 = imag sign)

        snr_eff = h_pow / (noise_var + 1e-10)                   # (B, N_sc, N_sym)

        # Max-log LLR for QPSK
        llr_scale = 2.0 * math.sqrt(2.0)
        llr_b0 = llr_scale * x_eq_real * snr_eff                # (B, N_sc, N_sym)
        llr_b1 = llr_scale * x_eq_imag * snr_eff                # (B, N_sc, N_sym)

        llrs = torch.stack([llr_b0, llr_b1], dim=-1)            # (B, N_sc, N_sym, 2)

        # Soft symbols from LLRs (for next iteration feedback)
        # QPSK: bits map to constellation points (+/- const_scale, +/- const_scale)
        # soft_b0 = E[b0] = tanh(LLR_b0 / 2)
        soft_b0 = torch.tanh(llr_b0 / 2.0)                      # (B, N_sc, N_sym)
        soft_b1 = torch.tanh(llr_b1 / 2.0)                      # (B, N_sc, N_sym)

        x_soft_real = self.const_scale * soft_b0                # (B, N_sc, N_sym)
        x_soft_imag = self.const_scale * soft_b1                # (B, N_sc, N_sym)

        x_soft = torch.stack([x_soft_real, x_soft_imag], dim=-1)  # (B, N_sc, N_sym, 2)

        # Cast back to input dtype
        x_soft = x_soft.to(y.dtype)
        llrs = llrs.to(y.dtype)

        return x_soft, llrs


# ──────────────────────────────────────────────────────────────────────────
# Main VAMP-JCD Receiver Model
# ──────────────────────────────────────────────────────────────────────────

class VAMPJCDReceiver(nn.Module):
    """
    Deep-Unfolded VAMP Joint Channel Estimation and Data Detection (JCD) Receiver.

    Architecture:
        T=5 unfolded iterations (configurable via n_unfolded_layers),
        each performing:
          1. CFO residual compensation
          2. VAMP linear step (LMMSE channel estimation)
          3. Delay-Doppler domain denoising (learnable shrinkage)
          4. VAMP state update (damping + Onsager correction)
          5. Soft data detection (MMSE equalizer -> LLRs)
          6. CFO residual estimation from pilot mismatch

    The architecture is a hybrid model-driven + data-driven design:
    - The VAMP structure encodes the physics (measurement model y = Hx + w)
    - Learnable parameters (shrinkage thresholds, precisions, damping)
      adapt the algorithm to the LEO NTN channel statistics

    Input shapes:
        y: (B, n_subcarriers, n_symbols, 2)  — received IQ samples
        pilots: (B, n_pilots, 2)  — known pilot symbols
        pilot_mask: (n_subcarriers, n_symbols)  — boolean pilot locations

    Output shapes:
        h_hat: (B, n_subcarriers, n_symbols, 2)  — channel estimate
        x_soft: (B, n_subcarriers, n_symbols, 2)  — soft symbol estimates
        llrs: (B, n_subcarriers, n_symbols, n_bits)  — LLRs for decoder
        state: dict of intermediate values for analysis

    Parameter count (default config): 24 learnable params
    """

    def __init__(self, config: DU_VAMP_JCD_Config):
        super().__init__()
        self.config = config
        self.T = config.n_unfolded_layers
        self.N_sc = config.n_subcarriers
        self.N_sym = config.n_symbols
        self.N_p = config.n_pilots

        # ── VAMP Learnable Parameters ────────────────────────────────

        # Noise precision gamma_z (inverse noise variance) — log-parametrized
        # for positivity: gamma_z = exp(log_noise_precision)
        self.log_noise_precision = nn.Parameter(
            torch.tensor(math.log(1.0 / 0.1))   # init: SNR ~ 10 dB
        )

        # Channel prior precision gamma_h — log-parametrized
        self.log_channel_precision = nn.Parameter(
            torch.tensor(math.log(1.0 / 1.0))   # init: unit variance
        )

        # Damping factor alpha in (0,1) via sigmoid of unconstrained logit
        n_damp = 1 if config.n_damping_params == 1 else self.T
        self.logit_damping = nn.Parameter(
            torch.full((n_damp,), 0.0)           # init: alpha = 0.5
        )

        # Onsager correction coefficient beta (unconstrained)
        n_onsager = 1 if config.n_onsager_params == 1 else self.T
        self.onsager_coeff = nn.Parameter(
            torch.full((n_onsager,), 0.0)        # init: beta = 0 (no correction)
        )

        # ── Sub-modules ───────────────────────────────────────────────
        self.denoiser = build_denoiser(config)
        self.cfo_estimator = CFOEstimator(config)
        self.cfo_compensator = CFOCompensator(self.N_sym)
        self.dd_transform = DelayDopplerTransform()
        self.vamp_linear = VAMPLinearStep()
        self.detector = SoftMMSEDetector(config)

    def _compute_precisions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positive noise and channel precisions from log params."""
        gamma_z = torch.exp(self.log_noise_precision)   # scalar, positive
        gamma_h = torch.exp(self.log_channel_precision)  # scalar, positive
        return gamma_z, gamma_h

    def _get_damping(self, layer_idx: int) -> torch.Tensor:
        """Get damping factor alpha in (0, 1) for the given layer."""
        idx = 0 if self.config.n_damping_params == 1 else layer_idx
        return torch.sigmoid(self.logit_damping[idx])

    def _get_onsager(self, layer_idx: int) -> torch.Tensor:
        """Get Onsager correction coefficient for the given layer."""
        idx = 0 if self.config.n_onsager_params == 1 else layer_idx
        return self.onsager_coeff[idx]

    def _vamp_state_update(self, h_lin: torch.Tensor, h_den: torch.Tensor,
                           h_prev: torch.Tensor,
                           layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAMP state update: damping + Onsager correction.

        Args:
            h_lin: (B, N_sc, N_sym, 2) — LMMSE estimate from linear step
            h_den: (B, N_sc, N_sym, 2) — denoised estimate
            h_prev: (B, N_sc, N_sym, 2) — previous channel estimate
            layer_idx: current iteration index

        Returns:
            h_new: (B, N_sc, N_sym, 2) — damped channel estimate
            r1_new: (B, N_sc, N_sym, 2) — new VAMP reference (with stop-gradient)
            u1_new: (B, N_sc, N_sym, 2) — new Onsager correction
        """
        alpha = self._get_damping(layer_idx)      # scalar

        # Damping: blend denoised and previous estimate
        h_new = alpha * h_den + (1.0 - alpha) * h_prev   # (B, N_sc, N_sym, 2)

        # Onsager correction for next iteration
        beta = self._get_onsager(layer_idx)       # scalar
        u1_new = beta * (h_den - h_lin)            # (B, N_sc, N_sym, 2)

        # New reference is damped estimate (with stop-gradient to prevent
        # gradient flow through the state iteration)
        r1_new = h_new.detach()

        return h_new, r1_new, u1_new

    def forward(self,
                y: torch.Tensor,
                pilots: torch.Tensor,
                pilot_mask: torch.Tensor,
                use_checkpoint: bool = False,
                return_full_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Full forward pass of the deep-unfolded VAMP-JCD receiver.

        Args:
            y: (B, N_sc, N_sym, 2) — received IQ samples
            pilots: (B, N_p, 2) — known pilot symbols
            pilot_mask: (N_sc, N_sym) — boolean pilot locations
            use_checkpoint: if True, use gradient checkpointing (minimal benefit
                           for 24-param model, but provided for completeness)
            return_full_state: if True, return intermediate per-iteration states

        Returns:
            h_hat: (B, N_sc, N_sym, 2) — final channel estimate, residual NOT yet added
            x_soft: (B, N_sc, N_sym, 2) — final soft symbol estimates
            llrs: (B, N_sc, N_sym, n_bits) — final per-bit LLRs
            state: dict with keys:
                - 'h_per_iter': list of channel estimates per iteration
                - 'cfo_per_iter': list of CFO estimates per iteration
                - 'x_soft_per_iter': list of soft symbols per iteration
                - 'gamma_z': noise precision used
                - 'gamma_h': channel prior precision used

        Shape invariants:
            - Batch B >= 1; N_sc == config.n_subcarriers; N_sym == config.n_symbols
            - dtype in {float32, bfloat16}; float16 not supported (atan2/tanh in LLR path)
            - IO buffer: pilot_mask must be (N_sc, N_sym) boolean tensor
        """
        B = y.shape[0]
        device = y.device
        dtype = y.dtype

        if use_checkpoint:
            # Note: gradient checkpointing is provided for API compatibility.
            # With only 24 parameters, the memory savings are negligible.
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, y, pilots, pilot_mask, return_full_state,
                use_reentrant=False
            )
        return self._forward_impl(y, pilots, pilot_mask, return_full_state)

    def _forward_impl(self, y: torch.Tensor, pilots: torch.Tensor,
                      pilot_mask: torch.Tensor,
                      return_full_state: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Internal forward implementation (separated for checkpointing support)."""
        B = y.shape[0]
        device = y.device
        dtype = y.dtype

        # ── Step 0: Initial CFO Estimate ────────────────────────────
        # Compute coarse CFO from pilot autocorrelation
        delta_f = self.cfo_estimator(y, pilots, pilot_mask)          # (B,)

        # Apply coarse CFO correction
        y_corrected = self.cfo_compensator(y, delta_f)               # (B, N_sc, N_sym, 2)

        # ── Initialize VAMP State ───────────────────────────────────
        h_hat = torch.zeros(B, self.N_sc, self.N_sym, 2,
                            device=device, dtype=dtype)               # (B, N_sc, N_sym, 2)
        x_soft = torch.zeros(B, self.N_sc, self.N_sym, 2,
                             device=device, dtype=dtype)              # (B, N_sc, N_sym, 2)

        # VAMP auxiliary variables
        r1 = torch.zeros_like(h_hat)                                  # (B, N_sc, N_sym, 2)
        u1 = torch.zeros_like(h_hat)                                  # (B, N_sc, N_sym, 2)

        # Known symbol matrix: pilots at pilot locations, soft symbols elsewhere
        x_known = torch.zeros_like(y)                                 # (B, N_sc, N_sym, 2)

        # Place known pilots (broadcast pilot_mask over batch dim)
        # pilot_mask is (N_sc, N_sym) -> index dims 1,2
        pilots_reshaped = pilots.view(B, -1, 2)                       # (B, N_p, 2)
        x_known[:, pilot_mask, :] = pilots_reshaped                   # (B, N_sc, N_sym, 2)

        # ── Compute Precision Parameters ────────────────────────────
        gamma_z, gamma_h = self._compute_precisions()                 # scalars

        # ── Unfolded Iterations ──────────────────────────────────────
        state = {
            "h_per_iter": [],
            "cfo_per_iter": [],
            "x_soft_per_iter": [],
        }

        for t in range(self.T):
            # ── a) CFO Residual Correction ──
            y_corrected = self.cfo_compensator(y, delta_f)           # (B, N_sc, N_sym, 2)

            # ── b) VAMP Linear Step (LMMSE Channel Estimate) ──
            h_lin = self.vamp_linear(
                y_corrected, x_known, h_hat, r1, u1, gamma_z, gamma_h
            )                                                        # (B, N_sc, N_sym, 2)

            # ── c) Delay-Doppler Denoising ──
            # Transform to delay-Doppler
            h_dd = self.dd_transform(h_lin)                          # (B, N_sc, N_sym, 2)

            # Apply shrinkage/denoising
            h_dd_denoised = self.denoiser(h_dd, t)                   # (B, N_sc, N_sym, 2)

            # Transform back to time-frequency
            h_den = self.dd_transform.inverse(h_dd_denoised)          # (B, N_sc, N_sym, 2)

            # ── d) VAMP State Update ──
            h_new, r1_new, u1_new = self._vamp_state_update(
                h_lin, h_den, h_hat, t
            )
            h_hat = h_new                                             # (B, N_sc, N_sym, 2)
            r1 = r1_new                                               # (B, N_sc, N_sym, 2)
            u1 = u1_new                                               # (B, N_sc, N_sym, 2)

            # ── e) Soft Data Detection ──
            if self.config.use_soft_feedback:
                x_soft_detected, llrs_t = self.detector(
                    y_corrected, h_hat, gamma_z
                )                                                    # x_soft: (B, N_sc, N_sym, 2)
                                                                     # llrs: (B, N_sc, N_sym, 2)

                # Update x_known: keep pilots fixed, update data REs
                x_known = x_soft_detected.clone()                    # (B, N_sc, N_sym, 2)
                # Re-insert known pilots
                x_known[:, pilot_mask, :] = pilots_reshaped
                x_soft = x_soft_detected

            # ── f) CFO Residual Update ──
            if self.config.cfo_mode != "analytic":
                # Compute residual CFO from current corrected signal
                delta_f_residual = self.cfo_estimator.analytic_estimate(
                    y_corrected, pilots, pilot_mask
                )                                                    # (B,)
                # Apply learned corrections if enabled
                if (self.config.cfo_mode == "analytic_plus_learned"
                        and self.config.cfo_learned_scale):
                    delta_f_residual = delta_f_residual * self.cfo_estimator.cfo_scale
                if (self.config.cfo_mode == "analytic_plus_learned"
                        and self.config.cfo_learned_bias):
                    delta_f_residual = delta_f_residual + self.cfo_estimator.cfo_bias
                # Accumulate
                delta_f = delta_f + delta_f_residual                  # (B,)

            # ── Logging (detached for memory efficiency) ──
            if return_full_state:
                state["h_per_iter"].append(h_hat.detach().cpu())
                state["cfo_per_iter"].append(delta_f.detach().cpu())
                if self.config.use_soft_feedback:
                    state["x_soft_per_iter"].append(x_soft.detach().cpu())

        # ── Final Detection ──────────────────────────────────────────
        x_final, llrs_final = self.detector(
            y_corrected, h_hat, gamma_z
        )

        state["gamma_z"] = gamma_z.detach().cpu() if torch.is_tensor(gamma_z) else gamma_z
        state["gamma_h"] = gamma_h.detach().cpu() if torch.is_tensor(gamma_h) else gamma_h

        return h_hat, x_final, llrs_final, state


# ──────────────────────────────────────────────────────────────────────────
# Parameter Count Helper
# ──────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> None:
    """Print total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
    return trainable


# ──────────────────────────────────────────────────────────────────────────
# Quick self-test (run with `python model.py`)
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    cfg = DU_VAMP_JCD_Config(
        n_unfolded_layers=5,
        weight_tying=True,
        denoiser_type="delay_doppler_shrinkage",
        cfo_mode="analytic_plus_learned",
        use_soft_feedback=True,
    )
    print(f"Config parameter budget: {cfg.n_trainable_params}")

    model = VAMPJCDReceiver(cfg).to(device)
    print(f"Model actual trainable params: {count_params(model)}")

    # Create synthetic input
    B = 4
    N_sc = cfg.n_subcarriers   # 12
    N_sym = cfg.n_symbols      # 14
    N_p = cfg.n_pilots         # 24

    # Received signal: (B, 12, 14, 2)
    y = torch.randn(B, N_sc, N_sym, 2, device=device)

    # Pilot symbols: (B, 24, 2)
    pilots = torch.randn(B, N_p, 2, device=device)
    pilots = pilots / (pilots.norm(dim=-1, keepdim=True) + 1e-10)  # normalize

    # Pilot mask: True at symbol indices 3 and 10 for all subcarriers
    pilot_mask = torch.zeros(N_sc, N_sym, dtype=torch.bool, device=device)
    pilot_mask[:, 3] = True
    pilot_mask[:, 10] = True

    # Forward pass
    with torch.no_grad():
        h_hat, x_soft, llrs, state = model(y, pilots, pilot_mask, return_full_state=True)

    print(f"h_hat:  {h_hat.shape}  (expected [{B}, {N_sc}, {N_sym}, 2])")
    print(f"x_soft: {x_soft.shape}  (expected [{B}, {N_sc}, {N_sym}, 2])")
    print(f"llrs:   {llrs.shape}   (expected [{B}, {N_sc}, {N_sym}, 2])")

    # Verify shapes
    assert h_hat.shape == (B, N_sc, N_sym, 2), f"Bad h_hat shape: {h_hat.shape}"
    assert x_soft.shape == (B, N_sc, N_sym, 2), f"Bad x_soft shape: {x_soft.shape}"
    assert llrs.shape == (B, N_sc, N_sym, cfg.n_bits_per_symbol), f"Bad llrs shape: {llrs.shape}"

    print(f"\n[OK] VAMPJCDReceiver forward pass successful on {device}")
    print(f"     Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params")
    print(f"     Config budget: {cfg.n_trainable_params} (should match)")
