"""
Baseline Receivers for LEO NTN IoT
====================================

Implements conventional receiver algorithms as baselines for comparison
against the deep-unfolded VAMP-JCD receiver.

Baselines:
  1. LMMSEBaseline — Linear Minimum Mean-Square Error receiver
     with pilot-aided channel estimation and interpolation
  2. LSBaseline — Least-Squares channel estimation with linear
     interpolation between pilots
  3. GenieBound — Ideal known-channel bound (channel is known perfectly)

All baselines use the same interface as VAMPJCDReceiver:
    forward(y, pilots, pilot_mask) -> (h_hat, x_soft, llrs, state)

Shape conventions:
    - All complex signals use (..., 2) last-dim = (real, imag)
    - y: (B, N_sc, N_sym, 2) — received IQ
    - pilots: (B, N_p, 2) — known pilots
    - pilot_mask: (N_sc, N_sym) — boolean pilot locations
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict


# ──────────────────────────────────────────────────────────────────────────
# LMMSE Baseline Receiver
# ──────────────────────────────────────────────────────────────────────────

class LMMSEBaseline(nn.Module):
    """
    Conventional pilot-aided LMMSE receiver for NB-IoT NPUSCH.

    Processing chain:
        1. LS channel estimation at pilot positions
        2. Linear interpolation across time (between pilot symbols)
        3. Linear interpolation across frequency (between subcarriers)
        4. MMSE data detection per RE

    This is the primary baseline — the deep-unfolded VAMP-JCD receiver
    must outperform this at low SNR.

    Shape conventions:
        h_hat: (B, N_sc, N_sym, 2) — channel estimate
        x_soft: (B, N_sc, N_sym, 2) — soft symbol estimates
        llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs
    """

    def __init__(self, config):
        super().__init__()
        self.N_sc = config.n_subcarriers
        self.N_sym = config.n_symbols
        self.N_p = config.n_pilots
        self.n_bits = config.n_bits_per_symbol
        self.pilot_symbol_indices = config.pilot_symbol_indices
        self.const_scale = 1.0 / math.sqrt(2.0)

    def _ls_channel_estimate(self, y: torch.Tensor,
                               pilots: torch.Tensor,
                               pilot_mask: torch.Tensor) -> torch.Tensor:
        """
        Least-squares channel estimate at pilot positions.

        At each pilot RE: h_ls = conj(x_p) * y_p / |x_p|^2

        Args:
            y: (B, N_sc, N_sym, 2) — received signal
            pilots: (B, N_p, 2) — known pilots
            pilot_mask: (N_sc, N_sym) — boolean mask

        Returns:
            h_pilot_est: (B, N_sc, N_sym, 2) — LS channel estimate
                         (only valid at pilot positions; zeros elsewhere)
        """
        B = y.shape[0]

        # Extract pilot observations
        y_p = y[:, pilot_mask, :]                                  # (B, N_p, 2)

        # LS estimate: h_ls = conj(x_p) * y_p / |x_p|^2
        # conj(x) * y: same as complex multiply with conj of x
        pilot_power = pilots[..., 0]**2 + pilots[..., 1]**2        # (B, N_p)

        ls_real = (pilots[..., 0] * y_p[..., 0] + pilots[..., 1] * y_p[..., 1]) / (pilot_power + 1e-10)
        ls_imag = (pilots[..., 0] * y_p[..., 1] - pilots[..., 1] * y_p[..., 0]) / (pilot_power + 1e-10)

        # Place LS estimates at pilot positions
        h_est = torch.zeros(B, self.N_sc, self.N_sym, 2, device=y.device, dtype=y.dtype)
        h_est[:, pilot_mask, :] = torch.stack([ls_real, ls_imag], dim=-1)

        return h_est                                              # (B, N_sc, N_sym, 2)

    def _interpolate_channel(self, h_pilot: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation of channel estimates across time and frequency.

        Steps:
            1. Interpolate across time (between pilot symbols)
            2. Interpolate across frequency (between subcarriers)

        Args:
            h_pilot: (B, N_sc, N_sym, 2) — LS estimates at pilot positions

        Returns:
            h_interp: (B, N_sc, N_sym, 2) — interpolated channel estimate
        """
        B, N_sc, N_sym, _ = h_pilot.shape
        device = h_pilot.device
        dtype = h_pilot.dtype
        p0, p1 = self.pilot_symbol_indices

        # ── Step 1: Time-domain interpolation ──
        # Interpolate linearly between pilot symbol p0 and p1
        # For symbols before p0: hold p0 value
        # For symbols after p1: hold p1 value
        h_time = h_pilot.clone()                                   # (B, N_sc, N_sym, 2)

        if p1 > p0 + 1:
            # Linear interpolation between p0 and p1
            n_interp = p1 - p0 - 1
            # Weight for each interpolated symbol
            for k in range(1, n_interp + 1):
                alpha = k / (p1 - p0)
                h_time[:, :, p0 + k, 0] = ((1 - alpha) * h_pilot[:, :, p0, 0]
                                            + alpha * h_pilot[:, :, p1, 0])
                h_time[:, :, p0 + k, 1] = ((1 - alpha) * h_pilot[:, :, p0, 1]
                                            + alpha * h_pilot[:, :, p1, 1])

        # Extrapolate before first pilot and after last pilot
        for sym in range(p0):
            h_time[:, :, sym, :] = h_pilot[:, :, p0, :]           # hold first pilot
        for sym in range(p1 + 1, N_sym):
            h_time[:, :, sym, :] = h_pilot[:, :, p1, :]           # hold last pilot

        # ── Step 2: Frequency-domain interpolation ──
        # Since all subcarriers have pilots (NPUSCH Format 1), there's no
        # frequency interpolation needed — all SCs already have estimates.
        # This is a no-op for NB-IoT NPUSCH.

        return h_time                                              # (B, N_sc, N_sym, 2)

    def _mmse_detection(self, y: torch.Tensor,
                        h_hat: torch.Tensor,
                        noise_var: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MMSE data detection with LLR computation.

        Args:
            y: (B, N_sc, N_sym, 2) — received signal
            h_hat: (B, N_sc, N_sym, 2) — channel estimate
            noise_var: scalar — estimated noise variance

        Returns:
            x_soft: (B, N_sc, N_sym, 2) — soft symbols
            llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs
        """
        # Compute in float32 for numerical safety
        y = y.float()
        h_hat = h_hat.float()

        h_pow = h_hat[..., 0]**2 + h_hat[..., 1]**2                # (B, N_sc, N_sym)

        # MMSE equalizer
        conj_h_times_y_real = h_hat[..., 0] * y[..., 0] + h_hat[..., 1] * y[..., 1]
        conj_h_times_y_imag = h_hat[..., 0] * y[..., 1] - h_hat[..., 1] * y[..., 0]

        denom = h_pow + noise_var
        x_eq_real = conj_h_times_y_real / denom
        x_eq_imag = conj_h_times_y_imag / denom

        # LLR computation (max-log)
        snr_eff = h_pow / (noise_var + 1e-10)
        llr_scale = 2.0 * math.sqrt(2.0)

        llr_b0 = llr_scale * x_eq_real * snr_eff
        llr_b1 = llr_scale * x_eq_imag * snr_eff

        llrs = torch.stack([llr_b0, llr_b1], dim=-1)               # (B, N_sc, N_sym, 2)

        # Soft symbols
        soft_b0 = torch.tanh(llr_b0 / 2.0)
        soft_b1 = torch.tanh(llr_b1 / 2.0)

        x_soft_real = self.const_scale * soft_b0
        x_soft_imag = self.const_scale * soft_b1
        x_soft = torch.stack([x_soft_real, x_soft_imag], dim=-1)

        return x_soft.to(y.dtype), llrs.to(y.dtype)

    def forward(self, y: torch.Tensor, pilots: torch.Tensor,
                pilot_mask: torch.Tensor,
                noise_var: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Full LMMSE receiver forward pass.

        Args:
            y: (B, N_sc, N_sym, 2) — received IQ
            pilots: (B, N_p, 2) — known pilot symbols
            pilot_mask: (N_sc, N_sym) — boolean pilot locations
            noise_var: estimated noise variance (can be tuned)

        Returns:
            h_hat: (B, N_sc, N_sym, 2) — channel estimate, residual NOT yet added
            x_soft: (B, N_sc, N_sym, 2) — soft symbols
            llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs
            state: dict with 'method' = 'LMMSE'

        Shape invariants:
            - y.dtype in {float32, bfloat16}; float32 used internally for LLR computation
        """
        h_ls = self._ls_channel_estimate(y, pilots, pilot_mask)    # (B, N_sc, N_sym, 2)
        h_hat = self._interpolate_channel(h_ls)                     # (B, N_sc, N_sym, 2)

        x_soft, llrs = self._mmse_detection(y, h_hat, noise_var)

        return h_hat, x_soft, llrs, {"method": "LMMSE"}


# ──────────────────────────────────────────────────────────────────────────
# LS Baseline Receiver
# ──────────────────────────────────────────────────────────────────────────

class LSBaseline(nn.Module):
    """
    Least-Squares receiver with linear pilot interpolation.

    Simpler than LMMSE — does not use noise statistics. Provided as
    an additional baseline and diagnostic.

    Processing chain:
        1. LS channel estimation at pilot positions
        2. Linear interpolation (same as LMMSE)
        3. ZF data detection
    """

    def __init__(self, config):
        super().__init__()
        self.N_sc = config.n_subcarriers
        self.N_sym = config.n_symbols
        self.n_bits = config.n_bits_per_symbol
        self.pilot_symbol_indices = config.pilot_symbol_indices
        self.const_scale = 1.0 / math.sqrt(2.0)

    def _ls_channel_estimate(self, y, pilots, pilot_mask):
        """Same as LMMSE._ls_channel_estimate."""
        B = y.shape[0]
        y_p = y[:, pilot_mask, :]
        pilot_power = pilots[..., 0]**2 + pilots[..., 1]**2

        ls_real = (pilots[..., 0] * y_p[..., 0] + pilots[..., 1] * y_p[..., 1]) / (pilot_power + 1e-10)
        ls_imag = (pilots[..., 0] * y_p[..., 1] - pilots[..., 1] * y_p[..., 0]) / (pilot_power + 1e-10)

        h_est = torch.zeros(B, self.N_sc, self.N_sym, 2, device=y.device, dtype=y.dtype)
        h_est[:, pilot_mask, :] = torch.stack([ls_real, ls_imag], dim=-1)
        return h_est

    def _interpolate_channel(self, h_pilot):
        """Linear interpolation (same as LMMSE)."""
        B, N_sc, N_sym, _ = h_pilot.shape
        p0, p1 = self.pilot_symbol_indices
        h_time = h_pilot.clone()

        if p1 > p0 + 1:
            for k in range(1, p1 - p0):
                alpha = k / (p1 - p0)
                h_time[:, :, p0 + k] = ((1 - alpha) * h_pilot[:, :, p0, :]
                                        + alpha * h_pilot[:, :, p1, :])

        for sym in range(p0):
            h_time[:, :, sym, :] = h_pilot[:, :, p0, :]
        for sym in range(p1 + 1, N_sym):
            h_time[:, :, sym, :] = h_pilot[:, :, p1, :]

        return h_time

    def _zf_detection(self, y, h_hat):
        """
        Zero-forcing data detection with LLR computation.

        Args:
            y: (B, N_sc, N_sym, 2) — received signal
            h_hat: (B, N_sc, N_sym, 2) — channel estimate

        Returns:
            x_soft: (B, N_sc, N_sym, 2) — soft symbols
            llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs
        """
        y = y.float()
        h_hat = h_hat.float()

        h_pow = h_hat[..., 0]**2 + h_hat[..., 1]**2 + 1e-10

        # ZF equalizer: x = conj(h) * y / |h|^2
        x_eq_real = (h_hat[..., 0] * y[..., 0] + h_hat[..., 1] * y[..., 1]) / h_pow
        x_eq_imag = (h_hat[..., 0] * y[..., 1] - h_hat[..., 1] * y[..., 0]) / h_pow

        # LLRs via post-equalization SNR = |h|^2 / sigma^2 (approximate)
        # Since we don't estimate noise here, use a heuristic scaling
        llr_b0 = 4.0 * x_eq_real
        llr_b1 = 4.0 * x_eq_imag
        llrs = torch.stack([llr_b0, llr_b1], dim=-1)

        soft_b0 = torch.tanh(llr_b0 / 2.0)
        soft_b1 = torch.tanh(llr_b1 / 2.0)
        x_soft = torch.stack([self.const_scale * soft_b0,
                              self.const_scale * soft_b1], dim=-1)

        return x_soft.to(y.dtype), llrs.to(y.dtype)

    def forward(self, y, pilots, pilot_mask, noise_var=0.1):
        """
        Forward pass: LS channel estimation + ZF detection.

        Args:
            y: (B, N_sc, N_sym, 2)
            pilots: (B, N_p, 2)
            pilot_mask: (N_sc, N_sym)
            noise_var: unused (kept for API compatibility)

        Returns:
            h_hat, x_soft, llrs, state
        """
        h_ls = self._ls_channel_estimate(y, pilots, pilot_mask)
        h_hat = self._interpolate_channel(h_ls)
        x_soft, llrs = self._zf_detection(y, h_hat)
        return h_hat, x_soft, llrs, {"method": "LS_ZF"}


# ──────────────────────────────────────────────────────────────────────────
# Genie-Aided Bound (Ideal Known Channel)
# ──────────────────────────────────────────────────────────────────────────

class GenieBound(nn.Module):
    """
    Ideal genie-aided bound: the receiver knows the true channel perfectly.

    This provides the upper bound on performance. No real receiver can
    outperform this bound. The gap between GenieBound and any proposed
    receiver measures the channel estimation loss.

    Uses the exact channel for MMSE detection, with noise variance
    set to the true value.
    """

    def __init__(self, config):
        super().__init__()
        self.n_bits = config.n_bits_per_symbol
        self.const_scale = 1.0 / math.sqrt(2.0)

    def forward(self, y: torch.Tensor, h_true: torch.Tensor,
                noise_var: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Genie-aided detection with perfect channel knowledge.

        Args:
            y: (B, N_sc, N_sym, 2) — received signal
            h_true: (B, N_sc, N_sym, 2) — true channel (from simulator)
            noise_var: true noise variance

        Returns:
            x_soft: (B, N_sc, N_sym, 2) — soft symbols
            llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs
            h_hat: same as h_true (the "estimate" is perfect)
            state: dict with method info

        Shape invariants:
            - y and h_true must have matching shapes (B, N_sc, N_sym, 2)
            - dtype in {float32, bfloat16}; float32 used internally for LLR computation
        """
        y = y.float()
        h_true = h_true.float()

        h_pow = h_true[..., 0]**2 + h_true[..., 1]**2               # (B, N_sc, N_sym)

        # MMSE equalizer with known channel
        conj_h_times_y_real = h_true[..., 0] * y[..., 0] + h_true[..., 1] * y[..., 1]
        conj_h_times_y_imag = h_true[..., 0] * y[..., 1] - h_true[..., 1] * y[..., 0]

        denom = h_pow + noise_var
        x_eq_real = conj_h_times_y_real / denom
        x_eq_imag = conj_h_times_y_imag / denom

        # LLR computation
        snr_eff = h_pow / (noise_var + 1e-10)
        llr_scale = 2.0 * math.sqrt(2.0)

        llr_b0 = llr_scale * x_eq_real * snr_eff
        llr_b1 = llr_scale * x_eq_imag * snr_eff
        llrs = torch.stack([llr_b0, llr_b1], dim=-1)

        # Soft symbols
        soft_b0 = torch.tanh(llr_b0 / 2.0)
        soft_b1 = torch.tanh(llr_b1 / 2.0)
        x_soft = torch.stack([self.const_scale * soft_b0,
                              self.const_scale * soft_b1], dim=-1)

        return h_true, x_soft.to(y.dtype), llrs.to(y.dtype), {"method": "GENIE"}


# ──────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from channel import LEONTNChannelSimulator
    from dataclasses import dataclass

    @dataclass
    class _TestConfig:
        n_subcarriers: int = 12
        n_symbols: int = 14
        subcarrier_spacing_hz: float = 15000.0
        n_pilots: int = 24
        pilot_symbol_indices: tuple = (3, 10)
        n_bits_per_symbol: int = 2

    cfg = _TestConfig()
    sim = LEONTNChannelSimulator(cfg)

    # Generate test data
    batch = sim.generate_batch(batch_size=4, elevation_deg=45.0,
                                snr_db=0.0, doppler_hz=1200.0, seed=42)

    y = batch["y"]
    pilots = batch["pilots"]
    pilot_mask = batch["pilot_mask"]

    # Test LMMSE baseline
    lmmse = LMMSEBaseline(cfg)
    h_lmmse, x_lmmse, llrs_lmmse, _ = lmmse(y, pilots, pilot_mask)
    assert h_lmmse.shape == (4, 12, 14, 2), f"Bad LMMSE h shape: {h_lmmse.shape}"
    assert llrs_lmmse.shape == (4, 12, 14, 2), f"Bad LMMSE llrs shape: {llrs_lmmse.shape}"

    # Test LS baseline
    ls = LSBaseline(cfg)
    h_ls, x_ls, llrs_ls, _ = ls(y, pilots, pilot_mask)
    assert h_ls.shape == (4, 12, 14, 2), f"Bad LS h shape: {h_ls.shape}"

    # Test Genie bound
    genie = GenieBound(cfg)
    h_genie, x_genie, llrs_genie, _ = genie(y, batch["h_true"], noise_var=1.0)
    assert llrs_genie.shape == (4, 12, 14, 2), f"Bad genie llrs shape: {llrs_genie.shape}"

    print("[OK] All baselines produce correct shapes")
    print(f"     LMMSE:   {h_lmmse.shape}")
    print(f"     LS:      {h_ls.shape}")
    print(f"     Genie:   {llrs_genie.shape}")
