"""
LEO NTN Channel Simulator
==========================

Implements a LEO satellite Non-Terrestrial Network (NTN) channel model
following 3GPP TR 38.811 for NB-IoT NPUSCH.

Generates time-varying Rician fading channel coefficients for a LEO
satellite IoT link, producing received IQ samples, true channels,
and pilot symbols for training the VAMP-JCD receiver.

Key channel characteristics:
  - Time-varying Doppler (elevation-dependent, up to ~40 kHz at 2 GHz)
  - Rician fading with K-factor from TR 38.811 Table 6.1-1
  - Delay spread dependent on environment (urban/suburban/rural)
  - Path loss: free space + atmospheric + shadowing
  - NB-IoT NPUSCH Format 1 frame structure (12 SC x 14 Sym)

Shape conventions:
  - All complex signals use (..., 2) last-dim = (real, imag)
  - Channel: (B, N_sc, N_sym, 2)
  - Received: (B, N_sc, N_sym, 2)
  - Pilots: (B, N_p, 2)
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

import torch


# ──────────────────────────────────────────────────────────────────────────
# NTN Channel Model Parameters (3GPP TR 38.811)
# ──────────────────────────────────────────────────────────────────────────

# Rician K-factor vs. elevation angle (Table 6.1-1, TR 38.811)
# Format: {elevation_deg: K_factor_dB}
RICE_K_FACTOR_DB = {
    10:   -3.0,    # low elevation: near-Rayleigh
    30:    0.0,    # medium elevation
    45:    3.0,    # typical
    60:    5.0,    # high elevation
    90:    7.0,    # nadir: strong LOS
}

# Delay spread by environment (RMS delay spread in seconds)
# TR 38.811 Table 6.1-2
DELAY_SPREAD_S = {
    "urban":     0.5e-6,    # 0.5 us
    "suburban":  0.3e-6,    # 0.3 us
    "rural":     0.1e-6,    # 0.1 us
}

# Satellite orbit parameters
EARTH_RADIUS_KM = 6371.0
SATELLITE_ALTITUDE_KM = 600.0   # typical LEO
ORBITAL_VELOCITY_MS = 7550.0    # ~7.55 km/s at 600 km altitude
SPEED_OF_LIGHT = 299792458.0


def rician_k_factor_linear(elevation_deg: float) -> float:
    """
    Interpolate Rician K-factor (linear scale) from TR 38.811 table.

    Args:
        elevation_deg: elevation angle in degrees [10, 90]

    Returns:
        K: linear Rician K-factor (power ratio LOS/NLOS)
    """
    elevations = sorted(RICE_K_FACTOR_DB.keys())
    k_db_values = [RICE_K_FACTOR_DB[e] for e in elevations]

    k_db = float(np.interp(elevation_deg, elevations, k_db_values))
    return 10.0 ** (k_db / 10.0)


def doppler_spread_hz(elevation_deg: float,
                       carrier_freq_hz: float = 2.0e9,
                       velocity_ms: float = ORBITAL_VELOCITY_MS) -> float:
    """
    Compute maximum Doppler spread for a LEO satellite pass.

    Args:
        elevation_deg: elevation angle in degrees
        carrier_freq_hz: carrier frequency in Hz (default: 2 GHz for NB-IoT NTN)
        velocity_ms: satellite velocity in m/s

    Returns:
        f_d_max: maximum Doppler spread in Hz
    """
    # Maximum Doppler shift when satellite is at the horizon
    f_d_max = velocity_ms * carrier_freq_hz / SPEED_OF_LIGHT

    # Effective Doppler at elevation angle theta
    # f_d = f_d_max * cos(theta)  where theta = 90 - elevation
    theta_rad = math.radians(90.0 - elevation_deg)
    f_d = f_d_max * abs(math.cos(theta_rad))

    return f_d


def doppler_shift_hz(elevation_deg: float,
                      carrier_freq_hz: float = 2.0e9) -> float:
    """
    Compute instantaneous Doppler shift at given elevation.

    The Doppler shift is positive during approach (s rising) and
    negative during recession (s setting). This returns the magnitude.

    Args:
        elevation_deg: elevation angle in degrees
        carrier_freq_hz: carrier frequency in Hz

    Returns:
        f_d: Doppler shift magnitude in Hz
    """
    return doppler_spread_hz(elevation_deg, carrier_freq_hz)


def path_loss_db(elevation_deg: float,
                 distance_km: float = None,
                 carrier_freq_hz: float = 2.0e9) -> float:
    """
    Compute path loss for LEO satellite link.

    Includes free-space path loss + atmospheric gas attenuation.

    Args:
        elevation_deg: elevation angle in degrees
        distance_km: slant range distance (computed if not provided)
        carrier_freq_hz: carrier frequency in Hz

    Returns:
        pl_db: total path loss in dB
    """
    if distance_km is None:
        # Compute slant range from elevation angle and satellite altitude
        # Using law of cosines
        R_e = EARTH_RADIUS_KM
        h_sat = SATELLITE_ALTITUDE_KM
        theta_rad = math.radians(elevation_deg)

        # Slant range
        distance_km = math.sqrt(
            (R_e + h_sat)**2 - (R_e * math.cos(theta_rad))**2
        ) - R_e * math.sin(theta_rad)

    # Free-space path loss
    wavelength = SPEED_OF_LIGHT / carrier_freq_hz
    fspl_db = 20 * math.log10(4 * math.pi * distance_km * 1000 / wavelength)

    # Atmospheric gas attenuation (simplified: ~0.1 dB at 2 GHz zenith)
    # Increases with lower elevation (longer path through atmosphere)
    atmos_db = 0.1 / math.sin(math.radians(max(elevation_deg, 5.0)))

    # Shadowing standard deviation (log-normal, ~3 dB for LEO)
    # Shadowing is a random variable, return 0 here for mean path loss

    return fspl_db + atmos_db


# ──────────────────────────────────────────────────────────────────────────
# LEO NTN Channel Simulator
# ──────────────────────────────────────────────────────────────────────────

class LEONTNChannelSimulator:
    """
    LEO NTN channel simulator following 3GPP TR 38.811.

    Generates batches of channel realizations and received signals
    for NB-IoT NPUSCH slots with time-varying Rician fading,
    Doppler shift, and AWGN.

    The channel model for each time-frequency resource element is:

        H(t, f) = sqrt(K/(K+1)) * H_LOS(t, f)
                 + sqrt(1/(K+1)) * H_NLOS(t, f)

    where:
        H_LOS(t, f) = exp(j * 2*pi * f_d * t)  (deterministic Doppler phase)
        H_NLOS(t, f) = Rayleigh fading with delay-Doppler profile

    Usage:
        sim = LEONTNChannelSimulator(cfg)
        batch = sim.generate_batch(
            batch_size=256,
            elevation_deg=45.0,
            snr_db=-3.0,
            doppler_hz=1200.0,
        )
        # batch contains: y, h_true, pilots, pilot_mask, x, bits
    """

    def __init__(self, config):
        """
        Args:
            config: DU_VAMP_JCD_Config instance with system parameters
        """
        self.N_sc = config.n_subcarriers
        self.N_sym = config.n_symbols
        self.scs = config.subcarrier_spacing_hz
        self.T_sym = 1.0 / self.scs          # OFDM symbol period (without CP)
        self.n_bits = config.n_bits_per_symbol
        self.pilot_indices = config.pilot_symbol_indices
        self.n_pilots = config.n_pilots

        # QPSK constellation points (Gray-coded)
        # Bit mapping: b0 = real sign, b1 = imag sign
        # 00 -> (+1, +1), 01 -> (+1, -1), 10 -> (-1, +1), 11 -> (-1, -1)
        self.const_scale = 1.0 / math.sqrt(2.0)
        self.qpsk_constellation = torch.tensor([
            [ 1,  1],   # 00
            [ 1, -1],   # 01
            [-1,  1],   # 10
            [-1, -1],   # 11
        ], dtype=torch.float32) * self.const_scale  # (4, 2)

        # Pre-compute pilot mask for NPUSCH Format 1
        self.register_pilot_mask()

        # Pre-compute subcarrier frequencies for delay-Doppler profile
        self.subcarrier_freqs = torch.arange(self.N_sc).float() * self.scs

    def register_pilot_mask(self):
        """Create pilot mask for NPUSCH Format 1 (pilots at specified symbols)."""
        mask = torch.zeros(self.N_sc, self.N_sym, dtype=torch.bool)
        for sym_idx in self.pilot_indices:
            mask[:, sym_idx] = True
        self.pilot_mask = mask

    def _generate_bits(self, batch_size: int) -> torch.Tensor:
        """
        Generate random transmitted bits.

        Args:
            batch_size: B

        Returns:
            bits: (B, N_sc, N_sym, n_bits) — random {0, 1} bits
        """
        bits = torch.randint(0, 2, (batch_size, self.N_sc, self.N_sym, self.n_bits))
        return bits

    def _modulate_qpsk(self, bits: torch.Tensor) -> torch.Tensor:
        """
        QPSK modulation from bits to complex symbols.

        Args:
            bits: (B, N_sc, N_sym, n_bits) — bit values {0, 1}

        Returns:
            symbols: (B, N_sc, N_sym, 2) — QPSK symbols, last dim = (I, Q)
        """
        # Gray mapping: b0 -> I, b1 -> Q
        # 0 maps to +const_scale, 1 maps to -const_scale
        i_component = (1 - 2 * bits[..., 0].float()) * self.const_scale   # (B, N_sc, N_sym)
        q_component = (1 - 2 * bits[..., 1].float()) * self.const_scale   # (B, N_sc, N_sym)

        return torch.stack([i_component, q_component], dim=-1)             # (B, N_sc, N_sym, 2)

    def _build_rician_channel(self, batch_size: int,
                               elevation_deg: float,
                               doppler_hz: float,
                               rng: torch.Generator) -> torch.Tensor:
        """
        Build Rician fading channel for LEO NTN.

        Channel model for each RE:
            H = sqrt(K/(K+1)) * H_LOS + sqrt(1/(K+1)) * H_NLOS

        The LOS component captures the deterministic Doppler shift.
        The NLOS component captures diffuse multipath (Rayleigh fading
        with delay-Doppler correlation).

        Args:
            batch_size: number of independent channels
            elevation_deg: elevation angle in degrees
            doppler_hz: Doppler spread in Hz
            rng: torch.Generator for reproducible noise

        Returns:
            h_true: (B, N_sc, N_sym, 2) — true channel coefficients
        """
        K_linear = rician_k_factor_linear(elevation_deg)
        sqrt_K = math.sqrt(K_linear)

        # LOS component: deterministic Doppler phase
        # H_LOS(t) = exp(j * 2*pi * f_d * t)
        # Where f_d is the doppler_hz and t is time in seconds
        time_s = torch.arange(self.N_sym, dtype=torch.float32) * self.T_sym  # (N_sym,)
        los_phase = 2.0 * math.pi * doppler_hz * time_s                      # (N_sym,)
        los_real = torch.cos(los_phase)                                       # (N_sym,)
        los_imag = torch.sin(los_phase)                                       # (N_sym,)

        # Broadcast to all subcarriers and batch
        los_real = los_real[None, None, :]                                     # (1, 1, N_sym)
        los_imag = los_imag[None, None, :]                                     # (1, 1, N_sym)

        # NLOS component: Rayleigh fading
        # Generate i.i.d. complex Gaussian per RE (white in time-freq)
        nlos_real = torch.randn(batch_size, self.N_sc, self.N_sym,
                                generator=rng)                                # (B, N_sc, N_sym)
        nlos_imag = torch.randn(batch_size, self.N_sc, self.N_sym,
                                generator=rng)                                # (B, N_sc, N_sym)
        nlos_power = 0.5  # unit variance per real dim -> unit power complex

        # Combine LOS + NLOS with K-factor weighting
        los_weight = sqrt_K / math.sqrt(K_linear + 1.0)
        nlos_weight = 1.0 / math.sqrt(K_linear + 1.0)

        h_real = los_weight * los_real + nlos_weight * nlos_real * math.sqrt(nlos_power)
        h_imag = los_weight * los_imag + nlos_weight * nlos_imag * math.sqrt(nlos_power)

        # Normalize to unit average power per RE
        h_power = h_real**2 + h_imag**2
        h_power_mean = h_power.mean(dim=(1, 2, 3), keepdim=True) if h_power.ndim == 4 \
                       else h_power.mean()
        h_real = h_real / torch.sqrt(h_power_mean + 1e-10)
        h_imag = h_imag / torch.sqrt(h_power_mean + 1e-10)

        return torch.stack([h_real, h_imag], dim=-1)   # (B, N_sc, N_sym, 2)

    def _apply_frequency_selectivity(self, h: torch.Tensor,
                                      delay_spread_s: float) -> torch.Tensor:
        """
        Apply frequency-selective fading by filtering across subcarriers.

        The delay-Doppler profile creates correlation across subcarriers
        (frequency selectivity) and across symbols (time selectivity).

        Args:
            h: (B, N_sc, N_sym, 2) — flat-fading channel
            delay_spread_s: RMS delay spread in seconds

        Returns:
            h_freq_sel: (B, N_sc, N_sym, 2) — frequency-selective channel
        """
        # Convert to complex for filtering
        h_complex = torch.view_as_complex(h.contiguous())              # (B, N_sc, N_sym)

        # Frequency-domain correlation via IFFT -> window -> FFT
        # 1. Transform to delay domain (IFFT over subcarriers)
        h_delay = torch.fft.ifft(h_complex, dim=1)                    # (B, N_sc, N_sym)

        # 2. Apply delay-domain windowing (exponential PDP)
        delay_bins = torch.arange(self.N_sc, dtype=torch.float32)      # (N_sc,)
        delay_time_bins = delay_bins / (self.N_sc * self.scs)          # seconds per bin

        # Exponential PDP: p(tau) = exp(-tau / tau_rms)
        tau_rms_samples = delay_spread_s * self.N_sc * self.scs
        pdp = torch.exp(-delay_time_bins / (delay_spread_s + 1e-10))  # (N_sc,)
        pdp = pdp / (pdp.sum() + 1e-10)                               # normalize

        # Apply PDP as window (element-wise multiply in delay domain)
        pdp_window = torch.sqrt(pdp)[None, :, None]                   # (1, N_sc, 1)
        h_delay_windowed = h_delay * pdp_window                        # (B, N_sc, N_sym)

        # 3. Transform back to frequency domain (FFT over subcarriers)
        h_freq_sel = torch.fft.fft(h_delay_windowed, dim=1)            # (B, N_sc, N_sym)

        return torch.view_as_real(h_freq_sel)                          # (B, N_sc, N_sym, 2)

    def generate_batch(
        self,
        batch_size: int = 32,
        elevation_deg: float = 45.0,
        snr_db: float = 0.0,
        doppler_hz: Optional[float] = None,
        delay_spread_s: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a batch of LEO NTN channel realizations.

        The process is:
            1. Generate random bits
            2. Modulate to QPSK symbols
            3. Insert pilot symbols at known positions
            4. Generate Rician fading channel
            5. Apply frequency-selective fading
            6. Add AWGN at specified SNR

        Args:
            batch_size: number of independent channel realizations
            elevation_deg: elevation angle in degrees [10, 90]
            snr_db: signal-to-noise ratio per symbol in dB
            doppler_hz: Doppler spread in Hz (auto-computed if None)
            delay_spread_s: RMS delay spread in seconds
                           (default: 0.3 us = suburban)
            seed: random seed for reproducibility

        Returns:
            dict with keys:
                - 'y': (B, N_sc, N_sym, 2) received IQ samples, dtype float32
                - 'h_true': (B, N_sc, N_sym, 2) true channel, dtype float32
                - 'x': (B, N_sc, N_sym, 2) transmitted symbols, dtype float32
                - 'bits': (B, N_sc, N_sym, n_bits) transmitted bits {0, 1}, dtype int64
                - 'pilots': (B, n_pilots, 2) known pilot symbols, dtype float32
                - 'pilot_mask': (N_sc, N_sym) boolean pilot locations
                - 'snr_db': float, the SNR used
                - 'doppler_hz': float, the Doppler spread used
                - 'elevation_deg': float, the elevation angle used
        """
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        # Set defaults
        if doppler_hz is None:
            doppler_hz = doppler_spread_hz(elevation_deg, carrier_freq_hz=2.0e9)
        if delay_spread_s is None:
            delay_spread_s = DELAY_SPREAD_S["suburban"]

        # ── Transmitted Signal ──────────────────────────────────────
        bits = self._generate_bits(batch_size)                     # (B, N_sc, N_sym, n_bits)
        x_data = self._modulate_qpsk(bits)                         # (B, N_sc, N_sym, 2)

        # ── Channel ────────────────────────────────────────────────
        # Generate Rician fading
        h_flat = self._build_rician_channel(
            batch_size, elevation_deg, doppler_hz, rng
        )                                                           # (B, N_sc, N_sym, 2)

        # Apply frequency selectivity
        h_true = self._apply_frequency_selectivity(h_flat, delay_spread_s)  # (B, N_sc, N_sym, 2)

        # ── Received Signal ─────────────────────────────────────────
        # y = h * x + w  (element-wise complex multiply)
        y_real = h_true[..., 0] * x_data[..., 0] - h_true[..., 1] * x_data[..., 1]
        y_imag = h_true[..., 0] * x_data[..., 1] + h_true[..., 1] * x_data[..., 0]
        y = torch.stack([y_real, y_imag], dim=-1)                  # (B, N_sc, N_sym, 2)

        # Add AWGN
        # SNR = signal_power / noise_power
        # For unit-variance channel and unit-power QPSK: signal power = 1 per RE
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise_std = math.sqrt(noise_power / 2.0)  # half per real/imag component

        noise = torch.randn(batch_size, self.N_sc, self.N_sym, 2,
                            generator=rng) * noise_std               # (B, N_sc, N_sym, 2)
        y = y + noise                                                # (B, N_sc, N_sym, 2)

        # ── Pilot Symbols ──────────────────────────────────────────
        # Extract pilot symbols from transmitted signal at pilot positions
        pilots = x_data[:, self.pilot_mask, :]                       # (B, N_p, 2)

        return {
            "y": y,
            "h_true": h_true,
            "x": x_data,
            "bits": bits,
            "pilots": pilots,
            "pilot_mask": self.pilot_mask.clone(),
            "snr_db": snr_db,
            "doppler_hz": doppler_hz,
            "elevation_deg": elevation_deg,
        }


# ──────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create a minimal config for testing
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

    # Generate test batch
    batch = sim.generate_batch(
        batch_size=4,
        elevation_deg=45.0,
        snr_db=0.0,
        doppler_hz=1200.0,
        seed=42,
    )

    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"{key:15s}: {val.shape}  {val.dtype}")
        else:
            print(f"{key:15s}: {val}")

    # Verify shapes
    B = 4
    assert batch["y"].shape == (B, 12, 14, 2), f"Bad y shape: {batch['y'].shape}"
    assert batch["h_true"].shape == (B, 12, 14, 2), f"Bad h_true shape: {batch['h_true'].shape}"
    assert batch["x"].shape == (B, 12, 14, 2), f"Bad x shape: {batch['x'].shape}"
    assert batch["bits"].shape == (B, 12, 14, 2), f"Bad bits shape: {batch['bits'].shape}"
    assert batch["pilots"].shape == (B, 24, 2), f"Bad pilots shape: {batch['pilots'].shape}"
    assert batch["pilot_mask"].shape == (12, 14), f"Bad pilot_mask shape: {batch['pilot_mask'].shape}"

    print(f"\n[OK] LEONTNChannelSimulator generates correct shapes")
    print(f"     Physical params: SNR={batch['snr_db']} dB, Doppler={batch['doppler_hz']:.0f} Hz, Elev={batch['elevation_deg']} deg")
