"""
Training Harness for Deep-Unfolded VAMP-JCD Receiver
=====================================================

Provides loss functions, evaluation metrics, and training utilities
for the VAMP-JCD receiver on LEO NTN channel data.

Loss function:
    ℒ = λ₁ · NMSE(Ĥ, H_true)                  # Channel estimation loss
      + λ₂ · BCE(LLRs, bits)                   # Detection loss (binary cross-entropy)
      + λ₃ · ||θ||₁                            # Sparsity regularization on shrinkage

Evaluation metrics:
    - BLER (Block Error Rate) — after hard decision on LLRs
    - NMSE (Normalized Mean-Squared Error) — for channel estimation
    - Parameter count and MAC count per block
"""

import math
import time
from typing import Optional, Tuple, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────────

def channel_nmse_loss(h_pred: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """
    Normalized Mean-Squared Error for channel estimation.

    NMSE = ||h_pred - h_true||^2 / ||h_true||^2

    Computed per batch element, then averaged.

    Args:
        h_pred: (B, N_sc, N_sym, 2) — predicted channel
        h_true: (B, N_sc, N_sym, 2) — true channel

    Returns:
        nmse: scalar — normalized MSE averaged over batch
    """
    # Compute squared error
    error = h_pred - h_true                                          # (B, N_sc, N_sym, 2)
    error_power = error[..., 0]**2 + error[..., 1]**2                 # (B, N_sc, N_sym)
    mse_per_batch = error_power.reshape(error.shape[0], -1).mean(dim=1)  # (B,)

    # Normalize by true channel power
    true_power = h_true[..., 0]**2 + h_true[..., 1]**2               # (B, N_sc, N_sym)
    power_per_batch = true_power.reshape(h_true.shape[0], -1).mean(dim=1)  # (B,)

    nmse_per_batch = mse_per_batch / (power_per_batch + 1e-10)       # (B,)
    return nmse_per_batch.mean()


def detection_bce_loss(llrs: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss from LLRs.

    For each bit with true value b ∈ {0, 1} and LLR l:
        p(b=1) = sigmoid(l)
        CE = -b * log(p) - (1-b) * log(1-p)
           = log(1 + exp(-(2b-1) * l))

    Using the numerically stable log-sum-exp formulation.

    Args:
        llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs
        bits: (B, N_sc, N_sym, n_bits) — true bits {0, 1}

    Returns:
        bce: scalar — binary cross-entropy averaged over all bits

    Shape invariants:
        - llrs and bits must have matching shapes
        - llrs.dtype in {float32, bfloat16}; bits.dtype {int64, float32}
        - F.logsigmoid used internally for numerical stability
    """
    # Convert bits to {+1, -1} format
    b_pm1 = 2.0 * bits.float() - 1.0                                 # (B, N_sc, N_sym, n_bits)

    # Numerically stable BCE: log(1 + exp(-b * LLR))
    # Use F.logsigmoid for stability: -log(sigmoid(b * LLR)) = logsigmoid(-b * LLR)
    # Actually: log(1 + exp(-x)) = -F.logsigmoid(x)
    bce = -F.logsigmoid(b_pm1 * llrs.float())                         # (B, N_sc, N_sym, n_bits)

    return bce.mean()


def sparsity_regularization(model, lambda_3: float = 0.01) -> torch.Tensor:
    """
    L1 sparsity regularization on shrinkage thresholds.

    Encourages the delay-Doppler shrinkage thresholds to be sparse
    (many thresholds = 0), which corresponds to aggressive denoising.

    Args:
        model: VAMPJCDReceiver instance
        lambda_3: regularization strength

    Returns:
        reg: scalar regularization loss
    """
    reg = 0.0
    # Find shrinkage threshold parameters
    for name, param in model.named_parameters():
        if 'shrinkage' in name or 'threshold' in name:
            reg = reg + param.abs().sum()
    return lambda_3 * reg


def total_loss(h_pred: torch.Tensor, h_true: torch.Tensor,
               llrs: torch.Tensor, bits: torch.Tensor,
               model: Optional[nn.Module] = None,
               lambda_1: float = 1.0,
               lambda_2: float = 1.0,
               lambda_3: float = 0.01) -> Dict[str, torch.Tensor]:
    """
    Composite loss function for VAMP-JCD receiver training.

    ℒ = λ₁ · NMSE + λ₂ · BCE + λ₃ · L1(shrinkage)

    Args:
        h_pred: (B, N_sc, N_sym, 2) — predicted channel
        h_true: (B, N_sc, N_sym, 2) — true channel
        llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs
        bits: (B, N_sc, N_sym, n_bits) — true bits
        model: receiver model (for sparsity regularization)
        lambda_1: NMSE loss weight
        lambda_2: BCE loss weight
        lambda_3: sparsity regularization weight

    Returns:
        dict with keys: 'total', 'nmse', 'bce', 'sparsity'
    """
    nmse = channel_nmse_loss(h_pred, h_true)
    bce = detection_bce_loss(llrs, bits)
    sparsity = sparsity_regularization(model, lambda_3) if model is not None else torch.tensor(0.0)

    total = lambda_1 * nmse + lambda_2 * bce + sparsity

    return {
        "total": total,
        "nmse": nmse,
        "bce": bce,
        "sparsity": sparsity,
    }


# ──────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────────

def compute_bler(llrs: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """
    Compute Block Error Rate (BLER).

    A block is in error if any bit in the block is decoded incorrectly.
    BLER = (# blocks with at least one bit error) / (total blocks)

    Args:
        llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs
        bits: (B, N_sc, N_sym, n_bits) — true bits {0, 1}

    Returns:
        bler: scalar — block error rate
    """
    # Hard decision: sign(LLR) -> {0, 1}
    hard_bits = (llrs > 0).float()                                    # (B, N_sc, N_sym, n_bits)

    # Bit errors
    bit_errors = (hard_bits != bits.float()).any(dim=-1)               # (B, N_sc, N_sym)
    # Any error in block -> block error
    block_errors = bit_errors.any(dim=(1, 2))                          # (B,)

    return block_errors.float().mean()


def compute_ber(llrs: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """
    Compute Bit Error Rate (BER).

    Args:
        llrs: (B, N_sc, N_sym, n_bits) — per-bit LLRs
        bits: (B, N_sc, N_sym, n_bits) — true bits {0, 1}

    Returns:
        ber: scalar — bit error rate
    """
    hard_bits = (llrs > 0).float()
    return (hard_bits != bits.float()).float().mean()


# ──────────────────────────────────────────────────────────────────────────
# Training Harness
# ──────────────────────────────────────────────────────────────────────────

class TrainingHarness:
    """
    Training harness for the VAMP-JCD receiver.

    Handles data generation, loss computation, optimizer setup,
    and evaluation across SNR/Doppler conditions.

    Usage:
        harness = TrainingHarness(model, config)
        harness.train(num_epochs=50)
        results = harness.evaluate(snr_db_list=[-10, -5, 0, 5, 10])
    """

    def __init__(self, model: nn.Module, config, device: str = "cpu"):
        """
        Args:
            model: VAMPJCDReceiver instance
            config: DU_VAMP_JCD_Config
            device: "cpu" or "cuda"
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Channel simulator (imported lazily)
        from channel import LEONTNChannelSimulator
        self.simulator = LEONTNChannelSimulator(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # Training history
        self.history = defaultdict(list)

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: dict from LEONTNChannelSimulator.generate_batch()

        Returns:
            dict of loss components
        """
        # Move to device
        y = batch["y"].to(self.device)
        h_true = batch["h_true"].to(self.device)
        pilots = batch["pilots"].to(self.device)
        pilot_mask = batch["pilot_mask"].to(self.device)
        bits = batch["bits"].to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        h_pred, x_soft, llrs, state = self.model(y, pilots, pilot_mask)

        # Compute loss
        losses = total_loss(
            h_pred, h_true, llrs, bits,
            model=self.model,
            lambda_1=1.0, lambda_2=1.0, lambda_3=0.01,
        )

        # Backward pass
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, snr_db_list=None, elevation_deg=45.0,
                 doppler_hz=1200.0, batch_size=256, n_batches=5) -> Dict:
        """
        Evaluate model across SNR values.

        Args:
            snr_db_list: list of SNR values in dB
            elevation_deg: elevation angle in degrees
            doppler_hz: Doppler spread in Hz
            batch_size: number of samples per batch
            n_batches: number of batches to average

        Returns:
            dict of {snr: {metric: value}}
        """
        if snr_db_list is None:
            snr_db_list = [-10.0, -5.0, -3.0, 0.0, 3.0, 5.0, 10.0]

        self.model.eval()
        results = {}

        for snr in snr_db_list:
            bler_total = 0.0
            ber_total = 0.0
            nmse_total = 0.0

            for _ in range(n_batches):
                batch = self.simulator.generate_batch(
                    batch_size=batch_size,
                    elevation_deg=elevation_deg,
                    snr_db=snr,
                    doppler_hz=doppler_hz,
                )

                y = batch["y"].to(self.device)
                h_true = batch["h_true"].to(self.device)
                pilots = batch["pilots"].to(self.device)
                pilot_mask = batch["pilot_mask"].to(self.device)
                bits = batch["bits"].to(self.device)

                h_pred, x_soft, llrs, _ = self.model(y, pilots, pilot_mask)

                bler_total += compute_bler(llrs, bits).item()
                ber_total += compute_ber(llrs, bits).item()
                nmse_total += channel_nmse_loss(h_pred, h_true).item()

            results[snr] = {
                "bler": bler_total / n_batches,
                "ber": ber_total / n_batches,
                "nmse": nmse_total / n_batches,
            }
            print(f"  SNR={snr:+.1f} dB: BLER={results[snr]['bler']:.4f}, "
                  f"NMSE={results[snr]['nmse']:.4f}")

        self.model.train()
        return results

    def train(self, num_epochs: int = 50, batch_size: int = 256,
              eval_every: int = 10, snr_range_db=(-10.0, 10.0),
              doppler_hz: float = 1200.0) -> Dict:
        """
        Full training loop.

        Args:
            num_epochs: number of epochs
            batch_size: training batch size
            eval_every: evaluate every N epochs
            snr_range_db: (low, high) for random SNR sampling
            doppler_hz: fixed Doppler spread for training

        Returns:
            training history dict
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print(f"  Device: {self.device}")
        print()

        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()

            # Random SNR for this batch (uniform over range)
            snr_db = (snr_range_db[0] + (snr_range_db[1] - snr_range_db[0])
                      * torch.rand(1).item())

            # Random elevation angle
            elevation_deg = 10.0 + 80.0 * torch.rand(1).item()

            # Generate training batch
            batch = self.simulator.generate_batch(
                batch_size=batch_size,
                elevation_deg=float(elevation_deg),
                snr_db=float(snr_db),
                doppler_hz=doppler_hz,
            )

            # Training step
            losses = self.train_step(batch)

            # Logging
            for k, v in losses.items():
                self.history[k].append(v)

            # Learning rate scheduling
            self.scheduler.step(losses["total"])

            # Evaluation
            if (epoch + 1) % eval_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Loss={losses['total']:.4f} | "
                      f"NMSE={losses['nmse']:.4f} | "
                      f"BCE={losses['bce']:.4f} | "
                      f"LR={current_lr:.2e}")

            # Save best model
            if losses["total"] < best_loss:
                best_loss = losses["total"]

        print(f"\nTraining complete. Best loss: {best_loss:.4f}")
        return dict(self.history)


# ──────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from model import DU_VAMP_JCD_Config, VAMPJCDReceiver, count_params

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = DU_VAMP_JCD_Config(
        n_unfolded_layers=3,          # use fewer for quick testing
        weight_tying=True,
        denoiser_type="delay_doppler_shrinkage",
        cfo_mode="analytic_plus_learned",
    )

    model = VAMPJCDReceiver(cfg).to(device)
    count_params(model)
    print(f"Config param budget: {cfg.n_trainable_params}")

    # Quick training sanity check
    harness = TrainingHarness(model, cfg, device=device)

    # Single training step
    from channel import LEONTNChannelSimulator
    sim = LEONTNChannelSimulator(cfg)

    batch = sim.generate_batch(batch_size=4, elevation_deg=45.0,
                                snr_db=0.0, doppler_hz=1200.0, seed=42)

    losses = harness.train_step(batch)
    print(f"Train step: total={losses['total']:.4f}, nmse={losses['nmse']:.4f}, bce={losses['bce']:.4f}")

    # Quick evaluation
    print("\nQuick evaluation:")
    results = harness.evaluate(snr_db_list=[0.0, 5.0], n_batches=1, batch_size=4)
    print(f"[OK] Training harness functional")
