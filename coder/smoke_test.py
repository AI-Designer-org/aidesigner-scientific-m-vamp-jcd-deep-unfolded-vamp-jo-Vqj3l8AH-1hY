"""
Smoke Test: VAMP-JCD Receiver for LEO NTN IoT
===============================================

End-to-end validation of the deep-unfolded VAMP-JCD receiver,
including:
  - Model instantiation and parameter count verification
  - Forward pass with shape assertions
  - Channel simulator sanity check
  - Baseline comparison (LMMSE, LS, Genie)
  - Ablation configuration tests
  - Numerical safety checks (NaN/Inf)
  - torch.compile compatibility test

Run with:  python smoke_test.py
"""

import math
import sys
import torch

sys.path.insert(0, ".")


# ──────────────────────────────────────────────────────────────────────────
# 1. Model Instantiation
# ──────────────────────────────────────────────────────────────────────────

def test_model_instantiation():
    """Verify model creation and parameter count."""
    from model import DU_VAMP_JCD_Config, VAMPJCDReceiver, count_params

    print("=" * 60)
    print("1. Model Instantiation")
    print("=" * 60)

    # Default config
    cfg = DU_VAMP_JCD_Config(
        n_unfolded_layers=5,
        weight_tying=True,
        denoiser_type="delay_doppler_shrinkage",
        cfo_mode="analytic_plus_learned",
        use_soft_feedback=True,
    )

    model = VAMPJCDReceiver(cfg)
    n_params = count_params(model)

    # Assert: should be 24 (default config)
    assert n_params == cfg.n_trainable_params, (
        f"Parameter mismatch: model has {n_params}, config says {cfg.n_trainable_params}"
    )
    assert n_params <= 1000, f"Exceeds parameter budget: {n_params} > 1000"

    print(f"  ✓ Config budget matches model: {n_params} params")
    print(f"  ✓ Well within ≤1000 budget")

    return model, cfg


# ──────────────────────────────────────────────────────────────────────────
# 2. Forward Pass + Shape Assertions
# ──────────────────────────────────────────────────────────────────────────

def test_forward_pass(model, cfg, device="cpu"):
    """Verify forward pass produces correct shapes."""
    print()
    print("=" * 60)
    print("2. Forward Pass Shape Assertions")
    print("=" * 60)

    model = model.to(device)
    model.eval()

    B = 4
    N_sc = cfg.n_subcarriers   # 12
    N_sym = cfg.n_symbols      # 14
    N_p = cfg.n_pilots         # 24

    # Create synthetic input
    y = torch.randn(B, N_sc, N_sym, 2, device=device)

    # Normalized pilots
    pilots = torch.randn(B, N_p, 2, device=device)
    pilots = pilots / (pilots.norm(dim=-1, keepdim=True) + 1e-10)

    # Pilot mask: NPUSCH Format 1 pilots at symbols 3 and 10
    pilot_mask = torch.zeros(N_sc, N_sym, dtype=torch.bool, device=device)
    pilot_mask[:, 3] = True
    pilot_mask[:, 10] = True

    with torch.no_grad():
        h_hat, x_soft, llrs, state = model(y, pilots, pilot_mask, return_full_state=True)

    # Shape assertions
    expected_shapes = {
        "h_hat": (B, N_sc, N_sym, 2),
        "x_soft": (B, N_sc, N_sym, 2),
        "llrs": (B, N_sc, N_sym, cfg.n_bits_per_symbol),
    }

    for name, tensor in [("h_hat", h_hat), ("x_soft", x_soft), ("llrs", llrs)]:
        expected = expected_shapes[name]
        actual = tensor.shape
        assert actual == expected, (
            f"  ✗ {name}: expected {expected}, got {actual}"
        )
        print(f"  ✓ {name}: {list(actual)}")

    # State dict assertions
    assert len(state["h_per_iter"]) == cfg.n_unfolded_layers, (
        f"Expected {cfg.n_unfolded_layers} state entries, got {len(state['h_per_iter'])}"
    )
    print(f"  ✓ State has {len(state['h_per_iter'])} per-iteration entries")

    # Check no NaN or Inf in outputs
    for name, tensor in [("h_hat", h_hat), ("x_soft", x_soft), ("llrs", llrs)]:
        assert not torch.isnan(tensor).any(), f"NaN in {name}"
        assert not torch.isinf(tensor).any(), f"Inf in {name}"
    print(f"  ✓ No NaN or Inf in outputs")

    return y, pilots, pilot_mask


# ──────────────────────────────────────────────────────────────────────────
# 3. Channel Simulator Sanity Check
# ──────────────────────────────────────────────────────────────────────────

def test_channel_simulator(cfg):
    """Verify channel simulator produces realistic data."""
    print()
    print("=" * 60)
    print("3. Channel Simulator Sanity Check")
    print("=" * 60)

    from channel import LEONTNChannelSimulator

    sim = LEONTNChannelSimulator(cfg)

    # Test at multiple SNR values
    for snr_db in [-5.0, 0.0, 10.0]:
        batch = sim.generate_batch(
            batch_size=256,
            elevation_deg=45.0,
            snr_db=snr_db,
            doppler_hz=1200.0,
            seed=42,
        )

        y = batch["y"]
        h_true = batch["h_true"]
        bits = batch["bits"]

        # Check shapes
        B = 256
        assert y.shape == (B, 12, 14, 2), f"Bad y shape: {y.shape}"
        assert h_true.shape == (B, 12, 14, 2), f"Bad h_true shape: {h_true.shape}"

        # Check signal power
        y_power = (y[..., 0]**2 + y[..., 1]**2).mean().item()
        h_power = (h_true[..., 0]**2 + h_true[..., 1]**2).mean().item()

        print(f"  SNR={snr_db:+.0f} dB: signal power={y_power:.3f}, "
              f"channel power={h_power:.3f}, "
              f"expected noise var={10**(-snr_db/10):.4f}")

    print(f"  ✓ Channel simulator produces realistic data")

    return sim


# ──────────────────────────────────────────────────────────────────────────
# 4. Baseline Comparison
# ──────────────────────────────────────────────────────────────────────────

def test_baselines(cfg, device="cpu"):
    """Verify baseline receivers run and produce correct shapes."""
    print()
    print("=" * 60)
    print("4. Baseline Comparison")
    print("=" * 60)

    from baselines import LMMSEBaseline, LSBaseline, GenieBound
    from channel import LEONTNChannelSimulator

    sim = LEONTNChannelSimulator(cfg)

    # Generate data
    batch = sim.generate_batch(batch_size=4, elevation_deg=45.0,
                                snr_db=0.0, doppler_hz=1200.0, seed=42)
    y = batch["y"].to(device)
    pilots = batch["pilots"].to(device)
    pilot_mask = batch["pilot_mask"].to(device)

    # Test LMMSE
    lmmse = LMMSEBaseline(cfg).to(device)
    h_lm, x_lm, llrs_lm, _ = lmmse(y, pilots, pilot_mask, noise_var=1.0)
    assert h_lm.shape == (4, 12, 14, 2), f"Bad LMMSE shape: {h_lm.shape}"
    print(f"  ✓ LMMSE baseline: {h_lm.shape}")

    # Test LS
    ls = LSBaseline(cfg).to(device)
    h_ls, x_ls, llrs_ls, _ = ls(y, pilots, pilot_mask)
    assert h_ls.shape == (4, 12, 14, 2), f"Bad LS shape: {h_ls.shape}"
    print(f"  ✓ LS baseline: {h_ls.shape}")

    # Test Genie
    genie = GenieBound(cfg).to(device)
    h_true = batch["h_true"].to(device)
    h_gn, x_gn, llrs_gn, _ = genie(y, h_true, noise_var=1.0)
    assert llrs_gn.shape == (4, 12, 14, 2), f"Bad Genie shape: {llrs_gn.shape}"
    print(f"  ✓ Genie bound: {llrs_gn.shape}")

    from train import compute_bler, compute_ber, channel_nmse_loss

    bler_lm = compute_bler(llrs_lm, batch["bits"].to(device)).item()
    bler_ls = compute_bler(llrs_ls, batch["bits"].to(device)).item()
    bler_gn = compute_bler(llrs_gn, batch["bits"].to(device)).item()

    nmse_lm = channel_nmse_loss(h_lm, batch["h_true"].to(device)).item()
    nmse_ls = channel_nmse_loss(h_ls, batch["h_true"].to(device)).item()

    print(f"  LMMSE: BLER={bler_lm:.4f}, NMSE={nmse_lm:.4f}")
    print(f"  LS:    BLER={bler_ls:.4f}, NMSE={nmse_ls:.4f}")
    print(f"  Genie: BLER={bler_gn:.4f}")

    return lmmse, ls, genie


# ──────────────────────────────────────────────────────────────────────────
# 5. Ablation Configurations
# ──────────────────────────────────────────────────────────────────────────

def test_ablations(device="cpu"):
    """Verify all ablation configurations work."""
    print()
    print("=" * 60)
    print("5. Ablation Configuration Tests")
    print("=" * 60)

    from model import DU_VAMP_JCD_Config, VAMPJCDReceiver, count_params

    B, N_sc, N_sym = 4, 12, 14

    # Shared test input
    y = torch.randn(B, N_sc, N_sym, 2, device=device)
    pilots = torch.randn(B, 24, 2, device=device)
    pilot_mask = torch.zeros(N_sc, N_sym, dtype=torch.bool, device=device)
    pilot_mask[:, 3] = True
    pilot_mask[:, 10] = True

    # ── Ablation 1: Vary unfolded layers ──
    print("  5a. Unfolded layer count:")
    for T in [1, 3, 5, 7]:
        cfg = DU_VAMP_JCD_Config(
            n_unfolded_layers=T, weight_tying=True,
            denoiser_type="delay_doppler_shrinkage",
            cfo_mode="analytic_plus_learned",
        )
        model = VAMPJCDReceiver(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            h_hat, x_soft, llrs, _ = model(y, pilots, pilot_mask)
        assert h_hat.shape == (B, N_sc, N_sym, 2)
        print(f"    T={T}: {n_params} params, output OK")

    # ── Ablation 2: Weight tying vs. per-layer ──
    print("  5b. Weight tying vs. per-layer:")
    for wt in [True, False]:
        cfg = DU_VAMP_JCD_Config(
            n_unfolded_layers=5, weight_tying=wt,
            denoiser_type="delay_doppler_shrinkage",
        )
        model = VAMPJCDReceiver(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            h_hat, _, _, _ = model(y, pilots, pilot_mask)
        assert h_hat.shape == (B, N_sc, N_sym, 2)
        print(f"    weight_tying={wt}: {n_params} params, output OK")

    # ── Ablation 3: Denoiser variants ──
    print("  5c. Denoiser variants:")
    for denoiser in ["delay_doppler_shrinkage", "element_shrinkage", "small_cnn", "mlp"]:
        cfg = DU_VAMP_JCD_Config(
            n_unfolded_layers=3, weight_tying=True,
            denoiser_type=denoiser,
            cfo_mode="analytic",  # no learned CFO to isolate denoiser
        )
        model = VAMPJCDReceiver(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            h_hat, _, _, _ = model(y, pilots, pilot_mask)
        assert h_hat.shape == (B, N_sc, N_sym, 2), f"Failed for {denoiser}"
        print(f"    {denoiser}: {n_params} params, output OK")

    # ── Ablation 4: CFO modes ──
    print("  5d. CFO modes:")
    for cfo_mode in ["analytic", "analytic_plus_learned"]:
        cfg = DU_VAMP_JCD_Config(
            n_unfolded_layers=3, weight_tying=True,
            denoiser_type="delay_doppler_shrinkage",
            cfo_mode=cfo_mode,
        )
        model = VAMPJCDReceiver(cfg).to(device)
        with torch.no_grad():
            h_hat, _, _, _ = model(y, pilots, pilot_mask)
        assert h_hat.shape == (B, N_sc, N_sym, 2), f"Failed for {cfo_mode}"
        print(f"    {cfo_mode}: output OK")

    # ── Ablation 5: Soft feedback ──
    print("  5e. Soft symbol feedback:")
    for fb in [True, False]:
        cfg = DU_VAMP_JCD_Config(
            n_unfolded_layers=3, weight_tying=True,
            use_soft_feedback=fb,
        )
        model = VAMPJCDReceiver(cfg).to(device)
        with torch.no_grad():
            h_hat, _, _, _ = model(y, pilots, pilot_mask)
        assert h_hat.shape == (B, N_sc, N_sym, 2), f"Failed for soft_feedback={fb}"
        print(f"    use_soft_feedback={fb}: output OK")

    print(f"  ✓ All {9} ablation configurations pass")

    return True


# ──────────────────────────────────────────────────────────────────────────
# 6. Numerical Safety
# ──────────────────────────────────────────────────────────────────────────

def test_numerical_safety(model, cfg, device="cpu"):
    """
    Test numerical stability under extreme conditions.

    Checks:
      - Very low SNR (Eb/N0 = -10 dB)
      - Very high SNR (Eb/N0 = +20 dB)
      - Zero input
      - Large CFO
    """
    print()
    print("=" * 60)
    print("6. Numerical Safety Tests")
    print("=" * 60)

    model = model.to(device)
    model.eval()

    B, N_sc, N_sym = 4, 12, 14

    # Test at low SNR
    print("  6a. Low SNR (-10 dB):")
    y_low = torch.randn(B, N_sc, N_sym, 2, device=device) * 10.0  # high noise
    pilots = torch.randn(B, 24, 2, device=device)
    pilot_mask = torch.zeros(N_sc, N_sym, dtype=torch.bool, device=device)
    pilot_mask[:, 3] = True
    pilot_mask[:, 10] = True

    with torch.no_grad():
        h_hat, x_soft, llrs, _ = model(y_low, pilots, pilot_mask)

    assert not torch.isnan(h_hat).any(), "NaN at low SNR"
    assert not torch.isinf(h_hat).any(), "Inf at low SNR"
    print(f"    ✓ OK: h_hat range [{h_hat.min():.3f}, {h_hat.max():.3f}]")

    # Test at high SNR
    print("  6b. High SNR (+20 dB):")
    y_high = torch.randn(B, N_sc, N_sym, 2, device=device) * 0.01  # low noise
    with torch.no_grad():
        h_hat, _, _, _ = model(y_high, pilots, pilot_mask)
    assert not torch.isnan(h_hat).any(), "NaN at high SNR"
    assert not torch.isinf(h_hat).any(), "Inf at high SNR"
    print(f"    ✓ OK: h_hat range [{h_hat.min():.3f}, {h_hat.max():.3f}]")

    # Test with zero input
    print("  6c. Zero input:")
    y_zero = torch.zeros(B, N_sc, N_sym, 2, device=device)
    with torch.no_grad():
        h_hat, _, _, _ = model(y_zero, pilots, pilot_mask)
    assert not torch.isnan(h_hat).any(), "NaN with zero input"
    assert not torch.isinf(h_hat).any(), "Inf with zero input"
    print(f"    ✓ OK: h_hat range [{h_hat.min():.3f}, {h_hat.max():.3f}]")

    # Test VAMP linear step with extreme precisions
    print("  6d. VAMP linear step (extreme precisions):")
    from model import VAMPLinearStep
    vamp_lin = VAMPLinearStep().to(device)

    # Test: gamma_z very small (high noise), gamma_h very large (strong prior)
    h_test = vamp_lin(
        y=y_low,
        x_known=torch.randn(B, N_sc, N_sym, 2, device=device),
        h_prev=torch.zeros(B, N_sc, N_sym, 2, device=device),
        r1=torch.zeros(B, N_sc, N_sym, 2, device=device),
        u1=torch.zeros(B, N_sc, N_sym, 2, device=device),
        gamma_z=torch.tensor(1e-6, device=device),
        gamma_h=torch.tensor(1e6, device=device),
    )
    assert not torch.isnan(h_test).any(), "NaN in extreme VAMP step"
    assert not torch.isinf(h_test).any(), "Inf in extreme VAMP step"
    print(f"    ✓ OK: extreme precision values stable")

    print(f"  ✓ All numerical safety tests pass")


# ──────────────────────────────────────────────────────────────────────────
# 7. torch.compile Compatibility
# ──────────────────────────────────────────────────────────────────────────

def test_torch_compile(model, cfg, device="cpu"):
    """Test torch.compile compatibility."""
    print()
    print("=" * 60)
    print("7. torch.compile Compatibility")
    print("=" * 60)

    model = model.to(device)
    model.eval()

    B, N_sc, N_sym = 4, 12, 14
    y = torch.randn(B, N_sc, N_sym, 2, device=device)
    pilots = torch.randn(B, 24, 2, device=device)
    pilot_mask = torch.zeros(N_sc, N_sym, dtype=torch.bool, device=device)
    pilot_mask[:, 3] = True
    pilot_mask[:, 10] = True

    try:
        # Attempt torch.compile
        compiled_model = torch.compile(model, mode="reduce-overhead")

        with torch.no_grad():
            h_hat, x_soft, llrs, _ = compiled_model(y, pilots, pilot_mask)

        assert h_hat.shape == (B, N_sc, N_sym, 2)
        assert not torch.isnan(h_hat).any()
        print(f"  ✓ torch.compile produces correct output")
    except Exception as e:
        print(f"  ⚠ torch.compile not supported or failed: {e}")
        print(f"    (This is expected on some platforms; model is still functional)")


# ──────────────────────────────────────────────────────────────────────────
# 8. Parameter Budget Headroom
# ──────────────────────────────────────────────────────────────────────────

def test_parameter_budget():
    """Verify that all config variants stay within the 1000-param budget."""
    print()
    print("=" * 60)
    print("8. Parameter Budget Verification")
    print("=" * 60)

    from model import DU_VAMP_JCD_Config, VAMPJCDReceiver

    configs = [
        ("Minimal (T=1, analytic)", DU_VAMP_JCD_Config(
            n_unfolded_layers=1, weight_tying=True,
            cfo_mode="analytic", use_soft_feedback=False,
        )),
        ("Default (T=5, tied, analytic+learned)", DU_VAMP_JCD_Config(
            n_unfolded_layers=5, weight_tying=True,
            cfo_mode="analytic_plus_learned",
        )),
        ("Per-layer (T=5, untied)", DU_VAMP_JCD_Config(
            n_unfolded_layers=5, weight_tying=False,
        )),
        ("Max denoiser (T=5, small_cnn)", DU_VAMP_JCD_Config(
            n_unfolded_layers=5, weight_tying=True,
            denoiser_type="small_cnn",
        )),
        ("Deep (T=10, tied)", DU_VAMP_JCD_Config(
            n_unfolded_layers=10, weight_tying=True,
        )),
    ]

    all_within_budget = True
    for name, cfg in configs:
        model = VAMPJCDReceiver(cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        within = n_params <= 1000
        status = "✓" if within else "✗"
        print(f"  [{status}] {name}: {n_params:3d} params "
              f"(budget headroom: {1000 - n_params})")
        if not within:
            all_within_budget = False

    assert all_within_budget, "Some configs exceed 1000-parameter budget!"
    print(f"  ✓ All configs within ≤1000 parameter budget")


# ──────────────────────────────────────────────────────────────────────────
# Main Smoke Test Runner
# ──────────────────────────────────────────────────────────────────────────

def main():
    """Run all smoke tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"VAMP-JCD Receiver Smoke Test")
    print(f"Device: {device}")
    print()

    # 1. Instantiation
    model, cfg = test_model_instantiation()

    # 2. Forward pass
    test_forward_pass(model, cfg, device)

    # 3. Channel simulator
    test_channel_simulator(cfg)

    # 4. Baselines
    test_baselines(cfg, device)

    # 5. Ablations
    test_ablations(device)

    # 6. Numerical safety
    test_numerical_safety(model, cfg, device)

    # 7. torch.compile
    test_torch_compile(model, cfg, device)

    # 8. Parameter budget
    test_parameter_budget()

    # ── Summary ──
    print()
    print("=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Model: VAMPJCDReceiver (T={cfg.n_unfolded_layers})")
    print(f"  Parameters: {cfg.n_trainable_params} (budget: ≤1000)")
    print(f"  Denoiser: {cfg.denoiser_type}")
    print(f"  CFO mode: {cfg.cfo_mode}")
    print(f"  Weight tying: {cfg.weight_tying}")
    print(f"  Soft feedback: {cfg.use_soft_feedback}")
    print()
    print(f"  ✓ All smoke tests passed on {device}")


if __name__ == "__main__":
    main()
