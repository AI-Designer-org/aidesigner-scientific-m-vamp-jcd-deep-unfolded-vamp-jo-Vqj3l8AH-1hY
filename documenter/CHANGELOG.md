# Changelog

## [0.1.0] — 2026-05-26

### Added
- Initial implementation of VAMP-JCD: deep-unfolded VAMP Joint Channel Estimation and Data Detection receiver for LEO satellite IoT systems.
- Core architecture: 5-iteration unfolded VAMP with delay-Doppler shrinkage denoising (24 learnable parameters), CFO compensation, and soft MMSE detection.
- LEONTNChannelSimulator: 3GPP TR 38.811-compliant LEO NTN channel model with Rician fading, elevation-dependent Doppler, and frequency selectivity.
- Baseline receivers: LMMSE (pilot-aided + interpolation + MMSE), LS (pilot-aided + ZF), and Genie-aided bound.
- Training harness: composite loss function (NMSE + BCE + L1 sparsity), AdamW optimizer, ReduceLROnPlateau scheduler, per-SNR evaluation.
- Unit test suite (pytest):
  - Shape and gradient correctness for all modules
  - Domain-specific benchmarks (NMSE vs SNR, BLER vs Doppler, CFO accuracy)
  - Single-field ablation tests (depth, weight tying, denoiser type, CFO mode, soft feedback, damping, noise precision)
  - Numerical stability (extreme SNR, zero input, bf16 compatibility, gradient flow)
  - Parameter count and MAC count verification
  - Model determinism and reproducibility
- Domain-specific benchmarks: BLER vs Eb/N0 at 600/1200/2400 Hz Doppler, BLER vs Doppler spread at –3 dB, CFO estimation accuracy, delay-Doppler sparsity probe.
- Ablation runner with JSON output for 8 ablation experiments.
- Profiling infrastructure: parameter count, MAC estimation, inference speed measurement.
- Research evaluation: scorecard, claim grounding analysis, experiment coverage assessment, and rubric scoring.
- Documentation: README, ARCHITECTURE, TRAINING, BENCHMARKS, API reference, and CHANGELOG.
