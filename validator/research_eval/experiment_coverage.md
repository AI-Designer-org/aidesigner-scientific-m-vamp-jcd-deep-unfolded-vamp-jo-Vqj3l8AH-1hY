# Experiment Coverage: VAMP-JCD Receiver for LEO NTN IoT

## Required vs Implemented Experiments

Maps all upstream requirements (research stage Section 5, architect stage Steps 9-10) to implemented artifacts.

---

## 1. Baselines Required by Research

| Baseline | Required | Implemented | File | Status |
|----------|----------|-------------|------|--------|
| Conventional MMSE receiver (perfect CFO) | ✅ | ✅ | `coder/baselines.py::LMMSEBaseline` | implemented |
| Conventional MMSE receiver (estimated CFO) | ✅ | ✅ | `coder/baselines.py::LMMSEBaseline` (takes `noise_var` param) | implemented |
| LS channel estimation + linear interpolation | ✅ | ✅ | `coder/baselines.py::LSBaseline` | implemented |
| Ideal known-channel bound (genie-aided) | ✅ | ✅ | `coder/baselines.py::GenieBound` | implemented |
| DeepSig OmniPHY-style DNN receiver | ✅ | ❌ | Proprietary (not reproducible). TinyCNNDenoiser (62 params) as alternative. | alternative_provided |
| LR-FHSS Enhanced Decoder (Maldonado 2025) | ✅ | ❌ | Out of scope (NB-IoT focus) | out_of_scope |

## 2. Evaluation Requirements

| Requirement | Implemented | File | Status |
|-------------|-------------|------|--------|
| BLER vs Eb/N0 from -10 dB to +10 dB @ 600 Hz | ✅ | `run_benchmarks.py::benchmark_bler_vs_snr` | implemented |
| BLER vs Eb/N0 from -10 dB to +10 dB @ 1.2 kHz | ✅ | `run_benchmarks.py::benchmark_bler_vs_snr` | implemented |
| BLER vs Eb/N0 from -10 dB to +10 dB @ 2.4 kHz | ✅ | `run_benchmarks.py::benchmark_bler_vs_snr` | implemented |
| BLER vs Doppler spread at Eb/N0 = -3 dB | ✅ | `run_benchmarks.py::benchmark_bler_vs_doppler` | implemented |
| Parameter count (total trainable) | ✅ | `profile_model.py::parameter_count_report` | implemented |
| MAC count per received block | ✅ | `profile_model.py::profile_mac_estimate` | implemented (analytical) |
| FPGA resource utilization (LUT, DSP, BRAM) | ❌ | Requires FINN/Vitis AI toolchain | TODO_unverified |
| FPGA power consumption (W) | ❌ | Requires FINN/Vitis AI toolchain | TODO_unverified |
| FPGA per-block inference latency (ms) | ❌ | Requires FINN/Vitis AI toolchain | TODO_unverified |
| NMSE of channel estimation vs SNR | ✅ | `run_benchmarks.py::benchmark_bler_vs_snr` (includes NMSE) | implemented |
| Training data: synthetic LEO NTN channels (TR 38.811) | ✅ | `coder/channel.py::LEONTNChannelSimulator` | implemented |
| Test data: held-out geometries | ❌ | Simulator supports but benchmark at 45° only | TODO |

## 3. Single-Field Ablations (from Architect Step 9)

| Ablation | Field | Values | Implemented | File |
|----------|-------|--------|-------------|------|
| 1: Unfolded layer count (depth) | `n_unfolded_layers` | {1, 2, 3, 5, 7, 10} | ✅ | `ablation_runner.py`, `test_model.py::TestAblations::test_ablation_unfolded_layers` |
| 2: Weight tying | `weight_tying` | {True, False} | ✅ | `ablation_runner.py`, `test_model.py::TestAblations::test_ablation_weight_tying` |
| 3: Denoiser type | `denoiser_type` | {delay_doppler_shrinkage, element_shrinkage, small_cnn, mlp} | ✅ | `ablation_runner.py`, `test_model.py::TestAblations::test_ablation_denoiser_type` |
| 4: CFO estimation mode | `cfo_mode` | {analytic, analytic_plus_learned} | ✅ | `ablation_runner.py`, `test_model.py::TestAblations::test_ablation_cfo_mode` |
| 5: Soft symbol feedback | `use_soft_feedback` | {True, False} | ✅ | `ablation_runner.py`, `test_model.py::TestAblations::test_ablation_soft_feedback` |
| 6: Damping strategy | `n_damping_params` | {1, 5} | ✅ | `ablation_runner.py`, `test_model.py::TestAblations::test_ablation_damping_strategy` |
| 7: Quantization sensitivity | `quantization_target` | {int8, fp32} | ❌ | Requires FINN/Vitis AI or simulated quantization |
| 8: Noise precision | `learn_noise_precision` | {True, False} | ✅ | `ablation_runner.py`, `test_model.py::TestAblations::test_ablation_noise_precision` |

## 4. Synthetic Benchmarks Implemented

| Benchmark | Implemented | File | Description |
|-----------|-------------|------|-------------|
| Channel estimation NMSE vs SNR | ✅ | `test_model.py::TestDomainBenchmarks::test_channel_nmse_vs_snr` | Checks NMSE at 6 SNR points |
| BLER at multiple Doppler spreads | ✅ | `test_model.py::TestDomainBenchmarks::test_bler_vs_doppler` | BLER at 600/1200/2400 Hz |
| Parameter count budget | ✅ | `test_model.py::TestDomainBenchmarks::test_parameter_count_budget` | ≤1000 verification |
| VAMP-JCD vs LMMSE at low SNR | ✅ | `test_model.py::TestDomainBenchmarks::test_vamp_jcd_outperforms_lmmse_at_low_snr` | Central hypothesis test |
| CFO estimation accuracy | ✅ | `test_model.py::TestDomainBenchmarks::test_cfo_analytic_estimate_vs_true` | Estimate vs true CFO |
| Genie bound baseline | ✅ | `test_model.py::TestDomainBenchmarks::test_genie_bound_baseline` | Perfect channel upper bound |
| Delay-Doppler sparsity probe | ✅ | `test_model.py::TestDomainBenchmarks::test_delay_doppler_sparsity_preservation` | Energy concentration |
| Channel physics validation | ✅ | `test_model.py::TestChannelPhysics` | K-factor, Doppler, SNR, correlation |
| Gradient flow and training | ✅ | `test_model.py::TestGradients` | Backprop through all params |
| Numerical stability | ✅ | `test_model.py::TestNumerics` | Extreme SNR, zero input, bf16 |
| Inference speed profiling | ✅ | `test_model.py::TestProfiling::test_inference_speed` | Latency measurement |
| Model determinism | ✅ | `test_model.py::TestResearchQuality::test_model_is_reproducible` | Fixed-seed reproducibility |

## 5. Metrics Reported

| Metric | Reported | How |
|--------|----------|-----|
| BLER (Block Error Rate) | ✅ | Hard decision on LLRs, averaged over batch |
| BER (Bit Error Rate) | ✅ | Per-bit error rate |
| NMSE (Normalized MSE) | ✅ | Channel estimation NMSE |
| Parameter count | ✅ | `count_params()` per model |
| MAC count | ✅ | Analytical estimate |
| Inference time (ms/block) | ✅ | Timed forward pass |
| CFO estimation error | ✅ | Absolute error vs true CFO |
| Delay-Doppler concentration | ✅ | Energy fraction in top bins |

## 6. TODO: Unverified Items

| Item | Reason | Unblocking Action |
|------|--------|-------------------|
| BLER improvement over MMSE (≥2 dB) | Model not trained | Train on synthetic LEO NTN data |
| FPGA resource utilization | No FINN/Vitis AI toolchain | Implement INT8 quantization + HLS |
| FPGA power (<5W) | No FPGA measurement setup | Board-level or tool-estimated power |
| FPGA latency (<1ms) | No FPGA measurement setup | Post-implementation timing analysis |
| Generalization across elevation angles | Benchmark at 45° only | Run benchmarks at 15°, 30°, 60°, 90° |
| Quantization sensitivity (<0.5 dB loss) | Quantization not implemented | Simulated quantization in PyTorch first |
| MAC budget compliance at T=5 | Estimated ~15k vs 10k budget | Use T=3 or implement FFT pruning |

## 7. Can the Benchmark Suite Distinguish from Trivial Baseline?

**Question:** Does the validation suite have enough signal to tell if VAMP-JCD
is better than a trivial baseline (e.g., always-predict-zero channel)?

**Answer: Partially.**

**What the suite CAN detect:**
- If VAMP-JCD produces NMSE > 1.0 (worse than always predicting zero channel)
- If BLER is near 0.75 (random guessing for QPSK)
- If CFO estimates are consistently wrong (bounded by analytic component)
- If numerical instability produces NaN/Inf

**What the suite CANNOT detect (without trained weights):**
- Whether VAMP-JCD outperforms LMMSE (the central hypothesis)
- Whether the deep unfolding architecture provides real benefit over the base algorithm
- Whether the 24 learnable parameters actually learn useful behavior

**Gap:** The central distinguishing experiment (trained VAMP-JCD vs LMMSE at low SNR)
is defined in `test_model.py::TestDomainBenchmarks::test_vamp_jcd_outperforms_lmmse_at_low_snr`
and `run_benchmarks.py::benchmark_full_comparison`, but both operate on randomly
initialized models. The comparison will become meaningful only after training.
