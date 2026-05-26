# Research Quality Rubric: Deep-Unfolded VAMP-JCD Receiver for LEO NTN IoT

## Domain-Specific Assessment Framework

**Primary Domain:** Scientific ML — Hybrid model-driven + data-driven physical layer receiver
**Secondary Domain:** Time Series — IQ samples as complex-valued time series on OFDM resource grid
**Sub-field:** Non-Terrestrial Network (NTN) Physical Layer — LEO Satellite IoT Receiver Chain

### Task Level: Level 2 (papers/context provided)

The upstream research stage provides a comprehensive literature review with 7+ reference works
(DVAMPNet, DUI-SISO-SIC, DeepSig OmniPHY, LR-FHSS Enhanced Receiver, Lin-Shen joint CFO,
CNN+LSTM channel prediction, Choquenaira-Florez survey). The architecture is evaluated for
novelty against this specific context.

---

## Scoring Dimensions (0–5)

### 1. Novelty (score: 4/5)

**Domain-Specific Questions for Scientific ML (Physical Layer Comms):**
- ✅ Does the benchmark suite include the specific baselines identified in the literature review?
  → Yes: LMMSE, LS, Genie bound. Full DNN (DeepSig) not reproducible but identified as limitation.
- ✅ Is the novelty claim grounded in a gap analysis against cited prior work?
  → Yes: research stage provides explicit gap analysis for DVAMPNet, DUI-SISO-SIC, Lin-Shen.
- ✅ Does the architecture avoid reinventing standard signal processing?
  → Yes: VAMP unfolding is the right level of novelty — algorithmic structure preserved, only
    hyperparameters learned. Architect explains 12 grounded design decisions.
- ⚠️ Are the proposed ablation experiments sufficient to distinguish the method from baselines?
  → Partially: denoiser ablation tests if DD sparsity helps; depth ablation tests convergence.
    But without trained weights, no ablation can currently distinguish anything.

**What's missing:**
- Experimental validation of the novelty claim (trained model results against baselines)
- FPGA implementation results (the key novelty — SWaP-constrained deployment)

### 2. Experimental Comprehensiveness (score: 3/5)

**Domain-Specific Questions for Scientific ML / LEO NTN:**
- ✅ BLER vs SNR curves at multiple Doppler spreads (600, 1200, 2400 Hz)
- ✅ BLER vs Doppler spread at fixed SNR (-3 dB operating point)
- ✅ NMSE of channel estimation vs SNR
- ✅ CFO estimation accuracy benchmark
- ✅ Channel sparsity probe (delay-Doppler vs time-frequency concentration)
- ✅ Full baseline comparison infrastructure (LMMSE, LS, Genie)
- ✅ 8 single-field ablations with evaluation infrastructure
- ⚠️ MAC count estimation (analytical, not profiler-based)
- ❌ FPGA resource utilization or power measurements (TODO: unverified)
- ❌ Held-out elevation angle performance (all benchmarks at 45°)

**Critical gaps:**
1. No trained model — all benchmarks run on randomly initialized weights
2. No channel-coded BLER (hard decisions on LLRs without Turbo/LDPC decoder)
3. No learning curves, convergence analysis, or training dynamics study
4. No held-out geometry evaluation
5. Simulator uses simplified Rician model, not full 3GPP TR 38.821

### 3. Theoretical Foundation (score: 4/5)

**Domain-Specific Questions for Scientific ML:**
- ✅ Physics constraints: VAMP linear step enforces exact measurement model y = Hx + w
- ✅ Function vs operator learning correctly identified (function learning)
- ✅ Symmetry/equivariance: Phase rotation equivariance correctly identified
- ✅ Regular 2D grid mesh correctly identified (time-frequency resource grid)
- ✅ VAMP state evolution theory cited and justified for non-i.i.d. matrices
- ✅ Delay-Doppler sparsity grounded in OFDM channel estimation literature
- ✅ LMMSE over ZF justified via estimation theory

**Gaps:**
- Small-system behavior (N=168 REs) — VAMP guarantees are asymptotic
- No proof of convergence for the joint CFO+channel+detection system
- Onsager correction with learnable β is empirical
- Damping-Onsager interaction not analyzed

### 4. Result Analysis (score: 1/5)

**What exists:**
- Benchmark infrastructure generates all required curves
- Numerical stability validated across extreme conditions
- Shape and gradient flow assertions pass

**Critical gaps:**
- No trained model results. All BLER/NMSE values are from random initialization and are not
  scientifically meaningful for comparing architecture quality.
- No error bars or confidence intervals on any benchmark measurement.
- No statistical significance testing between receiver variants.
- No learning curve analysis (loss vs epoch).
- Gap between VAMP-JCD and Genie bound at random init is meaningless.

**This dimension is the weakest and will remain so until model training produces meaningful weights.**

### 5. Implementation Reproducibility (score: 4/5)

**Domain-Specific Questions for Scientific ML / Physical Layer:**
- ✅ Complete model implementation (24 files across 4 modules)
- ✅ Seeded random number generation (seed=42 throughout)
- ✅ Deterministic inference verified in test suite
- ✅ All baselines share same interface as main model
- ✅ Cross-platform tests (CPU + CUDA with auto-detection)
- ✅ JSON output for all benchmarks with metadata
- ✅ Channel simulator implements 3GPP TR 38.811 parameters

**Gaps:**
- No Docker container or pinned dependency versions
- Channel simulator uses simplified Rician model (not full 3GPP TR 38.821)
- No single-command end-to-end reproduce pipeline

### 6. Writing Readiness (score: 3/5)

- ✅ Docstrings on every class and method with shape conventions
- ✅ ASCII architecture diagram
- ✅ Research-to-architecture traceability table
- ✅ Module-level docstrings explaining purpose
- ✅ Consistent naming conventions throughout

**Gaps:**
- No paper draft or method write-up
- No figures (trained model results required)
- Related work exists only in research stage
- No ablation interpretation

---

## Summary of Blocking Gaps

| Gap | Severity | Affects |
|-----|----------|---------|
| No trained model → benchmarks produce random-init values | **HIGH** | All novelty and performance claims |
| FPGA resource/power/latency not measured | **HIGH** | SWaP claims (research hypothesis) |
| Held-out elevation angle not tested | MEDIUM | Generalization claim |
| MAC budget exceeded at T=5 (~15k vs 10k budget) | MEDIUM | FPGA deployment feasibility |
| Quantization sensitivity not measured | LOW | FPGA deployment study |

## Recommended Next Steps (Priority Order)

1. **P0:** Train the model on synthetic LEO NTN data and measure BLER vs SNR against baselines
2. **P0:** Sweep unfolded iterations (T=1..10) to find optimal depth for FPGA deployment
3. **P1:** Evaluate at held-out elevation angles (15°, 75°) after training at 45°
4. **P1:** Run denoiser ablation to validate delay-Doppler sparsity assumption
5. **P2:** Profile INT8 quantization sensitivity using simulated quantization
6. **P2:** Conduct full training hyperparameter study (SNR range, batch size, learning rate)
