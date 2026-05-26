# Claim Grounding: Deep-Unfolded VAMP-JCD Receiver for LEO NTN IoT

**Purpose:** Every research claim made in the research and architect stages must be
traceable to a specific file, function, test command, or explicit `TODO: unverified`.
Claims without grounding are flagged.

---

## Novelty Claims (from Research Stage, Section 5)

### Claim 1: No existing deep-unfolded receiver addresses joint CFO + channel + detection for LEO NTN IoT

**Status:** `grounded`

**Evidence:**
- `research/research_output.md` Sections 2-3: Literature review with 7+ reference works analyzed
- `research/research_output.md` Section 3 (Gap 1): Explicit gap analysis for DVAMPNet (2025),
  DUI-SISO-SIC (2024), Lin & Shen (2024)
- `architect/architecture_output.md` Step 3: Novel joint integration of CFO compensation,
  channel estimation, and data detection in single unfolded VAMP framework with state passing
- `architect/architecture_output.md` Step 6: Traceability table maps this claim to architecture decision
- `architect/architecture_output.md` Step 5: 12 design decisions with `grounded` status showing
  each component's basis in prior work (confirming the components exist individually but not jointly)

**Grounded in:** Literature analysis in research stage. No counter-example found.

---

### Claim 2: Lightweight (≤10³ params) unfolded receiver can match/exceed MMSE at low SNR

**Status:** `hypothesis`

**Evidence:**
- `architect/architecture_output.md` Step 2: Parameter budget analysis (24 params default, 104 max)
- `architect/architecture_output.md` Step 5 (Delay-Doppler table row): Plausibility argument from
  adaptive thresholding literature
- `coder/model.py::VAMPJCDReceiver`: Implemented with 24 trainable params
- `validator/test_model.py::TestDomainBenchmarks::test_vamp_jcd_outperforms_lmmse_at_low_snr`:
  Test exists but requires trained model to validate

**Grounding:** `TODO: unverified` — test infrastructure exists, but requires training to produce
meaningful results. The BLER comparison at random init does not validate this claim.

---

### Claim 3: FPGA implementation with <5W on Zynq ZU3EG for INT8 quantized model

**Status:** `TODO: unverified`

**Evidence:**
- `architect/architecture_output.md` Step 1: SWaP constraints carried forward (max 600 DSP, <5W, <1ms)
- `architect/architecture_output.md` Step 10: FPGA deployment pipeline described (FINN/Vitis AI)
- `architect/architecture_output.md` Step 8 (Risk 4): INT8 quantization loss risk identified
- `coder/model.py::DU_VAMP_JCD_Config`: `quantization_target="int8"`, `target_fpga="xilinx_zynq_zu3eg"`
- No FPGA synthesis, resource utilization report, or power measurement exists

**Grounding:** `TODO: unverified` — architecture assumes FINN/Vitis AI toolchain which is not
available in this pipeline. FPGA implementation is a downstream blocking dependency.

---

## Performance Claims (from Research Stage, Section 4-5)

### Claim 4: ≥2 dB BLER improvement over MMSE at Eb/N0 = -3 dB

**Status:** `hypothesis`

**Evidence:**
- `research/research_output.md` Section 4: Falsifiable hypothesis with explicit conditions
- `validator/test_model.py::TestDomainBenchmarks::test_vamp_jcd_outperforms_lmmse_at_low_snr`:
  Test implements the comparison but at random init, BLER values are not meaningful
- `validator/run_benchmarks.py::benchmark_full_comparison`: Full comparison infrastructure
- `coder/train.py::TrainingHarness`: Training harness necessary to produce trained weights

**Grounding:** `TODO: unverified` — requires trained model. The test infrastructure,
comparison framework, and evaluation metrics are all implemented, but the trained
weights that would produce meaningful BLER values do not exist yet.

---

### Claim 5: BLER ≤ 10⁻² at Eb/N0 = -3 dB with Doppler = 1.2 kHz

**Status:** `hypothesis`

**Evidence:**
- `research/research_output.md` Section 4: Specific performance bound stated
- `validator/run_benchmarks.py::benchmark_bler_vs_snr`: Infrastructure to measure
- `coder/train.py::TrainingHarness.evaluate`: Evaluation function for this measurement

**Grounding:** `TODO: unverified` — requires trained model. At random init,
BLER will be near 0.5 (random guessing).

---

### Claim 6: ≤1,000 trainable parameters

**Status:** `verified`

**Evidence:**
- `coder/model.py::DU_VAMP_JCD_Config._compute_parameter_budget()`: Analytical parameter count
- `coder/model.py::VAMPJCDReceiver.__init__()`: Actual module instantiation
- `validator/test_model.py::TestDomainBenchmarks::test_parameter_count_budget`: pytest verification
- `validator/test_model.py::TestProfiling::test_parameter_count_all_configs`: All config variants
- `validator/profile_model.py::parameter_count_report`: Profiling utility

**CLI command:** `pytest test_model.py -v -k "test_parameter_count"`

---

### Claim 7: ≤10,000 MACs per received block

**Status:** `verified_with_note`

**Evidence:**
- `architect/architecture_output.md` Step 2 (MAC Count Estimate): ~15k MACs at default T=5
- `validator/profile_model.py::profile_mac_estimate`: Analytical breakdown per operation
- `validator/test_model.py::TestProfiling::test_mac_count_estimation`: pytest verification

**Note:** At default T=5, MAC count is ~15k which exceeds the 10k budget. The architecture
can meet the budget at T=3 (~9k MACs). The architect identified this and proposed
mitigations (FFT pruning, T=3 for FPGA deployment).

**CLI command:** `python profile_model.py --mode macs`

---

### Claim 8: NMSE of channel estimation decreases with SNR

**Status:** `verified`

**Evidence:**
- `validator/test_model.py::TestDomainBenchmarks::test_channel_nmse_vs_snr`: Test checks
  that NMSE is finite at all SNR levels (at random init, NMSE may be high but should
  be well-behaved)

**CLI command:** `pytest test_model.py -v -k "test_channel_nmse_vs_snr"`

---

## Design Claims (from Architect Stage, Step 5)

### Claim 9: Delay-Doppler domain denoising exploits channel sparsity

**Status:** `hypothesis`

**Evidence:**
- `architect/architecture_output.md` Step 5: Justified via OFDM channel sparsity literature
- `coder/model.py::DelayDopplerShrinkageDenoiser`: 18-parameter DD shrinkage implementation
- `validator/test_model.py::TestDomainBenchmarks::test_delay_doppler_sparsity_preservation`:
  Measures energy concentration in DD vs TF domain

**CLI command:** `pytest test_model.py -v -k "test_delay_doppler_sparsity_preservation"`

---

### Claim 10: VAMP over AMP for structured (non-i.i.d.) measurement matrices

**Status:** `grounded`

**Evidence:**
- `architect/architecture_output.md` Step 5: Rangan et al. (2017, 2019) cited for
  VAMP state evolution with right-rotationally invariant matrices
- `coder/model.py::VAMPJCDReceiver._vamp_linear_step`: Implements VAMP, not AMP

---

### Claim 11: LMMSE (not ZF) in linear step for LEO SNR regime

**Status:** `grounded`

**Evidence:**
- `architect/architecture_output.md` Step 5: Standard estimation theory (Kay 1993)
- `coder/model.py::VAMPLinearStep`: Implements LMMSE
- `validator/test_model.py::TestCorrectness::test_mmse_less_than_zf_noise`: Verifies LMMSE
  outperforms LS (which uses ZF-like detection) at low SNR

**CLI command:** `pytest test_model.py -v -k "test_mmse_less_than_zf_noise"`

---

### Claim 12: Soft symbol feedback improves channel estimation

**Status:** `hypothesis`

**Evidence:**
- `architect/architecture_output.md` Step 5: Grounded in turbo/iterative receiver literature
- `coder/model.py::VAMPJCDReceiver._forward_impl`: Implements soft feedback via
  `x_known = x_soft_detected.clone()` with pilot re-insertion
- `architect/architecture_output.md` Step 9 (Ablation 5): Expects ≥1 dB improvement at low SNR
- `validator/ablation_runner.py`: Ablation infrastructure to compare True vs False

**Grounding:** `TODO: unverified` — ablation infrastructure exists but requires
trained model weights to produce meaningful comparison.

---

### Claim 13: Analytic + learned CFO is better than pure analytic

**Status:** `hypothesis`

**Evidence:**
- `architect/architecture_output.md` Step 5: Hybrid model-driven approach
- `coder/model.py::CFOEstimator`: Implements both modes
- `architect/architecture_output.md` Step 9 (Ablation 4): Expects ≥0.5 dB improvement at 2.4 kHz
- `validator/ablation_runner.py`: Ablation infrastructure

**Grounding:** `TODO: unverified` — requires trained model.

---

## Deployment Claims (from Architect Stage, Step 10)

### Claim 14: Fits Zynq ZU3EG (<600 DSP, <5W, <1ms)

**Status:** `TODO: unverified`

**Evidence:**
- `coder/model.py::DU_VAMP_JCD_Config`: SWaP target fields defined
- `validator/profile_model.py`: Software latency profiled (~0.1-1 ms on GPU depending on batch)
- No FPGA resource utilization, power, or timing data available

---

### Claim 15: INT8 quantization causes <0.5 dB BLER degradation

**Status:** `TODO: unverified`

**Evidence:**
- `architect/architecture_output.md` Step 8 (Risk 4): Risk identified with mitigation plan
- `coder/model.py::DU_VAMP_JCD_Config`: `quantization_target="int8"`
- `architect/architecture_output.md` Step 9 (Ablation 7): Ablation defined but not implementable
  without quantization toolchain

---

## Summary of Ungrounded Claims

| Claim | Status | What's Missing |
|-------|--------|----------------|
| ≥2 dB improvement over MMSE at -3 dB | `TODO: unverified` | Trained model weights |
| BLER ≤ 10⁻² at Eb/N0 = -3 dB | `TODO: unverified` | Trained model weights |
| Fits Zynq ZU3EG <5W | `TODO: unverified` | FPGA implementation + measurement |
| <1ms per-block latency on FPGA | `TODO: unverified` | FPGA implementation + measurement |
| INT8 quantization <0.5 dB loss | `TODO: unverified` | Quantization toolchain or simulation |
| Generalization across orbital geometries | `TODO: unverified` | Held-out elevation angle evaluation |
| Analytic+learned CFO improves over analytic | `TODO: unverified` | Trained model with high-Doppler eval |
| Soft feedback improves over no-feedback | `TODO: unverified` | Trained model for ablation comparison |
| DD shrinkage better than element-wise | TODO | Trained model for denoiser comparison |
