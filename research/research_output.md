# ML-Enhanced Receiver Chain for LEO Satellite IoT Systems

## Research Synthesis & Novelty Analysis

---

## 1. Domain Identification

| Domain | Relevance | Rationale |
|---|---|---|
| **Scientific ML (primary)** | Direct | Hybrid model-driven + data-driven receiver design; deep unfolding of iterative signal processing algorithms; physics-constrained learning for physical layer communications |
| **Time Series (secondary)** | Direct | IQ samples and channel estimates are complex-valued time series; temporal correlation is key for Doppler tracking and channel prediction |
| **LM (tertiary)** | Partial | Transformer/attention architectures as sequence models for received symbol processing; potential for long-range dependency modeling in fading channels |

**Sub-field:** Non-Terrestrial Network (NTN) Physical Layer — specifically the **IoT LEO satellite receiver chain** (synchronization → channel estimation → equalization → detection → decoding).

---

## 2. Landscape Summary

### 2A. Conventional LEO IoT Receiver Chain (Baseline)

A conventional LEO IoT receiver (e.g., NB-IoT NTN Release 17–18, LR-FHSS) consists of:

| Stage | Conventional Algorithm | LEO-Specific Challenge |
|---|---|---|
| **Coarse synchronization** | Autocorrelation-based (Schmidl-Cox, etc.) | Doppler shift up to 24 ppm (~40 kHz at 2 GHz) breaks correlation peaks |
| **Carrier frequency offset (CFO) estimation** | ML-based (Moose, Fitz, etc.) or GNSS-assisted pre-compensation | Rapidly time-varying Doppler due to ~7.5 km/s satellite motion; residual CFO after pre-compensation still large |
| **Timing synchronization** | Cross-correlation with known sequences | Large differential delay across cell (up to several ms) |
| **Channel estimation** | LS/MMSE on pilot symbols | Low SNR (–10 to –3 dB typical for IoT); short contact windows (~10 min); pilot overhead competes with data |
| **Equalization / detection** | MMSE or ZF equalizer | Severe link budget (50–70 dB extra path loss vs. terrestrial); co-channel interference from dense constellations |
| **Decoding** | Turbo/LDPC/RS Viterbi | Low SNR pushes codec to error floor |

**Performance weak point:** At low SNR (below ~–5 dB Eb/N0) with high Doppler spreads, MMSE channel estimation degrades significantly. The Gaussian assumption breaks, and pilot-aided interpolation fails to track the rapidly time-varying channel.

### 2B. ML Approaches That Address These Weaknesses

| ML Paradigm | How It Works | LEO IoT Applicability | Existing Work |
|---|---|---|---|
| **Deep unfolding (algorithm unrolling)** | Unfold iterative optimization (AMP, ISTA, VAMP) into trainable NN layers with learnable parameters | **Strong fit** — preserves algorithmic structure, modest parameter count, retains low latency; AMP's O(N) per iteration suits on-board compute | DVAMPNet (2025) for satellite activity detection; unfolded MMSE-SIC for MIMO (2024); none for full LEO IoT receiver chain |
| **Learned channel estimation** | Replace LS/MMSE with CNN/ResNet that denoises pilot observations | Effective at low SNR, but CNN FLOPS can exceed CubeSat DSP budgets | DeepSig OmniPHY (2024, terrestrial 5G only, 2–3× UL throughput); no LEO NTN variant exists |
| **End-to-end neural receiver** | Replace entire receiver (synch → equalization → detection) with a single DNN | Most ambitious; requires large models, high training data volume, and generalization across varying orbital geometries | DeepSig (terrestrial 5G); none for LEO satellite IoT |
| **Hybrid model-driven + data-driven** | Use model (e.g., state evolution) to design NN structure; data-driven to learn residual errors | **Sweet spot** for LEO — strong inductive bias reduces data needs; modest compute | Lin & Shen (2024) for CFO + channel joint estimation (OFDM, not NTN) |
| **RL for adaptive receiver** | Reinforcement learning to select detection/equalization strategy per channel realization | Too slow for PHY-layer symbol-rate decisions; more suited to higher-layer link adaptation | None for LEO IoT PHY receiver |

### 2C. Complexity / Properties Table

| Approach | Time Complexity | Space Complexity | Params | Hardware Fit | LEO IoT Suitability | Expressiveness |
|---|---|---|---|---|---|---|
| **MMSE (baseline)** | O(N³) per subcarrier block | O(N²) | 0 (analytic) | Well-optimized DSP libs | Poor at low SNR, high Doppler | Limited by Gaussian assumption |
| **Deep unfolding (VAMP/LISTA, T=5–10)** | O(T N) | O(T N) | O(10²–10³) | DSP-friendly; small MAC count | **Good** — light, fast, interpretable | Matches or exceeds MMSE at low SNR |
| **CNN denoiser (2–3 conv layers)** | O(C K N) | O(C K) | O(10³–10⁴) | FPGA CNN accelerator (Zynq) | Moderate — CNN FLOPS manageable if small | Strong at low SNR if trained in-domain |
| **Transformer receiver** | O(N²) per head | O(N²) | O(10⁵–10⁶) | Needs GPU; unsuitable for CubeSat | **Poor** — compute budget exceeds SWaP | High but unnecessary for LEO IoT |
| **End-to-end DNN** | O(L N) | O(L N) | O(10⁵–10⁷) | Needs GPU/VPU; marginal for LEO | **Poor** — training data hard to get; generalization fragile | Very high but overkill |

---

## 3. Novelty Gaps

### Gap 1: Deep-Unfolded Channel Estimation + Equalization for LEO NTN IoT

**Problem:** Deep-unfolded receivers (LISTA, VAMP-Net, DU-SIC) have shown strong results for *terrestrial* MIMO-OFDM (O(10²) parameters replacing O(N³) MMSE). However, no work applies deep unfolding to the *LEO satellite IoT* receiver chain, which differs fundamentally:
- Doppler spreads 10–100× larger than terrestrial (up to 40 kHz)
- SNR 10–20 dB lower (IoT link budget)
- On-board compute ~100–1000× more constrained (CubeSat FPGA vs. base-station GPU)
- Pilot structure is NB-IoT NPUSCH or LR-FHSS, not terrestrial PUSCH

**What exists:** DVAMPNet (2025, IEEE IoT J.) addresses *activity detection* for grant-free access, not the full receiver chain. DUI-SISO-SIC (2024) is for 5G coded MIMO, not NTN. The unfolding literature has not characterized how unfolded receivers behave under the joint stress of extreme Doppler + low SNR + limited pilots.

**What remains missing:**
- An unfolded receiver architecture that integrates CFO tracking, channel estimation, and data detection into a single lightweight unrolled framework matched to NB-IoT NTN or LR-FHSS pilot structures
- Characterization of the performance–complexity Pareto frontier under LEO-relevant SWaP constraints (FPGA LUT/DSP count, power budget)
- Domain adaptation or online learning strategy to handle changing orbital geometries (different elevation angles → different delay–Doppler profiles)

### Gap 2: ML for LR-FHSS Headerless Frame Recovery

**Problem:** The Enhanced LR-FHSS Decoder (Maldonado et al., 2025) uses a classification-based FHS Locator that applies algorithmic search. Explicit neural-network-based headerless frame detection remains unexplored, despite the 2025 survey by Florez et al. calling ML out as the next direction.

**What exists:** Algorithmic FHS pattern matching + Doppler compensation (Maldonado, 2025). No neural LR-FHSS receiver.

**What remains missing:**
- A lightweight CNN or small transformer that maps spectrogram/STFT features directly to header occupancy likelihood, replacing the exhaustive FHS search
- Performance assessment at high device density (>1000 simultaneous transmissions)

### Gap 3: TinyML On-Board Neural Receivers

**Problem:** Choquenaira-Florez et al. (2025) identify that almost all ML-for-satellite-IoT papers ignore on-board compute constraints. Space-grade FPGAs (Xilinx Zynq UltraScale+) have ~600 DSP slices and ~5W power budget for inference. No receiver-ML paper reports actual FPGA resource utilization or measured inference latency for satellite- relevant models.

**What exists:** FPGA accelerator frameworks (FINN, Tensil, Vitis AI) for CNNs in Earth observation. No comparable work for *communications* receiver models.

**What remains missing:**
- A quantized (INT8) unfolded receiver mapped to a Zynq-class FPGA with measured LUT/DSP/BRAM/power/latency characterization
- Pruning + knowledge distillation to reduce unfolded MMSE/VAMP networks from FP32 to INT8 with <0.5 dB loss

### Gap 4: Generalization Across Orbital Geometries

**Problem:** An ML receiver trained on one satellite pass (e.g., 45° elevation, descending node, 600 km altitude) can fail on a different geometry (e.g., 15° elevation grazing pass, ascending node). No work characterizes or mitigates this for LEO IoT receivers.

**What exists:** Domain adaptation for wireless channel modeling (general domain adaptation literature) but not for LEO PHY-layer receiver models.

**What remains missing:**
- Systematic measurement of receiver model degradation across elevation angles, orbital altitudes, and frequency bands
- A lightweight adaptation mechanism (e.g., fine-tuning only 1–2 normalization layers at inference) that can run on board during pass transition

---

## 4. Recommended Direction (Falsifiable Hypothesis)

### Hypothesis

> A deep-unfolded receiver that integrates CFO compensation, channel estimation, and data detection into a single lightweight iterative neural architecture (≤10³ parameters, ≤10⁴ MACs per received block) can outperform a conventional MMSE receiver by ≥2 dB in the LEO satellite IoT regime (Eb/N0 ≤ 0 dB, Doppler spread ≥ 1 kHz, pilot overhead ≤ 10%) while fitting within a Zynq-class FPGA power budget (<5W).

### Justification

**Why this direction:**
1. **Deep unfolding** retains the signal-processing structure (iterative MMSE/AMP updates) so the inductive bias matches the physics — unlike black-box neural receivers, it generalizes outside the training distribution
2. The **parameter count is deliberately constrained** (≤10³) to fit FPGA DSP/memory budgets, addressing the #1 gap identified in the 2025 survey (Choquenaira-Florez et al.)
3. **No existing work** deep-unfolds a joint CFO+channel+detection receiver for the LEO IoT NTN scenario — the pilot structure, Doppler profile, and SNR regime are distinct from terrestrial 5G

### Expected Observable Behavior

- At Eb/N0 = –3 dB with Doppler spread = 1.2 kHz, the unfolded receiver achieves BLER ≤ 10⁻² while MMSE achieves BLER ≥ 10⁻¹
- The unfolded receiver has ≤ 1,000 trainable parameters and ≤ 10,000 MACs per received resource block
- The FPGA implementation (e.g., Xilinx Zynq ZU3EG) consumes < 5W and achieves per-block latency < 1 ms
- At Eb/N0 ≥ 6 dB, the unfolded receiver converges to within 0.5 dB of the ideal known-channel bound

### Falsification Condition

The hypothesis is false if either:
- The unfolded architecture **cannot be constrained** to ≤ 10³ parameters without >1 dB loss relative to a full unfolded receiver (i.e., the Pareto tradeoff is unfavorable), OR
- The unfolded receiver **does not outperform MMSE** by ≥ 1.5 dB at low SNR (Eb/N0 < 0 dB) under LEO-specific Doppler and pilot configurations, OR
- The FPGA implementation **exceeds 5W** or **per-block latency exceeds 10 ms** for the quantized model (i.e., the SWaP constraint cannot be met)

---

## 5. Research Lifecycle Contract

```yaml
task_level: level_2
domain: Scientific ML
subdomain: "Non-Terrestrial Network (NTN) Physical Layer — LEO Satellite IoT Receiver Design"

research_question:
  "Can a deep-unfolded receiver jointly performing CFO compensation, channel estimation,
   and data detection, constrained to ≤1,000 parameters and ≤10,000 MACs per block,
   outperform conventional MMSE by ≥2 dB in the LEO satellite IoT low-SNR regime
   (Eb/N0 ≤ 0 dB, Doppler spread ≥ 1 kHz) while fitting Zynq-class FPGA power budgets?"

novelty_claims:
  - claim: >
      No existing deep-unfolded receiver (DVAMPNet, DUI-SISO-SIC, LISTA variants)
      addresses the joint CFO + channel + detection problem under the specific
      LEO NTN IoT scenario: Doppler spread ≥ 1 kHz, Eb/N0 ≤ 0 dB, NB-IoT NPUSCH
      or LR-FHSS pilot structure, and CubeSat SWaP constraints.
    status: grounded
    evidence:
      - "DVAMPNet (2025, IEEE IoT J.) addresses activity detection only, not full receiver chain"
      - "DUI-SISO-SIC (2024, VTC Fall) targets terrestrial 5G coded MIMO, not NTN"
      - "Lin & Shen (2024, IEEE Commun. Lett.) joint CFO+channel but for generic OFDM, not NTN IoT"
      - "Choquenaira-Florez et al. (2025, Comput. Netw.) survey confirms absence"

  - claim: >
      A lightweight (≤10³ params) unfolded receiver can match or exceed MMSE
      at low SNR with a small fraction of the compute budget, making it feasible
      for space-grade FPGA deployment.
    status: hypothesis  # plausible from terrestrial deep unfolding results, unmeasured for LEO
    evidence:
      - "Unfolded LISTA for sparse recovery uses O(10²) params vs O(N³) for LASSO"
      - "No work has measured this tradeoff under LEO-specific Doppler + pilot constraints"

  - claim: >
      FPGA resource utilization and power consumption for an INT8-quantized unfolded
      receiver can be <5W on Zynq-class devices.
    status: "TODO: unverified"
    evidence:
      - "Remote Sensing 2024 survey confirms FINN/Tensil for Earth-observation CNNs, not comms receivers"
      - "No published FPGA implementation of a communications receiver (folded or unfolded)
         with measured resource/power for satellite hardware exists"

known_related_work:
  - work: "DVAMPNet — Hou et al. (2025), IEEE IoT J."
    covers: "Unfolded VAMP + ResCNN attention for device activity detection and channel estimation in asynchronous grant-free satellite IoT random access; 5.7 dB NMSE gain at 30 dB SNR"
    leaves_open: "Does not address CFO compensation, data detection, or full receiver chain; assumes perfect synchronization; uses 30 dB SNR (not LEO IoT regime of –5 to 0 dB)"

  - work: "DUI-SISO-SIC — AAU (2024), IEEE VTC Fall"
    covers: "Deep-unfolded iterative MMSE-SIC + LDPC decoding for 5G-compliant coded MIMO-OFDM; learnable parameters reduce ordering complexity"
    leaves_open: "Terrestrial only; no CFO tracking; relies on 5G DMRS pilot structure (dense pilots); no characterization at LEO SNR or Doppler extremes"

  - work: "DeepSig OmniPHY-5G (2024) — live 5G Open RAN deployment"
    covers: "Neural channel estimation + equalization for PUSCH; 2–3× cell-edge UL throughput on Intel Xeon; O-RAN compliant; demonstrated in Viettel live network"
    leaves_open: "Terrestrial 5G only; full DNN (not unfolded) parameters >10⁵; runs on x86, not FPGA; no satellite Doppler or long-delay characterization; proprietary model, not reproducible"

  - work: "LR-FHSS Enhanced Receiver — Maldonado et al. (2025), Comput. Netw."
    covers: "Headerless frame recovery via algorithmic FHS pattern search; 50% PDR improvement; accounts for Doppler in LEO"
    leaves_open: "Algorithmic classification, not ML; does not extend to full receiver chain (channel estimation, equalization); no learnable components"

  - work: "CNN+LSTM channel prediction for LEO IoT — Ying et al. (2024), IEEE TWC"
    covers: "CSI prediction for multibeam precoding; handles fast satellite motion"
    leaves_open: "Prediction (not estimation/detection); separate from receiver architecture; focuses on precoder design, not receiver chain"

  - work: "Choquenaira-Florez et al. (2025) survey — Comput. Netw."
    covers: "Comprehensive survey of ML for satellite IoT; identifies on-board compute constraints as #1 gap; no neural receiver work for LEO IoT exists"
    leaves_open: "Survey does not propose receiver architecture; confirms gap"

baseline_requirements:
  - "Conventional MMSE receiver with perfect CFO knowledge (upper bound) and with estimated CFO (realistic bound)"
  - "Least-squares channel estimation with linear interpolation between pilots (3GPP NR-NTN baseline)"
  - "DeepSig OmniPHY-style DNN receiver (where available/reproducible; otherwise a CNN of comparable parameter count)"
  - "For LR-FHSS: the algorithmic Enhanced Decoder (Maldonado et al. 2025) as baseline"
  - "Ideal known-channel bound (genie-aided)"

evaluation_requirements:
  - "BLER vs. Eb/N0 curves from –10 dB to +10 dB at fixed Doppler spread = 600 Hz, 1.2 kHz, 2.4 kHz"
  - "BLER vs. Doppler spread at Eb/N0 = –3 dB (the LEO IoT operating point)"
  - "Parameter count (total trainable) and MAC count per received resource block"
  - "FPGA resource utilization: LUT count, DSP slice count, BRAM count, on-chip power (W) for INT8 quantized model using FINN or Vitis AI"
  - "Per-block inference latency (ms) on target FPGA"
  - "NMSE of channel estimation vs. SNR"
  - "Training data: synthetically generated LEO NTN channel realizations following 3GPP TR 38.811/38.821 (NTN channel models) with varying elevation angles"
  - "Test data: held-out geometies (unseen elevation angle bands, different orbital altitudes)"

blocking_unknowns:
  - "Is there an existing reproducible open-source LEO NTN channel simulator with Python interface (e.g., Sionna-NTN or similar)? If not, building one becomes a prerequisite."
  - "Does the target FPGA board (e.g., Zynq ZU3EG) have sufficient DSP slices for even a minimal unfolded architecture (3–5 unfolded layers with complex multiplications)?"
  - "What is the actual degradation of a receiver trained at 45° elevation when tested at 15° elevation? If degradation exceeds 3 dB, domain adaptation is a blocking sub-problem."
  - "Are published NB-IoT NTN pilot patterns (NPUSCH Format 1/2) sufficiently detailed in 3GPP TR 36.763 to allow a faithful simulation, or are there non-public implementation-specific details?"
  - "Is the LR-FHSS physical layer specification fully open, or does it require Semtech proprietary information?"

claim_status:
  grounded:
    - "No existing deep-unfolded receiver addresses joint CFO+channel+detection for LEO NTN IoT (DVAMPNet, DUI-SISO-SIC, Lin-Shen all leave this gap)"
    - "Conventional MMSE degrades significantly at LEO IoT SNR regime (<0 dB, high Doppler)"
    - "On-board compute constraints are the #1 identified gap in ML-for-SIoT (2025 survey)"
  hypotheses:
    - "A ≤1,000-parameter unfolded receiver can outperform MMSE by ≥2 dB at low SNR under LEO Doppler"
    - "The unfolded architecture can be quantized to INT8 and fit within Zynq-class SWaP (<5W, <600 DSP slices)"
  "TODO: unverified":
    - "FPGA resource utilization for a communications receiver (unfolded) mapped with FINN/Tensil has not been reported"
    - "Amount of performance degradation from orbital geometry mismatch (train vs. test elevation angle) is uncharacterized"
    - "Availability and fidelity of open-source LEO NTN channel simulators for ML training pipeline"
```

---

## 6. Suggested First Steps

1. **Build the simulator**: Implement a LEO NTN IoT link-level simulator in Sionna (NVIDIA's differentiable link-level simulator) or MATLAB/Julia, following 3GPP TR 38.811. Include:
   - Time-varying Doppler (elevation-dependent)
   - NB-IoT NPUSCH or LR-FHSS pilot structure
   - LEO-specific path loss and Rician fading with K-factor from TR 38.811

2. **Design unfolded architecture**: Start with a VAMP-based unfolded channel estimator (3–5 iterations, learnable shrinkage/thresholding). Extend with a small MLP that predicts CFO residual from pilot observations. Connect to a soft MMSE detector. Keep parameter count < 1,000.

3. **Measure the Pareto frontier**: Sweep number of unfolded iterations (2–10) and hidden dimension per layer. Plot BLER vs. parameter count vs. MAC count. Identify the knee where marginal gain diminishes.

4. **FPGA deployment study**: Use FINN (Xilinx) to map the optimal unfolded model to INT8. Report LUT/DSP/BRAM/power/latency. Verify against Zynq ZU3EG datasheet.

---

## Sources

- [DVAMPNet (2025, IEEE IoT J.)](https://ieeexplore.ieee.org/abstract/document/11206356)
- [LEO Random Access with Doppler — Shen et al. (2024/2025)](http://export.arxiv.org/abs/2412.20806)
- [Improved Receiver Chain via Error Location Inference (2025)](https://arxiv.org/html/2509.08869v1)
- [DeepSig + Viettel Neural Receiver (2024)](https://www.deepsig.ai/viettel-high-tech-and-deepsig-announce-industry-first-ai-ml-based-neural-receiver-operating-in-live-5g-network/)
- [DeepSig OmniPHY Commercial Release (2024)](https://www.deepsig.ai/deepsig-announces-industrys-first-neural-receiver-software-for-open-ran-5g-networks/)
- [ML for Satellite IoT Survey — Florez et al. (2025)](https://www.sciencedirect.com/science/article/abs/pii/S1389128625000313)
- [LR-FHSS Enhanced Receiver — Maldonado et al. (2025)](https://www.sciencedirect.com/science/article/abs/pii/S1389128624008508)
- [On-Board ML Inference Hardware Survey — Diana & Dini (2024)](https://www.mdpi.com/2072-4292/16/21/3957)
- [Model-Driven DL Receiver with CFO — Lin & Shen (2024)](https://ieeexplore.ieee.org/document/10403945)
- [Deep-Unfolded SIC Receiver (2024, VTC Fall)](https://vbn.aau.dk/en/publications/deep-unfolded-iterative-soft-input-soft-output-sic-receiver-for-c/)
- [CNN+LSTM Channel Prediction for LEO IoT — Ying et al. (2024)](https://ethicseido.com/en/Iode/DocumentDetail?repoid=arXiv_repo&catId=cs&id=oai%3AarXiv.org%3A2405.17150)
- [IoT-NTN 3GPP Rel-18 Analysis (TR 21.918)](http://mp.weixin.qq.com/s?__biz=MzAwNDAyODM0NA==&mid=2657845068&idx=6&sn=fb49171b14fb3eefdda2f3d4e068e808)
- [Fine CFO Estimation with ML (2025, Scientific Reports)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12050312/)
