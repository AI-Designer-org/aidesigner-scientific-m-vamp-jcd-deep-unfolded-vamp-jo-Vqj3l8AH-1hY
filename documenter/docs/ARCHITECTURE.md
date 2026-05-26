# Architecture

## 1. Motivation

Conventional LEO satellite IoT receivers (NB-IoT NTN Release 17–18) use pilot-aided MMSE channel estimation followed by MMSE/ZF data detection. At terrestrial SNR levels (5–15 dB), this chain performs near the Cramer-Rao bound. However, the LEO IoT link budget operates at Eb/N0 ≤ 0 dB, often as low as –10 dB, where the Gaussian assumptions underlying MMSE break down. Simultaneously, the satellite's velocity of ~7.5 km/s produces Doppler spreads up to 40 kHz (at 2 GHz), causing rapid channel variation within a single OFDM slot (14 symbols at 15 kHz subcarrier spacing). The combination of low SNR and high Doppler renders conventional MMSE ineffective.

The deep unfolding literature offers a principled alternative: unroll iterative estimation algorithms (AMP, VAMP, ISTA) into neural network layers with learnable parameters. This preserves the algorithmic structure (strong inductive bias matching the physics) while allowing data-driven adaptation. However, existing unfolded receivers (DVAMPNet 2025 for device activity detection, DUI-SISO-SIC 2024 for terrestrial 5G MIMO, Lin & Shen 2024 for joint CFO+channel in generic OFDM) each address only a subset of the receiver chain and have not been characterized under LEO NTN-specific constraints.

The Choquenaira-Florez et al. (2025) survey identifies on-board compute constraints as the single largest gap in ML-for-satellite-IoT research. No existing work reports FPGA resource utilization for a neural receiver in the satellite context.

**Hypothesis:** A deep-unfolded VAMP receiver integrating CFO compensation, channel estimation, and data detection into ≤1,000 parameters and ≤10,000 MACs per block can outperform conventional MMSE by ≥2 dB in the LEO IoT regime (Eb/N0 ≤ 0 dB, Doppler spread ≥ 1 kHz) while fitting within Zynq-class FPGA power budgets (<5W).

## 2. At a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│           DEEP-UNFOLDED VAMP-JCD RECEIVER ARCHITECTURE              │
│         (NB-IoT NPUSCH, 12 SC x 14 Sym x 5 Iterations)             │
└─────────────────────────────────────────────────────────────────────┘

Input: Y in C^{12x14}  (received IQ after coarse synch & CP removal)
            |
            v
+---------------------------+
|  Pilot Extraction         |
|  Y_p = Y[pilot_mask]      |
|  X_p = known_pilot_symbols |
+----------+----------------+
           |
           v
+---------------------------+
|  Coarse CFO Estimation    |  <-- analytic: autocorrelation of pilots
|  Delta_f0 = f(Y_p, X_p)  |      (parameter-free)
+----------+----------------+
           |
           v
+---------------------------+
|  CFO Correction           |
|  Y_0 = Y * exp(-j*2pi*Delta_f0*t) |
+----------+----------------+
           |
    +------+------+
    |  +-------------------------------------+
    |  |                                     v
    |  |    +-------------------------------------+
    |  |    |  Layer t = 1 .. T   (T=5)           |
    |  |    |                                     |
    |  |    |  +-----------------------------+    |
    |  |    |  | a) CFO Residual Correction  |    |
    |  |    |  |    Delta_f_{t-1} -> apply   |    |
    |  |    |  +----------+------------------+    |
    |  |    |             v                       |
    |  |    |  +-----------------------------+    |
    |  |    |  | b) VAMP Linear Step         |    |
    |  |    |  |    LMMSE channel estimate   |    |  gamma_z (noise precision)
    |  |    |  |    H_lin = (gamma_z|X|^2    |    |  gamma_h (channel prior)
    |  |    |  |           + gamma_h)^{-1}   |    |
    |  |    |  |           * (gamma_z X* Y   |    |
    |  |    |  |           + gamma_h r1-u1)  |    |
    |  |    |  +----------+------------------+    |
    |  |    |             v                       |
    |  |    |  +-----------------------------+    |
    |  |    |  | c) Delay-Doppler Denoiser   |    |  theta_t (learnable
    |  |    |  |    IFFT(freq)->FFT(time)    |    |  shrinkage
    |  |    |  |    Soft threshold (real,imag)|    |  thresholds)
    |  |    |  |    IFFT(time)->FFT(freq)    |    |
    |  |    |  +----------+------------------+    |
    |  |    |             v                       |
    |  |    |  +-----------------------------+    |
    |  |    |  | d) VAMP State Update        |    |  alpha (damping)
    |  |    |  |    H_t = alpha*H_den        |    |  beta (Onsager)
    |  |    |  |          + (1-alpha)*H_{t-1} |    |
    |  |    |  |    u1 = beta*(H_den-H_lin)  |    |
    |  |    |  +----------+------------------+    |
    |  |    |             v                       |
    |  |    |  +-----------------------------+    |
    |  |    |  | e) Soft MMSE Detection      |    |
    |  |    |  |    MMSE equalizer per RE    |    |
    |  |    |  |    LLR computation          |    |
    |  |    |  |    Soft symbol feedback     |-------> iter t+1
    |  |    |  |    (pilot positions fixed)  |    |
    |  |    |  +----------+------------------+    |
    |  |    |             v                       |
    |  |    |  +-----------------------------+    |
    |  |    |  | f) CFO Residual Est.       |    |
    |  |    |  |    Pilot phase error        |    |
    |  |    |  |    Delta_f_t = Delta_f_{t-1}|    |  s, b (scale, bias)
    |  |    |  |                 + learned   |    |
    |  |    |  +-----------------------------+    |
    |  |    |                                     |
    |  -----+  (feedback: H_t, X_soft, Delta_f_t) |
    |                                             |
    +---------------------------------------------+
                        |
                        v
+-------------------------------+
|  Final Detection & LLR Output |
|  X_final, LLRs for decoder   |
|  H_final (channel estimate)   |
+-------------------------------+

LEARNABLE PARAMETERS (total: 24, shared across T=5 iterations):

+------------------------------------------------------------------+
| theta_dd[3x3x2] = 18 delay-Doppler shrinkage thresholds          |
| gamma_z          =  1 noise precision scalar                      |
| gamma_h          =  1 channel prior precision scalar              |
| alpha            =  1 damping factor (sigmoid-parametrized)       |
| beta             =  1 Onsager correction coefficient              |
| s, b             =  2 CFO scale and bias                          |
| TOTAL            = 24 trainable parameters                        |
+------------------------------------------------------------------+
```

| Property | Value |
|---|---|
| Parameter count (default config) | 24 (well within ≤1,000 budget) |
| Time complexity | O(T x N_sc x N_sym x log(N_sc x N_sym)) per block, T=5 |
| Space complexity | O(N_sc x N_sym) intermediate activations |
| Hardware requirements | GPU for training (2 GB VRAM sufficient); CPU inference; FPGA (Zynq ZU3EG) target for deployment |
| Custom kernels | None required — uses standard torch.fft and torch.nn operations |

## 3. The core component

### 3.1 Intuition

The VAMP-JCD receiver unfolds 5 iterations of the VAMP algorithm into a neural network. Each iteration corresponds to one pass through a "linear estimation → denoising → state update" loop. Think of it as an iterative refinement: start with a rough channel estimate from pilots, use the physics (the measurement model y = Hx + w) to project it back to the feasible set, then remove noise in the delay-Doppler domain where the channel is sparse, then use the improved channel to make better symbol decisions, which in turn improves the channel estimate in the next iteration. The 24 learnable parameters control only the thresholds for noise removal, the damping rate, and a small CFO correction — the heavy lifting (matrix inversions, FFTs) is done by fixed signal-processing operations that encode the physics.

The CFO correction works similarly: a coarse analytic estimate from pilot autocorrelation provides a strong prior, and a learned scalar scale+bias adjusts the residual at each iteration. This hybrid design keeps the inductive bias strong while allowing data-driven refinement.

### 3.2 Equations

**Received signal model (per resource element):**

y[t, f] = h[t, f] · x[t, f] · exp(j · 2π · Δf · t) + w[t, f]

where y is the received IQ sample, h is the channel coefficient, x is the transmitted symbol, Δf is the CFO in cycles/symbol, and w is AWGN with precision γ_z = 1/σ²_w.

**VAMP Linear Step (LMMSE channel estimate):**

h_lin = (γ_z · |x|² + γ_h)⁻¹ · (γ_z · conj(x) · y + γ_h · r1 − u1)

where γ_h is the channel prior precision, r1 is the VAMP reference from the previous denoiser step, and u1 is the Onsager correction.

**Delay-Doppler Transform:**

H_dd(τ, ν) = Σ_{f} Σ_{t} h(t, f) · exp(−j·2π·(f·τ − t·ν))

Implemented as IFFT over subcarriers (delay) followed by FFT over symbols (Doppler).

**Soft Thresholding in Delay-Doppler Domain:**

H_dd_shrunk = sign(H_dd) · ReLU(|H_dd| − θ(τ_bin, ν_bin))

where θ(τ_bin, ν_bin) is the learnable threshold for the delay-Doppler bin. There are 3 delay bins × 3 Doppler bins × 2 (real/imag) = 18 thresholds with weight tying (shared across iterations).

**VAMP State Update:**

h_new = α · h_den + (1 − α) · h_prev
u1_{next} = β · (h_den − h_lin)
r1_{next} = h_new  (with stop-gradient)

where α ∈ (0, 1) is the damping factor (sigmoid-parametrized) and β is the learnable Onsager correction coefficient.

**Soft MMSE Detection:**

x_eq = conj(h) · y / (|h|² + 1/γ_z)
LLR(b₀) = 2·√2 · Re(x_eq) · SNR_eff
LLR(b₁) = 2·√2 · Im(x_eq) · SNR_eff
SNR_eff = |h|² · γ_z

Soft symbols (for next iteration): x_soft = tanh(LLR/2) mapped to QPSK constellation.

### 3.3 Reference implementation walk-through

The forward pass is implemented in `VAMPJCDReceiver._forward_impl()` in `coder/model.py`. Here is the annotated core loop (simplified from lines 1000–1114):

```python
def _forward_impl(self, y, pilots, pilot_mask, return_full_state):
    B = y.shape[0]

    # Step 0: Initial CFO estimate from pilot autocorrelation (analytic, 0 params)
    delta_f = self.cfo_estimator(y, pilots, pilot_mask)  # (B,)

    # Apply coarse CFO correction by phase rotation per OFDM symbol
    y_corrected = self.cfo_compensator(y, delta_f)       # (B, 12, 14, 2)

    # Initialize VAMP state: channel, auxiliary variables
    h_hat = torch.zeros(B, 12, 14, 2, device=device)
    r1 = torch.zeros_like(h_hat)
    u1 = torch.zeros_like(h_hat)
    x_known[pilot_mask] = pilots  # anchor known pilots

    gamma_z = exp(log_noise_precision)   # noise precision, 1 param
    gamma_h = exp(log_channel_precision) # channel prior precision, 1 param

    for t in range(self.T):  # T=5 unfolded iterations
        # a) CFO residual correction (apply latest CFO estimate)
        y_corrected = self.cfo_compensator(y, delta_f)

        # b) VAMP linear step: LMMSE channel estimate
        h_lin = self.vamp_linear(y_corrected, x_known, h_hat, r1, u1, gamma_z, gamma_h)

        # c) Delay-Doppler denoising:
        #    IFFT over subcarriers (delay) -> FFT over symbols (Doppler)
        h_dd = self.dd_transform(h_lin)                    # (B, 12, 14, 2)
        h_dd_denoised = self.denoiser(h_dd, t)             # soft-threshold, 18 params
        h_den = self.dd_transform.inverse(h_dd_denoised)   # back to time-frequency

        # d) VAMP state update: damping + Onsager correction
        alpha = sigmoid(logit_damping[0])    # damping in (0, 1), 1 param
        h_new = alpha * h_den + (1 - alpha) * h_hat
        beta = onsager_coeff[0]              # Onsager strength, 1 param
        u1 = beta * (h_den - h_lin)
        r1 = h_new.detach()                  # stop-gradient for state evolution
        h_hat = h_new

        # e) Soft data detection (if feedback enabled)
        x_soft_detected, llrs_t = self.detector(y_corrected, h_hat, gamma_z)
        x_known = x_soft_detected.clone()    # data REs updated
        x_known[pilot_mask] = pilots         # pilots re-inserted

        # f) CFO residual: re-estimate from corrected signal, apply learned scale+bias
        delta_f_residual = self.cfo_estimator.analytic_estimate(y_corrected, ...)
        delta_f_residual = delta_f_residual * self.cfo_estimator.cfo_scale  # 1 param
        delta_f_residual = delta_f_residual + self.cfo_estimator.cfo_bias    # 1 param
        delta_f = delta_f + delta_f_residual

    # Final detection
    x_final, llrs_final = self.detector(y_corrected, h_hat, gamma_z)
    return h_hat, x_final, llrs_final, state
```

Key shapes at each stage:
- Input `y`: (B, 12, 14, 2) — 12 subcarriers × 14 OFDM symbols, last dim = (real, imag)
- `h_lin`, `h_hat`, `h_den`: (B, 12, 14, 2) — channel estimates (time-frequency)
- `h_dd`: (B, 12, 14, 2) — channel in delay-Doppler domain
- `llrs`: (B, 12, 14, 2) — per-bit LLRs for QPSK (2 bits/symbol)
- `delta_f`: (B,) — CFO estimate per batch element in cycles/symbol

## 4. Tensor shape evolution

| Stage | Shape | Notes |
|---|---|---|
| Input (received IQ) | (B, 12, 14, 2) | float32, last dim = (real, imag) |
| After CFO correction | (B, 12, 14, 2) | phase rotation per OFDM symbol |
| VAMP linear step (LMMSE) | (B, 12, 14, 2) | per-RE LMMSE estimate |
| Delay-Doppler transform | (B, 12, 14, 2) | IFFT over SCs, FFT over symbols |
| Shrinkage denoising | (B, 12, 14, 2) | soft-threshold per delay-Doppler bin |
| Inverse DD transform | (B, 12, 14, 2) | IFFT over symbols, FFT over SCs |
| VAMP state update | (B, 12, 14, 2) | damping blend + Onsager correction |
| Soft MMSE detection | (B, 12, 14, 2) | equalized symbols |
| LLR output | (B, 12, 14, 2) | QPSK: 2 bits per symbol |

All intermediate tensors maintain the same spatial dimensions (12, 14). No reshaping or flattening occurs. The last dimension is always 2 for (real, imag) or (LLR_b0, LLR_b1).

## 5. Design decisions

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| **Deep unfolding (not black-box DNN)** | End-to-end ResNet/Transformer receiver | Preserves algorithmic structure of iterative VAMP. The update equations encode y=Hx+w; at test time with out-of-distribution geometries, the architecture degrades gracefully to base VAMP. Black-box DNNs hallucinate. | Parameter count capped at 24; cannot learn entirely new signal representations |
| **VAMP over AMP** | AMP (approximate message passing) | AMP diverges for structured (non-i.i.d.) measurement matrices. NB-IoT pilot matrix is structured. VAMP's state evolution converges for right-rotationally invariant matrices (Rangan et al. 2017, 2019). | Slightly higher per-iteration complexity (LMMSE inverse vs element-wise) |
| **Delay-Doppler domain denoising** | Element-wise thresholding, CNN denoising | LEO satellite channel is approximately sparse in delay-Doppler (few propagation paths). Shrinkage in this domain exploits sparsity directly. CNN denoiser is 62 params vs 18 for DD shrinkage with similar expected performance. | Requires forward+inverse FFT each iteration (~2,500 MACs) |
| **Weight tying across iterations** | Per-layer thresholds | Forces the denoiser to learn a single shrinkage function for all VAMP iterations. Regularizes the model (24 vs 104 params) and prevents overfitting. | May limit expressivity; if ablation shows >0.3 dB gap, switch to per-layer |
| **LMMSE (not ZF) in linear step** | Zero-forcing equalization | MMSE accounts for noise via gamma_z; ZF does not. At LEO IoT SNR (–10 to 0 dB), noise amplification from ZF is catastrophic. | Requires 1 learnable parameter (gamma_z) vs ZF's 0 |
| **Analytic + learned CFO (not pure MLP)** | Pure MLP CFO estimator, pure analytic | The analytic estimate provides a strong prior (unbiased, consistent). Learned scale/bias corrects only the residual error at 2 extra params. An MLP (270 params) would need far more data. | May not capture nonlinear CFO-ICI coupling at extreme Doppler (>2.4 kHz) |
| **Soft MMSE detection with feedback** | Hard decision feedback, no feedback | Soft symbols preserve gradient flow and reduce error propagation at low SNR. Hard decisions are unreliable at Eb/N0 < 0 dB. | 336 extra MACs per iteration for soft symbol computation |
| **Damping via sigmoid-parametrized alpha** | Clamped alpha, learnable alpha directly | Sigmoid gives alpha in (0, 1) naturally without clipping. Avoids gradient issues at boundaries. | One extra exp() per forward pass (negligible) |
| **Pilot mask enforcement** | No re-insertion of pilots | Prevents the iterative process from drifting at known positions. Anchors estimation to pilot observations. | Slightly more complex gradient flow (masked assignment) |
| **No dropout** | Dropout in unfolded layers | Incompatible with deterministic VAMP state evolution. Architecture already regularized by ultra-low parameter count. | None |
| **No layer normalization** | LayerNorm on IQ samples | IQ samples have known physical statistics (bounded power, known noise). Normalization destroys this information. | No benefit from normalizing physical quantities |

## 6. Domain-specific considerations

### A. Scientific ML

| Consideration | Implementation |
|---|---|
| **Physics constraints** | The VAMP linear step enforces the exact measurement model y = Hx + w. No neural network output can violate the observation equation — the LMMSE step projects back to the feasible set. Physics is enforced via algorithm structure, not a soft penalty. |
| **Function vs. operator learning** | Function learning: maps received signal to transmitted message. Not operator learning (mapping between function spaces). |
| **Symmetry/equivariance** | Phase rotation equivariance: a constant phase shift on all subcarriers produces the same shift in the channel estimate. No explicit SE(3) equivariance needed (2D time-frequency, not 3D). |
| **Mesh type** | Regular 2D grid (12 × 14). FFT-based transforms are natural for this mesh. |
| **Rollout stability** | VAMP state evolution is contractive under standard conditions (Rangan et al. 2017) for T < 10. Each OFDM slot processed independently; no error accumulation across slots. |

### B. Time Series (Secondary Domain)

| Consideration | Implementation |
|---|---|
| **Temporal ordering** | Non-causal within a slot (bidirectional FFT over symbols for Doppler domain). The receiver has the whole slot before processing. A causal streaming variant would use a causal sliding-window FFT. |
| **Channel mixing vs. independence** | Channel mixing is inherent — subcarriers are coupled through delay-Doppler transforms. This is correct for OFDM where subcarrier coupling arises from delay spread. |
| **Stationarity** | LEO channel is non-stationary within a pass (Doppler shift, delay spread, K-factor change with elevation angle). Architecture handles this by processing each slot independently. |
| **Prediction head** | Direct multi-step: all 168 REs in the slot processed simultaneously (not auto-regressively). |

### C. Physical Layer Communications

| Consideration | Implementation |
|---|---|
| **Complex baseband representation** | All signals are complex-valued. Architecture uses 2-channel (real, imag) representation throughout. Conjugates, Hermitian operations, complex multiplies are explicit. |
| **Modulation awareness** | QPSK LLR computation uses the modulation constellation explicitly. Extending to higher-order modulations (16-QAM) requires only changing the LLR function. |
| **Pilot structure dependence** | Designed for NB-IoT NPUSCH Format 1 (24 pilots at symbols 3 and 10). Adapting to other patterns (LR-FHSS, NR-NTN DMRS) requires only changing `pilot_mask` and `pilot_symbol_indices` config fields. |
| **Channel code interface** | Output is per-bit LLRs for a soft-input channel decoder (Turbo, LDPC, or polar). The decoder is not included — this is a receiver front-end. |

## 7. Known limitations

- **No trained model results available.** All benchmark measurements in ablation_results.json are from randomly initialized weights. BLER ≈ 1.0 at all tested SNR levels is expected for an untrained model. The central hypothesis (≥2 dB improvement over MMSE at –3 dB) cannot be validated without training. See [TRAINING.md](TRAINING.md) for the training recipe.
- **MAC budget exceeded at T=5.** Estimated ~15k MACs per block vs the 10k target. The architecture meets the budget at T=3 (~9k MACs). The T=3 variant should be characterized for FPGA deployment.
- **FPGA resource utilization unmeasured.** The INT8 quantization target and Zynq ZU3EG SWaP targets are defined in the config but no synthesis, resource utilization, or power measurements exist. This requires the FINN/Vitis AI toolchain, which is not part of this pipeline.
- **Held-out elevation angle evaluation not performed.** All benchmarks use 45° elevation. Generalization to 15° and 75° (where the channel has different K-factor and Doppler profiles) is untested.
- **Channel simulator uses simplified Rician model** following 3GPP TR 38.811, not the full TR 38.821 specification. Elevation-dependent delay spread and atmospheric attenuation variations are not modeled.
- **BLER measurement is uncoded.** Hard decisions on LLRs without a Turbo/LDPC decoder. True BLER with channel coding will differ.
- **DeepSig OmniPHY comparison not possible.** The full DNN neural receiver (10⁵+ params) is proprietary and not reproducible. The TinyCNNDenoiser (62 params) serves as an alternative but is not equivalent to a production neural receiver.
- **VAMP state evolution guarantees are asymptotic.** The rigorous convergence proofs assume the large-system limit. For the small-system regime (N=168 REs), the learned Onsager correction (β) is an empirical relaxation without formal guarantees.
