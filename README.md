> **Project layout** — this bundle contains five stage directories from the
> AI-Designer pipeline:
> `research/` (literature survey), `architect/` (blueprint + `ModelConfig`),
> `coder/` (PyTorch implementation), `validator/` (tests + benchmarks), and
> `documenter/` (this README plus `docs/` and `CHANGELOG.md`).
> An optional `paper/` directory holds the NeurIPS-format writeup when the
> paper-generation step was triggered.
>
> The original research request that produced this bundle is preserved
> verbatim in [`prompt.md`](prompt.md) — if any URLs in the prompt were
> fetched server-side for additional context, their cleaned contents are
> appended there too.

---

# VAMP-JCD: Deep-Unfolded VAMP Joint Channel Estimation and Data Detection Receiver for LEO Satellite IoT

A hybrid model-driven + data-driven receiver chain for NB-IoT Non-Terrestrial Network (NTN) physical layer, using deep-unfolded Vector Approximate Message Passing (VAMP) with only 24 learnable parameters.

This work addresses the receiver design problem for low-Earth orbit satellite IoT systems, where conventional MMSE receivers degrade significantly at low SNR (Eb/N0 ≤ 0 dB) under high Doppler spreads (≥1 kHz). The VAMP-JCD receiver integrates CFO compensation, channel estimation, and data detection into a single lightweight iterative architecture designed to fit within CubeSat FPGA power budgets (<5W). This is the first deep-unfolded receiver to jointly address CFO + channel estimation + data detection for the LEO NTN IoT scenario (NB-IoT NPUSCH Format 1, 12 subcarriers × 14 symbols per slot).

> **Status:** The architecture, implementation, and benchmark infrastructure are complete. Model training on synthetic LEO NTN channel data (3GPP TR 38.811) is required before BLER measurements become scientifically meaningful. See [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md#research-quality-evaluation) for known gaps.

## Highlights

- **24 learnable parameters** — delay-Doppler shrinkage denoising with weight tying across 5 unfolded VAMP iterations; see [docs/ARCHITECTURE.md#3-the-core-component](documenter/docs/ARCHITECTURE.md#3-the-core-component)
- **Joint CFO + channel + detection** — per-iteration residual CFO tracking with analytic-anchored learned correction; see [docs/ARCHITECTURE.md#5-design-decisions](documenter/docs/ARCHITECTURE.md#5-design-decisions)
- **Delay-Doppler domain sparsity** — transforms channel estimate to delay-Doppler domain where the LEO satellite channel is sparse (few propagation paths); see [docs/ARCHITECTURE.md#33-reference-implementation-walk-through](documenter/docs/ARCHITECTURE.md#33-reference-implementation-walk-through)
- **Baseline comparison infrastructure** — LMMSE, LS, and Genie-aided bounds with shared interface; see [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md)
- **8 single-field ablations** — depth, weight tying, denoiser type, CFO mode, soft feedback, damping strategy, quantization sensitivity, noise precision; see [docs/BENCHMARKS.md#ablation-study](documenter/docs/BENCHMARKS.md#ablation-study)

## Quick start

```bash
pip install -r requirements.txt
python coder/model.py            # smoke test — prints param count + output shapes
python coder/smoke_test.py       # extended smoke test suite
pytest validator/test_model.py -v --tb=short  # full unit-test suite
```

## Repository layout

```
coder/
  model.py          — VAMP-JCD receiver, config dataclass, denoiser variants,
                     CFO estimator/compensator, delay-Doppler transform,
                     VAMP linear step, soft MMSE detector
  channel.py        — LEO NTN channel simulator (3GPP TR 38.811 Rician fading)
  baselines.py      — LMMSE, LS, Genie-aided bound receivers
  train.py          — Training harness, loss functions (NMSE+BCE+sparsity),
                     evaluation metrics (BLER, BER)
  smoke_test.py     — End-to-end verification (instantiation, forward pass,
                     numerics, torch.compile)

validator/
  test_model.py     — Comprehensive pytest suite (shapes, gradients, numerics,
                     domain benchmarks, ablations, profiling)
  run_benchmarks.py — BLER vs SNR, BLER vs Doppler, NMSE comparison scripts
  ablation_runner.py— Single-field ablation experiment runner
  profile_model.py  — Parameter count, MAC count, inference speed profiling
  research_eval/    — Scorecard, claim grounding, experiment coverage, rubric
  ablation_results.json — All ablation run results

research/           — Landscape summary, novelty gaps, hypothesis formulation
architect/          — ModelConfig, pseudocode, inductive bias justifications,
                     traceability table
```

## Documentation

- [docs/ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md) — full design, equations, inductive biases
- [docs/TRAINING.md](documenter/docs/TRAINING.md) — training recipe, environment, troubleshooting
- [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md) — results, ablations, profiling, research-quality evaluation
- [docs/API.md](documenter/docs/API.md) — module-level API reference

## Citation

```bibtex
@misc{vampjcd-leo-iot-2026,
  title  = {VAMP-JCD: Deep-Unfolded VAMP Joint Channel Estimation and Data Detection
            for LEO Satellite IoT},
  author = {TODO},
  year   = {2026},
  note   = {Generated via ml-designer pipeline}
}
```
