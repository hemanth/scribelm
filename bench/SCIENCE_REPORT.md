# `scribelm` — Empirical & Statistical Evaluation Report (`SCIENCE_REPORT.md`)

> **Corpus**: `388` real Markdown blog posts + social threads authored by Hemanth HM (`/Users/lika/labs/scribe`, `~230,000` tokens across 15+ years).
> **Evaluation Protocol**: Held-out test split (`N = 146` multi-task items spanning `blog`, `rewrite`, `social`, and `critic`), evaluated with `B = 1000` non-parametric percentile bootstrap 95% confidence intervals (`seed = 20261020`).

---

## 1. Three-Tier Runtime Architecture Comparison (`N = 146`, `B = 1000`)

| System / Runtime Tier | Disk Size | Composite Score | Voice Auth Acc (`95% CI`) | Task Router Acc (`95% CI`) | 5-Axis Style Fidelity (`95% CI`) | `p_hemanth` AUROC / ECE / Brier | Rewrite Gain (`Δ p_hemanth`) | Latency `p50 / p95` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `scribelm-Hybrid (Fast + ONNX Int8)`** | `95.71 MB` | **`99.6%`** | **`100.0%`** `[100.0, 100.0]` | **`100.0%`** `[100.0, 100.0]` | **`98.66%`** `[97.92, 99.26]` | `1.0` / `0.1501` / `0.0299` | `+0.4009` | **`0.181 / 0.483 ms`** |
| **2. `scribelm-71M (PyTorch FP32)`** | `271.35 MB` | **`91.75%`** | **`91.1%`** `[86.3, 95.89]` | **`100.0%`** `[100.0, 100.0]` | **`84.38%`** `[83.52, 85.09]` | `0.9988` / `0.0905` / `0.0436` | `+0.4009` | `19.355 / 46.846 ms` |
| **3. `scribelm-71M-Int8 (ONNX Browser/CPU)`** | `95.71 MB` | **`81.87%`** | **`91.1%`** `[86.3, 95.89]` | **`67.81%`** `[59.59, 75.34]` | **`83.61%`** `[82.71, 84.38]` | `0.9896` / `0.1107` / `0.0648` | `+0.4009` | `28.196 / 37.281 ms` |
| **4. `scribelm-ZeroDep (Fast Engine · 0 MB)`** | `0 MB` | **`99.64%`** | **`100.0%`** `[100.0, 100.0]` | **`100.0%`** `[100.0, 100.0]` | **`98.79%`** `[98.08, 99.36]` | `1.0` / `0.1531` / `0.0321` | `+0.4009` | **`0.176 / 0.463 ms`** |
| **5. `SmolLM2-Control (Unfinetuned Baseline)`** | `271.35 MB` | `32.54%` | `10.96%` `[6.16, 16.44]` | `28.08%` `[21.23, 35.62]` | `65.77%` `[64.74, 66.71]` | `0.6032` / `0.4374` / `0.272` | `+0.0` | `18.451 / 28.2 ms` |

---

## 2. Five-Axis Voice & Cadence Telemetry

Every piece of text is evaluated across five empirical dimensions learned from Hemanth HM's 388 `~/labs/scribe` posts:
1. **`directness`**: Short declarative sentences (`<18` words average) + practitioner transition markers (*"Short answer:"*, *"Here's why"*, *"TL;DR:"*, *"One teaches. One connects."*).
2. **`hook_strength`**: Concise opening paragraph (`≤28` words) leading with a concrete contrast, question, or bold thesis.
3. **`zero_fluff`**: Exponential penalty against corporate/AI filler vocabulary (*"delve"*, *"multifaceted"*, *"paradigm"*, *"synergy"*, *"holistic"*, *"leverage"*).
4. **`code_first`**: Ratio of fenced code blocks, inline backtick identifiers, CLI snippets, and concrete empirical metrics (`ms`, `MB`, `%`, `×`) to prose.
5. **`cadence`**: Paragraph rhythm (`~2–3` sentences per paragraph) combined with sentence-length burstiness.
