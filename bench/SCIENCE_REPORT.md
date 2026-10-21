# `scribelm` — Empirical & Statistical Evaluation Report (`SCIENCE_REPORT.md`)

> **Corpus**: `388` real Markdown blog posts + social threads authored by Hemanth HM (`/Users/lika/labs/scribe`, `~230,000` tokens across 15+ years).
> **Evaluation Protocol**: Held-out test split (`N = 146` multi-task items spanning `blog`, `rewrite`, `social`, and `critic`), evaluated with `B = 1000` non-parametric percentile bootstrap 95% confidence intervals (`seed = 20261020`).

---

## 1. Three-Tier Runtime Architecture Comparison (`N = 146`, `B = 1000`)

| System / Runtime Tier | Disk Size | Composite Score | Voice Auth Acc (`95% CI`) | Task Router Acc (`95% CI`) | 5-Axis Style Fidelity (`95% CI`) | `p_hemanth` AUROC / ECE / Brier | Rewrite Gain (`Δ p_hemanth`) | Latency `p50 / p95` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `scribelm-Hybrid (Fast + ONNX Int8)`** | `157.84 MB` | **`99.59%`** | **`100.0%`** `[100.0, 100.0]` | **`100.0%`** `[100.0, 100.0]` | **`98.63%`** `[97.89, 99.26]` | `1.0` / `0.1498` / `0.0297` | `+0.402` | **`0.666 / 1.728 ms`** |
| **2. `scribelm-135M (PyTorch FP32)`** | `514.45 MB` | **`91.7%`** | **`96.58%`** `[93.84, 99.32]` | **`91.1%`** `[86.3, 95.89]` | **`85.79%`** `[84.62, 86.89]` | `1.0` / `0.037` / `0.0266` | `+0.402` | `155.177 / 209.586 ms` |
| **3. `scribelm-135M-Int8 (ONNX Browser/CPU)`** | `157.84 MB` | **`89.57%`** | **`92.47%`** `[88.36, 96.58]` | **`89.73%`** `[84.25, 94.52]` | **`85.54%`** `[84.24, 86.83]` | `1.0` / `0.068` / `0.0551` | `+0.402` | `135.594 / 160.153 ms` |
| **4. `scribelm-ZeroDep (Fast Engine · 0 MB)`** | `0 MB` | **`99.64%`** | **`100.0%`** `[100.0, 100.0]` | **`100.0%`** `[100.0, 100.0]` | **`98.79%`** `[98.08, 99.36]` | `1.0` / `0.1531` / `0.0321` | `+0.402` | **`0.584 / 1.443 ms`** |
| **5. `SmolLM2-Control (Unfinetuned Baseline)`** | `514.45 MB` | `58.4%` | `91.1%` `[86.3, 95.89]` | `7.53%` `[3.42, 11.64]` | `65.66%` `[64.62, 66.62]` | `0.4482` / `0.3873` / `0.2311` | `+0.0` | `148.337 / 185.63 ms` |

---

## 2. Five-Axis Voice & Cadence Telemetry

Every piece of text is evaluated across five empirical dimensions learned from Hemanth HM's 388 `~/labs/scribe` posts:
1. **`directness`**: Short declarative sentences (`<18` words average) + practitioner transition markers (*"Short answer:"*, *"Here's why"*, *"TL;DR:"*, *"One teaches. One connects."*).
2. **`hook_strength`**: Concise opening paragraph (`≤28` words) leading with a concrete contrast, question, or bold thesis.
3. **`zero_fluff`**: Exponential penalty against corporate/AI filler vocabulary (*"delve"*, *"multifaceted"*, *"paradigm"*, *"synergy"*, *"holistic"*, *"leverage"*).
4. **`code_first`**: Ratio of fenced code blocks, inline backtick identifiers, CLI snippets, and concrete empirical metrics (`ms`, `MB`, `%`, `×`) to prose.
5. **`cadence`**: Paragraph rhythm (`~2–3` sentences per paragraph) combined with sentence-length burstiness.
