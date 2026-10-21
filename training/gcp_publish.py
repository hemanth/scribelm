#!/usr/bin/env python3
"""
GCP Post-Benchmark Publisher (`training/gcp_publish.py`).

1. Syncs the empirical benchmark table from `bench/results.json` into `docs/index.html`.
2. Generates the Hugging Face Model Card (`artifacts/scribelm/README.md`) including
   live qualitative outputs from the intact 30-layer `134.8M` `ScribeLMForAuthoring` model.
3. Uploads `artifacts/scribelm` (`model.safetensors`, `onnx/model_quantized.onnx`, tokenizer,
   and `modeling_scribelm.py`) directly from GCP to `https://huggingface.co/gnumanth/scribelm`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts" / "scribelm"
BENCH_RESULTS = ROOT / "bench" / "results.json"
DOCS_INDEX = ROOT / "docs" / "index.html"


def sync_docs_and_card() -> None:
    res = json.loads(BENCH_RESULTS.read_text(encoding="utf-8"))
    systems = {s["id"]: s for s in res["systems"]}
    s_hyb = systems["scribelm_hybrid"]
    s_zd = systems["scribelm_zerodep"]
    s_pt = systems["scribelm_pt_fp32"]
    s_ox = systems["scribelm_onnx_int8"]
    s_ct = systems["smollm2_control"]

    def fmt_row(idx: str, s: dict, bold: bool = True) -> str:
        tag_o = "<strong>" if bold else ""
        tag_c = "</strong>" if bold else ""
        v_m = s["voice_accuracy"]["mean"]
        v_ci = s["voice_accuracy"]["ci95"]
        t_m = s["task_accuracy"]["mean"]
        t_ci = s["task_accuracy"]["ci95"]
        sf_m = s["style_fidelity"]["mean"]
        sf_ci = s["style_fidelity"]["ci95"]
        auroc = s["calibration"]["auroc"]
        ece = s["calibration"]["ece"]
        p50 = s["latency"]["p50_ms"]
        label = f"{idx}. {s['name']}" if idx else s["name"]
        fp = f"{s['footprint_mb']} MB" if s["footprint_mb"] > 0 else "0 MB"
        return f"""              <tr>
                <td>{tag_o}{label}{tag_c}</td>
                <td>{tag_o}{fp}{tag_c}</td>
                <td>{tag_o}{s['composite_accuracy_pct']}%{tag_c}</td>
                <td>{tag_o}{v_m}% [{v_ci[0]}, {v_ci[1]}]{tag_c}</td>
                <td>{tag_o}{t_m}% [{t_ci[0]}, {t_ci[1]}]{tag_c}</td>
                <td>{tag_o}{sf_m}% [{sf_ci[0]}, {sf_ci[1]}]{tag_c}</td>
                <td>{tag_o}{auroc:.4f} / {ece:.4f}{tag_c}</td>
                <td>{tag_o}{p50} ms{tag_c}</td>
              </tr>"""

    tbody_html = (
        '<tbody id="bench-tbody">\n'
        + "\n".join(
            [
                fmt_row("1", s_hyb, True),
                fmt_row("2", s_zd, True),
                fmt_row("3", s_pt, True),
                fmt_row("4", s_ox, True),
                fmt_row("", s_ct, False),
            ]
        )
        + "\n            </tbody>"
    )

    html_src = DOCS_INDEX.read_text(encoding="utf-8")
    html_updated = re.sub(
        r'<tbody id="bench-tbody">[\s\S]*?</tbody>',
        tbody_html,
        html_src,
        count=1,
    )
    DOCS_INDEX.write_text(html_updated, encoding="utf-8")
    print("Updated docs/index.html benchmark table.")

    samples_md = ""
    for sp in res.get("qualitative_samples", []):
        samples_md += (
            f"### {sp['title']} (`{sp.get('task', 'rewrite')}`)\n"
            f"- **Input Draft**: `{sp['draft']}`\n"
            f"- **Neural Generation (`134.8M`, `p_hemanth={sp['neural_p_hemanth']}`)**:\n\n"
            f"```markdown\n{sp['neural_generation']}\n```\n\n"
        )

    model_card = f"""---
license: mit
language:
  - en
library_name: transformers
tags:
  - scribelm
  - hemanth-hm
  - causal-lm
  - style-transfer
  - writing-assistant
  - onnx
  - wasm
datasets:
  - gnumanth/scribelm-dataset
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
pipeline_tag: text-generation
---

# `gnumanth/scribelm` — 134.8M Multi-Head Authoring & Voice Critic Model (Hemanth HM Style)

`scribelm` (`ScribeLMForAuthoring`) is an intact **30-layer, 134.85M-parameter** causal transformer and 5-axis voice critic fine-tuned on **388 Markdown blog posts (`~230,000` tokens across 15+ years)** from [h3manth.com/scribe](https://h3manth.com/scribe).

## Architecture (`ScribeLMForAuthoring`)

- **Backbone**: Intact 30-layer `HuggingFaceTB/SmolLM2-135M-Instruct` (`d_model=576`, `n_heads=9`, `n_kv_heads=3`, `intermediate_size=1536`, `vocab_size=49,152`, eager attention for deterministic browser ONNX export, bottom 20 layers frozen to preserve fluent English & code syntax).
- **Four Joint Heads**:
  1. `lm_head`: Causal autoregressive language modeling head for writing blog posts, rewriting dense specs, and drafting social launch threads.
  2. `voice_head`: Calibrated `p_hemanth` authenticity classifier (`AUROC = {s_pt['calibration']['auroc']}`, `ECE = {s_pt['calibration']['ece']}`).
  3. `style_head`: 5-axis style telemetry regressor predicting `(directness, hook_strength, zero_fluff, code_first, cadence)`.
  4. `task_head`: 4-way writing task router `(blog, rewrite, social, critic)` (`{s_pt['task_accuracy']['mean']}%` accuracy).

## Three Runtime Tiers (`N = 146` Held-Out Test Split · `B = 1,000` Bootstrap 95% CIs)

| Runtime Tier | Artifact | Size | Composite | Voice Auth Acc | Task Acc | 5-Axis Fidelity | `p_hemanth` AUROC | `p50` Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1A (PyTorch FP32)** | `model.safetensors` | `{s_pt['footprint_mb']} MB` | **`{s_pt['composite_accuracy_pct']}%`** | `{s_pt['voice_accuracy']['mean']}%` | `{s_pt['task_accuracy']['mean']}%` | `{s_pt['style_fidelity']['mean']}%` | `{s_pt['calibration']['auroc']}` | `{s_pt['latency']['p50_ms']} ms` |
| **Tier 1B (ONNX Int8 Browser/CPU)** | `onnx/model_quantized.onnx` | `{s_ox['footprint_mb']} MB` | **`{s_ox['composite_accuracy_pct']}%`** | `{s_ox['voice_accuracy']['mean']}%` | `{s_ox['task_accuracy']['mean']}%` | `{s_ox['style_fidelity']['mean']}%` | `{s_ox['calibration']['auroc']}` | `{s_ox['latency']['p50_ms']} ms` |
| **Tier 2 (Zero-Dep JS/Py Engine)** | `scribelm` npm / PyPI | `0 MB` | **`{s_zd['composite_accuracy_pct']}%`** | `{s_zd['voice_accuracy']['mean']}%` | `{s_zd['task_accuracy']['mean']}%` | `{s_zd['style_fidelity']['mean']}%` | `{s_zd['calibration']['auroc']}` | `{s_zd['latency']['p50_ms']} ms` |
| **Hybrid (`Fast + ONNX Int8`)** | Combined | `{s_hyb['footprint_mb']} MB` | **`{s_hyb['composite_accuracy_pct']}%`** | `{s_hyb['voice_accuracy']['mean']}%` | `{s_hyb['task_accuracy']['mean']}%` | `{s_hyb['style_fidelity']['mean']}%` | `{s_hyb['calibration']['auroc']}` | `{s_hyb['latency']['p50_ms']} ms` |

## Live Qualitative Generations (`134.8M` Neural Checkpoint)

{samples_md}

## Usage (Python + Transformers)

```python
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

repo_id = "gnumanth/scribelm"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, config=config, trust_remote_code=True)

res = model.generate_text(
    tokenizer,
    "Rewrite this into Hemanth HM style:\\nIn contemporary AI agent architectures, practitioners debate whether modular Skill definitions supersede MCP.",
    max_new_tokens=96,
)
print(res["text"])
print(res["p_hemanth"], res["style_scores"])
```
"""
    (ARTIFACT_DIR / "README.md").write_text(model_card, encoding="utf-8")
    print("Wrote artifacts/scribelm/README.md")


def upload_to_hf() -> None:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    api.create_repo(repo_id="gnumanth/scribelm", repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id="gnumanth/scribelm",
        repo_type="model",
        folder_path=str(ARTIFACT_DIR),
        commit_message="Publish GCP-trained intact 30-layer 134.8M ScribeLMForAuthoring FP32 + Int8 ONNX",
    )
    print("Uploaded artifacts/scribelm to https://huggingface.co/gnumanth/scribelm")


if __name__ == "__main__":
    sync_docs_and_card()
    upload_to_hf()
