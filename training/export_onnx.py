#!/usr/bin/env python3
"""
Browser-Safe ONNX Export & QUInt8 Dynamic Quantization for `scribelm`.

Exports `ScribeLMForAuthoring` with eager attention (`opset_version=17`, `dynamo=False`)
and quantizes weights to `QUInt8` (`onnx/model_quantized.onnx`, ~68 MB) for
client-side WebAssembly SIMD (`onnxruntime-web`) and edge CPU execution.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from modeling_scribelm import ScribeLMConfig, ScribeLMForAuthoring

ARTIFACT_DIR = ROOT / "artifacts" / "scribelm"
ONNX_DIR = ARTIFACT_DIR / "onnx"


class ScribeLMOnnxWrapper(nn.Module):
    """Pure Tensor-In, Tensor-Tuple-Out wrapper for WASM-safe ONNX export."""

    def __init__(self, core: ScribeLMForAuthoring) -> None:
        super().__init__()
        self.core = core
        self.core.causal_lm.config._attn_implementation = "eager"

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.core.causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        logits = out.logits
        last_hidden = out.hidden_states[-1]
        mask_f = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        voice_logit = self.core.voice_head(pooled).squeeze(-1)
        p_hemanth = torch.sigmoid(voice_logit)
        style_scores = torch.sigmoid(self.core.style_head(pooled))
        task_logits = self.core.task_head(pooled)
        next_token_logits = logits[:, -1, :]
        return next_token_logits, p_hemanth, style_scores, task_logits


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ScribeLMConfig.from_pretrained(ARTIFACT_DIR)
    model = ScribeLMForAuthoring.from_pretrained(ARTIFACT_DIR, config=cfg, dtype=torch.float32)
    model.eval()

    wrapper = ScribeLMOnnxWrapper(model).eval()

    dummy_ids = torch.randint(0, 1000, (1, 24), dtype=torch.long)
    dummy_mask = torch.ones((1, 24), dtype=torch.long)

    fp32_path = ONNX_DIR / "model_fp32.onnx"
    int8_path = ONNX_DIR / "model_quantized.onnx"

    print(f"Exporting FP32 ONNX to {fp32_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["next_token_logits", "p_hemanth", "style_scores", "task_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "next_token_logits": {0: "batch"},
            "p_hemanth": {0: "batch"},
            "style_scores": {0: "batch"},
            "task_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(fp32_path), load_external_data=True)
    onnx.save_model(onnx_model, str(fp32_path), save_as_external_data=False)
    for ext_f in ONNX_DIR.glob("*.data"):
        ext_f.unlink(missing_ok=True)

    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    print(f"Consolidated FP32 ONNX: {fp32_mb:.2f} MB")

    print(f"Quantizing to QUInt8 -> {int8_path}...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )
    int8_mb = int8_path.stat().st_size / (1024 * 1024)
    ratio = fp32_mb / max(1e-6, int8_mb)
    print(f"Quantized Int8 ONNX: {int8_mb:.2f} MB ({ratio:.2f}x smaller)")

    # Verify with ONNX Runtime across multiple sequence lengths
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    for seq_len in (12, 32, 64):
        ids_np = np.random.randint(0, 1000, size=(1, seq_len), dtype=np.int64)
        mask_np = np.ones((1, seq_len), dtype=np.int64)
        outputs = sess.run(None, {"input_ids": ids_np, "attention_mask": mask_np})
        print(f"Verified ORT seq_len={seq_len}: p_hemanth={float(outputs[1][0]):.4f}, style={outputs[2][0].round(3).tolist()}")

    # Remove the intermediate FP32 ONNX to save disk space while keeping model.safetensors + model_quantized.onnx
    fp32_path.unlink(missing_ok=True)

    meta = {
        "fp32_mb": round(fp32_mb, 2),
        "int8_mb": round(int8_mb, 2),
        "compression_ratio": round(ratio, 2),
    }
    (ONNX_DIR / "onnx_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("ONNX export and quantization complete.")


if __name__ == "__main__":
    main()
