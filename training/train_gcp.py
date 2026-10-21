#!/usr/bin/env python3
"""
Fine-tuning & Probability Calibration pipeline for `scribelm` (`ScribeLMForAuthoring`, ~70.8M params).

Trains jointly on Hemanth HM's 388 posts from `/Users/lika/labs/scribe`:
- Causal LM generation (`blog`, `rewrite`, `social`, `critic`)
- Calibrated `p_hemanth` voice authenticity probability (`BCEWithLogitsLoss`)
- 5-axis style telemetry (`directness`, `hook_strength`, `zero_fluff`, `code_first`, `cadence`)
- 4-way writing task router (`blog`, `rewrite`, `social`, `critic`)
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from modeling_scribelm import STYLE_AXES, TASK_TO_ID, ScribeLMConfig, ScribeLMForAuthoring

SEED = 20261020
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts" / "scribelm"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class ScribeDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_len: int = 192) -> None:
        self.items: List[Dict[str, torch.Tensor]] = []
        eos = tokenizer.eos_token or "<|im_end|>"
        for row in rows:
            prompt = row["prompt"]
            target = row["target_text"].strip() + eos
            p_ids = tokenizer.encode(prompt, add_special_tokens=False)
            t_ids = tokenizer.encode(target, add_special_tokens=False)

            max_p = max_len // 2
            if len(p_ids) > max_p:
                p_ids = p_ids[:max_p]
            avail_t = max_len - len(p_ids)
            t_ids = t_ids[:avail_t]

            input_ids = p_ids + t_ids
            labels = [-100] * len(p_ids) + t_ids
            attn = [1] * len(input_ids)

            pad_len = max_len - len(input_ids)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            if pad_len > 0:
                input_ids = input_ids + [pad_id] * pad_len
                labels = labels + [-100] * pad_len
                attn = attn + [0] * pad_len

            sm = row["style_metrics"]
            # For critic_neg rows, the input text is corporate fluff (is_hemanth_target=0)
            is_hem = float(row.get("is_hemanth_target", 1))
            style_vec = [float(sm[k]) for k in STYLE_AXES]
            task_id = TASK_TO_ID.get(row["task"], 0)

            self.items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "voice_labels": torch.tensor(is_hem, dtype=torch.float32),
                "style_labels": torch.tensor(style_vec, dtype=torch.float32),
                "task_labels": torch.tensor(task_id, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


def compute_calibration_metrics(probs: List[float], labels: List[int], n_bins: int = 10) -> Dict[str, float]:
    n = max(1, len(probs))
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n

    # 10-bin ECE
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bucket = [
            (p, y)
            for p, y in zip(probs, labels)
            if (lo <= p < hi) or (b == n_bins - 1 and lo <= p <= hi)
        ]
        if bucket:
            avg_p = sum(x[0] for x in bucket) / len(bucket)
            avg_y = sum(x[1] for x in bucket) / len(bucket)
            ece += (len(bucket) / n) * abs(avg_p - avg_y)

    # Mann-Whitney U AUROC
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if not pos or not neg:
        auroc = 1.0
    else:
        wins = 0.0
        for pp in pos:
            for nn_p in neg:
                if pp > nn_p:
                    wins += 1.0
                elif pp == nn_p:
                    wins += 0.5
        auroc = wins / (len(pos) * len(neg))

    return {
        "auroc": round(auroc, 4),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
    }


@torch.no_grad()
def evaluate_model(model: ScribeLMForAuthoring, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_lm_loss = 0.0
    total_style_mae = 0.0
    task_correct = 0
    total_items = 0
    voice_probs: List[float] = []
    voice_targets: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        voice_labels = batch["voice_labels"].to(device)
        style_labels = batch["style_labels"].to(device)
        task_labels = batch["task_labels"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            voice_labels=voice_labels,
            style_labels=style_labels,
            task_labels=task_labels,
        )
        bsz = input_ids.size(0)
        total_items += bsz
        total_lm_loss += float(out["lm_loss"].item()) * bsz
        total_style_mae += float(torch.abs(out["style_scores"] - style_labels).mean().item()) * bsz
        preds_task = torch.argmax(out["task_logits"], dim=-1)
        task_correct += int((preds_task == task_labels).sum().item())

        voice_probs.extend(out["p_hemanth"].cpu().tolist())
        voice_targets.extend([int(round(x)) for x in voice_labels.cpu().tolist()])

    cal = compute_calibration_metrics(voice_probs, voice_targets)
    avg_lm = total_lm_loss / max(1, total_items)
    ppl = math.exp(min(12.0, avg_lm))
    return {
        "val_lm_loss": round(avg_lm, 4),
        "val_perplexity": round(ppl, 4),
        "val_task_accuracy": round(task_correct / max(1, total_items), 4),
        "val_style_mae": round(total_style_mae / max(1, total_items), 4),
        "voice_auroc": cal["auroc"],
        "voice_ece": cal["ece"],
        "voice_brier": cal["brier"],
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training `scribelm` on device: {device}")

    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "val.jsonl")

    # Up-sample curated & AI/JS/Python rewrite/critic examples for high-signal convergence
    curated_and_recent = [
        r for r in train_rows
        if r["category"] in ("ai", "javascript", "node", "python") or r["id"].startswith("rewrite_")
    ]
    sampled_train = curated_and_recent[:320] + train_rows[:240]
    random.shuffle(sampled_train)
    print(f"Active training slice: {len(sampled_train)} examples | Validation: {len(val_rows)} examples")

    cfg = ScribeLMConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ScribeLMForAuthoring(cfg, preload_pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized ScribeLMForAuthoring: {n_params / 1e6:.2f}M parameters")

    train_ds = ScribeDataset(sampled_train, tokenizer, max_len=160)
    val_ds = ScribeDataset(val_rows, tokenizer, max_len=160)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=12, shuffle=False)

    epochs = 2
    optimizer = AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(4, int(0.08 * total_steps)),
        num_training_steps=total_steps,
    )

    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                voice_labels=batch["voice_labels"].to(device),
                style_labels=batch["style_labels"].to(device),
                task_labels=batch["task_labels"].to(device),
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | "
                    f"Loss: {running_loss / step:.4f} | LM: {float(out['lm_loss'].item()):.4f} | "
                    f"Task: {float(out['task_loss'].item()):.4f}"
                )

        metrics = evaluate_model(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = round(running_loss / max(1, len(train_loader)), 4)
        history.append(metrics)
        print(f"Validation Epoch {epoch}: {json.dumps(metrics)}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu")
    model_cpu.config.auto_map = {
        "AutoConfig": "modeling_scribelm.ScribeLMConfig",
        "AutoModel": "modeling_scribelm.ScribeLMForAuthoring",
    }
    model_cpu.save_pretrained(ARTIFACT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(ARTIFACT_DIR)
    shutil.copy2(ROOT / "training" / "modeling_scribelm.py", ARTIFACT_DIR / "modeling_scribelm.py")

    cal_info = {
        "model_id": "gnumanth/scribelm",
        "parameters": n_params,
        "training_seconds": round(time.time() - t0, 2),
        "epochs": epochs,
        "final_validation": history[-1],
        "history": history,
    }
    (ARTIFACT_DIR / "calibration_metrics.json").write_text(json.dumps(cal_info, indent=2), encoding="utf-8")
    print(f"Saved fine-tuned checkpoint to {ARTIFACT_DIR} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
