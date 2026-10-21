"""
Custom Hugging Face PreTrainedConfig and PreTrainedModel for `scribelm`.

Architecture: `ScribeLMForAuthoring` (~70.8M parameters)
- Backbone: 12-layer Llama/SmolLM2 Causal Transformer (`hidden_size=576`, `num_attention_heads=9`,
  `num_key_value_heads=3`, `intermediate_size=1536`, `vocab_size=49152`) initialized from
  `HuggingFaceTB/SmolLM2-135M-Instruct`.
- Joint Multi-Task Heads:
  1. `lm_head`: Causal language modeling head generating Hemanth HM blog posts, rewrites, and social threads.
  2. `voice_head`: Calibrated `p_hemanth` logit scoring authentic practitioner voice vs corporate/LLM fluff.
  3. `style_head`: 5-axis style telemetry head predicting `(directness, hook_strength, zero_fluff, code_first, cadence)`.
  4. `task_head`: 4-way writing task router `(blog, rewrite, social, critic)`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel


TASK_NAMES = ["blog", "rewrite", "social", "critic"]
TASK_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}
STYLE_AXES = ["directness", "hook_strength", "zero_fluff", "code_first", "cadence"]


class ScribeLMConfig(PretrainedConfig):
    model_type = "scribelm"

    def __init__(
        self,
        base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        vocab_size: int = 49152,
        hidden_size: int = 576,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 9,
        num_key_value_heads: int = 3,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 100000.0,
        num_tasks: int = 4,
        num_style_axes: int = 5,
        dropout: float = 0.1,
        layer_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.num_tasks = num_tasks
        self.num_style_axes = num_style_axes
        self.dropout = dropout
        self.layer_indices = layer_indices or [0, 1, 2, 3, 9, 12, 15, 18, 26, 27, 28, 29]


class ScribeLMForAuthoring(PreTrainedModel):
    config_class = ScribeLMConfig
    _tied_weights_keys = ["causal_lm.lm_head.weight"]
    all_tied_weights_keys = {"causal_lm.lm_head.weight": "causal_lm.model.embed_tokens.weight"}

    def __init__(self, config: ScribeLMConfig, preload_pretrained: bool = False) -> None:
        super().__init__(config)
        self.config = config
        self.all_tied_weights_keys = {"causal_lm.lm_head.weight": "causal_lm.model.embed_tokens.weight"}

        llama_cfg = AutoConfig.from_pretrained(config.base_model_name)
        llama_cfg.num_hidden_layers = config.num_hidden_layers
        llama_cfg._attn_implementation = "eager"

        if preload_pretrained:
            full_lm = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                dtype=torch.float32,
            )
            full_lm.config._attn_implementation = "eager"
            keep = config.layer_indices[: config.num_hidden_layers]
            full_lm.model.layers = nn.ModuleList([full_lm.model.layers[i] for i in keep])
            full_lm.config.num_hidden_layers = config.num_hidden_layers
            self.causal_lm = full_lm
        else:
            llama_cfg.torch_dtype = torch.float32
            self.causal_lm = AutoModelForCausalLM.from_config(llama_cfg).float()

        self.causal_lm = self.causal_lm.float()
        h = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.voice_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )
        self.style_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, config.num_style_axes),
        )
        self.task_head = nn.Linear(h, config.num_tasks)
        self.causal_lm.lm_head.weight = self.causal_lm.model.embed_tokens.weight

    def tie_weights(self, **kwargs: Any) -> None:
        self.causal_lm.lm_head.weight = self.causal_lm.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        voice_labels: Optional[torch.Tensor] = None,
        style_labels: Optional[torch.Tensor] = None,
        task_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        out = self.causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        logits = out.logits
        last_hidden = out.hidden_states[-1]

        mask_f = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        pooled_drop = self.dropout(pooled)

        voice_logit = self.voice_head(pooled_drop).squeeze(-1)
        style_pred = torch.sigmoid(self.style_head(pooled_drop))
        task_logits = self.task_head(pooled_drop)

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "next_token_logits": logits[:, -1, :],
            "voice_logit": voice_logit,
            "p_hemanth": torch.sigmoid(voice_logit),
            "style_scores": style_pred,
            "task_logits": task_logits,
        }

        total_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["lm_loss"] = lm_loss
            total_loss = lm_loss

        if voice_labels is not None:
            v_loss = F.binary_cross_entropy_with_logits(
                voice_logit.view(-1),
                voice_labels.view(-1).to(voice_logit.dtype),
            )
            result["voice_loss"] = v_loss
            total_loss = v_loss if total_loss is None else total_loss + 0.35 * v_loss

        if style_labels is not None:
            s_loss = F.smooth_l1_loss(style_pred, style_labels.to(style_pred.dtype))
            result["style_loss"] = s_loss
            total_loss = s_loss if total_loss is None else total_loss + 0.45 * s_loss

        if task_labels is not None:
            t_loss = F.cross_entropy(task_logits, task_labels.view(-1).long())
            result["task_loss"] = t_loss
            total_loss = t_loss if total_loss is None else total_loss + 0.20 * t_loss

        if total_loss is not None:
            result["loss"] = total_loss

        return result

    @torch.no_grad()
    def generate_text(
        self,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 96,
        temperature: float = 0.25,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> Dict[str, Any]:
        self.eval()
        device = next(self.parameters()).device
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        first_out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        p_hemanth = float(first_out["p_hemanth"][0].item())
        style_vec = [round(float(x), 4) for x in first_out["style_scores"][0].tolist()]
        task_probs = torch.softmax(first_out["task_logits"][0], dim=-1).tolist()
        pred_task_idx = int(torch.argmax(first_out["task_logits"][0]).item())

        generated: List[int] = []
        cur_ids = input_ids
        cur_mask = attention_mask

        eos_id = tokenizer.eos_token_id
        for step in range(max_new_tokens):
            if step == 0:
                next_logits = first_out["next_token_logits"][0].clone()
            else:
                step_out = self.causal_lm(
                    input_ids=cur_ids,
                    attention_mask=cur_mask,
                    use_cache=False,
                )
                next_logits = step_out.logits[0, -1, :].clone()

            # Repetition penalty over recently generated tokens
            for seen_id in set(generated[-48:]):
                if next_logits[seen_id] > 0:
                    next_logits[seen_id] /= repetition_penalty
                else:
                    next_logits[seen_id] *= repetition_penalty

            if temperature <= 1e-4:
                next_token = int(torch.argmax(next_logits).item())
            else:
                scaled = next_logits / temperature
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                sorted_logits[remove_mask] = -float("inf")
                probs = torch.softmax(sorted_logits, dim=-1)
                sampled_idx = int(torch.multinomial(probs, num_samples=1).item())
                next_token = int(sorted_indices[sampled_idx].item())

            if eos_id is not None and next_token == eos_id and len(generated) >= 8:
                break
            generated.append(next_token)
            next_t = torch.tensor([[next_token]], dtype=torch.long, device=device)
            cur_ids = torch.cat([cur_ids, next_t], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones_like(next_t)], dim=1)

        text_out = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "text": text_out,
            "p_hemanth": round(p_hemanth, 4),
            "predicted_task": TASK_NAMES[pred_task_idx],
            "task_probs": {TASK_NAMES[i]: round(float(task_probs[i]), 4) for i in range(len(TASK_NAMES))},
            "style_scores": {STYLE_AXES[i]: style_vec[i] for i in range(len(STYLE_AXES))},
        }
