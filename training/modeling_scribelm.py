"""
Custom Hugging Face PreTrainedConfig and PreTrainedModel for `scribelm`.

Architecture: `ScribeLMForAuthoring` (~134.8M parameters)
- Backbone: Intact 30-layer Llama/SmolLM2 Causal Transformer (`hidden_size=576`, `num_attention_heads=9`,
  `num_key_value_heads=3`, `intermediate_size=1536`, `vocab_size=49152`) initialized from
  `HuggingFaceTB/SmolLM2-135M-Instruct` with bottom-20-layer grammar preservation and top-10-layer
  Hemanth HM style adaptation.
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
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

_TF_V5 = int(transformers.__version__.split(".")[0]) >= 5

TASK_NAMES = ["blog", "rewrite", "social", "critic"]
TASK_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}
STYLE_AXES = ["directness", "hook_strength", "zero_fluff", "code_first", "cadence"]

HEMANTH_SYSTEM_PROMPT = (
    "You are Hemanth HM (h3manth.com/scribe). Write direct, punchy, zero-fluff practitioner prose "
    "with short paragraphs, concrete metrics, and bullet points."
)


class ScribeLMConfig(PretrainedConfig):
    model_type = "scribelm"

    def __init__(
        self,
        base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        vocab_size: int = 49152,
        hidden_size: int = 576,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 9,
        num_key_value_heads: int = 3,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 100000.0,
        num_tasks: int = 4,
        num_style_axes: int = 5,
        dropout: float = 0.1,
        freeze_bottom_layers: int = 20,
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
        self.freeze_bottom_layers = freeze_bottom_layers


class ScribeLMForAuthoring(PreTrainedModel):
    config_class = ScribeLMConfig
    _tied_weights_keys = (
        {"causal_lm.lm_head.weight": "causal_lm.model.embed_tokens.weight"}
        if _TF_V5
        else ["causal_lm.lm_head.weight"]
    )

    def __init__(self, config: ScribeLMConfig, preload_pretrained: bool = False) -> None:
        super().__init__(config)
        self.config = config
        if _TF_V5:
            self.all_tied_weights_keys = {"causal_lm.lm_head.weight": "causal_lm.model.embed_tokens.weight"}

        llama_cfg = AutoConfig.from_pretrained(config.base_model_name)
        llama_cfg.num_hidden_layers = config.num_hidden_layers
        llama_cfg._attn_implementation = "eager"

        if preload_pretrained:
            full_lm = AutoModelForCausalLM.from_pretrained(config.base_model_name)
            full_lm.config._attn_implementation = "eager"
            self.causal_lm = full_lm.float()
        else:
            self.causal_lm = AutoModelForCausalLM.from_config(llama_cfg).float()

        self.causal_lm = self.causal_lm.float()
        self.causal_lm.lm_head.weight = self.causal_lm.model.embed_tokens.weight

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

    def tie_weights(self, **kwargs: Any) -> None:
        self.causal_lm.lm_head.weight = self.causal_lm.model.embed_tokens.weight

    def freeze_lower_layers(self, num_freeze: Optional[int] = None) -> None:
        n_f = self.config.freeze_bottom_layers if num_freeze is None else num_freeze
        self.causal_lm.model.embed_tokens.weight.requires_grad = False
        for name, p in self.causal_lm.named_parameters():
            if "model.layers." in name:
                layer_idx = int(name.split("model.layers.")[1].split(".")[0])
                if layer_idx < n_f:
                    p.requires_grad = False

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
        user_prompt: str,
        max_new_tokens: int = 120,
        repetition_penalty: float = 1.12,
    ) -> Dict[str, Any]:
        self.eval()
        device = next(self.parameters()).device
        messages = [
            {"role": "system", "content": HEMANTH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(device)

        gen_ids = self.causal_lm.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )
        new_tokens = gen_ids[0][enc["input_ids"].shape[1] :]
        text_out = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        full_mask = torch.ones_like(gen_ids)
        scored_out = self.forward(
            input_ids=gen_ids,
            attention_mask=full_mask,
        )
        p_hemanth = float(scored_out["p_hemanth"][0].item())
        style_vec = [round(float(x), 4) for x in scored_out["style_scores"][0].tolist()]
        task_probs = torch.softmax(scored_out["task_logits"][0], dim=-1).tolist()
        pred_task_idx = int(torch.argmax(scored_out["task_logits"][0]).item())

        return {
            "text": text_out,
            "p_hemanth": round(p_hemanth, 4),
            "predicted_task": TASK_NAMES[pred_task_idx],
            "task_probs": {TASK_NAMES[i]: round(float(task_probs[i]), 4) for i in range(len(TASK_NAMES))},
            "style_scores": {STYLE_AXES[i]: style_vec[i] for i in range(len(STYLE_AXES))},
        }
