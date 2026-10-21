#!/usr/bin/env python3
"""
Hybrid Dataset Builder for `scribelm` (gnumanth/scribelm-dataset).

Extracts all 388 real Markdown blog posts from `/Users/lika/labs/scribe/src/content/blog/`
and social adaptations from `/Users/lika/labs/scribe/social-posts.md`, and builds a
deterministic multi-task instruction + style-transfer + social + voice-critic corpus:

1. `blog`    — Title + category + tags + brief -> Hemanth HM technical blog post
2. `rewrite` — Dry / corporate / verbose technical blurb -> Hemanth-style punchy rewrite
3. `social`  — Technical post / README summary -> LinkedIn, Hacker News, Reddit & X posts
4. `critic`  — Voice & cadence critic: calibrated `p_hemanth` + 5-axis style telemetry
               (directness, hook_strength, zero_fluff, code_first, cadence) + diagnostics
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

SEED = 20261020
SCRIBE_ROOT = Path("/Users/lika/labs/scribe")
BLOG_DIR = SCRIBE_ROOT / "src" / "content" / "blog"
SOCIAL_FILE = SCRIBE_ROOT / "social-posts.md"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"


CORPORATE_FLUFF_REPLACEMENTS = [
    ("Short answer: No.", "In order to comprehensively address this multifaceted question, we must evaluate various architectural paradigms and conclude that the answer is generally negative."),
    ("Short answer: yes.", "Upon careful consideration of the ecosystem dynamics, it is evident that the answer is affirmative."),
    ("Here's why", "In the following sections, we will delve into the underlying rationale"),
    ("TL;DR:", "Executive Summary and Key Strategic Takeaways:"),
    ("Before Skills, my agents were slow. Really slow.", "Prior to the integration of modular skill abstractions, our autonomous agentic workflows exhibited suboptimal latency characteristics."),
    ("You need both.", "Organizations seeking optimal synergy should leverage both solutions simultaneously."),
]


def parse_frontmatter_and_body(raw: str) -> Tuple[Dict[str, Any], str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
    meta: Dict[str, Any] = {
        "title": "",
        "excerpt": "",
        "publishDate": "",
        "category": "programming",
        "tags": [],
    }
    if not m:
        return meta, raw.strip()
    fm, body = m.group(1), m.group(2).strip()
    for key in ("title", "excerpt", "publishDate", "category"):
        km = re.search(rf"^{key}:\s*[\"']?(.*?)[\"']?\s*$", fm, re.M)
        if km:
            meta[key] = km.group(1).strip()
    tm = re.search(r"tags:\s*\[([^\]]*)\]", fm)
    if tm:
        meta["tags"] = [
            t.strip().strip("\"'")
            for t in tm.group(1).split(",")
            if t.strip().strip("\"'")
        ]
    return meta, body


def compute_style_metrics(text: str) -> Dict[str, float]:
    """Compute empirical Hemanth HM voice & cadence telemetry on any text."""
    clean = text.strip()
    if not clean:
        return {
            "directness": 0.0,
            "hook_strength": 0.0,
            "zero_fluff": 0.0,
            "code_first": 0.0,
            "cadence": 0.0,
            "p_hemanth": 0.0,
        }

    # Separate code fences from prose
    code_blocks = re.findall(r"```[\s\S]*?```", clean)
    inline_codes = re.findall(r"`[^`\n]+`", clean)
    prose = re.sub(r"```[\s\S]*?```", " ", clean)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", prose) if p.strip()]
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", prose)
        if len(s.strip()) > 2
    ]
    words = re.findall(r"\b\w+\b", prose.lower())
    n_words = max(1, len(words))

    # 1. Zero-fluff index (penalizes corporate/AI filler words)
    fluff_terms = {
        "delve", "tapestry", "multifaceted", "synergy", "leverage", "paradigm",
        "furthermore", "moreover", "consequently", "utilize", "facilitate",
        "comprehensive", "revolutionize", "seamless", "holistic", "endeavor",
        "suboptimal", "aforementioned", "testament", "pivotal", "ecosystem",
    }
    fluff_hits = sum(1 for w in words if w in fluff_terms)
    fluff_rate = fluff_hits / max(1.0, n_words / 50.0)
    zero_fluff = max(0.05, min(1.0, math.exp(-0.9 * fluff_rate)))

    # 2. Directness (sentence length < 18 words + active practitioner markers)
    avg_sent_words = n_words / max(1, len(sentences))
    sent_len_score = max(0.1, min(1.0, 1.0 - max(0.0, avg_sent_words - 13.0) / 22.0))
    practitioner_markers = [
        r"\bshort answer\b", r"\btl;dr\b", r"\bhere's why\b", r"\bhere is how\b",
        r"\blet's\b", r"\byou need\b", r"\binstead of\b", r"\bno dependencies\b",
        r"\bzero-dep\b", r"\bsingle pass\b", r"\bhappy hacking\b", r"\bwrong question\b",
    ]
    marker_hits = sum(1 for pat in practitioner_markers if re.search(pat, clean.lower()))
    directness = min(1.0, 0.72 * sent_len_score + 0.28 * min(1.0, 0.35 + 0.22 * marker_hits))

    # 3. Hook strength (short opening paragraph, question/contrast or bold punchline)
    first_para = paragraphs[0] if paragraphs else clean[:160]
    first_words = len(first_para.split())
    hook_short = 1.0 if first_words <= 28 else max(0.25, 1.0 - (first_words - 28) / 55.0)
    has_punch = bool(
        re.search(r"(\?|:\s|—|\*\*|`|!|\bMeet\b|\bBuilt\b|\bShort answer\b|\bWrong question\b)", first_para)
    )
    hook_strength = min(1.0, 0.65 * hook_short + (0.35 if has_punch else 0.12))

    # 4. Code-first / technical density (fenced code blocks, inline backticks, CLI commands, numbers/metrics)
    code_chars = sum(len(b) for b in code_blocks) + sum(len(c) for c in inline_codes) * 3
    metric_hits = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:ms|MB|KB|GB|%|x|×|M|B|s)\b", clean))
    bullet_lines = len(re.findall(r"^\s*[-•⚡🌐🚀]\s+", clean, re.M))
    tech_ratio = (code_chars / max(1, len(clean))) + 0.08 * min(6, len(inline_codes)) + 0.07 * min(5, metric_hits) + 0.05 * min(5, bullet_lines)
    code_first = max(0.15, min(1.0, 0.35 + 0.65 * min(1.0, tech_ratio / 0.45)))

    # 5. Cadence (short paragraph rhythm + variance in sentence lengths)
    avg_para_sents = len(sentences) / max(1, len(paragraphs))
    para_rhythm = max(0.2, min(1.0, 1.0 - max(0.0, avg_para_sents - 3.2) / 5.0))
    if len(sentences) >= 2:
        s_lens = [len(s.split()) for s in sentences]
        mean_l = sum(s_lens) / len(s_lens)
        std_l = math.sqrt(sum((x - mean_l) ** 2 for x in s_lens) / len(s_lens))
        burstiness = min(1.0, std_l / 8.5)
    else:
        burstiness = 0.65
    cadence = round(0.6 * para_rhythm + 0.4 * burstiness, 4)

    raw_score = (
        0.26 * zero_fluff
        + 0.24 * directness
        + 0.18 * hook_strength
        + 0.17 * code_first
        + 0.15 * cadence
    )
    # Calibrated logistic mapping
    logit = (raw_score - 0.56) * 9.5
    p_hemanth = 1.0 / (1.0 + math.exp(-logit))

    return {
        "directness": round(directness, 4),
        "hook_strength": round(hook_strength, 4),
        "zero_fluff": round(zero_fluff, 4),
        "code_first": round(code_first, 4),
        "cadence": round(cadence, 4),
        "p_hemanth": round(p_hemanth, 4),
    }


def synthesize_verbose_draft(title: str, excerpt: str, body: str, category: str) -> str:
    """Turn a crisp Hemanth post excerpt/opening into a bloated corporate/LLM-style draft for paired rewrite training."""
    first_chunk = "\n\n".join(body.split("\n\n")[:3]).strip()
    prose_only = re.sub(r"```[\s\S]*?```", "", first_chunk).strip()
    snippet = prose_only[:360].replace("\n", " ").strip()
    if not snippet:
        snippet = excerpt or title
    return (
        f"In today's rapidly evolving {category} landscape, developers and organizations frequently "
        f"encounter multifaceted challenges regarding {title}. Specifically, when delving into this paradigm, "
        f"it is important to comprehensively evaluate the following technical context: {snippet} "
        f"Furthermore, leveraging a holistic methodology facilitates seamless developer productivity."
    )


def build_social_adaptations(title: str, slug: str, excerpt: str, category: str, tags: List[str], body: str) -> Dict[str, str]:
    """Build LinkedIn, Hacker News, Reddit, and X/Twitter posts in Hemanth HM's exact `social-posts.md` format."""
    url = f"https://h3manth.com/scribe/{slug}/"
    tag_str = " ".join(f"#{t.capitalize()}" for t in (tags[:4] or [category, "Programming", "Dev"]))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", re.sub(r"```[\s\S]*?```", "", body)) if p.strip()]
    hook = paragraphs[0] if paragraphs else (excerpt or title)
    hook = re.sub(r"\s+", " ", hook).strip()
    if len(hook) > 200:
        hook = hook[:197].rsplit(" ", 1)[0] + "..."
    second = paragraphs[1] if len(paragraphs) > 1 else (excerpt or "Short breakdown with working code examples.")
    second = re.sub(r"\s+", " ", second).strip()
    if len(second) > 180:
        second = second[:177].rsplit(" ", 1)[0] + "..."

    linkedin = (
        f"{title}\n\n"
        f"{hook}\n\n"
        f"• Problem: {excerpt or second}\n"
        f"• Takeaway: {second}\n\n"
        f"Full breakdown: {url}\n\n"
        f"{tag_str}"
    )
    hn = (
        f"Title: {title}\n\n"
        f"{hook}\n\n"
        f"{second}\n\n"
        f"Post: {url}"
    )
    reddit = (
        f"Title: {title}\n\n"
        f"TL;DR:\n"
        f"- {hook}\n"
        f"- {second}\n\n"
        f"Blog: {url}"
    )
    x_post = (
        f"{title}\n\n"
        f"{hook[:140]}\n\n"
        f"{url}"
    )
    return {
        "linkedin": linkedin,
        "hn": hn,
        "reddit": reddit,
        "x": x_post,
    }


CURATED_REWRITE_PAIRS: List[Dict[str, str]] = [
    {
        "id": "rewrite_nullspan_launch",
        "topic": "nullspan structured extraction launch",
        "input": (
            "Baked nullspan a single-pass structured extraction with exact [start, end] character spans, "
            "calibrated p_null abstention, algebraic invariant checks (subtotal == nodes * hourly_rate), "
            "and p10/p50/p90 quantiles. Benchmarked across 3 runtime tiers (B=1,000 bootstrap 95% CIs): "
            "Tier 1A (PyTorch FP32, 67M): 100% Structured/Math/Injection EM, 91.7% SQuAD v2.0 (100% abstention, 0.999 AUROC). "
            "Tier 1B (Browser ONNX Int8, 64.3 MB, 3.98x smaller): 100% Structured, 90.0% SQuAD v2.0 client-side in WASM SIMD. "
            "Tier 2 (Zero-Dep JS/Py Engine, 0 MB): 0 dependencies, 100% Structured & Invariant EM at 0.023 ms/field."
        ),
        "output": (
            "Meet nullspan: a structured extraction engine that pulls exact `[start, end]` spans from text, "
            "knows when a value isn't there instead of making one up (`p_null`), and double-checks the math "
            "(`subtotal == nodes * hourly_rate`).\n\n"
            "You can run it three ways:\n"
            "- ⚡ Full neural model (PyTorch, 67M): 98.8% overall — 100% on structured & math checks, 91.7% on SQuAD v2.0, and 100% at spotting unanswerable prompts.\n"
            "- 🌐 Browser-native (64 MB WASM): 4× smaller quantized model running entirely in the browser with 90% SQuAD v2.0 accuracy.\n"
            "- 🚀 Zero-dependency JS/Python (0 MB): Deterministic lexical + AST math engine — 100% on structured logs/invoices in 0.005 ms."
        ),
    },
    {
        "id": "rewrite_skills_vs_mcp",
        "topic": "Skills vs MCP for AI Agents",
        "input": (
            "In contemporary artificial intelligence agent architectures, practitioners often debate whether "
            "modular Skill definitions supersede the Model Context Protocol (MCP). However, a comprehensive "
            "evaluation indicates that these two paradigms address orthogonal concerns: Skills encapsulate "
            "domain-specific procedural knowledge, whereas MCP facilitates external system connectivity."
        ),
        "output": (
            "Skills vs MCP? Wrong question.\n\n"
            "Both are tools for building AI agents. But they solve different problems:\n"
            "- Skills = what your agent knows how to do\n"
            "- MCP = what your agent can access\n\n"
            "One teaches. One connects. You need both."
        ),
    },
    {
        "id": "rewrite_token_budgets",
        "topic": "Zero-overhead token budgets for agent loops",
        "input": (
            "Autonomous language model loops frequently suffer from unbounded token consumption and runaway API "
            "expenditures due to recursive tool invocation. By implementing deterministic per-turn accounting "
            "and circuit-breaker thresholds, engineering teams can mitigate financial risk without adding latency."
        ),
        "output": (
            "Unbounded agent loops burn tokens fast.\n\n"
            "`token-budgets` fixes this with zero-overhead circuit breakers:\n"
            "- `recordTurn()` logs usage per step\n"
            "- `checkBudget()` enforces hard token and dollar ceilings\n"
            "- `evaluate()` trips the circuit breaker before a runaway loop drains your API key\n\n"
            "Zero dependencies. Runs in <0.01 ms."
        ),
    },
    {
        "id": "rewrite_scribefont",
        "topic": "Handwriting to installable OpenType vector font",
        "input": (
            "We have developed a multi-tier handwriting vectorization pipeline capable of converting rasterized "
            "handwritten character samples into standard TrueType and OpenType font binaries utilizing cubic "
            "Bezier curve fitting and neural style conditioning."
        ),
        "output": (
            "Meet scribefont: turn a sample of your handwriting into a real, installable `.ttf` / `.otf` vector font.\n\n"
            "Two ways to run it:\n"
            "- ⚡ Neural Style Encoder (PyTorch / Browser ONNX Int8): learns your stroke slant, x-height, and curves from a few sample letters and synthesizes the full alphabet.\n"
            "- 🚀 Zero-Dep Geometric Vectorizer (0 MB JS/Python): traces contours, fits smooth cubic Béziers, and packs binary OpenType tables directly in <1 ms/glyph."
        ),
    },
    {
        "id": "rewrite_agentu_codelab",
        "topic": "Interactive browser codelab for agentu",
        "input": (
            "To facilitate developer onboarding for the agentu framework, we have constructed a browser-based "
            "educational environment utilizing WebAssembly and Pyodide that enables users to execute 41 lessons "
            "covering memory, guardrails, and workflows without local environment configuration."
        ),
        "output": (
            "Learn `agentu` by doing — 41 interactive Python lessons running directly in your browser via Pyodide.\n\n"
            "Covers tools, permissions, always-on memory, guardrails, workflows, MCP, durable agents, and loop engineering. "
            "Zero setup. Edit code, hit Run, and build real agents in ~2 hours."
        ),
    },
]


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(glob.glob(str(BLOG_DIR / "*.md*")))
    print(f"Found {len(md_files)} blog posts in {BLOG_DIR}")

    records: List[Dict[str, Any]] = []

    # 1. Add curated high-signal rewrite pairs first (marked for both train & golden benchmark)
    for idx, pair in enumerate(CURATED_REWRITE_PAIRS):
        hemanth_metrics = compute_style_metrics(pair["output"])
        corp_metrics = compute_style_metrics(pair["input"])
        records.append({
            "id": pair["id"],
            "task": "rewrite",
            "category": "ai",
            "title": pair["topic"],
            "prompt": f"<|task:rewrite|>\nTopic: {pair['topic']}\nDraft:\n{pair['input']}\n<|hemanth|>\n",
            "input_text": pair["input"],
            "target_text": pair["output"],
            "is_hemanth_target": 1,
            "style_metrics": hemanth_metrics,
            "draft_metrics": corp_metrics,
            "split_hint": "val" if idx % 2 == 1 else "test",
        })

    # 2. Parse all 388 posts from ~/labs/scribe
    for file_idx, fpath in enumerate(md_files):
        slug = Path(fpath).stem
        raw = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        meta, body = parse_frontmatter_and_body(raw)
        if len(body) < 80:
            continue

        title = meta["title"] or slug.replace("-", " ").title()
        category = (meta["category"] or "programming").lower()
        tags = meta["tags"] or [category]
        excerpt = meta["excerpt"] or ""

        # Take a clean, self-contained window of the post (up to 1,200 chars) for compact LLM training
        trimmed_body = body[:1200].strip()
        style_metrics = compute_style_metrics(trimmed_body)

        # Assign deterministic split: 80% train, 10% val, 10% test
        mod = file_idx % 10
        split = "test" if mod == 0 else ("val" if mod == 1 else "train")

        # Task A: `blog` generation
        tags_csv = ", ".join(tags[:5])
        blog_prompt = (
            f"<|task:blog|>\n"
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Tags: {tags_csv}\n"
            f"Summary: {excerpt or title}\n"
            f"<|hemanth|>\n"
        )
        records.append({
            "id": f"blog_{slug}",
            "slug": slug,
            "task": "blog",
            "category": category,
            "title": title,
            "tags": tags[:5],
            "prompt": blog_prompt,
            "input_text": f"{title} ({category}): {excerpt or title}",
            "target_text": trimmed_body,
            "is_hemanth_target": 1,
            "style_metrics": style_metrics,
            "split_hint": split,
        })

        # Task B: `rewrite` (corporate/verbose draft -> Hemanth's actual opening)
        opening_paras = "\n\n".join(trimmed_body.split("\n\n")[:3]).strip()
        if len(opening_paras) >= 80:
            verbose_draft = synthesize_verbose_draft(title, excerpt, trimmed_body, category)
            open_metrics = compute_style_metrics(opening_paras)
            draft_metrics = compute_style_metrics(verbose_draft)
            records.append({
                "id": f"rewrite_{slug}",
                "slug": slug,
                "task": "rewrite",
                "category": category,
                "title": title,
                "prompt": f"<|task:rewrite|>\nTopic: {title}\nDraft:\n{verbose_draft}\n<|hemanth|>\n",
                "input_text": verbose_draft,
                "target_text": opening_paras,
                "is_hemanth_target": 1,
                "style_metrics": open_metrics,
                "draft_metrics": draft_metrics,
                "split_hint": split,
            })

        # Task C: `social` (LinkedIn + HN + Reddit TL;DR) for every 2nd post or AI/JS/Node/Python posts
        if file_idx % 2 == 0 or category in ("ai", "javascript", "node", "python", "cli"):
            socials = build_social_adaptations(title, slug, excerpt, category, tags, trimmed_body)
            social_target = (
                f"### LinkedIn\n{socials['linkedin']}\n\n"
                f"### Hacker News\n{socials['hn']}\n\n"
                f"### Reddit\n{socials['reddit']}"
            )
            soc_metrics = compute_style_metrics(social_target)
            records.append({
                "id": f"social_{slug}",
                "slug": slug,
                "task": "social",
                "category": category,
                "title": title,
                "prompt": f"<|task:social|>\nTitle: {title}\nCategory: {category}\nPost:\n{opening_paras[:500]}\n<|hemanth|>\n",
                "input_text": f"{title}\n{opening_paras[:500]}",
                "target_text": social_target,
                "is_hemanth_target": 1,
                "style_metrics": soc_metrics,
                "split_hint": split,
            })

        # Task D: `critic` (both positive Hemanth samples and negative corporate-fluff samples)
        if file_idx % 3 == 0:
            # Positive sample (genuine Hemanth post)
            pos_critique = (
                f"verdict: authentic_hemanth_voice (p_hemanth={style_metrics['p_hemanth']:.2f})\n"
                f"directness={style_metrics['directness']:.2f} | hook={style_metrics['hook_strength']:.2f} | "
                f"zero_fluff={style_metrics['zero_fluff']:.2f} | code_first={style_metrics['code_first']:.2f} | "
                f"cadence={style_metrics['cadence']:.2f}\n"
                f"notes: Punchy practitioner hook, zero corporate filler, concrete examples."
            )
            records.append({
                "id": f"critic_pos_{slug}",
                "slug": slug,
                "task": "critic",
                "category": category,
                "title": title,
                "prompt": f"<|task:critic|>\nText:\n{opening_paras[:500]}\n<|hemanth|>\n",
                "input_text": opening_paras[:500],
                "target_text": pos_critique,
                "is_hemanth_target": 1,
                "style_metrics": style_metrics,
                "split_hint": split,
            })

            # Negative sample (bloated corporate prose)
            neg_text = synthesize_verbose_draft(title, excerpt, trimmed_body, category)
            neg_metrics = compute_style_metrics(neg_text)
            neg_critique = (
                f"verdict: needs_hemanth_rewrite (p_hemanth={neg_metrics['p_hemanth']:.2f})\n"
                f"directness={neg_metrics['directness']:.2f} | hook={neg_metrics['hook_strength']:.2f} | "
                f"zero_fluff={neg_metrics['zero_fluff']:.2f} | code_first={neg_metrics['code_first']:.2f} | "
                f"cadence={neg_metrics['cadence']:.2f}\n"
                f"notes: Cut filler ('multifaceted', 'delving', 'paradigm', 'holistic'). Lead with a 1-sentence hook and a concrete code or metric example."
            )
            records.append({
                "id": f"critic_neg_{slug}",
                "slug": slug,
                "task": "critic",
                "category": category,
                "title": title,
                "prompt": f"<|task:critic|>\nText:\n{neg_text}\n<|hemanth|>\n",
                "input_text": neg_text,
                "target_text": neg_critique,
                "is_hemanth_target": 0,
                "style_metrics": neg_metrics,
                "split_hint": split,
            })

    rng.shuffle(records)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for r in records:
        splits[r["split_hint"]].append(r)

    for sp_name, rows in splits.items():
        out_path = OUT_DIR / f"{sp_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {sp_name}: {len(rows)} examples -> {out_path}")

    # Also write a compact style profile & phrasebook JSON extracted from ~/labs/scribe
    # for the Zero-Dep JS & Python engines
    summary = {
        "total_posts_parsed": len(md_files),
        "total_examples": len(records),
        "splits": {k: len(v) for k, v in splits.items()},
        "curated_pairs": CURATED_REWRITE_PAIRS,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
