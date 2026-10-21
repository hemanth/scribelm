---
license: mit
task_categories:
  - text-generation
  - text-classification
language:
  - en
size_categories:
  - 1K<n<10K
---

# `gnumanth/scribelm-dataset` — Hemanth HM Technical Writing, Style-Transfer & Voice Critic Dataset

Built from **388 real Markdown blog posts (`~230,000` tokens)** and social adaptations in `~/labs/scribe` (`h3manth.com/scribe`), spanning 1,341 multi-task examples (`train.jsonl`: 1,072, `val.jsonl`: 123, `test.jsonl`: 146) across 4 tasks:
- `blog`: Technical blog post generation from title, category, tags, and summary
- `rewrite`: Corporate / dense technical draft $\rightarrow$ Hemanth HM practitioner voice
- `social`: LinkedIn, Hacker News, Reddit (`TL;DR:`), and X launch threads
- `critic`: Calibrated `p_hemanth` authenticity labels + 5-axis style telemetry (`directness`, `hook_strength`, `zero_fluff`, `code_first`, `cadence`)
