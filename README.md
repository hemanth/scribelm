# scribelm

Write, rewrite, and critique technical prose in Hemanth HM's voice across PyTorch (`134.8M`), Browser WebAssembly ONNX (`Int8`), and a Zero-Dependency JS/Python engine.

- **Interactive Playground (Browser WASM ONNX + Zero-Dep Engine)**: [https://hemanth.github.io/scribelm/](https://hemanth.github.io/scribelm/)
- **Fine-Tuned Hugging Face Model (134.8M PyTorch + Int8 ONNX)**: [https://huggingface.co/gnumanth/scribelm](https://huggingface.co/gnumanth/scribelm)
- **Hugging Face Dataset (388 `~/labs/scribe` Posts · 1,341 Multi-Task Items)**: [https://huggingface.co/datasets/gnumanth/scribelm-dataset](https://huggingface.co/datasets/gnumanth/scribelm-dataset)
- **Empirical Science Report**: [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md)

```bash
npm install scribelm
pip install scribelm
```

## Quick start

```js
import scribe from 'scribelm';

const writer = scribe('Skills vs MCP? Organizations must leverage multifaceted paradigms.');
const clean = writer.rewrite();
const audit = writer.critique(clean.text);
```

`rewrite()` strips corporate/LLM filler and restructures dense specs into Hemanth HM's practitioner cadence. `write()` drafts full code-first Markdown blog posts. `social()` generates LinkedIn, Hacker News, Reddit (`TL;DR:`), and X launch threads. `critique()` returns calibrated `p_hemanth` and 5-axis style telemetry (`directness`, `hook_strength`, `zero_fluff`, `code_first`, `cadence`).

## Python + Hugging Face Neural Backend

```python
from scribelm import scribe

writer = scribe("Zero-Overhead Token Budgets", category="python")
post = writer.write()
social = writer.social(post["text"])
score = writer.critique(post["text"])
```

Pass `backend="hf"` or `backend="onnx"` to run the fine-tuned `134.8M` `ScribeLMForAuthoring` transformer checkpoint (`gnumanth/scribelm`).

## License

MIT © [Hemanth.HM](https://h3manth.com)
