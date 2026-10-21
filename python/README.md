# scribelm

Write, rewrite, and critique technical prose in Hemanth HM's voice.

```bash
pip install scribelm
```

## Quick start

```python
from scribelm import scribe

writer = scribe("Zero-Overhead Token Budgets", category="python")
post = writer.write()
audit = writer.critique(post["text"])
```

`rewrite()` strips corporate filler into Hemanth HM's practitioner voice. `write()` drafts code-first Markdown posts. `social()` generates LinkedIn, HN, Reddit, and X threads. `critique()` scores `p_hemanth` across 5 cadence axes.

## License

MIT © [Hemanth.HM](https://h3manth.com)
