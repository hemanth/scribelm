# scribelm

Write, rewrite, and critique technical prose in Hemanth HM's voice.

```bash
npm install scribelm
```

## Quick start

```js
import scribe from 'scribelm';

const writer = scribe('Baked nullspan a single-pass structured extraction with exact [start, end] spans.');
const clean = writer.rewrite();
const audit = writer.critique(clean.text);
```

`rewrite()` turns dense specs or corporate drafts into Hemanth HM's punchy voice. `write()` drafts Markdown blog posts. `social()` generates LinkedIn, HN, Reddit, and X threads. `critique()` scores `p_hemanth` across 5 cadence axes.

## License

MIT © [Hemanth.HM](https://h3manth.com)
