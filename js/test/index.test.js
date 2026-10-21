import test from 'node:test';
import assert from 'node:assert/strict';
import scribe, { critiqueText, rewriteText, writePost, generateSocial } from '../src/index.js';

test('critiqueText distinguishes authentic Hemanth voice from corporate fluff', () => {
  const hemanthText = `Skills vs MCP? Wrong question.\n\nBoth are tools for building AI agents. But they solve different problems:\n- Skills = what your agent knows how to do\n- MCP = what your agent can access\n\nOne teaches. One connects. You need both.`;
  const corpText = `In today's rapidly evolving landscape, organizations must comprehensively delve into multifaceted paradigms and leverage holistic synergy to facilitate suboptimal workflows.`;

  const pos = critiqueText(hemanthText);
  const neg = critiqueText(corpText);

  assert.equal(pos.verdict, 'authentic_hemanth_voice');
  assert.ok(pos.p_hemanth > 0.7, `Expected p_hemanth > 0.7, got ${pos.p_hemanth}`);
  assert.equal(neg.verdict, 'needs_hemanth_rewrite');
  assert.ok(neg.p_hemanth < 0.35, `Expected p_hemanth < 0.35, got ${neg.p_hemanth}`);
  assert.ok(neg.fluff_words_found.includes('multifaceted'));
});

test('rewriteText transforms dense nullspan launch blurb into Hemanth voice', () => {
  const draft = `Baked nullspan a single-pass structured extraction with exact [start, end] character spans, calibrated p_null abstention, algebraic invariant checks (subtotal == nodes * hourly_rate), and p10/p50/p90 quantiles.`;
  const w = scribe(draft);
  const res = w.rewrite();

  assert.ok(res.text.includes('Meet nullspan'));
  assert.ok(res.after_p_hemanth > res.before_p_hemanth);
});

test('writePost and generateSocial produce complete markdown and platform posts', () => {
  const w = scribe('Zero-Overhead Token Budgets', { category: 'ai', tags: ['ai', 'agents', 'budgets'] });
  const blog = w.write();
  assert.ok(blog.text.includes('# Zero-Overhead Token Budgets'));
  assert.ok(blog.text.includes('```python'));
  assert.ok(blog.p_hemanth > 0.65);

  const soc = w.social(blog.text);
  assert.ok(soc.linkedin.includes('#AI'));
  assert.ok(soc.reddit.includes('TL;DR:'));
  assert.ok(soc.hn.includes('Title:'));
});
