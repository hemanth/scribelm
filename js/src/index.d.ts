export interface StyleScores {
  directness: number;
  hook_strength: number;
  zero_fluff: number;
  code_first: number;
  cadence: number;
}

export interface CritiqueResult {
  verdict: 'authentic_hemanth_voice' | 'needs_hemanth_rewrite';
  p_hemanth: number;
  style_scores: StyleScores;
  fluff_words_found: string[];
  notes: string;
  latency_ms: number;
}

export interface RewriteResult {
  text: string;
  before_p_hemanth: number;
  after_p_hemanth: number;
  style_scores: StyleScores;
  latency_ms: number;
}

export interface BlogResult {
  title: string;
  category: string;
  tags: string[];
  text: string;
  p_hemanth: number;
  style_scores: StyleScores;
  latency_ms: number;
}

export interface SocialResult {
  linkedin: string;
  hn: string;
  reddit: string;
  x: string;
  text: string;
  p_hemanth: number;
  style_scores: StyleScores;
  latency_ms: number;
}

export interface ScribeInstance {
  write(topic?: string, opts?: Record<string, unknown>): BlogResult;
  rewrite(draft?: string, opts?: Record<string, unknown>): RewriteResult;
  social(content?: string, opts?: Record<string, unknown>): SocialResult;
  critique(text?: string): CritiqueResult;
  on(event: string, fn: (payload: unknown) => void): ScribeInstance;
}

export declare function scribe(input?: string, options?: Record<string, unknown>): ScribeInstance;
export declare function critiqueText(text: string): CritiqueResult;
export declare function rewriteText(draft: string, options?: Record<string, unknown>): RewriteResult;
export declare function writePost(input: string, options?: Record<string, unknown>): BlogResult;
export declare function generateSocial(input: string, options?: Record<string, unknown>): SocialResult;
export default scribe;
