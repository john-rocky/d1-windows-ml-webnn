// d1-3B's tokenizer in the page, read from the model's own tokenizer.json (LiquidAI/d1-3B): the added tokens are split
// out first, then the Split pre-tokenizer's regex (behavior Isolated), the byte-level alphabet, and BPE with
// ignore_merges, as Hugging Face tokenizers reads that file. No dependencies; runs in the browser and in Node.
//
//   const tok = await loadTokenizer('models/tokenizer.json');   // or fromJson(object)
//   tok.encode('<|startoftext|><|im_start|>user\nHello');         // -> [124894, 124899, ...]

// GPT-2's bytes_to_unicode: every byte maps to one printable character.
const BYTE_CHAR = (() => {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n += 1; }
  const table = new Array(256);
  bs.forEach((b, i) => { table[b] = String.fromCharCode(cs[i]); });
  return table;
})();

// The Split pattern of d1-3B's tokenizer.json, and the variant d1-on-windows' fix_tokenizer.py writes for the GenAI
// export. JavaScript has no `(?i:` group in older engines, so the known patterns are spelled with explicit cases.
const KNOWN_PATTERNS = new Map([
  ["'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s",
   "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s"],
  ["(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
   "'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],
]);

function escapeRegExp(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

export function fromJson(json) {
  const model = json.model;
  if (!model || model.type !== 'BPE') throw new Error(`tokenizer.json: model type ${model && model.type}, this page reads BPE only`);
  if (model.byte_fallback || model.continuing_subword_prefix || model.end_of_word_suffix) throw new Error('tokenizer.json: BPE options this page does not implement');
  if (json.normalizer) throw new Error('tokenizer.json: a normalizer, which this page does not implement');
  const pre = json.pre_tokenizer;
  const steps = pre && pre.type === 'Sequence' ? pre.pretokenizers : [pre];
  if (steps.length !== 2 || steps[0].type !== 'Split' || steps[1].type !== 'ByteLevel' || steps[0].behavior !== 'Isolated' || steps[0].invert || steps[1].use_regex || steps[1].add_prefix_space) {
    throw new Error('tokenizer.json: a pre-tokenizer other than Split(Isolated) + ByteLevel(no regex, no prefix space)');
  }
  const source = steps[0].pattern.Regex;
  const split = new RegExp(KNOWN_PATTERNS.get(source) || source, 'gu');
  const vocab = new Map(Object.entries(model.vocab));
  const ranks = new Map();
  model.merges.forEach((m, i) => { ranks.set(Array.isArray(m) ? m[0] + ' ' + m[1] : m, i); });
  const added = new Map();
  for (const t of json.added_tokens || []) {
    if (t.lstrip || t.rstrip || t.single_word || t.normalized) throw new Error(`tokenizer.json: added token ${t.content} with options this page does not implement`);
    added.set(t.content, t.id);
  }
  // Leftmost match, longest first (Hugging Face's AddedVocabulary uses leftmost-longest matching).
  const addedRe = added.size ? new RegExp([...added.keys()].sort((a, b) => b.length - a.length).map(escapeRegExp).join('|'), 'g') : null;
  return new BpeTokenizer(vocab, ranks, added, addedRe, split, !!model.ignore_merges);
}

export async function loadTokenizer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return fromJson(await res.json());
}

class BpeTokenizer {
  constructor(vocab, ranks, added, addedRe, split, ignoreMerges) {
    this.vocab = vocab; this.ranks = ranks; this.added = added; this.addedRe = addedRe; this.split = split;
    this.ignoreMerges = ignoreMerges; this.cache = new Map(); this.utf8 = new TextEncoder();
  }

  // The ids of a text, with no special tokens added (d1's prompt writes its own).
  encode(text) {
    const ids = [];
    let at = 0;
    if (this.addedRe) {
      this.addedRe.lastIndex = 0;
      for (const m of text.matchAll(this.addedRe)) {
        if (m.index > at) this.encodePlain(text.slice(at, m.index), ids);
        ids.push(this.added.get(m[0]));
        at = m.index + m[0].length;
      }
    }
    if (at < text.length) this.encodePlain(text.slice(at), ids);
    return ids;
  }

  encodePlain(text, ids) {
    // Split(Isolated): every match is a piece, and so is the text between two matches.
    let at = 0;
    for (const m of text.matchAll(this.split)) {
      if (m.index > at) this.encodeWord(text.slice(at, m.index), ids);
      if (m[0].length) this.encodeWord(m[0], ids);
      at = m.index + m[0].length;
    }
    if (at < text.length) this.encodeWord(text.slice(at), ids);
  }

  encodeWord(piece, ids) {
    let word = '';
    for (const b of this.utf8.encode(piece)) word += BYTE_CHAR[b];
    const hit = this.cache.get(word);
    if (hit) { for (const id of hit) ids.push(id); return; }
    const out = this.bpe(word);
    if (this.cache.size < 50000) this.cache.set(word, out);
    for (const id of out) ids.push(id);
  }

  bpe(word) {
    if (this.ignoreMerges && this.vocab.has(word)) return [this.vocab.get(word)];
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + ' ' + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = parts[best], b = parts[best + 1], merged = [];
      // Merge every occurrence of the best pair, left to right (GPT-2's loop; Hugging Face's heap gives the same result).
      for (let i = 0; i < parts.length;) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i += 2; } else { merged.push(parts[i]); i += 1; }
      }
      parts = merged;
    }
    return parts.map((p) => {
      const id = this.vocab.get(p);
      if (id === undefined) throw new Error(`tokenizer: no id for the piece ${JSON.stringify(p)}`);
      return id;
    });
  }
}
