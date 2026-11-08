// d1's prompt and read-out in the page: a line-by-line port of upstream's examples/python/d1-decisions.py
// (microsoft/onnxruntime-genai v0.19.0, MIT License, Copyright (c) Microsoft Corporation): the request checks, the
// option codes, the question block, the prompt prefix and suffix, the read-out groups, the probabilities and the answer.
// A state that is not a string is written as Python's json.dumps(state, ensure_ascii=False, indent=2) writes it, so
// JSON objects are Maps (key order kept) and numbers keep their JSON text (parseJson below).

export const BOS = '<|startoftext|>', IM_START = '<|im_start|>', IM_END = '<|im_end|>', IMAGE = '<image>';
const QUESTION_KEYS = ['type', 'instructions', 'criteria'];
const YES_FORMS = ['yes', 'Yes', 'YES'];
const NO_FORMS = ['no', 'No', 'NO'];
const LETTERS = (from, to) => Array.from({ length: to.charCodeAt(0) - from.charCodeAt(0) + 1 }, (_, i) => String.fromCharCode(from.charCodeAt(0) + i));
const FALLBACK_CODES = [
  ...LETTERS('A', 'Z'),
  ...Array.from({ length: 100 }, (_, i) => String(i).padStart(2, '0')),
  ...LETTERS('a', 'z'),
  ...Array.from({ length: 200 }, (_, i) => `#${i}`),
  ...LETTERS('A', 'Z').flatMap((a) => LETTERS('A', 'Z').map((b) => a + b)),
];

export class RequestError extends Error {}

// ---- JSON with Python's semantics ----------------------------------------------------------------------------------
// A number from the JSON text: an int (no '.', 'e' or 'E') or a float, as Python's json.loads reads it.
export class PyNum {
  constructor(text) { this.text = text; this.isInt = !/[.eE]/.test(text); }
  get value() { return Number(this.text); }
}

// JSON.parse, except that objects become Maps in key order (a repeated key keeps its first place and its last value,
// as a Python dict does) and numbers become PyNum.
export function parseJson(text) {
  let i = 0;
  const ws = () => { while (i < text.length && ' \t\n\r'.includes(text[i])) i += 1; };
  const fail = (what) => { throw new SyntaxError(`JSON: ${what} at character ${i}`); };
  function value() {
    ws();
    const c = text[i];
    if (c === '{') {
      i += 1; const m = new Map(); ws();
      if (text[i] === '}') { i += 1; return m; }
      for (;;) {
        ws(); if (text[i] !== '"') fail('a key');
        const k = str(); ws(); if (text[i] !== ':') fail("':'"); i += 1;
        m.set(k, value()); ws();
        if (text[i] === ',') { i += 1; continue; }
        if (text[i] === '}') { i += 1; return m; }
        fail("',' or '}'");
      }
    }
    if (c === '[') {
      i += 1; const a = []; ws();
      if (text[i] === ']') { i += 1; return a; }
      for (;;) {
        a.push(value()); ws();
        if (text[i] === ',') { i += 1; continue; }
        if (text[i] === ']') { i += 1; return a; }
        fail("',' or ']'");
      }
    }
    if (c === '"') return str();
    if (text.startsWith('true', i)) { i += 4; return true; }
    if (text.startsWith('false', i)) { i += 5; return false; }
    if (text.startsWith('null', i)) { i += 4; return null; }
    const m = /^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/.exec(text.slice(i));
    if (!m) fail('a value');
    i += m[0].length;
    return new PyNum(m[0]);
  }
  function str() {
    let j = i + 1;
    for (;;) {
      if (j >= text.length) fail('the end of a string');
      if (text[j] === '\\') { j += 2; continue; }
      if (text[j] === '"') break;
      j += 1;
    }
    const s = JSON.parse(text.slice(i, j + 1));
    i = j + 1;
    return s;
  }
  const v = value(); ws();
  if (i !== text.length) fail('extra text');
  return v;
}

// repr(float) of Python: the shortest digits that read back as the same double, fixed notation for 1e-4 <= |x| < 1e16
// (with ".0" for a whole number), else d.ddde+XX.
function pyFloat(x) {
  if (Number.isNaN(x)) return 'NaN';
  if (!Number.isFinite(x)) return x > 0 ? 'Infinity' : '-Infinity';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const sign = x < 0 ? '-' : '';
  const [mant, exp] = Math.abs(x).toExponential().split('e');
  const digits = mant.replace('.', '');
  const decpt = parseInt(exp, 10) + 1;
  if (decpt <= -4 || decpt > 16) {
    const e = decpt - 1;
    return `${sign}${digits[0]}${digits.length > 1 ? '.' + digits.slice(1) : ''}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`;
  }
  if (decpt <= 0) return `${sign}0.${'0'.repeat(-decpt)}${digits}`;
  if (decpt >= digits.length) return `${sign}${digits}${'0'.repeat(decpt - digits.length)}.0`;
  return `${sign}${digits.slice(0, decpt)}.${digits.slice(decpt)}`;
}

function pyNumText(n) {
  if (n instanceof PyNum) {
    if (n.isInt) { const t = BigInt(n.text).toString(); return t; }
    return pyFloat(n.value);
  }
  return Number.isInteger(n) ? String(n) : pyFloat(n);
}

// json.dumps(s, ensure_ascii=False) for a string.
function pyJsonString(s) {
  return '"' + s.replace(/[\x00-\x1f\\"]/g, (c) => ({ '\\': '\\\\', '"': '\\"', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f' })[c]
    || '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0')) + '"';
}

// json.dumps(value, ensure_ascii=False, indent=2)
export function pyDumps(v, indent = 2, level = 0) {
  if (v === null || v === undefined) return 'null';
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (typeof v === 'string') return pyJsonString(v);
  if (typeof v === 'number' || v instanceof PyNum) return pyNumText(v);
  const pad = '\n' + ' '.repeat(indent * (level + 1)), end = '\n' + ' '.repeat(indent * level);
  if (Array.isArray(v)) {
    if (!v.length) return '[]';
    return '[' + v.map((x) => pad + pyDumps(x, indent, level + 1)).join(',') + end + ']';
  }
  const entries = v instanceof Map ? [...v.entries()] : Object.entries(v);
  if (!entries.length) return '{}';
  return '{' + entries.map(([k, x]) => pad + pyJsonString(String(k)) + ': ' + pyDumps(x, indent, level + 1)).join(',') + end + '}';
}

// str() and repr() of the values a request can hold, for the f-strings of question_block and the error messages.
function pyRepr(v) {
  if (typeof v === 'string') {
    const q = v.includes("'") && !v.includes('"') ? '"' : "'";
    return q + v.replace(/\\/g, '\\\\').replace(new RegExp(q, 'g'), '\\' + q).replace(/\n/g, '\\n').replace(/\r/g, '\\r').replace(/\t/g, '\\t') + q;
  }
  return pyStr(v);
}
function pyStr(v) {
  if (v === null || v === undefined) return 'None';
  if (v === true) return 'True';
  if (v === false) return 'False';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || v instanceof PyNum) return pyNumText(v);
  if (Array.isArray(v)) return '[' + v.map(pyRepr).join(', ') + ']';
  if (v instanceof Map) return '{' + [...v.entries()].map(([k, x]) => pyRepr(k) + ': ' + pyRepr(x)).join(', ') + '}';
  return String(v);
}
function pyTruthy(v) {
  if (v === null || v === undefined || v === false || v === '') return false;
  if (v instanceof PyNum) return v.value !== 0;
  if (typeof v === 'number') return v !== 0;
  if (Array.isArray(v)) return v.length > 0;
  if (v instanceof Map) return v.size > 0;
  return true;
}
const isMap = (v) => v instanceof Map;
const typeName = (v) => (v === null ? 'NoneType' : Array.isArray(v) ? 'list' : isMap(v) ? 'dict' : typeof v === 'string' ? 'str'
  : typeof v === 'boolean' ? 'bool' : v instanceof PyNum ? (v.isInt ? 'int' : 'float') : typeof v);

// ---- upstream's functions ------------------------------------------------------------------------------------------
export function checkRequest(request) {
  if (!isMap(request) || request.size !== 2 || !request.has('state') || !request.has('questions')) {
    const got = isMap(request) ? '[' + [...request.keys()].sort().map(pyRepr).join(', ') + ']' : typeName(request);
    throw new RequestError(`a request has exactly "state" and "questions"; got ${got}`);
  }
  const questions = request.get('questions');
  if (!isMap(questions) || !questions.size) throw new RequestError('questions must be {name: question} with at least one question');
  const checked = new Map();
  for (const [name, q] of questions) checked.set(name, checkQuestion(name, q));
  return [request.get('state'), checked];
}

export function checkQuestion(name, question) {
  if (!isMap(question) || [...question.keys()].some((k) => !QUESTION_KEYS.includes(k))) {
    throw new RequestError(`question ${pyRepr(name)} must be an object with only type, instructions and criteria`);
  }
  const kind = question.has('type') ? question.get('type') : 'choice';
  const criteria = question.has('criteria') ? question.get('criteria') : null;
  if (!['noul', 'choice', 'score'].includes(kind)) throw new RequestError(`question ${pyRepr(name)} has unknown type ${pyRepr(kind)}; use noul, choice or score`);
  if (typeof question.get('instructions') !== 'string') throw new RequestError(`question ${pyRepr(name)} needs instructions: a string`);
  if (kind === 'choice' && (!isMap(criteria) || criteria.size < 2)) throw new RequestError(`choice ${pyRepr(name)} needs criteria {name: description} with at least 2 options`);
  if (kind === 'score' && (!Array.isArray(criteria) || criteria.length < 2 || criteria.length > 10)) throw new RequestError(`score ${pyRepr(name)} needs criteria: a list of 2 to 10 levels, lowest first`);
  if (kind === 'noul' && !(criteria === null || (isMap(criteria) && [...criteria.keys()].every((k) => k === 'true' || k === 'false')))) {
    throw new RequestError(`noul ${pyRepr(name)} criteria must be {"true": "...", "false": "..."} if given`);
  }
  return { type: kind, instructions: question.get('instructions'), criteria };
}

export function singleTokenIds(tok, texts) {
  const ids = [];
  for (const text of texts) {
    const e = tok.encode(text);
    if (e.length === 1 && !ids.includes(e[0])) ids.push(e[0]);
  }
  return ids;
}

export function optionCodes(labels) {
  labels = labels.map((l) => pyStr(l).trim());
  if (labels.length && labels.every((l) => [...l].length === 1 && /^\p{L}$/u.test(l))) return labels;
  if (labels.length <= 26) return labels.map((_, i) => String.fromCharCode(65 + i));
  return labels.map((_, i) => String(i).padStart(2, '0'));
}

export function optionAliases(tok, labels) {
  const used = [], aliases = [];
  for (const code of optionCodes(labels)) {
    let found = false;
    for (const candidate of [code, ...FALLBACK_CODES]) {
      const ids = tok.encode(candidate);
      if (ids.length === 1 && !used.includes(ids[0])) { used.push(ids[0]); aliases.push([candidate, ids[0]]); found = true; break; }
    }
    if (!found) throw new RequestError(`no single-token alias left for ${labels.length} options`);
  }
  return aliases;
}

export function choiceCodes(tok, q) { return q.type === 'choice' ? optionAliases(tok, [...q.criteria.keys()]) : null; }

export function readoutIds(tok, q, codes) {
  if (q.type === 'noul') {
    const yes = singleTokenIds(tok, YES_FORMS), no = singleTokenIds(tok, NO_FORMS);
    if (!yes.length || !no.length) throw new RequestError('the tokenizer has no single-token yes or no');
    return [yes, no];
  }
  if (q.type === 'score') {
    const groups = q.criteria.map((_, i) => singleTokenIds(tok, [String(i)]));
    if (!groups.every((g) => g.length)) throw new RequestError('the tokenizer has no single-token digit for every score level');
    return groups;
  }
  return codes.map(([code, id]) => [id, ...singleTokenIds(tok, [` ${code}`]).filter((i) => i !== id)]);
}

export function questionBlock(q, codes) {
  const { instructions, criteria } = q;
  if (q.type === 'choice') {
    const labels = [...criteria.keys()];
    const lines = codes.map(([code], n) => `${code} ${pyTruthy(criteria.get(labels[n])) ? pyStr(criteria.get(labels[n])) : labels[n].replaceAll('_', ' ')}`).join('\n');
    return `${instructions}\n\nOptions:\n${lines}\n\nReply with the option code only.`;
  }
  if (q.type === 'noul') {
    const extra = pyTruthy(criteria) ? `\nYes: ${pyStr(criteria.has('true') ? criteria.get('true') : null)}\nNo: ${pyStr(criteria.has('false') ? criteria.get('false') : null)}` : '';
    return `${instructions}${extra}\n\nReply with yes or no only.`;
  }
  const legend = criteria.map((text, i) => `${i} ${pyStr(text)}`).join('\n');
  return `${instructions}\n\n${legend}\n\nReply with a single digit 0-${criteria.length - 1} only.`;
}

export function promptPrefix(state, numImages = 0) {
  const body = state === null ? '' : `${typeof state === 'string' ? state : pyDumps(state)}\n\n\nQUESTION:\n`;
  return `${BOS}${IM_START}user\n${IMAGE.repeat(numImages)}${body}`;
}

export function promptSuffix(q, codes) { return `${questionBlock(q, codes)}${IM_END}\n${IM_START}assistant\n`; }

// Each option scores the best log-probability among its ids; a softmax over the scores. logits[id] for every id of the
// groups (a Map or an object); the log-softmax over the whole vocabulary shifts every option by the same amount, so the
// options' rows of the head are enough (d1-on-windows' npu/head.py).
export function probabilities(logitOf, groups) {
  const scores = groups.map((g) => Math.max(...g.map((id) => logitOf(id))));
  const top = Math.max(...scores);
  const w = scores.map((s) => Math.exp(s - top));
  const sum = w.reduce((a, b) => a + b, 0);
  return w.map((x) => x / sum);
}

export function answer(q, probs) {
  if (q.type === 'noul') return { type: 'noul', noul: probs[0] };
  const names = q.type === 'choice' ? [...q.criteria.keys()] : probs.map((_, i) => String(i));
  const table = {};
  names.forEach((n, i) => { table[n] = probs[i]; });
  let top = names[0];
  for (const n of names) if (table[n] > table[top]) top = n;  // a tie goes to the first option, as in d1
  if (q.type === 'choice') return { type: 'choice', choice: top, confidence: table[top], probabilities: table };
  const expected = probs.reduce((a, p, i) => a + i * p, 0);
  const legend = {};
  names.forEach((n, i) => { legend[n] = pyStr(q.criteria[i]); });
  return { type: 'score', score: expected, confidence: table[top], probabilities: table, legend };
}

// The option -> probability table under the reference's keys (d1-on-windows' run_genai.py probs_of).
export function probsOf(a) { return a.type === 'noul' ? { true: a.noul, false: 1 - a.noul } : a.probabilities; }

// Upstream's decide() up to the forward passes: per question its input ids (one string for a one-question request;
// otherwise the shared prefix once plus each question's own suffix) and its read-out groups.
export function plan(tok, request, contextLength = Infinity) {
  const [state, questions] = checkRequest(request);
  const qs = [...questions.entries()];
  const codes = qs.map(([, q]) => choiceCodes(tok, q));
  const groups = qs.map(([, q], n) => readoutIds(tok, q, codes[n]));
  const prefix = promptPrefix(state, 0);
  const suffixes = qs.map(([, q], n) => promptSuffix(q, codes[n]));
  if ((prefix + suffixes.join('')).includes('\x00')) throw new RequestError('the request holds a NUL character (\\x00)');
  let head, tails;
  if (suffixes.length === 1) { head = prefix + suffixes[0]; tails = [[]]; } else { head = prefix; tails = suffixes.map((s) => tok.encode(s)); }
  const headIds = tok.encode(head);
  const longest = headIds.length + Math.max(...tails.map((t) => t.length));
  if (longest > contextLength) throw new RequestError(`the prompt has ${longest} tokens; the model allows ${contextLength}`);
  return qs.map(([name, q], n) => ({ name, question: q, codes: codes[n], groups: groups[n], ids: headIds.concat(tails[n]) }));
}
