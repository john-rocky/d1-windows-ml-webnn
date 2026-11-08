// d1-3B's typed decisions in the page: the page builds d1's prompt (d1prompt.mjs, upstream's d1-decisions.py), turns it
// into token ids with the model's own tokenizer.json (d1tok.mjs), runs the static 128-token graph (export_onnx.py /
// make_forms.py: input ids and the index of the last token in, the hidden state at that token out) through ONNX
// Runtime Web's WebNN EP on the GPU, the NPU or the CPU, and reads the options' rows of the head (lm_head.npy, fetched
// row by row with HTTP range requests) to get each option's probability. No server code runs: serve.py only serves files.
//
// URL parameters: devices (default gpu,npu,cpu: one session each, created in that order), form_gpu / form_npu /
// form_cpu (folders under models/, default h16 / w8 / w8), label_gpu / label_npu / label_cpu, machine (a caption),
// head (default models/lm_head.npy), tokenizer (default models/tokenizer.json), example (0-2, loaded at start), power
// (WebNN's powerPreference for every context the page creates: default, high-performance or low-power; not set = the
// browser's default).
// Checks, which post their result to serve.py's /result/<tag>: check=ids (the page's ids against fixtures/npu_ids_128.json,
// no model), check=parity&device=<d> (the 311 questions that fit 128 tokens: answers from the page's head, the hidden
// states for the Python head, the 39- and 101-token times and the examples' times), tag=<result name>.
import * as ort from './ort/ort.all.min.mjs';
import { loadTokenizer } from './d1tok.mjs';
import { parseJson, plan, answer, probabilities, probsOf, RequestError } from './d1prompt.mjs';

const P = new URLSearchParams(location.search);
const LENGTH = 128, PAD = 124893;
const CHECK = P.get('check') || '';
const POWER = P.get('power') || '';
const TAG = P.get('tag') || '';
const DEVICES = (CHECK === 'parity' ? [P.get('device') || 'gpu'] : (P.get('devices') || 'gpu,npu,cpu').split(',')).filter((d) => ['gpu', 'npu', 'cpu'].includes(d));
const FORMS = { gpu: P.get('form_gpu') || 'h16', npu: P.get('form_npu') || 'w8', cpu: P.get('form_cpu') || 'w8' };
const LABELS = { gpu: P.get('label_gpu') || 'GPU', npu: P.get('label_npu') || 'NPU', cpu: P.get('label_cpu') || 'CPU' };
const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const errorText = (e) => String((e && (e.stack || e.message)) || e);

async function log(line) {
  const box = $('#log');
  box.textContent += (box.textContent ? '\n' : '') + line;
  if (TAG) { try { await fetch('/log/' + TAG, { method: 'POST', body: line }); } catch (e) { /* no server to tell */ } }
}

// ---- the head: the options' rows of lm_head.npy ----------------------------------------------------------------------
class Head {
  constructor(url) { this.url = url; this.rows = new Map(); }
  async range(start, end) {
    const ctl = new AbortController();
    const res = await fetch(this.url, { headers: { Range: `bytes=${start}-${end}` }, signal: ctl.signal });
    if (res.status !== 206) { ctl.abort(); throw new Error(`${this.url}: the server answered ${res.status} to a range request (serve.py from this folder serves ranges)`); }
    return new Uint8Array(await res.arrayBuffer());
  }
  async init() {
    const b = await this.range(0, 1023);
    if (b[0] !== 0x93 || String.fromCharCode(...b.slice(1, 6)) !== 'NUMPY') throw new Error(`${this.url} is not a .npy file`);
    const v1 = b[6] === 1, len = v1 ? b[8] | (b[9] << 8) : b[8] | (b[9] << 8) | (b[10] << 16) | (b[11] << 24);
    const at = v1 ? 10 : 12, text = new TextDecoder().decode(b.slice(at, at + len));
    const descr = /'descr':\s*'([^']+)'/.exec(text), shape = /'shape':\s*\((\d+),\s*(\d+)\)/.exec(text);
    if (!descr || descr[1] !== '<f4' || /'fortran_order':\s*True/.test(text) || !shape) throw new Error(`${this.url}: expected a C-order float32 matrix, header ${text}`);
    this.offset = at + len; this.vocab = +shape[1]; this.dim = +shape[2];
  }
  async ensure(ids) {
    const want = [...new Set(ids)].filter((id) => !this.rows.has(id));
    for (let i = 0; i < want.length; i += 24) {
      await Promise.all(want.slice(i, i + 24).map(async (id) => {
        if (id < 0 || id >= this.vocab) throw new Error(`token ${id} is outside the head (${this.vocab} rows)`);
        const start = this.offset + id * this.dim * 4;
        const bytes = await this.range(start, start + this.dim * 4 - 1);
        this.rows.set(id, new Float32Array(bytes.buffer, bytes.byteOffset, this.dim));
      }));
    }
  }
  logit(id, hidden) {
    const r = this.rows.get(id);
    let s = 0;
    for (let i = 0; i < r.length; i++) s += r[i] * hidden[i];
    return s;
  }
}

// ---- WebNN sessions ----------------------------------------------------------------------------------------------------
const sessions = {};
async function createSession(device) {
  const s = sessions[device] = { device, form: FORMS[device], state: 'loading', t0: performance.now() };
  renderDevices();
  try {
    if (!('ml' in navigator)) throw new Error('navigator.ml is undefined: WebNN is not enabled in this browser');
    const dir = `models/${s.form}`;
    const ctx = await navigator.ml.createContext(POWER ? { deviceType: device, powerPreference: POWER } : { deviceType: device });
    const manifest = await (await fetch(`${dir}/manifest.json`)).json();
    const externalData = Object.keys(manifest.files).map((name) => ({ path: name, data: `${dir}/${name}` }));
    s.session = await ort.InferenceSession.create(`${dir}/model.onnx`, {
      executionProviders: [{ name: 'webnn', context: ctx, deviceType: device }],
      externalData, graphOptimizationLevel: 'basic', logSeverityLevel: 2,
    });
    s.loadS = (performance.now() - s.t0) / 1000;
    await forward(s, [124894, 124899, 5922, 207]);  // one warm run ("<|startoftext|><|im_start|>user\n"), not shown
    s.state = 'ready';
    await log(`${device} (${s.form}${POWER ? `, ${POWER}` : ''}) ready: session ${s.loadS.toFixed(1)} s`);
  } catch (e) {
    s.state = 'failed'; s.error = errorText(e);
    await log(`${device} (${s.form}) failed after ${((performance.now() - s.t0) / 1000).toFixed(1)} s: ${s.error}`);
  }
  renderDevices();
  return s;
}
async function forward(s, ids) {
  if (ids.length > LENGTH) throw new RequestError(`the prompt has ${ids.length} tokens; this graph takes ${LENGTH}`);
  const input = new Int32Array(LENGTH).fill(PAD);
  input.set(ids);
  const res = await s.session.run({ input_ids: new ort.Tensor('int32', input, [1, LENGTH]), last: new ort.Tensor('int32', new Int32Array([ids.length - 1]), []) });
  return await res.hidden.getData();
}
// One decision: the forward pass and the read-out (the span of the static rows' times in the README).
async function decideOne(s, step, withAnswer = true) {
  const t = performance.now();
  const hidden = await forward(s, step.ids);
  const probs = probabilities((id) => head.logit(id, hidden), step.groups);
  const a = withAnswer ? answer(step.question, probs) : null;
  return { a, probs, ms: performance.now() - t, hidden };
}

// ---- the examples ----------------------------------------------------------------------------------------------------
const EXAMPLES = [
  { title: 'Support ticket', state: 'I was charged twice this month, please refund one of them.',
    questions: [
      { name: 'refund', type: 'noul', instructions: 'Is the customer asking for a refund?', options: '' },
      { name: 'team', type: 'choice', instructions: 'Which team should handle this?', options: 'billing: Charges, refunds, invoices\ntechnical: App or site faults\nfraud: Suspected unauthorised use' },
      { name: 'urgency', type: 'score', instructions: 'How urgent is this?', options: 'Can wait\nToday\nBlocking the customer now' },
    ] },
  { title: 'Expense line', state: '{\n  "merchant": "Hotel Granvia Kyoto",\n  "amount": 412.50,\n  "currency": "USD",\n  "memo": "2 nights, client workshop",\n  "receipt_attached": false\n}',
    questions: [
      { name: 'category', type: 'choice', instructions: 'Which expense category is this line?', options: 'travel: Flights, hotels, trains, taxis\nmeals: Restaurants and catering\nsoftware: Subscriptions and licences\noffice: Supplies and equipment' },
      { name: 'needs_receipt', type: 'noul', instructions: 'Lines over 75 USD need a receipt. Is this line missing one it needs?', options: '' },
    ] },
  { title: 'Product review', state: "The headphones sound great, but the left earcup cracked after two weeks and support hasn't replied.",
    questions: [
      { name: 'sentiment', type: 'choice', instructions: 'What is the overall sentiment of this review?', options: 'positive\nmixed\nnegative' },
      { name: 'defect', type: 'noul', instructions: 'Does the review report a product defect?', options: '' },
      { name: 'stars', type: 'score', instructions: 'How many stars would this reviewer give?', options: '1 star\n2 stars\n3 stars\n4 stars\n5 stars' },
    ] },
];

// ---- the editor ------------------------------------------------------------------------------------------------------
const qList = [];
function autosize(t) { t.style.height = 'auto'; t.style.height = `${t.scrollHeight + 2}px`; }
function addQuestion(q) {
  const box = el('div', 'qed');
  const name = el('input'); name.value = q.name; name.placeholder = 'name'; name.spellcheck = false;
  const type = el('select');
  for (const [v, t] of [['choice', 'choice'], ['noul', 'yes / no'], ['score', 'score']]) { const o = el('option', '', t); o.value = v; type.append(o); }
  type.value = q.type;
  const ins = el('input'); ins.value = q.instructions; ins.placeholder = 'the question'; ins.spellcheck = false;
  const x = el('button', 'x', '×'); x.title = 'remove';
  const opts = el('textarea'); opts.value = q.options; opts.spellcheck = false;
  const hint = () => { opts.placeholder = type.value === 'choice' ? 'label: description (one option per line)' : type.value === 'noul' ? 'optional: yes: …  /  no: …' : 'one level per line, lowest first'; };
  type.addEventListener('change', hint); hint();
  box.append(name, type, ins, x, opts);
  opts.addEventListener('input', () => autosize(opts));
  const entry = { box, name, type, ins, opts };
  x.addEventListener('click', () => { box.remove(); qList.splice(qList.indexOf(entry), 1); });
  qList.push(entry);
  $('#questions').append(box);
  autosize(opts);
}
function loadExample(i) {
  const ex = EXAMPLES[i];
  $('#state').value = ex.state;
  qList.splice(0).forEach((q) => q.box.remove());
  ex.questions.forEach(addQuestion);
  autosize($('#state'));
  document.querySelectorAll('#examples button').forEach((b, n) => b.classList.toggle('on', n === i));
  $('#answers').textContent = ''; $('#summary').textContent = ''; history.clear(); lastTop.clear();
}
// The request of the editor, as upstream reads a request file: {state, questions: {name: {type, instructions, criteria}}}.
function readRequest() {
  const text = $('#state').value;
  let state = text;
  const t = text.trim();
  if (!t) state = null;
  else if (t[0] === '{' || t[0] === '[') { try { state = parseJson(t); } catch (e) { state = text; } }
  const questions = new Map();
  for (const q of qList) {
    const lines = q.opts.value.split('\n').map((l) => l.trim()).filter(Boolean);
    const m = new Map([['type', q.type.value], ['instructions', q.ins.value]]);
    if (q.type.value === 'choice') {
      const c = new Map();
      for (const l of lines) { const k = l.indexOf(':'); if (k > 0) c.set(l.slice(0, k).trim(), l.slice(k + 1).trim() || null); else c.set(l, null); }
      m.set('criteria', c);
    } else if (q.type.value === 'score') {
      m.set('criteria', lines);
    } else if (lines.length) {
      const c = new Map();
      for (const l of lines) { const k = l.indexOf(':'); const key = l.slice(0, k).trim().toLowerCase(); if (k > 0 && (key === 'yes' || key === 'no')) c.set(key === 'yes' ? 'true' : 'false', l.slice(k + 1).trim()); }
      if (c.size) m.set('criteria', c);
    }
    questions.set(q.name.value.trim() || `q${questions.size + 1}`, m);
  }
  return new Map([['state', state], ['questions', questions]]);
}

// ---- the answers -----------------------------------------------------------------------------------------------------
const history = new Map();  // question name -> {device: ms}
let device = null;
function renderDevices() {
  const box = $('#devices');
  box.textContent = '';
  for (const d of DEVICES) {
    const s = sessions[d] || { state: 'waiting' };
    const b = el('button', 'dev' + (d === device ? ' on' : '') + (s.state === 'failed' ? ' failed' : ''));
    b.dataset.device = d;
    b.append(el('div', 'name', LABELS[d]));
    const st = s.state === 'ready' ? `WebNN ${d}${POWER ? ` ${POWER}` : ''} · ${s.form} · ready` : s.state === 'loading' ? `WebNN ${d} · loading ${((performance.now() - s.t0) / 1000).toFixed(0)} s`
      : s.state === 'failed' ? `failed: ${s.error.split('\n')[0]}` : `WebNN ${d} · waiting`;
    b.append(el('div', 'st', st));
    b.title = s.error || '';
    b.disabled = s.state !== 'ready';
    b.addEventListener('click', () => { device = d; renderDevices(); });
    box.append(b);
  }
  $('#decide').disabled = !(device && sessions[device] && sessions[device].state === 'ready');
  $('#decide').textContent = device ? `Decide on the ${LABELS[device]}` : 'Decide';
}
function fmtMs(ms) { return ms >= 100 ? ms.toFixed(0) : ms.toFixed(1); }
const lastTop = new Map();  // question name -> the top answer of its last decision (a change flashes the card)
function card(step, r, dev) {
  const c = el('div', 'card');
  const top = el('div', 'top');
  top.append(el('div', 'qname', `${step.name} · ${step.question.instructions}`));
  const ms = el('div', 'ms', fmtMs(r.ms)); ms.append(el('small', '', ` ms · ${LABELS[dev]}`));
  top.append(ms);
  c.append(top);
  const a = r.a, q = step.question;
  let names, win, ans;
  if (a.type === 'noul') { names = ['yes', 'no']; win = a.noul >= 0.5 ? 'yes' : 'no'; ans = el('div', 'ans', win); }
  else if (a.type === 'choice') { names = [...q.criteria.keys()]; win = a.choice; ans = el('div', 'ans', win); }
  else {
    names = q.criteria.map((_, i) => String(i));
    win = names.reduce((b, n) => (a.probabilities[n] > a.probabilities[b] ? n : b), names[0]);
    ans = el('div', 'ans', String(q.criteria[+win])); ans.append(el('small', '', `score ${a.score.toFixed(2)} of 0–${names.length - 1}`));
  }
  const body = el('div', 'body');
  const bars = el('div', 'bars');
  names.forEach((n, i) => {
    const p = r.probs[i];
    bars.append(el('div', 'lab' + (n === win ? ' win' : ''), a.type === 'score' ? String(q.criteria[i]) : n));
    const bar = el('div', 'bar'), fill = el('i'); fill.style.width = `${(p * 100).toFixed(1)}%`; bar.append(fill); bars.append(bar);
    bars.append(el('div', 'pct', `${(p * 100).toFixed(1)}%`));
  });
  body.append(ans, bars);
  c.append(body);
  const h = history.get(step.name) || {};
  h[dev] = r.ms; history.set(step.name, h);
  const line = el('div', 'hist');
  DEVICES.filter((d) => h[d] !== undefined).forEach((d, i) => {
    if (i) line.append(' · ');
    line.append(el('b', '', LABELS[d]), ` ${fmtMs(h[d])} ms`);
  });
  c.append(line);
  if (lastTop.has(step.name) && lastTop.get(step.name) !== win) c.classList.add('changed');
  lastTop.set(step.name, win);
  return c;
}
let busy = false;
async function decide() {
  if (busy || !device) return;
  busy = true; $('#decide').disabled = true;
  const s = sessions[device], box = $('#answers'), sum = $('#summary');
  try {
    const t0 = performance.now();
    const steps = plan(tok, readRequest(), LENGTH);
    const tokMs = performance.now() - t0;
    await head.ensure(steps.flatMap((st) => st.groups.flat()));  // rows load once per option set, outside the decisions
    const out = [];
    for (const st of steps) out.push([st, await decideOne(s, st)]);
    box.textContent = '';
    for (const [st, r] of out) box.append(card(st, r, device));
    const total = out.reduce((a, [, r]) => a + r.ms, 0);
    sum.textContent = `${out.length} decision${out.length > 1 ? 's' : ''} on the ${LABELS[device]} in ${fmtMs(total)} ms · ${out.map(([st]) => st.ids.length).join(' / ')} tokens · tokenizer ${tokMs.toFixed(1)} ms`;
  } catch (e) {
    box.textContent = ''; sum.textContent = '';
    box.append(el('div', 'err', e instanceof RequestError ? e.message : errorText(e)));
    if (!(e instanceof RequestError)) await log('decide failed: ' + errorText(e));
  }
  busy = false; renderDevices();
}

// ---- the recording helpers (the page is driven by CDP input events; headless Edge draws no pointer) ---------------------
function rect(sel) { const r = (typeof sel === 'string' ? $(sel) : sel).getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height }; }
function cursor(x, y) { const c = $('#cursor'); c.style.display = 'block'; c.style.left = `${x}px`; c.style.top = `${y}px`; }

// ---- checks ----------------------------------------------------------------------------------------------------------
function b64(f32) {
  const u = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = '';
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
  return btoa(s);
}
async function fixtureSteps() {
  const fixture = parseJson(await (await fetch('fixtures/requests.json')).text());
  const rows = [];
  for (const record of fixture.get('records')) {
    const request = record.get('request');
    if (request.has('images')) continue;
    for (const [qid, question] of request.get('questions')) {
      const q = `${record.get('id')}/${qid}`;
      try { rows.push({ q, ...plan(tok, new Map([['state', request.get('state')], ['questions', new Map([[qid, question]])]]))[0] }); }
      catch (e) { if (e instanceof RequestError) rows.push({ q, refused: e.message }); else throw e; }
    }
  }
  return rows;
}
async function checkIds(out) {
  const t0 = performance.now();
  const rows = await fixtureSteps();
  out.encode_ms = performance.now() - t0;
  const want = new Map((await (await fetch('fixtures/npu_ids_128.json')).json()).map((r) => [r.q, r]));
  const r = { here: 0, there: [...want.values()].filter((w) => w.ids).length, same: 0, refused_same: 0, longer: 0, mismatches: [] };
  for (const row of rows) {
    const w = want.get(row.q);
    if (row.refused) { if (w && w.refused) r.refused_same += 1; else r.mismatches.push({ q: row.q, why: 'refused here' }); continue; }
    if (row.ids.length > LENGTH) { r.longer += 1; if (w) r.mismatches.push({ q: row.q, why: `longer here (${row.ids.length})` }); continue; }
    r.here += 1;
    if (!w || !w.ids) { r.mismatches.push({ q: row.q, why: 'not there' }); continue; }
    const same = w.ids.length === row.ids.length && w.ids.every((x, i) => x === row.ids[i]) && JSON.stringify(w.group) === JSON.stringify(row.groups);
    if (same) r.same += 1; else r.mismatches.push({ q: row.q, why: 'ids or groups differ' });
  }
  out.ids = r;
  await log(`ids: ${r.same} of ${r.there} the same, ${r.here} here, ${r.refused_same} refused on both sides, ${r.longer} longer than ${LENGTH}, ${r.mismatches.length} mismatches`);
  return rows;
}
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
async function checkParity(out, rows) {
  const s = sessions[DEVICES[0]];
  if (s.state !== 'ready') { out.error = s.error; return; }
  out.session_s = s.loadS;
  const fits = rows.filter((r) => r.ids && r.ids.length <= LENGTH);
  await head.ensure(fits.flatMap((r) => r.groups.flat()));
  out.answers = [];
  const t0 = performance.now();
  for (const row of fits) {
    const r = await decideOne(s, row);
    out.answers.push({ q: row.q, tokens: row.ids.length, ms: r.ms, probs: probsOf(r.a), hidden: b64(r.hidden) });
    if (out.answers.length % 50 === 0) await log(`parity ${out.answers.length}`);
  }
  out.parity_s = (performance.now() - t0) / 1000;
  await log(`parity done (${out.answers.length}) in ${out.parity_s.toFixed(1)} s`);
  // The 39- and 101-token prompts of d1-on-windows' speed fixture (the table's times): one warm pass, 9 timed runs.
  const speed = (await (await fetch('fixtures/npu_speed_ids.json')).json()).filter((r) => r.ids && r.ids.length <= LENGTH);
  await head.ensure(speed.flatMap((r) => r.group.flat()));
  out.speed = [];
  for (let pass = 0; pass < 2; pass++) {
    for (const row of speed) {
      const r = await decideOne(s, { ids: row.ids, groups: row.group }, false);
      if (pass === 1) out.speed.push({ q: row.q, tokens: row.ids.length, ms: r.ms });
    }
  }
  const byLen = {};
  for (const x of out.speed) (byLen[x.tokens] = byLen[x.tokens] || []).push(x.ms);
  out.speed_median_ms = Object.fromEntries(Object.entries(byLen).map(([k, v]) => [k, median(v)]));
  // The examples as the demo runs them: one warm decision, then 9 timed, per question.
  out.examples = [];
  for (let i = 0; i < EXAMPLES.length; i++) {
    loadExample(i);
    const steps = plan(tok, readRequest(), LENGTH);
    await head.ensure(steps.flatMap((st) => st.groups.flat()));
    for (const st of steps) {
      const times = []; let last;
      for (let k = 0; k < 10; k++) { last = await decideOne(s, st); if (k) times.push(last.ms); }
      out.examples.push({ example: EXAMPLES[i].title, name: st.name, tokens: st.ids.length, median_ms: median(times), min_ms: Math.min(...times), max_ms: Math.max(...times), answer: last.a });
    }
  }
  await log(`speed medians ${JSON.stringify(out.speed_median_ms)}; examples ${out.examples.map((x) => `${x.name} ${x.median_ms.toFixed(1)}`).join(', ')}`);
}

// ---- start -----------------------------------------------------------------------------------------------------------
let tok, head;
async function main() {
  const out = { tag: TAG, check: CHECK, devices: DEVICES, forms: FORMS, power: POWER || null, started: new Date().toISOString(), userAgent: navigator.userAgent, ort: ort.env.versions };
  window.demo = { ready: false, out, loadExample, decide, rect, cursor, sessions, selectDevice: (d) => { device = d; renderDevices(); },
    state: () => ({ busy, device, summary: $('#summary').textContent, answers: [...document.querySelectorAll('#answers .card')].map((c) => c.innerText) }) };
  EXAMPLES.forEach((ex, i) => { const b = el('button', '', ex.title); b.addEventListener('click', () => loadExample(i)); $('#examples').append(b); });
  $('#addq').addEventListener('click', () => addQuestion({ name: `q${qList.length + 1}`, type: 'choice', instructions: '', options: '' }));
  $('#decide').addEventListener('click', decide);
  $('#state').addEventListener('input', () => autosize($('#state')));
  loadExample(+(P.get('example') || 0));
  let os = 'Windows';
  try {
    const h = await navigator.userAgentData.getHighEntropyValues(['platformVersion']);
    if (navigator.userAgentData.platform === 'Windows') os = parseInt(h.platformVersion, 10) >= 13 ? 'Windows 11' : 'Windows 10';
    else os = navigator.userAgentData.platform;
  } catch (e) { /* no client hints */ }
  const edge = /Edg\/(\d+)/.exec(navigator.userAgent), chrome = /Chrome\/(\d+)/.exec(navigator.userAgent);
  const browser = edge ? `Microsoft Edge ${edge[1]}` : chrome ? `Chrome ${chrome[1]}` : 'this browser';
  const machine = P.get('machine');
  $('#where').textContent = `LiquidAI/d1-3B · ${browser} on ${os}${machine ? ` · ${machine}` : ''} · WebNN through ONNX Runtime Web ${ort.env.versions.web || ort.env.versions.common} · prompt, tokenizer and option head in this page`;
  ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;
  ort.env.wasm.numThreads = 4;
  ort.env.logLevel = 'warning';
  renderDevices();
  const ticker = setInterval(renderDevices, 1000);
  try {
    tok = await loadTokenizer(P.get('tokenizer') || 'models/tokenizer.json');
    if (CHECK !== 'ids') {
      head = new Head(P.get('head') || 'models/lm_head.npy');
      await head.init();
    }
    await log(`tokenizer ready${head ? `, head ${head.vocab} x ${head.dim}` : ''}`);
    let rows = null;
    if (CHECK) rows = await checkIds(out);
    if (CHECK !== 'ids') {
      // A first look at the rows most option sets use, so that the first decision does not wait for them.
      await head.ensure(plan(tok, readRequest(), LENGTH).flatMap((st) => st.groups.flat()));
      for (const d of DEVICES) { await createSession(d); if (!device && sessions[d].state === 'ready') { device = d; renderDevices(); } }
    }
    if (CHECK === 'parity') await checkParity(out, rows);
  } catch (e) {
    out.fatal = errorText(e);
    await log('fatal: ' + out.fatal);
  }
  clearInterval(ticker); renderDevices();
  out.sessions = Object.fromEntries(Object.entries(sessions).map(([d, s]) => [d, { form: s.form, state: s.state, load_s: s.loadS, error: s.error }]));
  window.demo.ready = true;
  if (CHECK && TAG) {
    out.ended = new Date().toISOString();
    await fetch('/result/' + TAG, { method: 'POST', body: JSON.stringify(out) });
    await log('posted result');
  }
}
main();
export { card, EXAMPLES };  // for tests of the layout
