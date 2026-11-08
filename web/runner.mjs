// Runs d1-3B's static decision graph (export_onnx.py / make_forms.py) in ONNX Runtime Web with the WebNN EP, one
// question per run, and posts what happened back to serve.py: the browser and WebNN state, the session creation time or
// its full error text, the time of every run, and the hidden state of every parity question (base64 float32), which
// post_web.py turns into answers with d1-on-windows/npu/head.py and upstream's read-out.
//
// URL parameters: tag (result name), model (folder under models/), ep (webnn | wasm), device (cpu | gpu | npu),
// power (default | high-performance | low-power), passes (warm passes before the timed one, default 1),
// parity (ids file under fixtures/, default npu_ids_128.json; "none" skips), speed (default npu_speed_ids.json),
// opt (graphOptimizationLevel, default basic), threads (wasm threads, default 4).
import * as ort from './ort/ort.all.min.mjs';

const p = new URLSearchParams(location.search);
const tag = p.get('tag') || 'run';
const modelDir = 'models/' + (p.get('model') || 'w8');
const ep = p.get('ep') || 'webnn';
const device = p.get('device') || 'gpu';
const power = p.get('power') || 'default';
const passes = parseInt(p.get('passes') || '1', 10);
const parityFile = p.get('parity') || 'npu_ids_128.json';
const speedFile = p.get('speed') || 'npu_speed_ids.json';
const opt = p.get('opt') || 'basic';  // basic: constant folding turns the graph's shape arithmetic (Shape -> Gather -> Concat) into the constants the WebNN EP needs for Reshape / Slice / Expand; DequantizeLinear stays unfolded (ORT protects DQ unless session.disable_quant_qdq=1)
const PAD = 124893;
const out = { tag, ep, device, power, model: modelDir, opt, started: new Date().toISOString() };
const pre = document.getElementById('log');

async function log(line) {
  pre.textContent += '\n' + line;
  try { await fetch('/log/' + tag, { method: 'POST', body: line }); } catch (e) { /* the server is gone: nothing to do */ }
}
async function finish() {
  out.ended = new Date().toISOString();
  await fetch('/result/' + tag, { method: 'POST', body: JSON.stringify(out) });
  await log('posted result');
}
function b64(f32) {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
function errorText(e) { return (e && (e.stack || e.message)) ? String(e.stack || e.message) : String(e); }

async function main() {
  out.userAgent = navigator.userAgent;
  out.ort = ort.env.versions;
  out.hasML = 'ml' in navigator;
  out.crossOriginIsolated = self.crossOriginIsolated;
  await log(`ua ${navigator.userAgent} | navigator.ml ${out.hasML} | isolated ${self.crossOriginIsolated} | ort ${JSON.stringify(ort.env.versions)}`);
  let ctx = null;  // the page's own MLContext, handed to the WebNN EP so that the session runs where it was created
  if (ep === 'webnn') {
    if (!out.hasML) { out.error = 'navigator.ml is undefined (WebNN is not enabled in this browser)'; await log(out.error); return finish(); }
    try {
      const t = performance.now();
      ctx = await navigator.ml.createContext({ deviceType: device, powerPreference: power });
      out.context_ms = performance.now() - t;
      try { out.opSupportLimits = ctx.opSupportLimits(); } catch (e) { out.opSupportLimits_error = errorText(e); }
      try { out.context_accelerated = ctx.accelerated; } catch (e) { /* not in this build */ }
      await log(`createContext({deviceType: ${device}}) ok in ${out.context_ms.toFixed(1)} ms; input int64 ${JSON.stringify(out.opSupportLimits?.input?.dataTypes)}`);
    } catch (e) {
      out.context_error = errorText(e);
      await log('createContext failed: ' + out.context_error);
    }
  }
  const manifest = await (await fetch(modelDir + '/manifest.json')).json();
  out.manifest = manifest;
  const externalData = Object.keys(manifest.files).map((name) => ({ path: name, data: modelDir + '/' + name }));
  ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;  // './ort/' resolves against ort.all.min.mjs (= /ort/ort/)
  ort.env.wasm.numThreads = parseInt(p.get('threads') || '4', 10);
  ort.env.logLevel = 'warning';
  const options = {
    executionProviders: ep !== 'webnn' ? ['wasm']
      : ctx ? [{ name: 'webnn', context: ctx, deviceType: device }] : [{ name: 'webnn', deviceType: device, powerPreference: power }],
    externalData, graphOptimizationLevel: opt, logSeverityLevel: 2,
  };
  let session;
  const t0 = performance.now();
  try {
    session = await ort.InferenceSession.create(modelDir + '/model.onnx', options);
    out.session_ms = performance.now() - t0;
    await log(`session created in ${(out.session_ms / 1000).toFixed(1)} s; inputs ${session.inputNames} outputs ${session.outputNames}`);
  } catch (e) {
    out.session_ms = performance.now() - t0;
    out.session_error = errorText(e);
    await log('session failed after ' + (out.session_ms / 1000).toFixed(1) + ' s: ' + out.session_error);
    return finish();
  }
  const length = 128;
  async function forward(ids) {
    const input = new Int32Array(length).fill(PAD);
    input.set(ids);
    const feeds = { input_ids: new ort.Tensor('int32', input, [1, length]), last: new ort.Tensor('int32', new Int32Array([ids.length - 1]), []) };
    const t = performance.now();
    const res = await session.run(feeds);
    const data = await res.hidden.getData();
    return { ms: performance.now() - t, hidden: data };
  }
  // Speed: the 39- and 101-token prompts of d1-on-windows' speed fixture (467 does not fit a 128-token graph), warm passes first.
  const speed = (await (await fetch('fixtures/' + speedFile)).json()).filter((r) => r.ids && r.ids.length <= length);
  out.speed = [];
  try {
    for (let pass = 0; pass <= passes; pass++) {
      for (const row of speed) {
        const r = await forward(row.ids);
        if (pass === passes) out.speed.push({ q: row.q, tokens: row.ids.length, ms: r.ms, hidden: b64(r.hidden) });
      }
      await log(`speed pass ${pass} done (${speed.length} runs)`);
    }
  } catch (e) {
    out.run_error = errorText(e);
    await log('run failed: ' + out.run_error);
    return finish();
  }
  // Parity: every question of the fixture that fits, once.
  if (parityFile !== 'none') {
    const rows = await (await fetch('fixtures/' + parityFile)).json();
    out.parity = [];
    try {
      let n = 0;
      for (const row of rows) {
        if (!row.ids || row.ids.length > length) continue;
        const r = await forward(row.ids);
        out.parity.push({ q: row.q, tokens: row.ids.length, ms: r.ms, hidden: b64(r.hidden) });
        if (++n % 50 === 0) await log(`parity ${n}`);
      }
      await log(`parity done (${n})`);
    } catch (e) {
      out.run_error = errorText(e);
      await log('parity run failed: ' + out.run_error);
    }
  }
  await finish();
}
main().catch(async (e) => { out.fatal = errorText(e); await log('fatal: ' + out.fatal); await finish(); });
