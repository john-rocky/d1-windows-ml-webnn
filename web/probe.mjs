// WebNN state in this browser, without the big model: navigator.ml, createContext per deviceType, opSupportLimits,
// a one-op graph built with the WebNN API itself, and the same one-op graph through ONNX Runtime Web's WebNN EP.
// Posts one JSON to serve.py: /result/<tag>. URL parameters: tag.
import * as ort from './ort/ort.all.min.mjs';

const tag = new URLSearchParams(location.search).get('tag') || 'probe';
const out = { tag, started: new Date().toISOString(), userAgent: navigator.userAgent, hasML: 'ml' in navigator,
              isolated: self.crossOriginIsolated, ort: ort.env.versions, devices: {} };
const pre = document.getElementById('log');
async function log(line) {
  pre.textContent += '\n' + line;
  try { await fetch('/log/' + tag, { method: 'POST', body: line }); } catch (e) { /* server gone */ }
}
function errorText(e) { return (e && (e.stack || e.message)) ? `${e.name || 'Error'}: ${e.message}\n${e.stack || ''}` : String(e); }

async function rawWebNN(device) {
  const r = {};
  let t = performance.now();
  const ctx = await navigator.ml.createContext({ deviceType: device });
  r.context_ms = performance.now() - t;
  try { r.opSupportLimits = ctx.opSupportLimits(); } catch (e) { r.opSupportLimits_error = errorText(e); }
  for (const k of ['accelerated', 'deviceType', 'powerPreference']) { try { if (ctx[k] !== undefined) r['ctx_' + k] = ctx[k]; } catch (e) { /* none */ } }
  const builder = new MLGraphBuilder(ctx);
  const desc = (shape) => ({ dataType: 'float32', shape, dimensions: shape });
  const a = builder.input('a', desc([1, 4]));
  const eye = new Float32Array(16); for (let i = 0; i < 4; i++) eye[i * 5] = 1;
  const b = builder.constant(desc([4, 4]), eye);
  const c = builder.matmul(a, b);
  t = performance.now();
  const graph = await builder.build({ c });
  r.build_ms = performance.now() - t;
  const ta = await ctx.createTensor({ ...desc([1, 4]), writable: true });
  const tc = await ctx.createTensor({ ...desc([1, 4]), readable: true });
  ctx.writeTensor(ta, new Float32Array([0, 1, 2, 3]));
  ctx.dispatch(graph, { a: ta }, { c: tc });
  r.result = Array.from(new Float32Array(await ctx.readTensor(tc)));
  r.ok = r.result.join(',') === '0,1,2,3';
  return r;
}

async function ortWebNN(device) {
  const r = {};
  const t = performance.now();
  const session = await ort.InferenceSession.create('models/tiny/model.onnx', { executionProviders: [{ name: 'webnn', deviceType: device }] });
  r.session_ms = performance.now() - t;
  const res = await session.run({ x: new ort.Tensor('float32', new Float32Array([0, 1, 2, 3]), [1, 4]) });
  r.result = Array.from(await res.y.getData());
  r.ok = r.result.join(',') === '0,1,2,3';
  return r;
}

async function main() {
  await log(`ua ${navigator.userAgent} | navigator.ml ${out.hasML} | isolated ${self.crossOriginIsolated}`);
  ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;  // './ort/' resolves against ort.all.min.mjs (= /ort/ort/)
  for (const device of ['cpu', 'gpu', 'npu']) {
    const d = {};
    if (out.hasML) {
      try { d.webnn = await rawWebNN(device); } catch (e) { d.webnn = { ok: false, error: errorText(e) }; }
      await log(`${device} raw WebNN: ${JSON.stringify({ ok: d.webnn.ok, error: d.webnn.error, ms: d.webnn.context_ms })}`);
    }
    try { d.ort = await ortWebNN(device); } catch (e) { d.ort = { ok: false, error: errorText(e) }; }
    await log(`${device} ORT Web WebNN EP: ${JSON.stringify({ ok: d.ort.ok, error: d.ort.error })}`);
    out.devices[device] = d;
  }
  out.ended = new Date().toISOString();
  await fetch('/result/' + tag, { method: 'POST', body: JSON.stringify(out) });
  await log('posted result');
}
main().catch(async (e) => { out.fatal = errorText(e); await fetch('/result/' + tag, { method: 'POST', body: JSON.stringify(out) }); });
