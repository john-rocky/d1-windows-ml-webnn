# d1-3B through Windows ML and WebNN on a Windows laptop

[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B)'s decision step (415 questions with reference answers, and
prompts of 39, 101 and 467 tokens for the times) run through the two newest Microsoft paths for on-device models:

- **Windows ML**: ONNX Runtime from the `onnxruntime-windowsml` package, with execution providers (EPs) that Windows
  downloads and installs on request (`windowsml` on PyPI, its `EpCatalog`), driven by ONNX Runtime GenAI
  (`onnxruntime-genai-winml`) or by a plain ONNX Runtime session on a static graph.
- **WebNN** in Edge and Chrome: a web page with ONNX Runtime Web's WebNN EP, `deviceType` cpu / gpu / npu.

It is the same workload and the same scoring as [d1-on-windows](https://github.com/john-rocky/d1-on-windows), which
ran it through CUDA, WebGPU and OpenVINO on the same laptop; that repository's rows are the baseline here.

[Demo page](#demo-page): `web/demo.html` answers your own state, questions and options in Edge through WebNN, with
its own prompt building, tokenizer and option head in JavaScript, and shows the milliseconds of each decision.

Machine of the run recorded here (2026-11-07; the last WebNN row and the second timing of three d1-on-windows rows
after midnight; the WebNN rows on the Intel GPU on 2026-11-09, with the Windows build and the drivers below read again
that day): MSI Stealth 16 AI+ B3WI, Windows 11 Home 25H2 (26200.9711), Core Ultra 9 386H, Intel(R) Graphics (driver
32.0.101.8724), Intel(R) NPU (driver 32.0.100.4841), NVIDIA GeForce RTX 5080 Laptop GPU (16 GB, driver 591.74), 32 GB
RAM.

## Results

Answers are scored with d1-on-windows' `score.py` against the reference probabilities: "same top" counts the questions
whose most likely option matches the reference, and a change of the top option counts against a row only when it is
not a near tie (a question whose reference marks its top two options as too close to call). GenAI rows answer the 415
questions (the request file has 416 prompts; upstream's script refuses one question that has no instructions, and the
Windows ML rows' `score.json` list it under `refused`); the static 128-token graphs answer the 311 that fit 128 tokens
(they have no 467-token time). Times are the
median of 9 runs after one warm pass, with the laptop otherwise quiet when each timing started (system CPU under 15 %
over three 5-second samples, no other process holding RTX memory). GenAI rows time upstream's `decide()` per question;
static rows time the forward pass and the read-out. "vs d1-on-windows' row" divides by that repository's row for the
same device (recorded 2026-11-05); "vs the 2026-11-07 re-timing" divides by the same d1-on-windows configuration timed
between 14:50 and 15:22 on 2026-11-07, in the same run of timings as the Windows ML rows.

| Device | Path | EP | Model | Answers (same top / scored) | 39 / 101 / 467 tokens, ms | vs d1-on-windows' row (39 / 101 / 467) | vs the 2026-11-07 re-timing (101) |
|---|---|---|---|---|---|---|---|
| RTX 5080 Laptop | d1-on-windows' row, re-timed 2026-11-07 | CUDA (PyPI ORT 1.31.1, GenAI 0.19.0) | GenAI cuda-fp16 | d1-on-windows' record | 18.4 / 21.7 / 61.2 | ×1.17 / ×1.18 / ×0.94 | — |
| RTX 5080 Laptop | Windows ML, ORT 1.30.0, GenAI-winml | WebGPU 0.5.0 | GenAI cpu-fp32 | 415/415, answers from the README run's own cpu-fp32 folder | 69.4 / 117 / 384 | ×4.42 / ×6.35 / ×5.93 | ×5.39 |
| RTX 5080 Laptop | WebNN, Edge 156 (signed-in session), ORT Web 1.31.1, deviceType gpu | DirectML (RTX) | static 128, int8 weights | 311/311 | 149 / 146 / — | ×9.51 / ×7.96 / — | ×6.75 |
| RTX 5080 Laptop | WebNN, Edge 156 (signed-in session), ORT Web 1.31.1, deviceType gpu | DirectML (RTX) | static 128, fp16 | 311/311 | 18.3 / 17.0 / — | ×1.17 / ×0.92 / — | ×0.78 |
| RTX 5080 Laptop | WebNN, Chrome for Testing 157, ORT Web 1.31.1, deviceType gpu | DirectML (RTX) | static 128, int8 weights | 311/311 | 143 / 147 / — | ×9.14 / ×7.97 / — | ×6.76 |
| RTX 5080 Laptop | WebNN, Chrome for Testing 157, ORT Web 1.31.1, deviceType gpu | DirectML (RTX) | static 128, fp16 | 311/311 | 17.3 / 17.3 / — | ×1.10 / ×0.94 / — | ×0.80 |
| RTX 5080 Laptop | WebNN, Chrome for Testing 159 (canary), ORT Web 1.31.1, deviceType gpu | WebGPU 0.5.0 (RTX) | static 128, int8 weights | 311/311 | 211 / 210 / — | ×13.41 / ×11.40 / — | ×9.67 |
| RTX 5080 Laptop | WebNN, Chrome for Testing 159 (canary), ORT Web 1.31.1, deviceType gpu | WebGPU 0.5.0 (RTX) | static 128, fp16 | 310/311 (1 not a near tie) | 141 / 142 / — | ×9.00 / ×7.72 / — | ×6.55 |
| Intel GPU | d1-on-windows' row, re-timed 2026-11-07 | OpenVINO 1.8.1 (PyPI) | GenAI cpu-fp32 | d1-on-windows' record | 130 / 176 / 622 | ×1.21 / ×1.26 / ×1.57 | — |
| Intel GPU | Windows ML, ORT 1.30.0, GenAI-winml | OpenVINO 1.7.0 | GenAI cpu-fp32 | 415/415 | 144 / 195 / 685 | ×1.34 / ×1.39 / ×1.73 | ×1.11 |
| Intel GPU | Windows ML, ORT 1.28.3 | OpenVINO 1.7.0 | static 128, fp16 | 311/311 | 103 / 102 / — | ×0.95 / ×0.73 / — | ×0.58 |
| Intel GPU | WebNN, Edge 157 (signed-in session), ORT Web 1.31.1, deviceType gpu, Edge started with `--webnn-ort-ep-device=OpenVINOExecutionProvider,0x8086,0xb0a0` | OpenVINO 1.7.0 (Intel GPU) | static 128, fp16 | 311/311 | 108 / 103 / — | ×1.00 / ×0.74 / — | — |
| Intel GPU | WebNN, Edge 157 (signed-in session), ORT Web 1.31.1, deviceType gpu, Edge started with `--webnn-ort-ep-device=DmlExecutionProvider,0x8086,0xb0a0` | DirectML (Intel GPU) | static 128, fp16 | 311/311 | 109 / 107 / — | ×1.01 / ×0.76 / — | — |
| Intel GPU | WebNN, Edge 157 (signed-in session), ORT Web 1.31.1, deviceType gpu, Edge started with `--webnn-ort-ep-device=OpenVINOExecutionProvider,0x8086,0xb0a0` | OpenVINO 1.7.0 (Intel GPU) | static 128, int8 weights | 311/311 | 52.8 / 52.9 / — | ×0.49 / ×0.38 / — | — |
| Intel GPU | WebNN, Edge 157 (signed-in session), ORT Web 1.31.1, deviceType gpu, Edge started with `--webnn-ort-ep-device=DmlExecutionProvider,0x8086,0xb0a0` | DirectML (Intel GPU) | static 128, int8 weights | 311/311 | 831 / 829 / — | ×7.69 / ×5.92 / — | — |
| CPU | d1-on-windows' row, re-timed 2026-11-07 | OpenVINO 1.8.1 (PyPI) | GenAI cpu-int8 | d1-on-windows' record | 471 / 1346 / 5245 | ×2.23 / ×2.27 / ×2.28 | — |
| CPU | Windows ML, ORT 1.30.0, GenAI-winml | OpenVINO 1.7.0 | GenAI cpu-int8 | 390/391 (1 near tie), the first 392 questions (memory, see below) | 480 / 1411 / 5379 | ×2.28 / ×2.38 / ×2.33 | ×1.05 |
| CPU | Windows ML, ORT 1.30.0, GenAI-winml | CPU | GenAI cpu-int8 | 415/415 | 811 / 1885 / 8273 | ×3.84 / ×3.17 / ×3.59 | ×1.40 |
| CPU | Windows ML, ORT 1.28.3, GenAI-winml | CPU | GenAI cpu-int8 without state_window | 414/415 (1 not a near tie) | 812 / 1886 / 8291 | ×3.85 / ×3.18 / ×3.60 | ×1.40 |
| CPU | Windows ML, ORT 1.28.3 | CPU | static 128, int8 weights | 311/311 | 2257 / 2258 / — | ×10.70 / ×3.80 / — | ×1.68 |
| CPU | WebNN, Edge 156, ORT Web 1.31.1, deviceType cpu | OpenVINO 1.7.0 (CPU) | static 128, int8 weights | 310/311 (1 not a near tie) | 534 / 532 / — | ×2.53 / ×0.90 / — | ×0.40 |
| CPU | WebNN, Chrome for Testing 157, ORT Web 1.31.1, deviceType cpu | OpenVINO 1.7.0 (CPU) | static 128, int8 weights | 310/311 (1 not a near tie) | 528 / 527 / — | ×2.50 / ×0.89 / — | ×0.39 |
| CPU | WebNN, Chrome for Testing 159 (canary), ORT Web 1.31.1, deviceType cpu | OpenVINO 1.7.0 (CPU) | static 128, int8 weights | 310/311 (1 not a near tie) | 527 / 526 / — | ×2.50 / ×0.89 / — | ×0.39 |
| Intel NPU | d1-on-windows' row, re-timed 2026-11-07 | OpenVINO 1.8.1 (PyPI ORT 1.31.1) | static 128, d1-on-windows' NPU graph | d1-on-windows' record | 54.3 / 54.0 / — | ×1.00 / ×1.00 / — | — |
| Intel NPU | Windows ML, ORT 1.28.3 | OpenVINO 1.7.0 | static 128, d1-on-windows' NPU graph | 311/311 | 54.1 / 54.1 / — | ×1.00 / ×1.00 / — | ×1.00 |
| Intel NPU | Windows ML, ORT 1.28.3 | OpenVINO 1.7.0 | static 128, fp16 | 311/311 | 96.5 / 96.8 / — | ×1.78 / ×1.79 / — | ×1.79 |
| Intel NPU | WebNN, Edge 156, ORT Web 1.31.1, deviceType npu | OpenVINO 1.7.0 (NPU) | static 128, int8 weights | 311/311 | 55.2 / 55.2 / — | ×1.02 / ×1.02 / — | ×1.02 |
| Intel NPU | WebNN, Edge 156 (signed-in session), ORT Web 1.31.1, deviceType npu | OpenVINO 1.7.0 (NPU) | static 128, fp16 | 311/311 | 96.8 / 97.2 / — | ×1.79 / ×1.79 / — | ×1.80 |
| Intel NPU | WebNN, Chrome for Testing 157, ORT Web 1.31.1, deviceType npu | OpenVINO 1.7.0 (NPU) | static 128, int8 weights | 311/311 | 55.2 / 55.4 / — | ×1.02 / ×1.02 / — | ×1.03 |
| Intel NPU | WebNN, Chrome for Testing 159 (canary), ORT Web 1.31.1, deviceType npu | OpenVINO 1.7.0 (NPU) | static 128, int8 weights | 311/311 | 55.7 / 56.1 / — | ×1.03 / ×1.04 / — | ×1.04 |

Each row's files are in [`results/`](results). Windows ML rows: answers `<row>.jsonl`, score `<row>.score.json`, times
`speed_<row>.jsonl` (the OpenVINO CPU row's answers and score: `wml130_ov_cpu_int8_392.*`). WebNN rows: answers, score
and the page's log (`.log`) under `web_…`, `webs1_…` (Edge's signed-in session) or `weblp_…` (Chrome's fp16 rows on the
RTX, run with `powerPreference: 'low-power'`), and times from a separate timing run, `speed_webt_…` (`speed_webs1t_…`
for Edge's signed-in session). The re-timed d1-on-windows rows have times only (`speed_ctl_…`). In those files, the
folders of the run are written as `<models>` (the model folders) and `<work>` (the working folder).
The Intel GPU WebNN rows come from the demo page's headless check (see "Demo page"): answers
`demo_igpu_<ov|dml>_<h16|w8>.jsonl` (`ov` = the OpenVINO EP, `dml` = DirectML, `h16` = fp16, `w8` = int8 weights), score
`.score.json`, the check's summary with the times (`.check.json`), the page's log (`.log`) and the lines of Edge's log
that name the device it used (`.selected.txt`); the same files for the two checks on the RTX timed before and after them
(`demo_gpu_h16_before`, `demo_gpu_h16_after`) and for the check with `powerPreference: 'low-power'`
(`demo_lowpower_gpu_h16`).

Windows ML's TensorRT-RTX EP was also run on the RTX. Everything about it is left out of this repository (its rows, its
result files, and its entry in the list of Windows ML EPs that each Windows ML result file carries, `winml_eps` or
`registered`): the EP is built on NVIDIA TensorRT for RTX, whose license agreement (section 2.13) requires NVIDIA's
prior written permission before results of that software are shared.

The re-timed d1-on-windows rows against their records of 2026-11-05 (39 / 101 / 467 tokens): ×1.17 / ×1.18 / ×0.94 on
the RTX, ×1.21 / ×1.26 / ×1.57 on the Intel GPU, ×2.23 / ×2.27 / ×2.28 on the CPU, and ×1.00 on the NPU. Windows was
updated in between (26200.6899 to 26200.9711); the power settings were the same (on AC, Balanced). The cause is not
separated, so read each row against the re-timing of its device (the last column).

The WebNN rows were timed after the Windows ML rows: the int8-weight rows from 20:18 to 20:28 (Edge's on the RTX excepted), and the fp16
rows and Edge's int8-weight row on the RTX from 23:47 to 00:11. A second timing of three d1-on-windows rows (RTX, NPU,
CPU) from 00:03 to 00:09 gave 19.7 / 21.1 / 63.9 ms on the RTX and 54.9 / 54.6 ms on the NPU (101 tokens within 3 % of
the 14:50 rows), and 632 / 1,895 / 7,437 ms on the CPU (×1.34 to ×1.42, cause not separated). So the last column holds
for the Windows ML rows (timed together with their re-timing) and for the WebNN RTX and NPU rows. It does not hold for
the WebNN CPU rows: there, only their own times count.

The Intel GPU WebNN rows were timed on 2026-11-09 from 05:00 to 05:10 (Edge 157.0.4341.42), between two runs of the same
check on the RTX with the fp16 graph as a control: 18.2 ms before and 18.5 ms after for 101 tokens (Edge 156's row
above: 17.0 ms). No d1-on-windows row was timed with them, so their last column is empty.

Edge's rows marked "signed-in session" ran from the laptop's desktop. Its int8-weight cpu and npu rows ran from an ssh
session, where Edge's GPU process restarted twice and the page then answered (see "What did not work").

- RTX: GenAI on Windows ML's WebGPU EP, 117 ms for 101 tokens, is in the class of d1-on-windows' WebGPU row (113 ms
  there on 2026-11-05). GenAI's Windows ML build has no CUDA EP, so d1-on-windows' cuda-fp16 folder does not run on
  Windows ML (see "What did not work").
- Intel GPU: the static fp16 graph on Windows ML's OpenVINO EP, 102 ms (×0.58), runs where d1-on-windows' OpenVINO IR
  of the same wrapper (fp16 weights, 128 tokens) failed to compile in OpenVINO Runtime. GenAI with the same EP, 195 ms,
  is ×1.11 of the re-timing (the OpenVINO EP from PyPI).
- CPU: GenAI with Windows ML's OpenVINO EP, 1,411 ms (×1.05), and with the CPU EP, 1,885 ms (×1.40). Both OpenVINO
  EP builds hold much more memory on long prompts (see "What did not work"); the CPU EP answers all 415 questions.
- Intel NPU: d1-on-windows' NPU graph runs unchanged on Windows ML's OpenVINO EP at the same speed, 54.1 ms (×1.00).
- WebNN on the RTX: `deviceType` gpu is DirectML on the RTX in Edge 156 and Chrome 157. The static fp16 graph takes
  17.0 ms in Edge and 17.3 ms in Chrome for 101 tokens (×0.78 and ×0.80 of the CUDA re-timing), for prompts up to 128
  tokens; the int8-weight graph takes 146 to 147 ms. In Chrome 159 the same `deviceType` is the WebGPU EP 0.5.0: 210 ms
  with int8 weights, and 142 ms with fp16, where one answer changes (see "What did not work"). `powerPreference:
  'low-power'` still picks the RTX in both Chromes and in Edge 157 (Edge's log names DirectML on the RTX as the
  device it used); Chromium's WebNN code for ONNX Runtime does not read the power preference yet (a TODO in its
  source, crbug.com/425487285).
- WebNN on the Intel GPU: Edge reaches it with Chromium's switch `--webnn-ort-ep-device=<EP name>,<vendor id>,<device
  id>`, which makes WebNN hand ONNX Runtime that one device whatever `deviceType` the page asks for (ONNX Runtime's CPU
  fallback stays on). On this laptop's Intel(R) Graphics (`0x8086,0xb0a0`) the OpenVINO EP takes 103 ms with fp16 and
  52.9 ms with int8 weights for 101 tokens (×0.74 and ×0.38 of d1-on-windows' Intel GPU row), and DirectML 107 ms and
  829 ms. The int8-weight graph runs on the OpenVINO EP here, where it fails on Windows ML's OpenVINO EP (see "What did
  not work"). The cause was not separated; one difference is that through WebNN the EP gets a graph Chromium rebuilds
  from the page's WebNN operations, not the ONNX file.
- WebNN on the NPU: the int8-weight graph takes 55.2 to 56.1 ms in all three browsers (d1-on-windows' NPU row: 54.0 ms
  in the re-timing) and the fp16 graph 97.2 ms in Edge, as on Windows ML (96.8 ms).
- WebNN on the CPU: `deviceType` cpu is the OpenVINO EP's CPU device, 526 to 532 ms with one changed answer (see "What
  did not work").

The WebNN block under "Run it" reproduces the Edge rows. Run as written on 2026-11-08 (Edge was version 157 by then),
its five runs scored as the rows do (310/311 on the CPU, 311/311 on the others), and their 101-token times were within
6 % of the rows. The Chrome rows ran the same page in Chrome for Testing 157.0.8103.38 and
159.0.8147.0 (a canary build), unpacked from the zips, with `--enable-features=WebMachineLearningNeuralNetwork` as
for Edge. Two more things were needed: read access for app containers on the unpacked folder (`icacls <folder> /grant
"*S-1-15-2-1:(OI)(CI)(RX)" "*S-1-15-2-2:(OI)(CI)(RX)" /T`), and, when started from an ssh session,
`--disable-features=NetworkServiceSandbox` (without it the page never loaded; with it, it did).

## What you need

- Windows 11 x64 (this run: 25H2, 26200.9711). The `windowsml` package finds the EPs in Windows' catalog and installs
  the ones that are missing: here the WebGPU EP was not installed, and its first `ensure_ready()` took 16.8 s and
  needed no restart.
- Python 3.12 (`py -3.12`), git, and about 42 GB of free disk in this folder (41 GB after block 5 here; 39.4 GB after
  block 4).
- 32 GB of RAM with the page file on, and no other heavy job: GenAI's int8 builder and the NPU graph's quantization each
  peaked at 22 to 23 GiB in use and 29 GiB committed, and Edge's NPU compile of the fp16 graph held about 26 GiB.
- Edge for the WebNN part (the rows: 156.0.4314.58; the re-run of the WebNN block on 2026-11-08: 157; the Intel GPU rows on
  2026-11-09: 157.0.4341.42).
- A folder with LiquidAI/d1-3B from Hugging Face (this run: revision 051bcc464b).

## Run it

Run these blocks in order, in one PowerShell window, from this folder. Each Python package brings its own ONNX
Runtime, so blocks 2, 3 and 4 each create their own virtual environment (block 1 uses `py -3.12` directly; the WebNN
block reuses block 2's `.venv-build`).

```powershell
# The fixtures, upstream's decision script with its Windows ML helper, d1-on-windows' scripts, and the model folder
foreach ($f in "rebuild_requests.py","requests_public.json","reference_probs.json") { curl.exe -sSLO "https://raw.githubusercontent.com/john-rocky/hf-to-litertlm/main/d1_3b_work/fixtures/$f" }
py -3.12 rebuild_requests.py --out requests.json
New-Item -ItemType Directory -Force upstream | Out-Null
foreach ($f in "d1-decisions.py","common.py","winml.py") { curl.exe -sSL -o "upstream\$f" "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/v0.19.0/examples/python/$f" }
git clone -q https://github.com/john-rocky/d1-on-windows
git -C d1-on-windows checkout -q 5d5c3268d6d052e5e99ac7e6fdee75a5939d2a3f
Copy-Item d1-on-windows\results\speed_requests.json .
# A folder with LiquidAI/d1-3B from https://huggingface.co/LiquidAI/d1-3B (revision 051bcc464b here)
$model = "<folder with LiquidAI/d1-3B>"
```

`rebuild_requests.py` downloads one public file and checks the SHA-256 of each result. See
[its README](https://github.com/john-rocky/hf-to-litertlm/tree/main/d1_3b_work/fixtures). `speed_requests.json` holds
the timed prompts (9 each of 39, 101, 467, 1024 and 3470 tokens; the times here use the first three).

```powershell
# The model folders: GenAI 0.19.0's builder as in d1-on-windows, the static 128-token ONNX graphs, and d1-on-windows'
# NPU graph as an EPContext model
py -3.12 -m venv .venv-build
$py = ".venv-build\Scripts\python.exe"
& $py -m pip install onnxruntime-genai==0.19.0 torch transformers==5.22.1 onnx==1.23.2 onnx-ir==1.1.0 openvino==2026.4.1 nncf==3.4.0
& $py -m onnxruntime_genai.models.builder -i $model -o cpu-fp32 -p fp32 -e cpu -c cache --extra_options prune_lm_head=true
& $py -m onnxruntime_genai.models.builder -i $model -o cpu-int8 -p int8 -e cpu -c cache --extra_options prune_lm_head=true
foreach ($d in "cpu-fp32","cpu-int8") { & $py d1-on-windows\fix_tokenizer.py $d }
# The static graphs for WebNN and the EPs: 128 tokens, fp16 (h16), and int8 weights with fp32 activations (w8)
& $py web\export_onnx.py --model $model --length 128 --dtype float16 --out x\f16-128
& $py web\make_forms.py --src x\f16-128\model.onnx --form h16 --out x\h16-128
& $py web\make_forms.py --src x\f16-128\model.onnx --form w8 --out x\w8-128
& $py web\check_onnx.py --model x\h16-128\model.onnx --probe x\f16-128\probe.npz
& $py web\check_onnx.py --model x\w8-128\model.onnx --probe x\f16-128\probe.npz
# d1-on-windows' NPU graph: 128 tokens, int8 weights, 16-bit activations, no head; then its EPContext wrapper
& $py d1-on-windows\npu\export_ir.py --model $model --length 128 --head hidden --out ir-128
& $py d1-on-windows\npu\make_ids.py --upstream upstream -m cpu-int8 --requests requests.json --out ids-128.json --max_tokens 128
& $py d1-on-windows\npu\make_ids.py --upstream upstream -m cpu-int8 --requests speed_requests.json --out speed-ids-128.json --max_tokens 128
# With output sent to a file on a non-UTF-8 code page, the quantizer's progress display stops with UnicodeEncodeError
$env:PYTHONUTF8 = "1"
& $py d1-on-windows\npu\quant_ir.py --ir ir-128\model.xml --ids ids-128.json --out ir-128-q --subset 100 --bits 16
& $py d1-on-windows\npu\wrap_ir.py ir-128-q\model.xml
```

`export_onnx.py` uses d1-on-windows' wrapper: input ids padded on the right to 128 tokens, the index of the last real
token as a second input, no cache, and the hidden state at that token as the output. The head runs on the CPU
(`d1-on-windows\npu\head.py`, the options' rows only), as in d1-on-windows' NPU rows. `check_onnx.py` runs each form on
the ONNX Runtime CPU EP against PyTorch's hidden state on one probe prompt.

A 128-token graph answers the 311 questions that fit it, and times the 39- and 101-token prompts.

```powershell
# Windows ML as `windowsml[with-ort]` installs it (ONNX Runtime 1.28.3): the EP catalog, upstream's example, and the
# static graphs on each EP
py -3.12 -m venv .venv-wml
$py = ".venv-wml\Scripts\python.exe"
& $py -m pip install "windowsml[with-ort]==2.5.96" onnxruntime-genai-winml==0.19.0 onnx==1.23.2
& $py winml_probe.py --mode catalog --out catalog.json
# Upstream's example on this ONNX Runtime: the model does not load (state_window, see "What did not work")
& $py -c "import json; r = json.load(open('speed_requests.json'))['records'][0]['request']; json.dump(r, open('request.json', 'w'))"
& $py upstream\d1-decisions.py -m cpu-int8 -r request.json -e cpu --use_winml
# The same model folder without the state_window attributes (all 0) loads and answers
& $py strip_state_window.py cpu-int8 cpu-int8-nosw
& $py run_genai.py --upstream upstream -m cpu-int8-nosw --winml cpu --requests requests.json --out wml_cpu_int8_nosw.jsonl
& $py d1-on-windows\score.py wml_cpu_int8_nosw.jsonl reference_probs.json
# The static graphs: answers on each EP, then the times (one warm pass, 9 timed runs per prompt length)
foreach ($r in @(
    @("ov_gpu_h16", "x\h16-128\model.onnx", "OpenVINOExecutionProvider", "GPU"),
    @("ov_npu_h16", "x\h16-128\model.onnx", "OpenVINOExecutionProvider", "NPU"),
    @("ov_npu_hid16", "ir-128-q\model_ctx.onnx", "OpenVINOExecutionProvider", "NPU"),
    @("cpu_w8", "x\w8-128\model.onnx", "CPUExecutionProvider", ""))) {
  $tag, $graph, $ep, $hw = $r
  $dev = @(); if ($hw) { $dev = @("--hw", $hw) }
  & $py run_static.py --upstream upstream --model $graph --ep $ep @dev --winml --head ir-128\lm_head.npy --ids ids-128.json --out "wml_$tag.jsonl"
  & $py d1-on-windows\score.py "wml_$tag.jsonl" reference_probs.json
  & $py run_static.py --upstream upstream --model $graph --ep $ep @dev --winml --head ir-128\lm_head.npy --ids speed-ids-128.json --passes 2 --out "speed_wml_$tag.jsonl"
  & $py d1-on-windows\speed_lines.py "speed_wml_$tag.jsonl"
}
```

`winml_probe.py` lists each EP of the catalog with its state before and after `ensure_ready()`, its package and
version, and the devices ONNX Runtime then offers. `run_static.py` registers the catalog's EPs (`--winml`) and runs the
graph on one device of one EP; `--hw` picks the device type when the EP has several.

```powershell
# Windows ML with ONNX Runtime 1.30.0 (onnxruntime-windowsml from PyPI): GenAI on WebGPU, OpenVINO and the CPU, answers and times
py -3.12 -m venv .venv-wml130
$py = ".venv-wml130\Scripts\python.exe"
& $py -m pip install windowsml==2.5.96 onnxruntime-windowsml==1.30.0.202609102321 onnxruntime-genai-winml==0.19.0
$three = (39, 101, 467 | ForEach-Object { $t = $_; 0..8 | ForEach-Object { "speed_${t}_$_" } }) -join ","
foreach ($r in @(
    @("webgpu_fp32", "cpu-fp32", "--winml", "WebGpuExecutionProvider", "--ep_option", "adapterIndex=0"),
    @("ov_gpu_fp32", "cpu-fp32", "--winml", "OpenVINO", "--ep_option", "device_type=GPU"),
    @("ov_cpu_int8", "cpu-int8", "--winml", "OpenVINO", "--ep_option", "device_type=CPU"),
    @("cpu_int8", "cpu-int8", "--winml", "cpu"))) {
  $tag, $folder, $ep = $r[0], $r[1], $r[2..($r.Count - 1)]
  # OpenVINO on the CPU holds memory that grows with the longest prompt so far (see "What did not work"): the first 392 questions only
  $more = @(); if ($tag -eq "ov_cpu_int8") { $more = @("--max", "392") }
  & $py run_genai.py --upstream upstream -m $folder @ep --requests requests.json @more --out "wml130_$tag.jsonl"
  & $py d1-on-windows\score.py "wml130_$tag.jsonl" reference_probs.json
  & $py run_genai.py --upstream upstream -m $folder @ep --requests speed_requests.json --ids $three --passes 2 --out "speed_wml130_$tag.jsonl"
  & $py d1-on-windows\speed_lines.py "speed_wml130_$tag.jsonl"
}
```

`run_genai.py --winml <EP>` registers the catalog's EPs with upstream's `winml.py` (the same calls as `d1-decisions.py
--use_winml`) and then runs upstream's `decide()` per question, as d1-on-windows' `run_genai.py` does for CUDA,
WebGPU and OpenVINO. `--winml cpu` uses the CPU EP of the same ONNX Runtime.

```powershell
# Run this block in a PowerShell window on the laptop's own desktop (signed in). Over ssh, Edge's GPU process restarts
# and WebNN turns itself off (see "What did not work").
# WebNN in Edge: ONNX Runtime Web 1.31.1's WebNN EP on the static graphs, one browser run per device and graph. The NPU
# run of the fp16 graph compiles for 1.5 to 4 minutes before its first answer (80.7 s with the laptop otherwise quiet,
# 250.8 s while another process was exporting a model).
curl.exe -sSL -o onnxruntime-web-1.31.1.tgz https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.31.1.tgz
tar -xzf onnxruntime-web-1.31.1.tgz
New-Item -ItemType Directory -Force webroot\ort, webroot\models, webroot\fixtures, webresults | Out-Null
Copy-Item package\dist\* webroot\ort\ -Recurse -Force
Copy-Item web\index.html, web\runner.mjs webroot\
Copy-Item ids-128.json webroot\fixtures\npu_ids_128.json
Copy-Item speed-ids-128.json webroot\fixtures\npu_speed_ids.json
foreach ($f in "h16","w8") { if (-not (Test-Path webroot\models\$f)) { New-Item -ItemType Junction -Path webroot\models\$f -Target "$PWD\x\$f-128" | Out-Null } }
$server = Start-Process ".venv-build\Scripts\python.exe" -ArgumentList "web\serve.py --root webroot --results webresults --port 8780" -WindowStyle Hidden -PassThru
Start-Sleep 5
# serve.py does not share its port: if another program already listens on 8780, the server stops here, and the pages
# would talk to that program instead.
$runs = if ($server.HasExited) { Write-Host "Port 8780 is taken (Get-NetTCPConnection -LocalPort 8780 -State Listen): stop that program, then run this block again."; @() } else { "cpu:w8", "gpu:w8", "gpu:h16", "npu:w8", "npu:h16" }
$edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
$py = ".venv-build\Scripts\python.exe"
foreach ($run in $runs) {
  $device, $graph = $run.Split(":")
  $tag = "edge_$($device)_$graph"
  $url = "http://127.0.0.1:8780/index.html?tag=$tag&device=$device&model=$graph"
  $browser = Start-Process $edge -ArgumentList "--user-data-dir=`"$PWD\profiles\$tag`" --no-first-run --headless=new --enable-features=WebMachineLearningNeuralNetwork `"$url`"" -PassThru
  $t0 = Get-Date
  while (-not (Test-Path "webresults\$tag.done") -and ((Get-Date) - $t0).TotalMinutes -lt 45) { Start-Sleep 5 }
  Get-CimInstance Win32_Process -Filter "Name = 'msedge.exe'" | Where-Object { $_.CommandLine -like "*profiles\$tag*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  if (-not (Test-Path "webresults\$tag.done")) { Write-Host "No result for $tag after 45 minutes; the page's log is webresults\$tag.log"; continue }
  & $py web\post_web.py --upstream upstream --result "webresults\$tag.json" --ids ids-128.json --speed_ids speed-ids-128.json --head ir-128\lm_head.npy --out "web_$tag.jsonl" --speed_out "speed_web_$tag.jsonl"
  & $py d1-on-windows\score.py "web_$tag.jsonl" reference_probs.json
  & $py d1-on-windows\speed_lines.py "speed_web_$tag.jsonl"
}
Stop-Process -Id $server.Id
```

The page (`web/runner.mjs`) creates an `MLContext` for the device, hands it to ONNX Runtime Web's WebNN EP, times
the 39- and 101-token prompts after one warm pass, answers the 311 questions that fit 128 tokens, and posts the hidden
states back to `serve.py`. `post_web.py` turns them into answers with the same head and read-out as `run_static.py`.
`webresults\<tag>.log` holds the page's lines and any error in full.

## Demo page

`web/demo.html` runs the decision step for your own input: type a state (text, or JSON), one or more questions with
their type (choice, noul or score) and options, pick the GPU, the NPU or the CPU, and press Decide. The page builds
upstream's prompt, turns it into tokens, runs the static 128-token graph through WebNN, and reads the answer from the
options' rows of the head. It shows each option's probability and the milliseconds of each decision (the forward pass
and the read-out, the span of the static rows above). Three examples fill the form. Python only serves the files
(`serve.py` on 127.0.0.1).

- `web/d1prompt.mjs`: upstream's request checks, prompt and read-out (`d1-decisions.py`, GenAI 0.19.0) in JavaScript:
  the same error messages, the state written the way Python's `json.dumps(state, ensure_ascii=False, indent=2)` writes
  it (key order, and floats such as `1000.0`), the option codes and token groups of each type, the softmax over the
  options. One question is encoded as one string; several share the encoded prefix, as upstream does.
- `web/d1tok.mjs`: byte-level BPE from the model's own `tokenizer.json` (its pre-tokenizer regex, merges and added
  tokens).
- The head: the page fetches only the options' rows of `lm_head.npy` (HTTP range requests, 8 KiB a row) and takes the
  dot products itself.

A question whose prompt is longer than 128 tokens is refused with its token count (the graphs are static).

The block below ran as written on 2026-11-09 on the laptop above (Edge 157, headless, in the signed-in session). The
page's prompts and tokens match d1-on-windows' `npu_ids_128.json` for all 311 questions that fit 128 tokens (the same
token ids and option groups); the question upstream refuses is refused here too, and the same 104 questions are longer
than 128 tokens. The answers are scored with `score.py` like the rows above, and the read-out column runs
d1-on-windows' head (`npu/head.py`) and upstream's read-out on the hidden states the page returned. An example's time
is the median of 9 decisions after one warm one, for each of the examples' 8 questions.

| Device | WebNN deviceType | Model | Answers (same top / scored) | The page's read-out vs Python's (same top, largest difference) | One question of the examples, ms | Session ready after, s (headless, one device a page) |
|---|---|---|---|---|---|---|
| RTX 5080 Laptop | gpu | static 128, fp16 | 311/311 | 311/311, 1.6e-06 | 17.5–20.0 | 36 |
| Intel NPU | npu | static 128, int8 weights | 311/311 | 311/311, 1.5e-06 | 55.6–57.1 | 85 |
| CPU | cpu | static 128, int8 weights | 310/311 (1 not a near tie) | 311/311, 1.7e-06 | 528–532 | 21 |

The first decision after a pause takes longer: on the GPU, 0.18 to 0.2 s headless (in one run the one after it too,
0.1 s) and 1.0 s in the app window; on the NPU the first three are slower, 97, 75 and 64 ms, in the runs here.

```powershell
# After the WebNN block (it fills webroot\ort, webroot\models and webroot\fixtures), in the same window on the
# laptop's own desktop. The page loads the GPU, NPU and CPU sessions one after another (the NPU session took 73 to 85 s
# to be ready in the runs here); the buttons on the right switch between them.
Copy-Item web\demo.html, web\demo.mjs, web\d1prompt.mjs, web\d1tok.mjs webroot\
Copy-Item "$model\tokenizer.json" webroot\models\tokenizer.json
Copy-Item requests.json webroot\fixtures\requests.json
if (-not (Test-Path webroot\models\lm_head.npy)) { New-Item -ItemType HardLink -Path webroot\models\lm_head.npy -Target "$PWD\ir-128\lm_head.npy" | Out-Null }
$server = Start-Process ".venv-build\Scripts\python.exe" -ArgumentList "web\serve.py --root webroot --results webresults --port 8780" -WindowStyle Hidden -PassThru
Start-Sleep 5
# As in the WebNN block: if another program already listens on 8780, the server stops here.
$runs = if ($server.HasExited) { Write-Host "Port 8780 is taken (Get-NetTCPConnection -LocalPort 8780 -State Listen): stop that program, then run this block again."; @() } else { "ids", "gpu", "npu", "cpu" }
$edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
$py = ".venv-build\Scripts\python.exe"
# The checks in the table above: one headless run each, the answers scored like the rows above
foreach ($run in $runs) {
  $tag = "demo_$run"
  $query = if ($run -eq "ids") { "check=ids" } else { "check=parity&device=$run" }
  Remove-Item "webresults\$tag.*" -ErrorAction SilentlyContinue
  $browser = Start-Process $edge -ArgumentList "--user-data-dir=`"$PWD\profiles\$tag`" --no-first-run --headless=new --enable-features=WebMachineLearningNeuralNetwork `"http://127.0.0.1:8780/demo.html?$query&tag=$tag`"" -PassThru
  $t0 = Get-Date
  while (-not (Test-Path "webresults\$tag.done") -and ((Get-Date) - $t0).TotalMinutes -lt 20) { Start-Sleep 5 }
  Get-CimInstance Win32_Process -Filter "Name = 'msedge.exe'" | Where-Object { $_.CommandLine -like "*profiles\$tag*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  if ($run -eq "ids") { Select-String -Path "webresults\$tag.log" -Pattern "ids:" } else {
    & $py web\check_demo.py --result "webresults\$tag.json" --d1 d1-on-windows --upstream upstream --head ir-128\lm_head.npy --reference reference_probs.json --ids ids-128.json --out "$tag.jsonl" }
}
# The page itself, as an app window
if (-not $server.HasExited) { Start-Process $edge -ArgumentList "--user-data-dir=`"$PWD\profiles\demo`" --no-first-run --enable-features=WebMachineLearningNeuralNetwork --app=http://127.0.0.1:8780/demo.html" }
# Close the window when done, then: Stop-Process -Id $server.Id
```

The Intel GPU rows in the table at the top came from this page's check with Edge started with Chromium's switch
`--webnn-ort-ep-device=<EP name>,<vendor id>,<device id>`; with it, WebNN picks that one device whatever `deviceType`
the page asks for, with ONNX Runtime's CPU fallback still on, and if no registered device matches, the session fails.
`0x8086,0xb0a0` are the vendor and device ids of this laptop's Intel(R) Graphics: Edge's log, written with
`--enable-logging --v=0 --webnn-ort-logging-level=INFO`, lists every device it can use as `Registered OrtEpDevice` and
the one a page used as `Selected OrtEpDevice #0`. The page's `power` argument (`&power=low-power`) passes WebNN's
`powerPreference`. The block below runs those checks between two checks on the RTX, as when the rows were timed.
Run as written on 2026-11-09, its seven checks scored 311/311 on the same devices (Edge's log), and their 101-token
times were within 5 % of the rows.

```powershell
# After the demo block (close its app window first), in the same window on the laptop's own desktop. One headless check
# each: the RTX before, the Intel GPU through the OpenVINO EP and DirectML (fp16, int8 weights), powerPreference
# 'low-power' with no switch, the RTX after. Each check's device lines from Edge's log go to <tag>.selected.txt.
$edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
$py = ".venv-build\Scripts\python.exe"
if (-not $server -or $server.HasExited) { $server = Start-Process $py -ArgumentList "web\serve.py --root webroot --results webresults --port 8780" -WindowStyle Hidden -PassThru; Start-Sleep 5 }
$runs = if ($server.HasExited) { Write-Host "Port 8780 is taken (Get-NetTCPConnection -LocalPort 8780 -State Listen): stop that program, then run this block again."; @() } else {
  "demo_gpu_h16_before::h16", "demo_igpu_ov_h16:OpenVINOExecutionProvider:h16", "demo_igpu_dml_h16:DmlExecutionProvider:h16", "demo_igpu_ov_w8:OpenVINOExecutionProvider:w8",
  "demo_igpu_dml_w8:DmlExecutionProvider:w8", "demo_lowpower_gpu_h16:low-power:h16", "demo_gpu_h16_after::h16" }
foreach ($run in $runs) {
  $tag, $how, $form = $run.Split(":")
  $switch = if ($how -like "*ExecutionProvider") { "--webnn-ort-ep-device=$how,0x8086,0xb0a0" } else { "" }
  $power = if ($how -eq "low-power") { "&power=low-power" } else { "" }
  Remove-Item "webresults\$tag.*", "$tag.*" -ErrorAction SilentlyContinue
  Remove-Item "profiles\$tag" -Recurse -Force -ErrorAction SilentlyContinue
  $browser = Start-Process $edge -ArgumentList "--user-data-dir=`"$PWD\profiles\$tag`" --no-first-run --headless=new --enable-features=WebMachineLearningNeuralNetwork --enable-logging --v=0 --webnn-ort-logging-level=INFO $switch `"http://127.0.0.1:8780/demo.html?check=parity&device=gpu&form_gpu=$form$power&tag=$tag`"" -PassThru
  $t0 = Get-Date
  while (-not (Test-Path "webresults\$tag.done") -and ((Get-Date) - $t0).TotalMinutes -lt 20) { Start-Sleep 5 }
  Get-CimInstance Win32_Process -Filter "Name = 'msedge.exe'" | Where-Object { $_.CommandLine -like "*profiles\$tag*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Get-ChildItem "profiles\$tag" -Recurse -Filter chrome_debug.log | Select-String -Pattern "Selected OrtEpDevice" | ForEach-Object { $_.Line } | Out-File "$tag.selected.txt" -Encoding utf8
  if (-not (Test-Path "webresults\$tag.done")) { Write-Host "No result for $tag after 20 minutes; the page's log is webresults\$tag.log"; continue }
  # The device WebNN used: the EP and the hardware of "Selected OrtEpDevice #0"
  Select-String -Path "$tag.selected.txt" -Pattern "Selected OrtEpDevice #0" | ForEach-Object { $l = $_.Line; $i = [Math]::Max(0, $l.IndexOf("OrtHardwareDevice")); "$tag $([regex]::Match($l, 'ep_name: \w+').Value), $($l.Substring($i, [Math]::Min(170, $l.Length - $i)))" }
  & $py web\check_demo.py --result "webresults\$tag.json" --d1 d1-on-windows --upstream upstream --head ir-128\lm_head.npy --reference reference_probs.json --ids ids-128.json --out "$tag.jsonl"
}
# Then: Stop-Process -Id $server.Id
```

## What did not work

```
# GenAI 0.19.0's builder folders do not load on Windows ML as `windowsml[with-ort]` installs it (ONNX Runtime 1.28.3):
# cpu-fp32 on the OpenVINO EP (GPU) and the WebGPU EP, cpu-int8 on the OpenVINO EP (CPU) and the CPU EP, all with this error
RuntimeError: Load model from ...\cpu-int8\model.onnx failed:This is an invalid model. In Node, ("/model/layers.0/conv/CausalConvWithState", CausalConvWithState, "com.microsoft", -1) : ("/model/layers.0/conv/Mul_1/output_0": tensor(float),"model.layers.0.conv.conv.weight": tensor(float),"","past.0.conv": tensor(float),) -> ("/model/layers.0/conv/CausalConvWithState/output_0": tensor(float),"/model/layers.0/conv/CausalConvWithState/output_1": tensor(float),) , Error Unrecognized attribute: state_window for operator CausalConvWithState

# The static graph with int8 weights (w8) on the Intel GPU through Windows ML's OpenVINO EP; the fp16 graph (h16) works
onnxruntime.capi.onnxruntime_pybind11_state.EPFail: [ONNXRuntimeError] : 11 : EP_FAIL : Exception from src\inference\src\cpp\core.cpp:120:
Exception from src\inference\src\dev\plugin.cpp:54:
Check 'false' failed at src\plugins\intel_gpu\src\plugin\program_builder.cpp:168:
[GPU] ProgramBuilder build failed!
Exception from src\plugins\intel_gpu\src\graph\impls\onednn\utils.cpp:426:
[GPU] Unexpected layout format f16:b_fs_yx_fsv16:1x2048x128:nopad

# Windows ML's OpenVINO EP DLL with ONNX Runtime 1.31.1 from PyPI (NPU, GPU and CPU alike); ONNX Runtime then runs on the CPU EP
[E:onnxruntime:Default, ov_factory.cc:85 OpenVINOEpPluginFactory::CreateEp] Experimental function OrtEpApi_ReleaseEpContextConfig_SinceV28 is not available in this build

# The cuda-fp16 folder in the next entry is d1-on-windows' (its builder line `-p fp16 -e cuda`); the blocks here do
# not build it.
# Upstream's example with -e follow_config on d1-on-windows' cuda-fp16 folder: the Windows ML build of GenAI has no CUDA EP
RuntimeError: CUDA execution provider is not enabled in this build.
```

The first one comes from GenAI's builder: since 0.16.0 it writes the `state_window` attribute also when it is 0, and
ONNX Runtime knows the attribute from 1.29.0. `state_window_repro.py` shows it with one node of each op;
`strip_state_window.py` removes the attributes (all 0) from a model folder. With ONNX Runtime 1.30.0 (block 4) the
folders load as they are. Reported as
[microsoft/onnxruntime-genai#2902](https://github.com/microsoft/onnxruntime-genai/issues/2902).

The second is the same error as the static graph on the Intel GPU through OpenVINO Runtime in d1-on-windows; there it
also hit the fp16 graph. The third is ONNX Runtime 1.31 removing an experimental API that this DLL (plugin
1.7.0+72b3270.ci.72) looks up (microsoft/onnxruntime#32265; on the plugin's side onnxruntime/onnxruntime-ep-openvino#4,
fixed in 1.8.1); the same DLL works with ONNX Runtime 1.25.2, 1.28.3 and 1.30.0, and the OpenVINO EP from PyPI
(`onnxruntime-ep-openvino` 1.8.1) works with 1.31.1.

On the CPU, both OpenVINO EP builds (Windows ML's 1.7.0 and PyPI's 1.8.1) keep memory that grows with the longest
prompt seen so far and do not give it back when the generator is dropped. With LiquidAI/LFM2-350M int8 and one prompt
each of 256 to 4,096 tokens, the process holds 1.9 to 9.5 GiB after each prompt, against 1.1 to 2.0 GiB with the CPU
EP. In one run of the questions on the int8 folder with Windows ML's EP, the process reached 33.2 GiB
at the 413th prompt (3,446 tokens), where it was stopped: the laptop had no free memory left. One prompt of 1,491 tokens
leaves the process at 17.3 GiB, with Windows ML's EP (1.7.0) and with the EP from PyPI (1.8.1, d1-on-windows' CPU row)
alike. The CPU EP of the same ONNX Runtime answered all 415 questions, with a sampled peak of 6.1 GiB. Setting the CPU
plugin's runtime cache to 0 (`CPU_RUNTIME_CACHE_CAPACITY`) moves the LFM2-350M numbers by less than 0.03 GiB and leaves
d1-3B's growth during the first long prompt unchanged, although the option does reach the plugin (a bad value is
rejected by name). Block 4 therefore runs the
OpenVINO CPU answers on the first 392 questions (prompts of up to 378 tokens).

```
# Edge 156 started from a session with no desktop (ssh, session 0): the GPU process dies and restarts three times, then
# WebNN is off for gpu and for the NPU's fp16 graph. Edge in a signed-in session runs all five pairs (block 5 as
# written, 2026-11-08, by then Edge 157).
UnknownError: Failed to execute 'createContext' on 'ML': WebNN service connection error.
UnknownError: Failed to execute 'createContext' on 'ML': WebNN is disabled due to some unresolvable issues.
```

In Chrome 159, `deviceType` gpu is the WebGPU EP 0.5.0. With the fp16 graph one answer changes and is not a near tie
(`semif_0aa24a6701ac9c56c4a0/answer`: reference B 0.4885, insufficient 0.4406; this row insufficient 0.4698, B 0.4537).
The same fp16 graph gives 311/311 on DirectML and OpenVINO.

WebNN's `cpu` device is the OpenVINO EP here. It runs the int8-weight graph, but one answer changes and is not a near
tie (`tv4_027/answer`: reference b 0.4116, d 0.3839; this row d 0.4016, b 0.3969), in all three browsers. The same graph
gives 311/311 on the CPU EP, the NPU and the RTX.

## License

MIT for the code here. This repository holds no weights: d1-3B is under the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/main/LICENSE).
