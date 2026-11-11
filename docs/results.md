# Results

The sections named in the text are on these pages: [Results](results.md), [What you need and Run it](run-it.md), [Demo page](demo-page.md), [What did not work](what-did-not-work.md).

[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B)'s decision step (415 questions with reference answers, and
prompts of 39, 101 and 467 tokens for the times) run through the two newest Microsoft paths for on-device models:

- **Windows ML**: ONNX Runtime from the `onnxruntime-windowsml` package, with execution providers (EPs) that Windows
  downloads and installs on request (`windowsml` on PyPI, its `EpCatalog`), driven by ONNX Runtime GenAI
  (`onnxruntime-genai-winml`) or by a plain ONNX Runtime session on a static graph.
- **WebNN** in Edge and Chrome: a web page with ONNX Runtime Web's WebNN EP, `deviceType` cpu / gpu / npu.

It is the same workload and the same scoring as [d1-on-windows](https://github.com/john-rocky/d1-on-windows), which
ran it through CUDA, WebGPU and OpenVINO on the same laptop; that repository's rows are the baseline here.

[Demo page](demo-page.md): `web/demo.html` answers your own state, questions and options in Edge through WebNN, with
its own prompt building, tokenizer and option head in JavaScript, and shows the milliseconds of each decision.

Machine of the run recorded here (2026-11-07; the last WebNN row and the second timing of three d1-on-windows rows
after midnight; the WebNN rows on the Intel GPU and the Windows ML row with 2-D convolutions on 2026-11-09, with the
Windows build and the drivers below read again that day): MSI Stealth 16 AI+ B3WI, Windows 11 Home 25H2 (26200.9711),
Core Ultra 9 386H, Intel(R) Graphics (driver 32.0.101.8724), Intel(R) NPU (driver 32.0.100.4841), NVIDIA GeForce RTX
5080 Laptop GPU (16 GB, driver 591.74), 32 GB RAM.

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
| Intel GPU | Windows ML, ORT 1.28.3 | OpenVINO 1.7.0 | static 128, int8 weights, 1-D convolutions as 2-D | 311/311 | 50.0 / 49.9 / — | ×0.46 / ×0.36 / — | — |
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

Each row's files are in [`results/`](../results). Windows ML rows: answers `<row>.jsonl`, score `<row>.score.json`, times
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

The Windows ML row with 2-D convolutions was timed on 2026-11-09 from 08:02 to 08:03, by the last five lines of block
3 below (from `conv1d_to_2d.py` on), run as written, between two runs of the static fp16 row above as a control:
100.2 ms before and 96.1 ms after for 101 tokens (the row: 102 ms; both under it, the second by 5.8 %). No
d1-on-windows row was timed with it, so its last column is empty.

Edge's rows marked "signed-in session" ran from the laptop's desktop. Its int8-weight cpu and npu rows ran from an ssh
session, where Edge's GPU process restarted twice and the page then answered (see "What did not work").

- RTX: GenAI on Windows ML's WebGPU EP, 117 ms for 101 tokens, is in the class of d1-on-windows' WebGPU row (113 ms
  there on 2026-11-05). GenAI's Windows ML build has no CUDA EP, so d1-on-windows' cuda-fp16 folder does not run on
  Windows ML (see "What did not work").
- Intel GPU: the static fp16 graph on Windows ML's OpenVINO EP, 102 ms (×0.58), runs where d1-on-windows' OpenVINO IR
  of the same wrapper (fp16 weights, 128 tokens) failed to compile in OpenVINO Runtime. GenAI with the same EP, 195 ms,
  is ×1.11 of the re-timing (the OpenVINO EP from PyPI). The int8-weight graph, as block 2 writes it, does not compile
  on this EP (see "What did not work"); with its 22 1-D convolutions rewritten as 2-D convolutions (`conv1d_to_2d.py`),
  it answers all 311 and takes 49.9 ms for 101 tokens (×0.36 of d1-on-windows' row).
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
  829 ms. The int8-weight graph runs on the OpenVINO EP here, where it fails on Windows ML's OpenVINO EP as block 2
  writes it (see "What did not work"). The difference that decides it is the form of the graph's 22 1-D convolutions:
  ONNX Runtime Web's WebNN EP hands each 1-D convolution to WebNN as a 2-D convolution between two reshapes, and the
  ONNX file with its convolutions rewritten that way compiles on Windows ML's OpenVINO EP too (see the Intel GPU bullet
  above). The error needs the 1-D form together with what follows the convolution in the int8-weight graph (the
  multiply and the int8-weight MatMul); the fp16 graph, with the same 1-D convolutions, compiles.
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
