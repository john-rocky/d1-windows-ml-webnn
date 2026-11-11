# Run it

The sections named in the text are on these pages: [Results](results.md), [What you need and Run it](run-it.md), [Demo page](demo-page.md), [What did not work](what-did-not-work.md).

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
# The int8-weight graph on the Intel GPU: its 22 1-D convolutions rewritten as 2-D convolutions (the form ONNX Runtime
# Web's WebNN EP gives them); the graph as exported does not compile there (see "What did not work")
& $py conv1d_to_2d.py --model x\w8-128\model.onnx --out x\w8-128\model_c2d.onnx
& $py run_static.py --upstream upstream --model x\w8-128\model_c2d.onnx --ep OpenVINOExecutionProvider --hw GPU --winml --head ir-128\lm_head.npy --ids ids-128.json --out wml_ov_gpu_w8c2d.jsonl
& $py d1-on-windows\score.py wml_ov_gpu_w8c2d.jsonl reference_probs.json
& $py run_static.py --upstream upstream --model x\w8-128\model_c2d.onnx --ep OpenVINOExecutionProvider --hw GPU --winml --head ir-128\lm_head.npy --ids speed-ids-128.json --passes 2 --out speed_wml_ov_gpu_w8c2d.jsonl
& $py d1-on-windows\speed_lines.py speed_wml_ov_gpu_w8c2d.jsonl
```

`winml_probe.py` lists each EP of the catalog with its state before and after `ensure_ready()`, its package and
version, and the devices ONNX Runtime then offers. `run_static.py` registers the catalog's EPs (`--winml`) and runs the
graph on one device of one EP; `--hw` picks the device type when the EP has several.
`conv1d_to_2d.py` writes `model_c2d.onnx` next to `model.onnx` and leaves the weight files as they are: a 1-D
convolution's weight has the same bytes as its 2-D form, so only `model.onnx` changes: each of those convolutions gets
a reshape before and after it, and its weight a fourth dimension of 1.

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
