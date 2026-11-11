# Demo page

The sections named in the text are on these pages: [Results](results.md), [What you need and Run it](run-it.md), [Demo page](demo-page.md), [What did not work](what-did-not-work.md).

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
