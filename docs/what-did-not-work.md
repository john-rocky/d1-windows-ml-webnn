# What did not work

The sections named in the text are on these pages: [Results](results.md), [What you need and Run it](run-it.md), [Demo page](demo-page.md), [What did not work](what-did-not-work.md).

```
# GenAI 0.19.0's builder folders do not load on Windows ML as `windowsml[with-ort]` installs it (ONNX Runtime 1.28.3):
# cpu-fp32 on the OpenVINO EP (GPU) and the WebGPU EP, cpu-int8 on the OpenVINO EP (CPU) and the CPU EP, all with this error
RuntimeError: Load model from ...\cpu-int8\model.onnx failed:This is an invalid model. In Node, ("/model/layers.0/conv/CausalConvWithState", CausalConvWithState, "com.microsoft", -1) : ("/model/layers.0/conv/Mul_1/output_0": tensor(float),"model.layers.0.conv.conv.weight": tensor(float),"","past.0.conv": tensor(float),) -> ("/model/layers.0/conv/CausalConvWithState/output_0": tensor(float),"/model/layers.0/conv/CausalConvWithState/output_1": tensor(float),) , Error Unrecognized attribute: state_window for operator CausalConvWithState

# The static graph with int8 weights (w8) on the Intel GPU through Windows ML's OpenVINO EP, as exported; the fp16
# graph (h16) works, and so does w8 with its 1-D convolutions rewritten as 2-D (conv1d_to_2d.py, in block 3)
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
