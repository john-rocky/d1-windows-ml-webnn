"""Check an exported static d1-3B graph on the ONNX Runtime CPU EP against the PyTorch hidden state of export_onnx.py's probe.

    python check_onnx.py --model <dir>/model.onnx --probe <fp32 dir>/probe.npz
"""
import argparse
import sys
import time

import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--probe", required=True)
parser.add_argument("--opt", default="basic")
args = parser.parse_args()
probe = np.load(args.probe)
options = ort.SessionOptions()
options.graph_optimization_level = {"disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[args.opt]
t = time.perf_counter()
session = ort.InferenceSession(args.model, sess_options=options, providers=["CPUExecutionProvider"])
load = time.perf_counter() - t
types = {i.name: i.type for i in session.get_inputs()}
feeds = {"input_ids": probe["input_ids"].astype(np.int32 if "int32" in types["input_ids"] else np.int64),
         "last": np.array(probe["last"]).astype(np.int32 if "int32" in types["last"] else np.int64)}
t = time.perf_counter()
hidden = session.run(None, feeds)[0].reshape(-1).astype(np.float32)
run = time.perf_counter() - t
ref = probe["hidden"].reshape(-1)
cos = float(hidden @ ref / (np.linalg.norm(hidden) * np.linalg.norm(ref)))
print(f"{args.model}: ort {ort.__version__} load {load:.1f} s run {run * 1000:.0f} ms | max|diff| {np.abs(hidden - ref).max():.4g} "
      f"rel {np.abs(hidden - ref).max() / np.abs(ref).max():.4g} cos {cos:.6f} | inputs {types}")
