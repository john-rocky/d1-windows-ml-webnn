"""Answer d1-3B questions on a static ONNX graph (web/export_onnx.py, make_forms.py, or d1-on-windows/npu/wrap_ir.py's EPContext
model) through ONNX Runtime with one execution provider device, the EPs coming from the Windows ML catalog (--winml) or
from a DLL path (--ep_dll). Output lines are run_genai.py's, so d1-on-windows/score.py and speed_lines.py read them.

    python run_static.py --upstream <dir> --model <dir>/model.onnx --ep OpenVINOExecutionProvider --hw NPU --winml \
        --ids ids.json --out out.jsonl [--passes 2] [--head lm_head.npy] [--opt all] [--ep_option k=v] [--vendor 0x8086]
"""

import argparse
import contextlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # run_genai.py
sys.path.insert(0, os.path.join(HERE, "d1-on-windows", "npu"))  # head.py of d1-on-windows (README step 1)
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnxruntime as ort

LEVELS = {"disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
          "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep", required=True, help="EP name as ort.get_ep_devices() reports it")
    parser.add_argument("--hw", default="", help="CPU, GPU or NPU: the hardware device type to pick")
    parser.add_argument("--vendor", default="", help="hardware vendor id, e.g. 0x10de (NVIDIA) or 0x8086 (Intel)")
    parser.add_argument("--winml", action="store_true", help="register the EPs of the Windows ML catalog (windowsml)")
    parser.add_argument("--ep_dll", default="")
    parser.add_argument("--ep_option", action="append", default=[])
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--pad", type=int, default=124893)
    parser.add_argument("--head", default="")
    parser.add_argument("--opt", default="all", choices=sorted(LEVELS))
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    from run_genai import load_upstream, probs_of
    from head import load_head, read_out

    up = load_upstream(args.upstream)
    head = load_head(args.head)
    stack = contextlib.ExitStack()
    reg = {}
    if args.winml:
        from windowsml import EpCatalog

        catalog = stack.enter_context(EpCatalog())
        for provider in catalog.find_all_providers():
            t = time.perf_counter()
            provider.ensure_ready()
            path = provider.library_path
            if path and os.path.exists(path):
                ort.register_execution_provider_library(provider.name, path)
                reg[provider.name] = {"version": provider.version, "library": path, "ensure_s": round(time.perf_counter() - t, 3)}
    elif args.ep_dll:
        ort.register_execution_provider_library(args.ep + "_lib", args.ep_dll)
        reg[args.ep] = {"library": args.ep_dll}
    devices = [d for d in ort.get_ep_devices() if d.ep_name == args.ep]
    if args.hw:
        devices = [d for d in devices if str(d.device.type).upper().endswith(args.hw.upper())]
    if args.vendor:
        devices = [d for d in devices if hex(d.device.vendor_id) == args.vendor.lower()]
    if not devices:
        listing = [(d.ep_name, str(d.device.type), hex(d.device.vendor_id), dict(d.ep_metadata)) for d in ort.get_ep_devices()]
        raise SystemExit(f"no EP device {args.ep} {args.hw} {args.vendor}; devices: {listing}")
    device = devices[0]
    options = ort.SessionOptions()
    options.graph_optimization_level = LEVELS[args.opt]
    options.add_provider_for_devices([device], dict(item.split("=", 1) for item in args.ep_option))
    t = time.perf_counter()
    session = ort.InferenceSession(args.model, sess_options=options)
    load_s = time.perf_counter() - t
    inputs = {i.name: i for i in session.get_inputs()}
    length = inputs["input_ids"].shape[1]
    id_type = np.int32 if "int32" in inputs["input_ids"].type else np.int64
    last_type = np.int32 if "int32" in inputs["last"].type else np.int64
    meta = {"_meta": True, "model": args.model, "ep": f"{args.ep} {args.hw} static {length} (ORT {ort.__version__})",
            "ep_options": dict(item.split("=", 1) for item in args.ep_option), "opt": args.opt, "load_s": load_s,
            "ort": ort.__version__, "providers": session.get_providers(), "registered": reg,
            "device": {"type": str(device.device.type), "vendor_id": hex(device.device.vendor_id), "device_id": hex(device.device.device_id),
                       "vendor": device.device.vendor, "ep_metadata": dict(device.ep_metadata)}}
    print(f"load {load_s:.1f} s on {meta['device']}, providers {session.get_providers()}, length {length}", flush=True)
    with open(args.ids, encoding="utf-8") as f:
        rows = json.load(f)

    def forward(ids):
        padded = np.full((1, length), args.pad, dtype=id_type)
        padded[0, : len(ids)] = ids
        return session.run(None, {"input_ids": padded, "last": np.array(len(ids) - 1, dtype=last_type)})[0].reshape(-1)

    for _ in range(args.passes - 1):
        for row in rows:
            if "ids" in row and len(row["ids"]) <= length:
                forward(row["ids"])
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps(meta, ensure_ascii=False) + "\n")
        done = 0
        for row in rows:
            if "refused" in row:
                out.write(json.dumps({"q": row["q"], "refused": row["refused"]}, ensure_ascii=False) + "\n")
                continue
            if len(row["ids"]) > length:
                continue
            t = time.perf_counter()
            probs = read_out(up, forward(row["ids"]), row["group"], head)
            ms = (time.perf_counter() - t) * 1000
            out.write(json.dumps({"q": row["q"], "ms": ms, "tokens": len(row["ids"]),
                                  "probs": probs_of(up.answer(row["question"], probs))}, ensure_ascii=False) + "\n")
            out.flush()
            done += 1
            if args.max and done >= args.max:
                break
    print(f"done {done} questions", flush=True)
    stack.close()


if __name__ == "__main__":
    main()
