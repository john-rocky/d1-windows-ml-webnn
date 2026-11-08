"""Windows ML from Python, step by step, with the time of each step and the full text of every failure.

--mode catalog: open the Windows ML EP catalog (the `windowsml` package), list every provider with its version, package and
ready state, time ensure_ready() twice per provider (the first call in this process, then a second one), register each
provider's library with the ONNX Runtime of this venv, then run a one-op MatMul on every EP device it exposes.
--mode dll: skip the catalog and register one EP library by path (to compare one EP DLL across ONNX Runtime builds).

    python winml_probe.py --mode catalog --out probe.json
    python winml_probe.py --mode dll --dll <path to onnxruntime_providers_openvino_plugin.dll> --name openvino_ep --out probe.json
"""
import argparse
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

sys.stdout.reconfigure(encoding="utf-8")


def say(**fields):
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def one_op_model():
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor("w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten().tolist())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = os.path.join(tempfile.mkdtemp(), "matmul.onnx")
    onnx.save(model, path)
    return path


def describe(device):
    hw = device.device
    row = {"ep": device.ep_name, "ep_vendor": getattr(device, "ep_vendor", None),
           "type": str(getattr(hw, "type", "")), "vendor": getattr(hw, "vendor", None),
           "vendor_id": hex(getattr(hw, "vendor_id", 0) or 0), "device_id": hex(getattr(hw, "device_id", 0) or 0)}
    for name in ("ep_metadata", "ep_options"):
        try:
            row[name] = dict(getattr(device, name))
        except Exception:
            pass
    try:
        row["hw_metadata"] = dict(hw.metadata)
    except Exception:
        pass
    return row


def run_devices(model, summary, only=None):
    devices = ort.get_ep_devices()
    summary["devices"] = [describe(d) for d in devices]
    say(step="devices", n=len(devices), devices=summary["devices"])
    results = []
    for device in devices:
        label = describe(device)
        if only and device.ep_name not in only:
            continue
        if device.ep_name == "CPUExecutionProvider" and only is None:
            pass
        row = {"ep": device.ep_name, "type": label["type"], "ov_device": label.get("ep_metadata", {}).get("ov_device"),
               "vendor_id": label["vendor_id"], "device_id": label["device_id"]}
        t = time.perf_counter()
        try:
            options = ort.SessionOptions()
            options.add_provider_for_devices([device], {})
            session = ort.InferenceSession(model, sess_options=options)
            row["session_s"] = round(time.perf_counter() - t, 3)
            out = session.run(None, {"x": np.arange(4, dtype=np.float32).reshape(1, 4)})[0]
            row["providers"] = session.get_providers()
            row["result"] = out.tolist()
            row["ok"] = bool(np.allclose(out, np.arange(4, dtype=np.float32).reshape(1, 4)))
        except Exception as error:
            row["session_s"] = round(time.perf_counter() - t, 3)
            row["ok"] = False
            row["error"] = str(error)
        results.append(row)
        say(step="one_op", **row)
    summary["one_op"] = results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["catalog", "dll"], default="catalog")
    parser.add_argument("--dll", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no_ensure", action="store_true", help="list and register only, no ensure_ready()")
    args = parser.parse_args()
    if args.mode == "dll" and not (args.dll and args.name):
        parser.error("--mode dll needs --dll <path> and --name <EP name>")
    if args.mode == "catalog" and (args.dll or args.name):
        parser.error("--dll and --name are for --mode dll; --mode catalog registers every EP of the catalog")

    summary = {"ort": ort.__version__, "ort_file": ort.__file__, "python": sys.version.split()[0], "mode": args.mode,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    say(step="start", **summary)
    model = one_op_model()
    try:
        if args.mode == "dll":
            t = time.perf_counter()
            try:
                ort.register_execution_provider_library(args.name, args.dll)
                summary["register"] = {"name": args.name, "dll": args.dll, "ok": True, "s": round(time.perf_counter() - t, 3)}
            except Exception as error:
                summary["register"] = {"name": args.name, "dll": args.dll, "ok": False, "error": str(error)}
            say(step="register", **summary["register"])
            run_devices(model, summary)
        else:
            import windowsml
            from windowsml import EpCatalog

            summary["windowsml"] = getattr(windowsml, "__version__", None)
            t0 = time.perf_counter()
            with EpCatalog() as catalog:
                summary["catalog_open_s"] = round(time.perf_counter() - t0, 3)
                t = time.perf_counter()
                providers = catalog.find_all_providers()
                summary["find_s"] = round(time.perf_counter() - t, 3)
                eps = []
                for ep in providers:
                    row = {}
                    for name in ("name", "version", "package_family_name", "library_path", "package_root_path"):
                        try:
                            row[name] = getattr(ep, name)
                        except Exception as error:
                            row[name] = f"<error {error}>"
                    for name in ("ready_state", "certification"):
                        try:
                            value = getattr(ep, name)
                            row[name] = getattr(value, "name", str(value))
                        except Exception as error:
                            row[name] = f"<error {error}>"
                    if not args.no_ensure:
                        for k in (1, 2):
                            t = time.perf_counter()
                            try:
                                result = ep.ensure_ready()
                                row[f"ensure_{k}_s"] = round(time.perf_counter() - t, 3)
                                row[f"ensure_{k}_result"] = str(result)
                            except Exception as error:
                                row[f"ensure_{k}_s"] = round(time.perf_counter() - t, 3)
                                row[f"ensure_{k}_error"] = f"{type(error).__name__}: {error}"
                        try:
                            row["ready_state_after"] = getattr(ep.ready_state, "name", str(ep.ready_state))
                            row["library_path_after"] = ep.library_path
                        except Exception as error:
                            row["after_error"] = str(error)
                    eps.append(row)
                    say(step="provider", **row)
                summary["providers"] = eps
                for row in eps:
                    path = row.get("library_path_after") or row.get("library_path")
                    if not path or not isinstance(path, str) or not os.path.exists(path):
                        row["register"] = "skipped: no library file"
                        continue
                    t = time.perf_counter()
                    try:
                        ort.register_execution_provider_library(row["name"], path)
                        row["register"] = "ok"
                        row["register_s"] = round(time.perf_counter() - t, 3)
                    except Exception as error:
                        row["register"] = f"FAILED: {error}"
                    say(step="register", name=row["name"], result=row["register"])
                run_devices(model, summary)
    except Exception:
        summary["fatal"] = traceback.format_exc()
        say(step="fatal", text=summary["fatal"])
    summary["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    say(step="end", out=args.out)
    failed = summary.get("fatal") or (args.mode == "dll" and not summary["register"]["ok"])
    if failed:
        sys.exit("failed: " + (summary["register"]["error"] if args.mode == "dll" and not summary.get("fatal") else "see fatal in " + args.out))


if __name__ == "__main__":
    main()
