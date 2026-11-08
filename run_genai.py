"""Run LiquidAI/d1-3B decisions through ONNX Runtime GenAI, one question per request, and time each one.

The decision step is upstream's: examples/python/d1-decisions.py (imported by path, not copied).
This file only loops over a fixture, times one prefill per question and writes one JSON line per question.

    python run_genai.py --upstream <dir with d1-decisions.py and common.py> -m <model dir> -e cpu \
        --requests requests.json --out results.jsonl [--ids id1,id2] [--ep_path path] [--max N] [--set_env K=V]
"""

import argparse
import importlib.util
import json
import os
import platform
import sys
import time


def load_upstream(folder):
    sys.path.insert(0, folder)  # d1-decisions.py's main() imports common from its own folder
    spec = importlib.util.spec_from_file_location("d1_decisions", os.path.join(folder, "d1-decisions.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probs_of(answer):
    """The option -> probability table of one upstream answer, under the reference's keys."""
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1.0 - answer["noul"]}
    return answer["probabilities"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-e", "--execution_provider", default="follow_config")
    parser.add_argument("--ep_path", default=None)
    parser.add_argument("--ep_option", action="append", default=[], help="key=value, repeatable")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids", default="", help="comma separated record ids; empty = all text records")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--passes", type=int, default=1, help="repeat the fixture; only the last pass is written")
    parser.add_argument("--openvino_device", default="", help="CPU, GPU or NPU: run through the pip OpenVINO plugin")
    parser.add_argument("--overlay", default="", help="JSON applied with og.Config.overlay, e.g. a thread count")
    parser.add_argument("--winml", default="", help="GenAI provider name (OpenVINO, NvTensorRtRtx, cpu): register the EPs of the "
                        "Windows ML catalog (the windowsml package) and run on this provider; --ep_option sets its options")
    args = parser.parse_args()

    up = load_upstream(args.upstream)
    import common  # upstream's common.py, from the folder load_upstream put on sys.path
    import onnxruntime_genai as og

    t0 = time.perf_counter()
    ep_options = dict(item.split("=", 1) for item in args.ep_option)
    winml_registered = {}
    if args.winml:
        # Windows ML the way upstream's CLI does it (--use_winml): common.register_ep -> examples/python/winml.py, which
        # makes every EP of the windowsml catalog ready, registers it with GenAI and closes the catalog. Then the
        # catalog is read once more for the record.
        from windowsml import EpCatalog

        t = time.perf_counter()
        if os.path.exists(os.path.join(args.upstream, "winml.py")):
            common.register_ep(args.winml, "", True)
            how = "upstream winml.py"
        else:
            how = "own code (no winml.py in the upstream folder)"
            with EpCatalog() as catalog:
                for provider in catalog.find_all_providers():
                    provider.ensure_ready()
                    if provider.library_path:
                        og.register_execution_provider_library(provider.name, provider.library_path)
        winml_registered = {"_how": how, "_register_s": round(time.perf_counter() - t, 3)}
        with EpCatalog() as catalog:
            for provider in catalog.find_all_providers():
                row = {"ready_state": str(provider.ready_state), "certification": str(provider.certification)}
                try:
                    row.update(version=provider.version, library=provider.library_path)
                except OSError as error:
                    row["error"] = str(error)
                winml_registered[provider.name] = row
        print("Windows ML EPs:", winml_registered, flush=True)
        config = og.Config(args.model_path)
        config.clear_providers()
        if args.winml != "cpu":
            config.append_provider(args.winml)
            for key, value in ep_options.items():
                config.set_provider_option(args.winml, key, value)
        args.execution_provider = f"Windows ML {args.winml}"
    elif args.openvino_device:
        import onnxruntime_ep_openvino as ov_ep

        og.register_execution_provider_library(ov_ep.get_ep_name(), ov_ep.get_library_path())
        config = og.Config(args.model_path)
        config.clear_providers()
        config.append_provider("OpenVINO")
        config.set_provider_option("OpenVINO", "device_type", args.openvino_device)
        for key, value in ep_options.items():
            config.set_provider_option("OpenVINO", key, value)
        args.execution_provider = f"OpenVINO {args.openvino_device}"
    else:
        common.register_ep(args.execution_provider, args.ep_path, False)
        config = common.get_config(args.model_path, args.execution_provider, args.ep_path, ep_options)
    if args.overlay:
        config.overlay(args.overlay)
    model = og.Model(config)
    load_s = time.perf_counter() - t0
    context = up.model_context_length(args.model_path)
    runner = up.D1Decisions(model, context)

    with open(args.requests, encoding="utf-8") as f:
        fixture = json.load(f)
    wanted = {i for i in args.ids.split(",") if i}
    for _ in range(args.passes - 1):  # warm up: the same questions, nothing written
        for record in fixture["records"]:
            request = record["request"]
            if "images" in request or (wanted and record["id"] not in wanted):
                continue
            for qid, question in request["questions"].items():
                try:
                    runner.decide({"state": request["state"], "questions": {qid: question}})
                except ValueError:
                    pass
    done = errors = 0
    with open(args.out, "w", encoding="utf-8") as out:
        header = {"_meta": True, "model": args.model_path, "ep": args.execution_provider, "ep_options": ep_options, "overlay": args.overlay,
                  "device_type": model.device_type, "model_type": model.type, "load_s": load_s, "og": og.__version__,
                  "python": platform.python_version(), "machine": platform.processor()}
        if winml_registered:
            header["winml_eps"] = winml_registered
        out.write(json.dumps(header) + "\n")
        for record in fixture["records"]:
            request = record["request"]
            if "images" in request or (wanted and record["id"] not in wanted):
                continue
            for qid, question in request["questions"].items():
                single = {"state": request["state"], "questions": {qid: question}}
                row = {"q": f"{record['id']}/{qid}"}
                t = time.perf_counter()
                try:
                    response = runner.decide(single)
                    row["ms"] = (time.perf_counter() - t) * 1000
                    row["tokens"] = response["usage"]["input_tokens"]
                    row["probs"] = probs_of(response["answers"][qid])
                except ValueError as error:  # upstream refuses the request before running the model
                    row["refused"] = str(error)
                except Exception as error:  # the runtime failed: keep the text, stop after a few
                    row["error"] = str(error)
                    errors += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                done += 1
                if args.max and done >= args.max:
                    print(f"stopped at --max {args.max}, errors {errors}")
                    return
                if errors >= 3:
                    print(f"stopped after {errors} runtime errors, last: {row['error'][:400]}")
                    sys.exit(3)
    print(f"done {done} questions, errors {errors}, load {load_s:.1f} s")


if __name__ == "__main__":
    main()
