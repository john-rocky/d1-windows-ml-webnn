"""Turn a runner.mjs result (serve.py's <tag>.json) into the lines of run_genai.py, so d1-on-windows/score.py scores it and
d1-on-windows/speed_lines.py times it. The page returns the hidden state at the last token; the options' rows of the head and
upstream's read-out run here (d1-on-windows/npu/head.py), and their time is measured here and added to each row's ms.

    python post_web.py --upstream <dir> --result <tag>.json --ids npu_ids_128.json --speed_ids npu_speed_ids.json \
        --head lm_head.npy --out <tag>.jsonl [--speed_out speed_<tag>.jsonl]
"""

import argparse
import base64
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # run_genai.py
sys.path.insert(0, os.path.join(HERE, "..", "d1-on-windows", "npu"))  # head.py of d1-on-windows (README step 1)
sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--speed_ids", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--speed_out", default="")
    args = parser.parse_args()

    from run_genai import load_upstream, probs_of
    from head import load_head, read_out

    up = load_upstream(args.upstream)
    head = load_head(args.head)
    result = json.load(open(args.result, encoding="utf-8"))
    meta = {"_meta": True, "model": result.get("model"), "ep": f"WebNN {result.get('device')} via ONNX Runtime Web {result.get('ort', {}).get('common', '')}",
            "ep_options": {"deviceType": result.get("device"), "powerPreference": result.get("power"), "graphOptimizationLevel": result.get("opt")},
            "load_s": (result.get("session_ms") or 0) / 1000, "userAgent": result.get("userAgent"), "source": args.result}
    for key in ("error", "context_error", "session_error", "run_error", "fatal"):
        if result.get(key):
            meta[key] = result[key]
    by_q = {}
    for path in (args.ids, args.speed_ids):
        for row in json.load(open(path, encoding="utf-8")):
            by_q[row["q"]] = row
    readout_ms = []

    def lines(rows):
        for row in rows:
            ref = by_q[row["q"]]
            hidden = np.frombuffer(base64.b64decode(row["hidden"]), dtype=np.float32)
            t = time.perf_counter()
            probs = read_out(up, hidden, ref["group"], head)
            answer = probs_of(up.answer(ref["question"], probs))
            extra = (time.perf_counter() - t) * 1000
            readout_ms.append(extra)
            yield {"q": row["q"], "ms": row["ms"] + extra, "page_ms": row["ms"], "tokens": row["tokens"], "probs": answer}

    for rows, path in ((result.get("parity") or [], args.out), (result.get("speed") or [], args.speed_out)):
        if not path:
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for line in lines(rows):
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    if readout_ms:
        print(f"read-out ms median {np.median(readout_ms):.3f} max {max(readout_ms):.3f} over {len(readout_ms)} rows")
    print(json.dumps({k: v for k, v in meta.items() if k != "_meta"}, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
