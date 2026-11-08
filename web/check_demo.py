"""Score a demo.html check (serve.py's <tag>.json from demo.html?check=parity): the page's own answers become the lines of
d1-on-windows' run_genai.py, so d1-on-windows/score.py scores them against the reference; the hidden states the page
returned go through d1-on-windows' Python head (npu/head.py) and upstream's read-out, and the two read-outs are compared.

    python check_demo.py --result <tag>.json --d1 <d1-on-windows folder> --upstream <folder with d1-decisions.py> \
        --head lm_head.npy --reference reference_probs.json --out <tag>.jsonl
"""

import argparse
import base64
import json
import os
import subprocess
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--d1", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--ids", default="", help="npu_ids_128.json: the questions' read-out groups for the Python head")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.d1)
    sys.path.insert(0, os.path.join(args.d1, "npu"))
    from run_genai import load_upstream, probs_of  # d1-on-windows
    from head import load_head, read_out  # d1-on-windows/npu

    result = json.load(open(args.result, encoding="utf-8"))
    device = (result.get("devices") or ["?"])[0]
    session = (result.get("sessions") or {}).get(device, {})
    meta = {"_meta": True, "source": args.result, "page": "demo.html", "ep": f"WebNN {device} via ONNX Runtime Web {(result.get('ort') or {}).get('web', '')}",
            "form": session.get("form"), "power": result.get("power"), "load_s": session.get("load_s"), "userAgent": result.get("userAgent")}
    for key in ("error", "fatal"):
        if result.get(key):
            meta[key] = result[key]
    if session.get("error"):
        meta["session_error"] = session["error"]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for row in result.get("answers") or []:
            f.write(json.dumps({"q": row["q"], "ms": row["ms"], "tokens": row["tokens"], "probs": row["probs"]}, ensure_ascii=False) + "\n")
    score = subprocess.run([sys.executable, os.path.join(args.d1, "score.py"), args.out, args.reference, "--json-out", args.out.replace(".jsonl", ".score.json")],
                           capture_output=True, text=True, encoding="utf-8")
    summary = json.loads(score.stdout) if score.returncode == 0 else {"score_error": score.stderr[-2000:]}

    # The page's read-out against the Python one, on the same hidden states.
    compare = {"rows": 0, "same_top": 0, "max_abs_dp": 0.0, "max_abs_dp_at": None}
    if args.ids and result.get("answers"):
        up = load_upstream(args.upstream)
        head = load_head(args.head)
        by_q = {row["q"]: row for row in json.load(open(args.ids, encoding="utf-8")) if row.get("ids")}
        for row in result["answers"]:
            ref = by_q.get(row["q"])
            if ref is None or "hidden" not in row:
                continue
            hidden = np.frombuffer(base64.b64decode(row["hidden"]), dtype=np.float32)
            py = probs_of(up.answer(ref["question"], read_out(up, hidden, ref["group"], head)))
            page = row["probs"]
            compare["rows"] += 1
            compare["same_top"] += max(py, key=py.get) == max(page, key=page.get)
            d = max(abs(py[k] - page[k]) for k in py)
            if d > compare["max_abs_dp"]:
                compare["max_abs_dp"], compare["max_abs_dp_at"] = d, row["q"]
    out = {"tag": result.get("tag"), "device": device, "form": session.get("form"), "power": result.get("power"), "load_s": session.get("load_s"),
           "ids": result.get("ids"), "same_top": summary.get("same_top"), "questions_scored": summary.get("questions_scored"),
           "flips": summary.get("flips"), "max_abs_dp_vs_reference": summary.get("max_abs_dp"), "page_vs_python_head": compare,
           "speed_median_ms": result.get("speed_median_ms"), "examples": result.get("examples"), "parity_s": result.get("parity_s"),
           "errors": {k: meta[k] for k in ("error", "fatal", "session_error") if k in meta} or None,
           "score_error": summary.get("score_error")}
    json.dump(out, open(args.out.replace(".jsonl", ".check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, ensure_ascii=False)[:3000])
    for x in out["examples"] or []:
        print(f"example {x['example']} / {x['name']}: {x['tokens']} tokens, median {x['median_ms']:.2f} ms (min {x['min_ms']:.2f}, max {x['max_ms']:.2f}), answer {json.dumps(x['answer'], ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    main()
