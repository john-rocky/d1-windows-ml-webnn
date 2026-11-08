"""d1-3B's text forward as an ONNX graph with one static prompt length, for WebNN (ONNX Runtime Web) and the Windows ML EPs.

The wrapper is the one of d1-on-windows/npu/export_ir.py: input_ids [1, N] right-padded, `last` = the index of the last real token,
plain causal attention, no cache, the head left out (the graph returns the hidden state at `last`; d1-on-windows/npu/head.py reads
the options' rows of the head). Inputs are int32: a WebNN backend may not take int64.
Also writes probe.npz (a random prompt, its `last` and the PyTorch hidden state) for check_onnx.py.

    python export_onnx.py --model <d1-3B folder> --length 128 --out <dir> [--dtype float16] [--fold]

--dtype float16 halves the RAM of the export (the machine of this run has 32 GB). Without --fold the exporter leaves each Linear
as MatMul(x, Transpose(W)) on the original tensor instead of writing a transposed copy of every weight; make_forms.py
transposes one tensor at a time while it writes the forms.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModel



class TextForward(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language = model.model.language_model

    def forward(self, input_ids, last):
        hidden = self.language.run(self.language.embed_tokens(input_ids))
        return hidden[0, last]


def plain_attention(length):
    """Causal group-query attention as explicit products (the same as export_ir.py)."""
    mask = torch.full((length, length), -1e4).triu(1)  # becomes the scores' dtype in the sum (fp16 holds -1e4)

    def attention(q, k, v, tree, window, scale):
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = (q @ k.transpose(-1, -2)) * scale + mask.to(q.dtype)
        return torch.softmax(scores, dim=-1) @ v

    return attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamo", action="store_true", help="the torch.export-based exporter instead of the TorchScript one")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--fold", action="store_true", help="constant folding in the exporter (copies every weight)")
    args = parser.parse_args()
    if not os.path.isfile(os.path.join(args.model, "config.json")):
        sys.exit(f"no config.json in --model {args.model} (the local folder of LiquidAI/d1-3B)")

    t = time.perf_counter()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=getattr(torch, args.dtype)).eval()
    print(f"loaded in {time.perf_counter() - t:.1f} s", flush=True)
    module = sys.modules[type(model.model.language_model.layers[2].self_attn).__module__]
    module.attention = plain_attention(args.length)
    if getattr(model.model, "vision_tower", None) is not None:
        model.model.vision_tower = None  # not part of the text forward; frees its memory before the export
    os.makedirs(args.out, exist_ok=True)
    wrapped = TextForward(model).eval()

    probe = torch.randint(1000, 5000, (1, args.length), dtype=torch.int32)
    last = torch.tensor(args.length // 2, dtype=torch.int32)
    padded = probe.clone()
    padded[0, args.length // 2 + 1:] = 124893
    with torch.no_grad():
        a = wrapped(probe, last).numpy()
        b = wrapped(padded, last).numpy()
    print("guard: a pad after the last token moves the hidden state by at most", float(np.abs(a - b).max()), flush=True)
    np.savez(os.path.join(args.out, "probe.npz"), input_ids=padded.numpy(), last=last.numpy(), hidden=b.astype(np.float32))

    path = os.path.join(args.out, "model.onnx")
    ids = torch.zeros((1, args.length), dtype=torch.int32)
    t = time.perf_counter()
    with torch.no_grad():
        if args.dynamo:
            program = torch.onnx.export(wrapped, (ids, last), dynamo=True, input_names=["input_ids", "last"],
                                        output_names=["hidden"], opset_version=args.opset)
            program.save(path, external_data=True)
        else:
            torch.onnx.export(wrapped, (ids, last), path, dynamo=False, input_names=["input_ids", "last"],
                              output_names=["hidden"], opset_version=args.opset, do_constant_folding=args.fold)
    print(f"exported in {time.perf_counter() - t:.1f} s (dynamo={args.dynamo})", flush=True)
    import onnx

    m = onnx.load(path, load_external_data=False)
    ops = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print("inputs", [(i.name, i.type.tensor_type.elem_type, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("outputs", [(o.name, o.type.tensor_type.elem_type, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
    print("ops", dict(sorted(ops.items(), key=lambda kv: -kv[1])), flush=True)
    print("files", {n: os.path.getsize(os.path.join(args.out, n)) for n in os.listdir(args.out)})


if __name__ == "__main__":
    main()
