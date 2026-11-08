"""Rewrite an exported static d1-3B ONNX graph (export_onnx.py) into the forms the browser and the Windows ML EPs load.
Tensors stream one at a time from the source's external data (memory maps), so the peak RAM stays near one tensor.

Every form folds MatMul(x, Transpose(W)) into MatMul(x, W^T) (export_onnx.py exports without constant folding) and makes
the graph output float32. Forms:
  h16:  the graph as exported (float16 when the export was float16).
  w8:   every MatMul weight becomes int8 with one symmetric scale per output channel, read back by
        DequantizeLinear(axis=1) in front of the MatMul; the embedding table stays float16 behind a Cast; everything
        else becomes float32 (a float16 export is converted: initializers, Cast targets, Constant values, I/O).
The tensors go into data_0.bin, data_1.bin, ... of at most --part_gib each (one ArrayBuffer per file in the browser),
with a manifest.json that lists them.

    python make_forms.py --src <dir>/model.onnx --form w8 --out <dir2>
"""

import argparse
import json
import os
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


def tensor_array(tensor, base):
    """The tensor's values as a numpy array, read from its external file through a memory map when it has one."""
    if tensor.data_location != TensorProto.EXTERNAL:
        return numpy_helper.to_array(tensor)
    info = {e.key: e.value for e in tensor.external_data}
    dtype = helper.tensor_dtype_to_np_dtype(tensor.data_type)
    count = int(np.prod(tensor.dims)) if len(tensor.dims) else 1
    data = np.memmap(os.path.join(base, info["location"]), dtype=dtype, mode="r", offset=int(info.get("offset", 0)), shape=(count,))
    return np.asarray(data).reshape(tuple(tensor.dims))


class Writer:
    """Appends tensors to data_<k>.bin files of bounded size and points each TensorProto at its bytes."""

    def __init__(self, out, part_bytes):
        self.out, self.part_bytes, self.k, self.offset = out, part_bytes, 0, 0
        self.file = open(os.path.join(out, "data_0.bin"), "wb")
        self.sizes = {}

    def put(self, name, array):
        array = np.ascontiguousarray(array)
        raw = array.tobytes()
        if self.offset and self.offset + len(raw) > self.part_bytes:
            self.file.close()
            self.sizes[f"data_{self.k}.bin"] = self.offset
            self.k, self.offset = self.k + 1, 0
            self.file = open(os.path.join(self.out, f"data_{self.k}.bin"), "wb")
        pad = (-self.offset) % 64
        if pad:
            self.file.write(b"\0" * pad)
            self.offset += pad
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = helper.np_dtype_to_tensor_dtype(array.dtype)
        tensor.dims.extend(array.shape)
        # The entries set_external_data would write (it refuses a tensor without raw_data, and the bytes go to the file).
        for key, value in (("location", f"data_{self.k}.bin"), ("offset", str(self.offset)), ("length", str(len(raw)))):
            entry = tensor.external_data.add()
            entry.key, entry.value = key, value
        tensor.data_location = TensorProto.EXTERNAL
        self.file.write(raw)
        self.offset += len(raw)
        return tensor

    def close(self):
        self.file.close()
        self.sizes[f"data_{self.k}.bin"] = self.offset
        return self.sizes


def fix_attrs(node, base, to_fp32):
    """Tensor attributes (Constant values) go inside model.onnx: the exporter writes the larger ones to their own files
    next to the source graph, which the new folder does not have. With to_fp32, float16 -> float32 in Cast targets and
    in those values."""
    for attr in node.attribute:
        if to_fp32 and node.op_type == "Cast" and attr.name == "to" and attr.i == F16:
            attr.i = F32
        if attr.type == onnx.AttributeProto.TENSOR and (attr.t.data_location == TensorProto.EXTERNAL or (to_fp32 and attr.t.data_type == F16)):
            array = tensor_array(attr.t, base)
            if to_fp32 and array.dtype == np.float16:
                array = array.astype(np.float32)
            attr.t.CopyFrom(numpy_helper.from_array(np.asarray(array), attr.t.name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--form", choices=["w8", "h16"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--part_gib", type=float, default=1.5)
    parser.add_argument("--min_elements", type=int, default=1024, help="smaller tensors stay inside model.onnx")
    args = parser.parse_args()

    t0 = time.perf_counter()
    base = os.path.dirname(os.path.abspath(args.src))
    model = onnx.load(args.src, load_external_data=False)
    graph = model.graph
    os.makedirs(args.out, exist_ok=True)
    writer = Writer(args.out, int(args.part_gib * 2**30))
    inits = {t.name: t for t in graph.initializer}
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    # MatMul(x, Transpose(W)) with W a 2-D initializer -> the transposed weight under the Transpose's output name.
    folded = {}
    for node in graph.node:
        if node.op_type == "Transpose" and node.input[0] in inits and len(inits[node.input[0]].dims) == 2:
            perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), [1, 0])
            if perm == [1, 0] and all(c.op_type == "MatMul" and c.input[1] == node.output[0] for c in consumers.get(node.output[0], [])):
                folded[node.output[0]] = (node, node.input[0])
    drop_nodes = {id(n) for n, _ in folded.values()}
    weight_names = set(folded) | {n.input[1] for n in graph.node if n.op_type == "MatMul" and n.input[1] in inits and len(inits[n.input[1]].dims) == 2}
    tables = {n.input[0] for n in graph.node if n.op_type == "Gather" and n.input[0] in inits and len(inits[n.input[0]].dims) == 2}
    still_used = {name for node in graph.node if id(node) not in drop_nodes for name in node.input}
    convert = args.form == "w8"
    new_inits, front, stats = [], [], {"weights": 0, "w8": 0, "tables": 0, "other": 0, "worst_rel_err": 0.0, "folded_transposes": len(folded)}

    def emit_weight(name, array):
        stats["weights"] += 1
        if args.form == "w8":
            w = array.astype(np.float32)
            scale = np.abs(w).max(axis=0) / 127.0
            scale[scale == 0] = 1.0
            q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
            stats["worst_rel_err"] = max(stats["worst_rel_err"], float(np.abs(q * scale - w).max() / (np.abs(w).max() + 1e-12)))
            new_inits.append(writer.put(name + "_q", q))
            new_inits.append(numpy_helper.from_array(scale.astype(np.float32), name + "_scale"))
            new_inits.append(numpy_helper.from_array(np.zeros(scale.shape, np.int8), name + "_zero"))
            front.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale", name + "_zero"], [name], name=name + "_dq", axis=1))
            stats["w8"] += 1
        else:
            new_inits.append(writer.put(name, array))

    for out_name, (node, src) in folded.items():
        emit_weight(out_name, tensor_array(inits[src], base).T)
    for tensor in graph.initializer:
        name = tensor.name
        if name not in still_used:
            continue  # only fed a folded Transpose
        array = tensor_array(tensor, base)
        if name in weight_names:
            emit_weight(name, array)
        elif name in tables and convert:
            new_inits.append(writer.put(name + "_f16", array.astype(np.float16)))
            front.append(helper.make_node("Cast", [name + "_f16"], [name], name=name + "_cast", to=F32))
            stats["tables"] += 1
        else:
            if convert and array.dtype == np.float16:
                array = array.astype(np.float32)
            new_inits.append(writer.put(name, array) if array.size >= args.min_elements else numpy_helper.from_array(np.asarray(array), name))
            stats["other"] += 1
        del array
    sizes = writer.close()
    nodes = [n for n in graph.node if id(n) not in drop_nodes]
    for node in nodes:
        fix_attrs(node, base, convert)
    if convert:
        for value in list(graph.input) + list(graph.output):
            if value.type.tensor_type.elem_type == F16:
                value.type.tensor_type.elem_type = F32
    # A float16 output gets a Cast so that the page and the Python runners always read float32.
    for value in graph.output:
        if value.type.tensor_type.elem_type == F16:
            producer = next(n for n in nodes if value.name in n.output)
            producer.output[list(producer.output).index(value.name)] = value.name + "_f16"
            nodes.append(helper.make_node("Cast", [value.name + "_f16"], [value.name], name=value.name + "_to_f32", to=F32))
            value.type.tensor_type.elem_type = F32
    graph.ClearField("value_info")
    graph.ClearField("initializer")
    graph.initializer.extend(new_inits)
    graph.ClearField("node")
    graph.node.extend(front + nodes)  # the new nodes read initializers only
    if args.form == "w8":
        opset = next(o for o in model.opset_import if o.domain in ("", "ai.onnx"))
        opset.version = max(opset.version, 13)  # DequantizeLinear with axis
    path = os.path.join(args.out, "model.onnx")
    onnx.save(model, path)
    onnx.checker.check_model(path)
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    manifest = {"form": args.form, "src": args.src, "files": sizes, "stats": stats, "ops": ops,
                "model_onnx_bytes": os.path.getsize(path), "seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
