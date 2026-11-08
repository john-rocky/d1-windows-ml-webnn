"""Rewrite each 1-D Conv of an ONNX graph (rank-3 input and weight) as a 2-D Conv between two Reshapes, the form ONNX
Runtime Web's WebNN EP gives a 1-D Conv: the input [N, C, L] becomes [N, C, L, 1], the weight [M, C/group, k] becomes
[M, C/group, k, 1] (kernel [k, 1], pads [b, 0, e, 0], strides [s, 1], dilations [d, 1]), and the output goes back to
[N, M, L']. Nothing else in the graph changes.

The int8-weight static graph (x\\w8-128) fails to compile on the Intel GPU through OpenVINO with
"[GPU] Unexpected layout format f16:b_fs_yx_fsv16:1x2048x128:nopad"; the rewritten graph compiles and answers.

The graph is read without its weight files. A conv weight stored in those files keeps its bytes (row-major
[M, C/group, k] and [M, C/group, k, 1] are the same bytes) and only its shape in model.onnx changes, so write the new
model.onnx into the folder that holds the original's weight files:

    python conv1d_to_2d.py --model x\\w8-128\\model.onnx --out x\\w8-128\\model_c2d.onnx
"""

import argparse
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite(model):
    inits = {t.name: t for t in model.graph.initializer}
    uses = {}
    for node in model.graph.node:
        for name in node.input:
            uses[name] = uses.get(name, 0) + 1
    nodes, count = [], 0
    for node in model.graph.node:
        weight = inits.get(node.input[1]) if node.op_type == "Conv" and len(node.input) > 1 else None
        if weight is None or len(weight.dims) != 3:
            nodes.append(node)
            continue
        if uses.get(weight.name, 0) != 1:
            raise SystemExit(f"{node.name}: the weight {weight.name} is used by {uses[weight.name]} nodes")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise SystemExit(f"{node.name}: auto_pad {attrs['auto_pad']} is not handled")
        kernel = list(attrs.get("kernel_shape", [weight.dims[2]]))
        pads = list(attrs.get("pads", [0, 0]))
        strides = list(attrs.get("strides", [1]))
        dilations = list(attrs.get("dilations", [1]))
        weight.dims.append(1)
        base = node.name or node.output[0]
        shape_in = numpy_helper.from_array(np.array([0, 0, -1, 1], np.int64), f"{base}_c2d_shape_in")
        shape_out = numpy_helper.from_array(np.array([0, 0, -1], np.int64), f"{base}_c2d_shape_out")
        model.graph.initializer.extend([shape_in, shape_out])
        x4, y4 = f"{base}_c2d_input", f"{base}_c2d_output"
        conv_attrs = {"kernel_shape": kernel + [1], "pads": [pads[0], 0, pads[1], 0], "strides": strides + [1],
                      "dilations": dilations + [1]}
        if "group" in attrs:
            conv_attrs["group"] = attrs["group"]
        nodes.append(helper.make_node("Reshape", [node.input[0], shape_in.name], [x4], name=f"{base}_reshape_input"))
        nodes.append(helper.make_node("Conv", [x4] + list(node.input[1:]), [y4], name=base, **conv_attrs))
        nodes.append(helper.make_node("Reshape", [y4, shape_out.name], [node.output[0]], name=f"{base}_reshape_output"))
        count += 1
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if os.path.dirname(os.path.abspath(args.model)) != os.path.dirname(os.path.abspath(args.out)):
        raise SystemExit("write --out into the folder of --model (the weight files stay where they are)")
    model = onnx.load(args.model, load_external_data=False)
    count = rewrite(model)
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    onnx.save(model, args.out)
    print(f"{args.out}: {count} 1-D convolutions rewritten as 2-D")


if __name__ == "__main__":
    main()
