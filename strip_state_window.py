"""Copy a GenAI model folder with every state_window attribute removed from model.onnx (each must be 0).

The other files are hard links (same volume), so the weights are not copied; model.onnx is read without its external
data and written back with the same external-data references.

    python strip_state_window.py <model folder> <new folder>
"""
import os
import sys

import onnx

if len(sys.argv) != 3:
    sys.exit(__doc__)
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for name in os.listdir(src):
    if name != "model.onnx" and os.path.isfile(os.path.join(src, name)) and not os.path.exists(os.path.join(dst, name)):
        os.link(os.path.join(src, name), os.path.join(dst, name))
model = onnx.load(os.path.join(src, "model.onnx"), load_external_data=False)
removed = {}
for node in model.graph.node:
    for attribute in list(node.attribute):
        if attribute.name == "state_window":
            value = onnx.helper.get_attribute_value(attribute)
            assert value == 0, f"{node.name}: state_window={value}"
            node.attribute.remove(attribute)
            removed[node.op_type] = removed.get(node.op_type, 0) + 1
onnx.save(model, os.path.join(dst, "model.onnx"))
print(f"removed state_window=0 from {removed}; wrote {os.path.join(dst, 'model.onnx')}")
