"""Write models/tiny/model.onnx for probe.mjs: y = x @ I, x float32 [1, 4] (the same one-op graph as winml_probe.py).

    python tiny_model.py <webroot>
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

if len(sys.argv) != 2:
    sys.exit(__doc__)
root = sys.argv[1]
os.makedirs(os.path.join(root, "models", "tiny"), exist_ok=True)
graph = helper.make_graph(
    [helper.make_node("MatMul", ["x", "w"], ["y"])],
    "g",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor("w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten().tolist())],
)
onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10),
          os.path.join(root, "models", "tiny", "model.onnx"))
print("wrote", os.path.join(root, "models", "tiny", "model.onnx"))
