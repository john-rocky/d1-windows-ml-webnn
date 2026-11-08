"""Does this ONNX Runtime load a CausalConvWithState / LinearAttention node that carries state_window=0?

The onnxruntime-genai 0.19.0 model builder writes state_window on both ops even when it is 0 (its default, "no
window"). This builds each op once without the attribute and once with state_window=0, creates a CPU session for
each, runs it, and prints one line per case. Where both load, it also prints the largest output difference.

    python state_window_repro.py
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

rng = np.random.default_rng(0)


def f32(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def causal_conv(attrs):
    # input (B=1, C=4, L=3), depthwise weight (C, 1, K=3), past_state (B, C, K-1); no bias
    node = helper.make_node("CausalConvWithState", ["x", "w", "", "past"], ["y", "present"],
                            domain="com.microsoft", ndim=1, activation="none", **attrs)
    feeds = {"x": f32(1, 4, 3), "w": f32(4, 1, 3), "past": f32(1, 4, 2)}
    outputs = [("y", [1, 4, 3]), ("present", [1, 4, 2])]
    return node, feeds, outputs


def linear_attention(attrs):
    # B=1, T=2, one query head and one key/value head, d_k = d_v = 2
    node = helper.make_node("LinearAttention", ["q", "k", "v", "past", "decay", "beta"], ["y", "present"],
                            domain="com.microsoft", q_num_heads=1, kv_num_heads=1, update_rule="gated_delta",
                            scale=1.0, **attrs)
    k = f32(1, 2, 2)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    feeds = {"q": f32(1, 2, 2), "k": k, "v": f32(1, 2, 2), "past": f32(1, 1, 2, 2),
             "decay": -np.abs(f32(1, 2, 1)), "beta": (1 / (1 + np.exp(-f32(1, 2, 1)))).astype(np.float32)}
    outputs = [("y", [1, 2, 2]), ("present", [1, 1, 2, 2])]
    return node, feeds, outputs


def model_bytes(node, feeds, outputs):
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape)) for name, value in feeds.items()],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in outputs])
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 17),
                                                                   helper.make_opsetid("com.microsoft", 1)])
    return model.SerializeToString()


print(f"onnxruntime {ort.__version__}, onnx {onnx.__version__}, CPUExecutionProvider")
for build in (causal_conv, linear_attention):
    results = {}
    for label, attrs in (("no state_window", {}), ("state_window=0", {"state_window": 0})):
        rng = np.random.default_rng(0)  # the same inputs for both cases
        node, feeds, outputs = build(attrs)
        try:
            session = ort.InferenceSession(model_bytes(node, feeds, outputs), providers=["CPUExecutionProvider"])
        except Exception as error:  # the error text is the result
            print(f"{node.op_type:20} {label:16} load FAILED: {error}")
            continue
        try:
            results[label] = session.run(None, feeds)
            print(f"{node.op_type:20} {label:16} load ok, run ok")
        except Exception as error:
            print(f"{node.op_type:20} {label:16} load ok, run FAILED: {error}")
    if len(results) == 2:
        diff = max(float(np.abs(a - b).max()) for a, b in zip(*results.values()))
        print(f"{node.op_type:20} max |difference| between the two cases: {diff}")
