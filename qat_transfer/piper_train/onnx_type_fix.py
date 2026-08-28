"""Post-process a torch.onnx.export()'d graph to fix int32/int64 index-type
mismatches that this torch/onnx version's TorchScript-based exporter emits
(reproduces on a plain FP32 export, unrelated to quantization -- ONNX
Runtime refuses to load the raw export with e.g. "Type parameter (Tind) of
Optype (Slice) bound to different types (tensor(int32) and tensor(int64))").

Surgical fix: insert Cast(to=INT64) on whichever integer inputs of
Slice/Gather/Expand/Reshape nodes are int32 while their siblings on the same
node are int64, so every node's index inputs share one consistent type.

Deliberately not using onnxsim for this: onnxsim's simplify() resolves the
crash too (as a side effect of its constant-folding/shape-inference passes),
but was observed on this model to also collapse SynthesizerTrn.infer()'s
genuinely input-length-dependent duration computation down to a fixed
length baked in from whatever shape onnxsim happened to trace with --
verified by comparing ONNX output duration against the pure-PyTorch ground
truth across inputs of varying phoneme length. This function only touches
the specific mismatched tensors and leaves the rest of the graph, including
all dynamic-shape behavior, untouched.
"""
import sys

import onnx
from onnx import TensorProto, helper, shape_inference

INDEX_TYPED_INPUTS = {
    "Slice": (1, 2, 3, 4),  # starts, ends, axes, steps
    "Gather": (1,),  # indices
    "Expand": (1,),  # shape
    "Reshape": (1,),  # shape
}


def _build_type_map(model: onnx.ModelProto) -> dict:
    inferred = shape_inference.infer_shapes(model)
    type_map = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if vi.type.HasField("tensor_type"):
            type_map[vi.name] = vi.type.tensor_type.elem_type
    for init in inferred.graph.initializer:
        type_map[init.name] = init.data_type
    return type_map


def fix_index_type_mismatches(model: onnx.ModelProto) -> int:
    """Mutates model in place. Returns the number of nodes fixed."""
    type_map = _build_type_map(model)
    graph = model.graph
    cast_cache: dict = {}  # original tensor name -> cast output name
    n_fixed = 0

    new_nodes = []
    for node in graph.node:
        input_idxs = INDEX_TYPED_INPUTS.get(node.op_type)
        if input_idxs is None:
            new_nodes.append(node)
            continue

        relevant = [i for i in input_idxs if i < len(node.input) and node.input[i]]
        dtypes = [type_map.get(node.input[i]) for i in relevant]
        int_dtypes = [d for d in dtypes if d in (TensorProto.INT32, TensorProto.INT64)]
        if len(set(int_dtypes)) <= 1:
            new_nodes.append(node)
            continue  # already consistent (or types unknown/non-integer -- leave alone)

        n_fixed += 1
        for i in relevant:
            name = node.input[i]
            if type_map.get(name) != TensorProto.INT32:
                continue
            if name not in cast_cache:
                cast_out = f"{name}_cast_i64"
                cast_node = helper.make_node("Cast", [name], [cast_out], to=TensorProto.INT64)
                new_nodes.append(cast_node)
                cast_cache[name] = cast_out
                type_map[cast_out] = TensorProto.INT64
            node.input[i] = cast_cache[name]
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return n_fixed


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    n = fix_index_type_mismatches(m)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"Fixed {n} node(s) with int32/int64 index mismatches. Wrote {dst}")
