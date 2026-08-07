#!/usr/bin/env python3
"""Give a model NHWC inputs and outputs without touching its interior.

Arm's pre-process shader writes the input tensor as NHWC int8 -- `(y * width +
x) * 3` int8x4 vectors -- and its post-process reads NHWC back. The exported
graph is NCHW because it came from PyTorch. Transposing 6.3MB per frame on the
CPU to bridge that would cost more than the inference.

So the transposes go in the graph instead, where they are free: QNN's layout
transformer runs the whole network in NHWC on the HTP anyway, so boundary
transposes cancel against the ones it would otherwise insert.

    tools/nhwc_io.py --model in.onnx --out out.onnx
"""
import argparse
import sys

import onnx
from onnx import helper


def dims(value):
    return [d.dim_value for d in value.type.tensor_type.shape.dim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.model)
    graph = model.graph
    nodes = []

    # --- input: NHWC -> NCHW feeding the original graph input ---
    src = graph.input[0]
    n, c, h, w = dims(src)
    nhwc_in = helper.make_tensor_value_info(
        src.name + "_nhwc", src.type.tensor_type.elem_type, [n, h, w, c])
    nodes.append(helper.make_node("Transpose", [nhwc_in.name], [src.name],
                                  name="nhwc_in", perm=[0, 3, 1, 2]))
    graph.input.remove(src)
    graph.input.insert(0, nhwc_in)
    print(f"input  {src.name} [{n},{c},{h},{w}] -> {nhwc_in.name} [{n},{h},{w},{c}]")

    # --- outputs: NCHW -> NHWC ---
    replacements = []
    for out in list(graph.output):
        n, c, h, w = dims(out)
        nhwc_out = helper.make_tensor_value_info(
            out.name + "_nhwc", out.type.tensor_type.elem_type, [n, h, w, c])
        nodes.append(helper.make_node("Transpose", [out.name], [nhwc_out.name],
                                      name="nhwc_" + out.name, perm=[0, 2, 3, 1]))
        replacements.append(nhwc_out)
        print(f"output {out.name} [{n},{c},{h},{w}] -> {nhwc_out.name} [{n},{h},{w},{c}]")

    del graph.output[:]
    graph.output.extend(replacements)

    # The input transpose must run first and the output ones last; ONNX
    # requires nodes in topological order.
    body = list(graph.node)
    del graph.node[:]
    graph.node.append(nodes[0])
    graph.node.extend(body)
    graph.node.extend(nodes[1:])

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
