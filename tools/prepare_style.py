#!/usr/bin/env python3
"""Retarget a fast-neural-style ONNX model to a game resolution and quantise it.

The model-zoo models are fixed at 224x224 and opset 9, which is not what a
frame looks like and not what the QNN importer prefers. This resizes the graph,
lifts the opset so the deprecated Upsample becomes Resize, folds away the
dynamic-shape arithmetic the upsampling used, then quantises to int8 QDQ with an
NHWC interface -- the same shape of pipeline the NSS model goes through, for the
same reasons.

    tools/prepare_style.py --model models/style_mosaic-9.onnx --height 360 --width 640
"""
import argparse
import pathlib
import sys

import numpy as np
import onnx
from onnx import version_converter


def retarget(model, h, w):
    """Point the graph at a new input size and let shape inference follow."""
    graph = model.graph
    initialisers = {i.name for i in graph.initializer}
    real_inputs = [i for i in graph.input if i.name not in initialisers]
    if len(real_inputs) != len(graph.input):
        del graph.input[:]
        graph.input.extend(real_inputs)

    for value in list(graph.input) + list(graph.output):
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 4:
            dims[0].dim_value = 1
            dims[2].dim_value = h
            dims[3].dim_value = w

    # Stale value_info would contradict the new size.
    del graph.value_info[:]
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out")
    args = ap.parse_args()

    src = pathlib.Path(args.model)
    out = pathlib.Path(args.out) if args.out else src.with_name(
        f"{src.stem}_{args.width}x{args.height}.onnx")

    model = onnx.load(str(src))
    print(f"opset {[o.version for o in model.opset_import]} -> 13")
    model = retarget(model, args.height, args.width)
    model = version_converter.convert_version(model, 13)

    # Constant folding collapses the Shape/Gather/Floor chain the old Upsample
    # used to compute its output size, which quantisation cannot see through.
    try:
        from onnxruntime.transformers.onnx_model import OnnxModel
        folded = OnnxModel(model)
        folded.topological_sort()
        model = folded.model
    except Exception as exc:                      # optional, not fatal
        print(f"  (skipping fold: {exc})")

    model = onnx.shape_inference.infer_shapes(model)

    # Resize with a dynamically computed size splits the graph: QNN cannot take
    # the Shape/Gather/Floor chain feeding it, so each Resize lands on the CPU
    # and every partition boundary costs a round trip. The resolution is fixed,
    # so the sizes are constants -- bake them in and drop the arithmetic.
    # Shape inference cannot give the Resize *outputs* -- their sizes are the
    # dynamic thing we are removing -- so derive them from each Resize's input,
    # which is known. This architecture upsamples 2x, twice. Freezing one makes
    # the next one's input knowable, so iterate until nothing new can be fixed.
    frozen = 0
    for _ in range(8):
        model = onnx.shape_inference.infer_shapes(model)
        shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
                  for vi in list(model.graph.value_info) + list(model.graph.input)
                  + list(model.graph.output)}
        progressed = False
        for node in model.graph.node:
            if node.op_type != "Resize" or (len(node.input) > 2 and node.input[2].endswith("_scales")):
                continue
            src = shapes.get(node.input[0])
            if not src or len(src) != 4 or not all(src):
                continue
            dims = [src[0], src[1], src[2] * 2, src[3] * 2]
            # scales, not sizes: QNN rejects a sizes-driven Resize even when the
            # sizes are constant, and takes the equivalent scales form.
            name = node.output[0] + "_scales"
            model.graph.initializer.append(
                onnx.helper.make_tensor(name, onnx.TensorProto.FLOAT, [4],
                                        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)))
            data = node.input[0]
            del node.input[:]
            node.input.extend([data, "", name])
            frozen += 1
            progressed = True
        if not progressed:
            break
    print(f"froze {frozen} Resize node(s) to constant scales")

    # Drop whatever only existed to compute those sizes.
    for _ in range(8):
        used = {i for n in model.graph.node for i in n.input}
        used |= {o.name for o in model.graph.output}
        dead = [n for n in model.graph.node
                if n.output and not any(o in used for o in n.output)]
        if not dead:
            break
        for n in dead:
            model.graph.node.remove(n)
    print(f"  {len(model.graph.node)} nodes after pruning")

    onnx.checker.check_model(model)
    onnx.save(model, str(out))

    from collections import Counter
    ops = Counter(n.op_type for n in model.graph.node)
    print(f"wrote {out}")
    print(f"  input  {[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"  ops    {dict(ops)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
