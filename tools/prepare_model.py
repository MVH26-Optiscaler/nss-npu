#!/usr/bin/env python3
"""Make an ONNX model acceptable to the QNN HTP backend.

HTP rejects dynamic shapes outright, and older exports often declare their
weights as graph inputs alongside the real input, which confuses both the
converter and anyone reading the signature. This fixes both, and upgrades the
opset since QNN's importer is happier on 13+.

    tools/prepare_model.py in.onnx out.onnx [--batch 1]
"""
import argparse
import sys

import onnx
from onnx import version_converter

TARGET_OPSET = 13


def describe(model, label):
    graph = model.graph
    initialisers = {i.name for i in graph.initializer}
    print(f"[{label}] opset={[o.version for o in model.opset_import]}")
    for value in list(graph.input) + list(graph.output):
        kind = "init " if value.name in initialisers else "io   "
        dims = [d.dim_value or d.dim_param or "?"
                for d in value.type.tensor_type.shape.dim]
        print(f"  {kind}{value.name:16} {dims}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    model = onnx.load(args.src)
    describe(model, "before")

    # Weights that are already initialisers do not belong in graph.input.
    initialisers = {i.name for i in model.graph.initializer}
    real_inputs = [i for i in model.graph.input if i.name not in initialisers]
    if len(real_inputs) != len(model.graph.input):
        dropped = len(model.graph.input) - len(real_inputs)
        del model.graph.input[:]
        model.graph.input.extend(real_inputs)
        print(f"dropped {dropped} initialiser(s) from graph inputs")

    # Pin every symbolic dimension. HTP compiles to a fixed graph.
    for value in list(model.graph.input) + list(model.graph.output):
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                dim.Clear()
                dim.dim_value = args.batch

    current = model.opset_import[0].version if model.opset_import else 0
    if current < TARGET_OPSET:
        model = version_converter.convert_version(model, TARGET_OPSET)
        print(f"opset {current} -> {TARGET_OPSET}")

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    describe(model, "after")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
