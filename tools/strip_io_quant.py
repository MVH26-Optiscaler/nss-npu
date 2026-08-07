#!/usr/bin/env python3
"""Make a QDQ model take int8 in and hand int8 back.

ORT wraps a quantised graph in a QuantizeLinear on the input and a
DequantizeLinear on each output, so a caller can keep passing floats. Those
three nodes stay on the CPU, and for NSS they convert 6.3M elements in and 3.3M
out on every frame -- pure overhead in a frame budget.

Arm's pipeline never wants floats anyway: the pre-process shader produces the
int8 tensor and post-process consumes int8, so the conversions exist only to
satisfy an interface we control. Removing them also cuts the input tensor from
25MB to 6.3MB.

    tools/strip_io_quant.py --model in.onnx --out out.onnx
"""
import argparse
import sys

import onnx
from onnx import TensorProto, helper

# Which ONNX element type a QuantizeLinear produces is decided by its
# zero-point operand, so read it rather than assuming int8.
ZP_TO_ELEM = {
    TensorProto.INT8: TensorProto.INT8,
    TensorProto.UINT8: TensorProto.UINT8,
    TensorProto.INT16: TensorProto.INT16,
    TensorProto.UINT16: TensorProto.UINT16,
}


def dims(value):
    return [d.dim_value for d in value.type.tensor_type.shape.dim]


def elem_type_of(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return ZP_TO_ELEM[init.data_type]
    raise SystemExit(f"zero-point {name} is not an initializer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.model)
    graph = model.graph

    # --- input: drop the leading QuantizeLinear ---
    src = graph.input[0]
    quant = [n for n in graph.node
             if n.op_type == "QuantizeLinear" and n.input[0] == src.name]
    if len(quant) != 1:
        raise SystemExit(f"expected one QuantizeLinear on {src.name}, found {len(quant)}")
    q = quant[0]

    new_input = helper.make_tensor_value_info(
        q.output[0], elem_type_of(graph, q.input[2]), dims(src))
    graph.node.remove(q)
    graph.input.remove(src)
    graph.input.insert(0, new_input)
    print(f"input  {src.name} float -> {new_input.name} "
          f"{TensorProto.DataType.Name(new_input.type.tensor_type.elem_type)}")

    # --- outputs: drop the trailing DequantizeLinear, preserving order ---
    replacements = []
    for out in graph.output:
        dequant = [n for n in graph.node
                   if n.op_type == "DequantizeLinear" and n.output[0] == out.name]
        if len(dequant) != 1:
            raise SystemExit(f"expected one DequantizeLinear for {out.name}")
        d = dequant[0]
        replacements.append(helper.make_tensor_value_info(
            d.input[0], elem_type_of(graph, d.input[2]), dims(out)))
        graph.node.remove(d)
        print(f"output {out.name} float -> {d.input[0]} "
              f"{TensorProto.DataType.Name(replacements[-1].type.tensor_type.elem_type)}")

    del graph.output[:]
    graph.output.extend(replacements)

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
