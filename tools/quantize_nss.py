#!/usr/bin/env python3
"""Quantise the exported NSS graph to int8 QDQ for the Hexagon NPU.

HTP is built for integer arithmetic; running NSS as fp32-with-fp16-precision
leaves most of the NPU idle. Arm ships NSS as int8 for that reason.

Calibration uses the real input tensors from Arm's NSS scenario rather than
random data, so the activation ranges are the ones the network actually sees.
The result is checked against Arm's reference outputs before it is written.

    tools/quantize_nss.py --model models/nss_v1_high_544x960.onnx \
                          --out models/nss_v1_high_544x960_int8.onnx \
                          --scenario ~/dev/q6a-backup/nss/scenario
"""
import argparse
import glob
import os
import pathlib
import sys

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import (
    get_qnn_qdq_config,
    qnn_preprocess_model,
)

INPUT_Q = (0.003912401385605335, -128)
OUTPUT_Q = (0.003937007859349251, -127)

# Arm's own I/O quantisation, from nss_v1_0_1_high_int8_metadata.json. Forcing
# these rather than letting calibration pick its own is what makes the int8
# interface byte-compatible with Arm's pre- and post-process shaders: the
# tensor the shader writes is the tensor the network reads, with no requantise
# in between. Calibration lands ~0.4% away, which is close enough to look
# right and wrong enough to shift every value.
IO_OVERRIDES = {
    "preprocess_tensor": INPUT_Q,
    "kpn_coefficients": OUTPUT_Q,
    "temporal_tensor": OUTPUT_Q,
}


def dequant(a, q):
    scale, zero = q
    return (a.astype(np.float32) - zero) * scale


def load_frames(scenario: pathlib.Path):
    """Every distinct input tensor the scenario shipped, as NCHW float."""
    seen, frames = set(), []
    for path in sorted(glob.glob(str(scenario / "*" / "out_input_tensor.npy"))):
        raw = np.load(path)
        key = hash(raw.tobytes())
        if key in seen:
            continue
        seen.add(key)
        frames.append(dequant(raw, INPUT_Q).transpose(0, 3, 1, 2).copy())
        print(f"  calibration frame from {pathlib.Path(path).parent.name}")
    if not frames:
        raise SystemExit(f"no out_input_tensor.npy under {scenario}")
    return frames


class Frames(CalibrationDataReader):
    def __init__(self, name, frames):
        self.name = name
        self.it = iter(frames)

    def get_next(self):
        item = next(self.it, None)
        return None if item is None else {self.name: item}


def check(model_path, frames, scenario: pathlib.Path, label):
    """Run the model on CPU and correlate against Arm's reference outputs."""
    import onnxruntime as ort

    ref = scenario / "out_high"
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    kpn, temporal = session.run(None, {name: frames[0]})

    print(f"  {label}:")
    for tensor, ref_name in ((kpn, "out_graph_0.npy"), (temporal, "out_graph_1.npy")):
        want = dequant(np.load(ref / ref_name), OUTPUT_Q)
        mine = tensor.transpose(0, 2, 3, 1)
        corr = np.corrcoef(mine.ravel(), want.ravel())[0, 1]
        err = np.abs(mine - want)
        print(f"    {ref_name}  corr {corr:.5f}  mean|err| {err.mean():.5f}")
    return temporal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--activations", default="int8", choices=["int8", "uint8", "uint16"],
                    help="int8 is what HTP is fastest at; uint16 trades speed for range")
    args = ap.parse_args()

    scenario = pathlib.Path(os.path.expanduser(args.scenario))
    print("calibration data:")
    frames = load_frames(scenario)

    input_name = onnx.load(args.model).graph.input[0].name

    # qnn_preprocess_model folds the graph into shapes the QNN QDQ config
    # expects; skipping it produces configs that silently fail to apply.
    prepared = args.out + ".prepared.onnx"
    changed = qnn_preprocess_model(args.model, prepared)
    src = prepared if changed else args.model
    print(f"qnn_preprocess_model changed the graph: {changed}")

    act = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8,
           "uint16": QuantType.QUInt16}[args.activations]
    # per_channel matters enormously here and is not optional: with per-tensor
    # weights the KPN head collapses to corr 0.26 against Arm's reference,
    # because its 36 sigmoid channels span very different ranges. Per-channel
    # takes it to 0.998.
    overrides = {
        # 0-d numpy arrays, not numpy scalars: the quantizer type-checks for
        # ndarray and rejects np.int8/np.float32 values outright.
        name: [{"scale": np.array(scale, dtype=np.float32),
                "zero_point": np.array(zero, dtype=np.int8)}]
        for name, (scale, zero) in IO_OVERRIDES.items()
    }
    config = get_qnn_qdq_config(
        src,
        Frames(input_name, frames),
        activation_type=act,
        weight_type=QuantType.QInt8,
        per_channel=True,
        init_overrides=overrides,
    )
    quantize(src, args.out, config)
    print(f"wrote {args.out}  activations={args.activations} weights=int8 per-channel")

    print("accuracy against Arm's reference:")
    check(args.model, frames, scenario, "fp32")
    check(args.out, frames, scenario, f"int8 qdq ({args.activations} activations)")

    if os.path.exists(prepared):
        os.remove(prepared)
    return 0


if __name__ == "__main__":
    sys.exit(main())
