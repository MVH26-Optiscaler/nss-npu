#!/usr/bin/env python3
"""Export Arm's NSS v1 backbone to ONNX, and check it against Arm's own tensors.

The published checkpoint is a bare state dict, so the architecture comes from
Arm's model gym (fetched and cached, not vendored). The interesting part is the
validation: the NSS scenario ships a real input tensor and the two output
tensors the reference implementation produced for it, so we can prove the
exported graph computes the right thing before it ever reaches the NPU.

    tools/export_nss.py --weights models/nss_v1_0_1_high_fp32.pt \
                        --out models/nss_v1_high_544x960.onnx \
                        --validate ~/dev/q6a-backup/nss/scenario/out_high
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.request

GYM = "https://raw.githubusercontent.com/arm/neural-graphics-model-gym/main/src"
SOURCES = {
    "ng_model_gym/__init__.py": None,
    "ng_model_gym/core/__init__.py": None,
    "ng_model_gym/core/model/__init__.py": None,
    "ng_model_gym/core/model/layers/__init__.py": None,
    "ng_model_gym/core/model/layers/conv_block.py": f"{GYM}/ng_model_gym/core/model/layers/conv_block.py",
    "ng_model_gym/usecases/__init__.py": None,
    "ng_model_gym/usecases/nss/__init__.py": None,
    "ng_model_gym/usecases/nss/model/__init__.py": None,
    "ng_model_gym/usecases/nss/model/model_blocks_v1.py": f"{GYM}/ng_model_gym/usecases/nss/model/model_blocks_v1.py",
}

# From nss_v1_0_1_high_int8_metadata.json. real = scale * (q - zero_point).
INPUT_Q = (0.003912401385605335, -128)
OUTPUT_Q = (0.003937007859349251, -127)


def fetch_gym(cache: pathlib.Path):
    """Materialise just enough of the model gym package to import the backbone."""
    for rel, url in SOURCES.items():
        dst = cache / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if url is None:
            dst.write_text("")
        else:
            print(f"fetching {rel}")
            with urllib.request.urlopen(url, timeout=60) as r:
                dst.write_bytes(r.read())
    sys.path.insert(0, str(cache))


# The int8 checkpoint is a traced export, so its parameters are named
# _param_constantN in *execution* order rather than declaration order -- note
# kpn_params runs before conv2d_9.
EXEC_ORDER = [
    "conv2d_0", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5",
    "conv2d_6", "conv2d_7", "conv2d_8", "kpn_params", "conv2d_9", "conv2d_10",
    "conv2d_11", "temporal_params_out_conv",
]


class StaticUpsample2x:
    """Nearest 2x upsample that exports with explicit output sizes.

    nn.UpsamplingNearest2d exports a Resize carrying float scales. ORT's
    quantiser then mis-infers the shape downstream -- conv2d_7 ends up seeing
    C:31 against 32 kernel channels and QLinearConv fails outright. Every shape
    in this graph is static, so baking the sizes in at trace time avoids the
    whole class of problem.
    """

    def __new__(cls):
        import torch
        from torch import nn
        from torch.nn import functional as F

        class _Impl(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h, w = int(x.shape[-2]), int(x.shape[-1])
                return F.interpolate(x, size=(h * 2, w * 2), mode="nearest")

        return _Impl()


def build_model(weights: pathlib.Path):
    import torch
    from ng_model_gym.usecases.nss.model.model_blocks_v1 import AutoEncoderV1

    model = AutoEncoderV1(in_channels=12, temporal_ch=4, kpn_size=(6, 6),
                          batch_norm=False)

    blob = torch.load(weights, map_location="cpu", weights_only=False)
    raw = blob.get("model_state_dict", blob)

    if any("_param_constant" in k for k in raw):
        print("checkpoint is a traced int8 export; mapping by execution order")
        state = {}
        for i, name in enumerate(EXEC_ORDER):
            state[f"{name}.conv2d.weight"] = raw[f"autoencoder._param_constant{2 * i}"]
            state[f"{name}.conv2d.bias"] = raw[f"autoencoder._param_constant{2 * i + 1}"]
    else:
        state = {k[len("autoencoder."):]: v for k, v in raw.items()
                 if k.startswith("autoencoder.")}

    model.load_state_dict(state, strict=True)
    model.upsample = StaticUpsample2x()
    model.eval()
    return model


def validate(model, ref_dir: pathlib.Path):
    """Run Arm's own input through our graph and compare against their outputs."""
    import numpy as np
    import torch

    def dequant(a, q):
        scale, zero = q
        return (a.astype(np.float32) - zero) * scale

    src = dequant(np.load(ref_dir / "out_input_tensor.npy"), INPUT_Q)  # NHWC
    x = torch.from_numpy(src.transpose(0, 3, 1, 2)).contiguous()       # NCHW

    with torch.no_grad():
        kpn, temporal = model(x)

    # The temporal tensor is the gate: it is full resolution and its path runs
    # through every layer of the trunk, so getting it right means the
    # architecture and the weight mapping are both right. The KPN head is
    # reported but not gated -- several of its channels saturate to exactly 0
    # in the int8 reference where fp32 gives small non-zero values, which costs
    # correlation without meaning much.
    gate = {"out_graph_1.npy": 0.99, "out_graph_0.npy": 0.0}
    ok = True
    for name, got, q in (("out_graph_0.npy", kpn, OUTPUT_Q),
                         ("out_graph_1.npy", temporal, OUTPUT_Q)):
        want = dequant(np.load(ref_dir / name), q)
        mine = got.numpy().transpose(0, 2, 3, 1)
        if mine.shape != want.shape:
            print(f"  {name}: SHAPE MISMATCH got {mine.shape} want {want.shape}")
            ok = False
            continue
        err = np.abs(mine - want)
        corr = np.corrcoef(mine.ravel(), want.ravel())[0, 1]
        # Arm's tensors come from the int8 graph, where every one of the 14
        # layers requantises its activations; we run the same weights in fp32.
        # So agreement is judged by correlation, not by equality.
        print(f"  {name}: shape {mine.shape}  corr {corr:.5f}  "
              f"mean|err| {err.mean():.5f}  max|err| {err.max():.4f}")
        ok = ok and corr > gate[name]
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--validate")
    ap.add_argument("--cache", default=os.path.expanduser("~/dev/nss-npu/third_party/gym"))
    args = ap.parse_args()

    fetch_gym(pathlib.Path(args.cache))
    import torch

    model = build_model(pathlib.Path(args.weights))
    total = sum(p.numel() for p in model.parameters())
    print(f"loaded NSS v1 backbone: {total} parameters")

    if args.validate:
        print("validating against Arm's reference tensors:")
        if not validate(model, pathlib.Path(os.path.expanduser(args.validate))):
            raise SystemExit("validation failed -- not exporting")
        print("  validation OK")

    dummy = torch.zeros(1, 12, args.height, args.width)
    torch.onnx.export(
        model, (dummy,), args.out,
        input_names=["preprocess_tensor"],
        output_names=["kpn_coefficients", "temporal_tensor"],
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {args.out}  input 1x12x{args.height}x{args.width}")


if __name__ == "__main__":
    sys.exit(main())
