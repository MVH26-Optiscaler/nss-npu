# nss-npu

Getting Arm NSS running through ONNX on the Hexagon NPU of a Galaxy S25 Ultra.

This is deliberately a standalone project. The eventual destination is the
XeSS shim in `~/dev/xess-shim`, which upscales The Witcher 3 under Proton on
this phone — but that adds a PE/unix boundary and a real-time budget on top of
questions worth answering with no game and no wine in the way.

## The device

    SM-S938U1        Galaxy S25 Ultra, US unlocked variant
    Android 16       SDK 36, One UI
    Snapdragon 8 Elite (sun), Adreno 830, Hexagon HTP V79
    bootloader       locked, verifiedbootstate=green

Everything here works unprivileged. No root, no unlocked bootloader.

## The NPU is reachable from an ordinary APK

Verified, not inferred. In `u:r:untrusted_app:s0`, uid 10356, from a sideloaded
debug APK:

    remote_session_control(unsigned pd, cdsp) = 0x0 (unsigned PD enabled)
    ADSP_LIBRARY_PATH=/data/user/0/dev.ynk.nssnpu/files/dsp
    remote_handle64_open(Calculator) = 0x0  handle=12970367451812963024
    *** DSP SESSION OPENED -- our code is on the NPU ***

Independently, `qnn-platform-validator` from QAIRT 2.32.6 reports from the
shell domain:

    Backend Hardware  : Supported
    Core Version      : Hexagon Architecture V79
    Unit Test         : Passed
    QNN is supported for backend DSP on the device.

### What it takes to get there

Four things, each of which silently produces a different failure:

1. **`<uses-native-library>` in the manifest.** `libcdsprpc.so` is listed in
   `/vendor/etc/public.libraries.txt`, but the app linker namespace still
   refuses it until the app declares it. Without this, `dlopen` fails with
   *"not accessible for the namespace"* and nothing else matters.
2. **Enable unsigned process domains before the first handle open**, via
   `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)`. The request
   id is **2** — from `enum session_control_req_id` in qualcomm/fastrpc's
   `inc/remote.h`. Guessing this wrong returns `0x14` and then every session
   fails with `0x80000600`.
3. **`ADSP_LIBRARY_PATH` pointing at a directory holding the Hexagon skels.**
   The app's own `filesDir` works; the skels ship as assets and get staged
   there at startup. This is what the DSP-side loader searches, and it is *not*
   consulted for anything else.
4. **The right URI, case included.** It is
   `file:///libCalculator_skel.so?Calculator_skel_handle_invoke&_modver=1.0&_dom=cdsp`
   — capital C in `Calculator_skel_handle_invoke`, which is not what the IDL
   naming convention would lead you to write. Read it out of the stub's strings
   rather than guessing.

### Two things that look like blockers and are not

**An app cannot `open()` the device node**, and this does not matter. All three
nodes return EACCES from an app:

    /dev/fastrpc-cdsp   O_RDONLY EACCES   O_RDWR EACCES

That is consistent with the vendor policy, which grants apps `ioctl` and `read`
on `vendor_qdsp_device` but deliberately not `open`:

    allow appdomain vendor_qdsp_device (chr_file (ioctl read))

The descriptor comes from elsewhere — `vendor.qti.hardware.dsp.IDspService`
runs as `vendor_dspservice`, which does have `open`, and the policy gives its
clients `fd (use)`. `libcdsprpc` handles this internally. Testing DSP
reachability by calling `open()` yourself measures the wrong thing and gives a
confidently wrong answer.

**`fastrpc_shell_3` is not found, and this does not matter either.** With
unsigned PD enabled the client looks for `fastrpc_shell_unsigned_3` in four
hardcoded paths — `/usr/lib/dsp/{,cdsp/}` (ENOENT, they are Linux paths) and
`/vendor/dsp/{,cdsp/}` (EACCES). All four fail and the session still comes up,
so the shell is supplied on the far side rather than loaded by the client.
`ADSP_LIBRARY_PATH` does not redirect this lookup; `openmode_shim.c` can, via
`FASTRPC_DSP_DIR`, but there is no need.

## ONNX on the NPU works, and it is 8x faster than the CPU

The ONNX model-zoo sub-pixel super-resolution net (4 convs, ReLU, pixel
shuffle, 224x224 -> 672x672 — a fair stand-in for an upscaler), run from the
same sideloaded APK:

    [onnxruntime 1.21.1]
      QNN / HTP (NPU)  session  169 ms   mean   4.28 ms   best   4.19 ms
      CPU              session    1 ms   mean  35.17 ms   best  34.82 ms

The 169 ms session build is real HTP graph compilation, which is what the
context-binary cache exists to amortise.

### ORT and QAIRT are a matched pair

This is the part that cost the most time, so it is worth stating precisely.
QAIRT 2.32.6 exposes **QNN interface 2.24.0**. ORT's QNN EP requires the
interface minor version to be **>= the one ORT was built against**, so a newer
ORT rejects an older QAIRT:

    ORT 1.22 – 1.28   "Unable to find a valid interface"  -> all nodes on CPU
    ORT 1.20, 1.21.1  "Found valid interface, version: 2.24.0"  -> runs on HTP

`skip_qnn_version_check=1` forces a newer ORT past the check, and it is a trap:
the interface is then accepted and the session dies later in `CreateContext`
with `QNN_CONTEXT_ERROR_INVALID_CONFIG` (`Failed to set context custom config,
err 5010`), because the newer ORT sets context configs this QNN does not know.
Pair the versions properly instead. The build pins ORT 1.21.1.

Failure is silent by default: without verbose logging ORT reports no error, the
session just runs entirely on CPU at CPU speed. Always confirm with
`session_state.cc VerifyEachNodeIsAssignedToAnEp` in the log before trusting a
number.

## NSS itself runs on the NPU

Arm publishes NSS as PyTorch weights plus a compiled VGF, not ONNX, so
`tools/export_nss.py` reconstructs it: it fetches the `AutoEncoderV1` backbone
from Arm's model gym (cached under `third_party/gym`, not vendored), loads the
published checkpoint into it, validates, and exports ONNX.

    loaded NSS v1 backbone: 148456 parameters
    nss_v1_high_544x960.onnx
      QNN / HTP (NPU)  session  1092 ms   mean   61.14 ms   best   53.09 ms
      CPU              session     6 ms   mean   90.09 ms   best   89.21 ms

The entire graph is taken by the EP — *"All nodes placed on
[QNNExecutionProvider]. Number of nodes: 1"* — so the whole network fuses into
a single QNN node with no CPU fallback.

### Two things the export needed

**The int8 checkpoint is a traced export.** Its parameters are named
`_param_constantN` in *execution* order, not declaration order, and those
differ: `kpn_params` runs before `conv2d_9`. Mapping by declaration order fails
on a shape mismatch at index 18, which is the useful clue.

**Validation against Arm's own tensors.** The NSS scenario ships a real
`out_input_tensor.npy` and the two outputs the reference produced from it, so
the export is checked before it ever reaches the device:

    out_graph_1.npy (temporal)  corr 0.99944   mean|err| 0.01105
    out_graph_0.npy (kpn)       corr 0.93499   mean|err| 0.06297

The temporal tensor is the gate: it is full resolution and its path runs
through every layer, so 0.9994 means the architecture and the weight mapping
are both right. The KPN head is reported but not gated — several of its
channels saturate to exactly 0 in the int8 reference where fp32 gives small
non-zero values, which costs correlation without meaning much. For reference,
`out_verify` is byte-identical to `out_high`, and the earlier hand-rolled
attempt in `out_ours2` scores 0.9513 against it.

### int8 QDQ: 2.6x faster, and closer to Arm's reference than fp32

    nss_v1_high_544x960_int8.onnx
      QNN / HTP (NPU)  session   959 ms   mean   23.78 ms   best   21.02 ms
      CPU              session    10 ms   mean   37.28 ms   best   36.39 ms

Accuracy against Arm's reference tensors actually *improves*, which is the
right result: the reference is itself int8, so a correctly quantised graph
should track it better than fp32 does.

                        kpn       temporal
    fp32              0.93499     0.99944
    int8 QDQ          0.99757     0.99997

**`per_channel=True` is not optional.** With per-tensor weights the KPN head
collapses to corr 0.26 while the temporal output still reads 0.996 — the 36
sigmoid channels span very different ranges, and one shared scale destroys the
small ones. Per-channel takes it to 0.998. A single output looking healthy is
not evidence that quantisation went well.

**Export Resize with explicit sizes, not scales.** `nn.UpsamplingNearest2d`
emits a Resize carrying float scales; ORT's quantiser then mis-infers shapes
downstream and `QLinearConv` fails outright with *"Input channels C is not
equal to kernel channels * group. C: 31 kernel channels: 32"*. Every shape here
is static, so `tools/export_nss.py` swaps in an upsample that bakes the output
sizes in at trace time. The fp32 graph runs fine either way, so this only
shows up once you quantise.

### What is left on the CPU

The int8 session puts the entire network on the NPU as a single QNN node, and
leaves exactly three nodes on the CPU:

    QuantizeLinear   (preprocess_tensor_QuantizeLinear)
    DequantizeLinear x2   (the two outputs)

That is pure boundary conversion, because the session is fed fp32: 6.3 M
elements quantised in and 3.3 M dequantised out, every frame. It is also
avoidable — Arm's pipeline already produces the input tensor as int8 from the
pre-process shader and consumes int8 in post-process, so the real integration
never needs the fp32 boundary at all. Removing the outer Q/DQ and taking int8
in and out should account for a good part of the remaining 23 ms.

### Why 61 ms was not the real number

NSS is ~5.9 G MACs per frame at 544x960. At 61 ms that is roughly 0.2 TFLOP/s
effective, which is far below what this NPU does — because we are running an
fp32 graph through `enable_htp_fp16_precision`, and HTP is built for int8.
Arm ships NSS as int8 for exactly this reason, and the quantisation parameters
are already in `nss_v1_0_1_high_int8_metadata.json`:

    input  _PreprocessTensor   scale 0.003912401385605335  zero_point -128
    output _KpnCoefficients    scale 0.003937007859349251  zero_point -127
    output _TemporalTensor     scale 0.003937007859349251  zero_point -127

Quantising to int8 QDQ with those exact parameters is the next step, and it
should also cut the per-inference transfer: the fp32 input tensor alone is
25 MB, against 6 MB for int8.

## Consequences for the port

GameNative runs as `untrusted_app`, and `untrusted_app` can drive the NPU. So
NSS inference can live **inside the game's own process** — no shell-domain
helper, no loopback transport, no copying frames between processes. The XeSS
shim's unix side can call libcdsprpc directly, and the interesting problem goes
back to being the one that was always interesting: getting tensors from D3D12
into DSP-visible memory without a round trip through the CPU.

## Next

1. **Take int8 in and out.** Strip the outer QuantizeLinear/DequantizeLinear so
   the graph's own input and outputs are int8, matching what Arm's pre- and
   post-process shaders already exchange. That removes ~10 M element
   conversions per frame from the CPU.
2. **Feed it real data.** Dump G-buffers from the XeSS shim (color, depth,
   motion vectors, jitter) and run NSS on them offline, comparing against the
   SGSR 2 output already working in-game.
3. **Zero-copy.** `AHardwareBuffer` is the currency: Vulkan can import and
   export it, and FastRPC can map dmabuf FDs onto the DSP. Getting this right
   is what decides whether the in-process design is actually fast.

Worth keeping in view: SGSR 2 currently costs ~1 ms of an 86 ms frame in
Witcher 3, so none of this will make the game faster. The point is NSS's
quality, and the result itself.

## Model constraints on HTP

- The HTP backend wants quantized QDQ models; float runs via
  `enable_htp_fp16_precision`.
- No dynamic shapes — render and output resolution are baked in, so each
  upscale ratio is a separate compiled model.
- Graph compilation is slow, so use the context-binary cache
  (`ep.context_enable` / `ep.context_file_path`). The binary is HTP-version
  specific.

## Layout

    models/         superres_fixed.onnx, the HTP smoke-test model
    third_party/    onnxruntime-android-qnn AARs (1.21.1 is the pinned one)
    tools/          prepare_model.py -- pins shapes, drops initialiser inputs
    probe/          DSP reachability probe + ORT/HTP benchmark
      jni/          probe_core.c is shared; _main is the CLI, _jni is the app
                    openmode_shim.c traces and rewrites fastrpc opens
      app/          minimal APK, no Gradle -- aapt2 + javac + d8 + apksigner
      build.sh      builds both, offline; bundles skels from QAIRT

## Building and running the probe

    probe/build.sh          # honours $QAIRT_SDK, $ANDROID_SDK, $ANDROID_NDK

    # app domain -- the one that matters
    adb install -r probe/build/nssprobe.apk
    adb shell am start -n dev.ynk.nssnpu/.ProbeActivity
    adb logcat -d -s nssprobe:I

    # shell domain, with syscall tracing
    adb push probe/build/nssprobe probe/build/openmode_shim.so /data/local/tmp/
    adb shell 'cd /data/local/tmp/qnn && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
        FASTRPC_TRACE=1 LD_PRELOAD=/data/local/tmp/openmode_shim.so \
        /data/local/tmp/nssprobe'

## Open detail

`Qnn_calculatorTest()` from `libQnnHtpV79CalculatorStub.so` returns -6 with
*"Unable to destroy the handle"* in both domains, while the direct
`remote_handle64_open` in the same process returns 0. Most likely the probe is
still holding a handle the stub expects to own. Harmless, but it means the
stub's own self-test is not currently a clean signal — use the handle open.
