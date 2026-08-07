#!/usr/bin/env bash
# Builds two copies of the same probe: a CLI binary for the shell domain and
# an APK for the untrusted_app domain. Gradle is deliberately not involved --
# this is aapt2 + javac + d8 + apksigner, so it works offline.
set -euo pipefail

SDK=${ANDROID_SDK:-$HOME/Library/Android/sdk}
NDK=${ANDROID_NDK:-$SDK/ndk/27.3.13750724}
BUILD_TOOLS=$SDK/build-tools/36.0.0
PLATFORM=$SDK/platforms/android-36/android.jar
API=29

here=$(cd "$(dirname "$0")" && pwd)
out=$here/build
host=$(uname | tr '[:upper:]' '[:lower:]')-x86_64
CC=$NDK/toolchains/llvm/prebuilt/$host/bin/aarch64-linux-android$API-clang

rm -rf "$out"
mkdir -p "$out/lib/arm64-v8a" "$out/classes" "$out/dex" "$out/res" "$out/assets/dsp"

echo "== native =="
$CC -O2 -Wall -o "$out/nssprobe" \
    "$here/jni/probe_core.c" "$here/jni/probe_main.c" -ldl
$CC -O2 -Wall -shared -fPIC -o "$out/lib/arm64-v8a/libnssprobe.so" \
    "$here/jni/probe_core.c" "$here/jni/probe_jni.c" -ldl -llog
$CC -O2 -Wall -shared -fPIC -o "$out/openmode_shim.so" \
    "$here/jni/openmode_shim.c" -ldl

# The QAIRT SDK supplies the Hexagon-side skel and its aarch64 stub. Without
# them the app can load libcdsprpc but has nothing to run on the DSP.
QAIRT=${QAIRT_SDK:-$HOME/Downloads/qairt/2.32.6.250402}
if [ -d "$QAIRT" ]; then
  A=$QAIRT/lib/aarch64-android
  H=$QAIRT/lib/hexagon-v79/unsigned
  # Hexagon-side skels are searched via ADSP_LIBRARY_PATH, so they ship as
  # assets and get staged to filesDir. The aarch64 halves are ordinary libs.
  cp "$H/libCalculator_skel.so" "$H/libQnnHtpV79Skel.so" "$out/assets/dsp/"
  cp "$A/libQnnHtpV79CalculatorStub.so" "$A/libQnnHtp.so" "$A/libQnnSystem.so" \
     "$A/libQnnHtpPrepare.so" "$A/libQnnHtpV79Stub.so" "$out/lib/arm64-v8a/"
  echo "   bundled QAIRT runtime (HTP V79) + calculator skel"
else
  echo "   QAIRT SDK not found at $QAIRT -- DSP tests will be skipped"
fi

# ONNX Runtime with the QNN EP. The AAR ships only libonnxruntime.so and the
# JNI shim; the QNN backends above are what it dlopens at session creation.
# ORT and QAIRT versions are a matched pair -- see README.
AAR=${ORT_AAR:-$here/../third_party/ort-qnn-1.21.1.aar}
ORT=$out/ort
if [ -f "$AAR" ]; then
  mkdir -p "$ORT"
  unzip -qo "$AAR" -d "$ORT"
  cp "$ORT"/jni/arm64-v8a/*.so "$out/lib/arm64-v8a/"
  echo "   bundled onnxruntime from $(basename "$AAR")"
else
  echo "   ORT AAR not found at $AAR"
fi

for m in superres_fixed nss_v1_high_544x960 nss_v1_high_544x960_int8; do
  cp "$here/../models/$m.onnx" "$out/assets/" 2>/dev/null \
    && echo "   bundled $m.onnx"
done

echo "== java =="
javac -source 8 -target 8 -nowarn -bootclasspath "$PLATFORM" \
      -classpath "$ORT/classes.jar" \
      -d "$out/classes" $(find "$here/app/java" -name '*.java')
"$BUILD_TOOLS/d8" --min-api $API --lib "$PLATFORM" --output "$out/dex" \
      "$ORT/classes.jar" $(find "$out/classes" -name '*.class')

echo "== package =="
"$BUILD_TOOLS/aapt2" link -I "$PLATFORM" \
      --manifest "$here/app/AndroidManifest.xml" \
      --min-sdk-version $API --target-sdk-version 36 \
      -A "$out/assets" \
      -o "$out/base.apk"

cd "$out"
cp dex/classes.dex .
zip -q base.apk classes.dex
zip -qr base.apk lib/arm64-v8a

keystore=$here/debug.keystore
if [ ! -f "$keystore" ]; then
  keytool -genkeypair -keystore "$keystore" -storepass android -keypass android \
          -alias probe -keyalg RSA -keysize 2048 -validity 10000 \
          -dname "CN=nss-npu probe" >/dev/null 2>&1
fi

"$BUILD_TOOLS/zipalign" -f 4 base.apk aligned.apk
"$BUILD_TOOLS/apksigner" sign --ks "$keystore" --ks-pass pass:android \
          --key-pass pass:android --out "$out/nssprobe.apk" aligned.apk
rm -f aligned.apk base.apk classes.dex

echo
echo "built:"
echo "  $out/nssprobe        (adb push /data/local/tmp)"
echo "  $out/nssprobe.apk    (adb install -r)"
