#!/usr/bin/env bash
# Builds the NSS inference server for the device. ORT and QNN come from the
# same artefacts the probe app bundles, so both paths run identical code.
set -euo pipefail

SDK=${ANDROID_SDK:-$HOME/Library/Android/sdk}
NDK=${ANDROID_NDK:-$SDK/ndk/27.3.13750724}
QAIRT=${QAIRT_SDK:-$HOME/Downloads/qairt/2.32.6.250402}
API=29

here=$(cd "$(dirname "$0")" && pwd)
root=$here/..
out=$here/build
host=$(uname | tr '[:upper:]' '[:lower:]')-x86_64
CC=$NDK/toolchains/llvm/prebuilt/$host/bin/aarch64-linux-android$API-clang

rm -rf "$out"; mkdir -p "$out"
unzip -qo "$root/third_party/ort-qnn-1.21.1.aar" 'jni/arm64-v8a/*' -d "$out"
cp "$out"/jni/arm64-v8a/*.so "$out/"
cp "$QAIRT"/lib/aarch64-android/libQnn{Htp,System,HtpPrepare,HtpV79Stub}.so "$out/"
cp "$QAIRT"/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so "$out/"

$CC -O2 -Wall -o "$out/nss_server" "$here/nss_server.c" \
    -I"$root/third_party/ort-headers/headers" \
    -L"$out" -lonnxruntime -ldl -llog
echo "built $out/nss_server"
