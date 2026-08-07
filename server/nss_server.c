// NSS inference on the Hexagon NPU, reachable over a socket.
//
// The XeSS shim is PE code running under Proton; ONNX Runtime and QNN are
// bionic ELF. Rather than bridge the two in-process (which needs a wine
// unixlib built against the Proton tree), the inference runs here and the shim
// talks to it over loopback. The protocol is deliberately dumb so the same
// wire format survives a later move in-process.
//
//   --bench            load the model, run it on a raw input, report timing
//   --serve [port]     accept frames on 127.0.0.1 and answer with the outputs
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "onnxruntime_c_api.h"

#define DEFAULT_PORT 47800
#define NSS_MAGIC 0x3153534eu /* "NSS1" */

static const OrtApi *ort;

/* Shape and element type come from the model, so this benchmarks any
 * single-input ONNX graph on the NPU rather than only NSS. */
static int64_t in_shape[8];
static size_t in_rank;
static ONNXTensorElementDataType in_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
static size_t in_elem_size = sizeof(float);

// NSS v1 "high": 960x540 render padded to 544, 12 packed channels in.
enum { IN_N = 1, IN_C = 12, IN_H = 544, IN_W = 960 };

struct header {
    uint32_t magic;
    uint32_t bytes;
};

static void die(const char *what, OrtStatus *status) {
    if (status) {
        fprintf(stderr, "%s: %s\n", what, ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
    } else {
        fprintf(stderr, "%s: %s\n", what, strerror(errno));
    }
    exit(1);
}

#define CHECK(call)                                        \
    do {                                                   \
        OrtStatus *_s = (call);                            \
        if (_s) die(#call, _s);                            \
    } while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

struct model {
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *mem;
    char *input_name;
    char *output_names[2];
    size_t output_count;
    size_t input_elems;
};

static void model_open(struct model *m, const char *path, const char *backend) {
    OrtSessionOptions *opts;

    CHECK(ort->CreateEnv(getenv("NSS_ORT_VERBOSE") ? ORT_LOGGING_LEVEL_VERBOSE
                                                   : ORT_LOGGING_LEVEL_WARNING,
                         "nss", &m->env));
    CHECK(ort->CreateSessionOptions(&opts));

    if (backend) {
        // QNN picks the accelerator by backend library: libQnnHtp.so is the
        // NPU, libQnnGpu.so the Adreno, libQnnCpu.so a reference path.
        /* Same settings the in-game bridge uses, so numbers are comparable:
         * a sustained DSP clock, and fp16 for graphs that are not quantised. */
        const char *keys[] = { "backend_path", "htp_performance_mode",
                               "enable_htp_fp16_precision" };
        const char *values[] = { backend, "sustained_high_performance", "1" };
        CHECK(ort->SessionOptionsAppendExecutionProvider(opts, "QNN", keys, values, 3));
    }

    CHECK(ort->CreateSession(m->env, path, opts, &m->session));
    ort->ReleaseSessionOptions(opts);

    OrtAllocator *alloc;
    CHECK(ort->GetAllocatorWithDefaultOptions(&alloc));
    CHECK(ort->SessionGetInputName(m->session, 0, alloc, &m->input_name));
    CHECK(ort->SessionGetOutputName(m->session, 0, alloc, &m->output_names[0]));
    {
        size_t out_count = 0;
        CHECK(ort->SessionGetOutputCount(m->session, &out_count));
        m->output_count = out_count > 2 ? 2 : (out_count ? out_count : 1);
        if (m->output_count > 1)
            CHECK(ort->SessionGetOutputName(m->session, 1, alloc, &m->output_names[1]));
    }
    CHECK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m->mem));

    {
        OrtTypeInfo *info = NULL;
        const OrtTensorTypeAndShapeInfo *shape_info = NULL;
        CHECK(ort->SessionGetInputTypeInfo(m->session, 0, &info));
        CHECK(ort->CastTypeInfoToTensorInfo(info, &shape_info));
        CHECK(ort->GetDimensionsCount(shape_info, &in_rank));
        if (in_rank > 8) in_rank = 8;
        CHECK(ort->GetDimensions(shape_info, in_shape, in_rank));
        CHECK(ort->GetTensorShapeElementCount(shape_info, &m->input_elems));
        CHECK(ort->GetTensorElementType(shape_info, &in_type));
        ort->ReleaseTypeInfo(info);
        in_elem_size = (in_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
                        in_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) ? 1 : sizeof(float);
    }

    printf("model %s\n  input  %s [", path, m->input_name);
    for (size_t i = 0; i < in_rank; i++)
        printf("%s%lld", i ? "," : "", (long long)in_shape[i]);
    printf("]  %s\n  outputs %s%s%s\n",
           in_elem_size == 1 ? "int8" : "float", m->output_names[0],
           m->output_names[1] ? ", " : "", m->output_names[1] ? m->output_names[1] : "");
}

/** Runs one frame. Caller owns `input`; outputs are returned as ORT values. */
static void model_run(struct model *m, float *input, OrtValue **outputs) {
    OrtValue *in = NULL;

    CHECK(ort->CreateTensorWithDataAsOrtValue(
            m->mem, input, m->input_elems * in_elem_size,
            in_shape, in_rank, in_type, &in));

    outputs[0] = outputs[1] = NULL;
    CHECK(ort->Run(m->session, NULL,
                   (const char *const *)&m->input_name, (const OrtValue *const *)&in, 1,
                   (const char *const *)m->output_names, m->output_count, outputs));
    ort->ReleaseValue(in);
}

static void model_close(struct model *m) {
    ort->ReleaseMemoryInfo(m->mem);
    ort->ReleaseSession(m->session);
    ort->ReleaseEnv(m->env);
}


static size_t value_bytes(OrtValue *v) {
    OrtTensorTypeAndShapeInfo *info;
    size_t count;
    CHECK(ort->GetTensorTypeAndShape(v, &info));
    CHECK(ort->GetTensorShapeElementCount(info, &count));
    ort->ReleaseTensorTypeAndShapeInfo(info);
    return count * sizeof(float);
}

static int bench(const char *model_path, const char *backend, const char *input_path,
                 const char *dump_prefix, int runs) {
    struct model m;
    float *input;
    FILE *f;

    model_open(&m, model_path, backend);

    input = malloc(m.input_elems * in_elem_size);
    if (!input) die("malloc", NULL);
    if (!(f = fopen(input_path, "rb"))) die(input_path, NULL);
    if (fread(input, in_elem_size, m.input_elems, f) != m.input_elems)
        die("short read on input", NULL);
    fclose(f);

    OrtValue *out[2];
    for (int i = 0; i < 3; i++) {           /* warm up graph + allocations */
        model_run(&m, input, out);
        for (size_t k = 0; k < m.output_count; k++) ort->ReleaseValue(out[k]);
    }

    double best = 1e9, total = 0;
    for (int i = 0; i < runs; i++) {
        double t = now_ms();
        model_run(&m, input, out);
        double dt = now_ms() - t;
        total += dt;
        if (dt < best) best = dt;
        if (i + 1 < runs)
            for (size_t k = 0; k < m.output_count; k++) ort->ReleaseValue(out[k]);
    }
    printf("%-28s mean %7.2f ms   best %7.2f ms   (%d runs)\n",
           backend ? backend : "CPU", total / runs, best, runs);

    if (dump_prefix) {
        for (size_t i = 0; i < m.output_count; i++) {
            char path[512];
            void *data;
            snprintf(path, sizeof path, "%s_%d.f32", dump_prefix, i);
            CHECK(ort->GetTensorMutableData(out[i], &data));
            f = fopen(path, "wb");
            if (!f) die(path, NULL);
            fwrite(data, 1, value_bytes(out[i]), f);
            fclose(f);
            printf("  wrote %s (%zu bytes)\n", path, value_bytes(out[i]));
        }
    }
    for (size_t k = 0; k < m.output_count; k++) ort->ReleaseValue(out[k]);
    free(input);
    model_close(&m);
    return 0;
}

static int read_exactly(int fd, void *buf, size_t len) {
    uint8_t *p = buf;
    while (len) {
        ssize_t n = read(fd, p, len);
        if (n <= 0) return -1;
        p += n;
        len -= n;
    }
    return 0;
}

static int write_exactly(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    while (len) {
        ssize_t n = write(fd, p, len);
        if (n <= 0) return -1;
        p += n;
        len -= n;
    }
    return 0;
}

static int serve(const char *model_path, const char *backend, int port) {
    struct model m;
    struct sockaddr_in addr = { 0 };
    int sock, one = 1;

    model_open(&m, model_path, backend);

    float *input = malloc(m.input_elems * sizeof(float));
    if (!input) die("malloc", NULL);

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) die("socket", NULL);
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (bind(sock, (struct sockaddr *)&addr, sizeof addr) < 0) die("bind", NULL);
    if (listen(sock, 4) < 0) die("listen", NULL);
    printf("listening on 127.0.0.1:%d\n", port);
    fflush(stdout);

    for (;;) {
        int client = accept(sock, NULL, NULL);
        if (client < 0) continue;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        printf("client connected\n");
        fflush(stdout);

        for (;;) {
            struct header h;
            OrtValue *out[2];
            double t;

            if (read_exactly(client, &h, sizeof h) < 0) break;
            if (h.magic != NSS_MAGIC || h.bytes != m.input_elems * sizeof(float)) {
                fprintf(stderr, "bad header magic=%08x bytes=%u\n", h.magic, h.bytes);
                break;
            }
            if (read_exactly(client, input, h.bytes) < 0) break;

            t = now_ms();
            model_run(&m, input, out);
            double infer = now_ms() - t;

            int failed = 0;
            for (int i = 0; i < 2 && !failed; i++) {
                void *data;
                size_t bytes = value_bytes(out[i]);
                struct header oh = { NSS_MAGIC, (uint32_t)bytes };
                CHECK(ort->GetTensorMutableData(out[i], &data));
                failed = write_exactly(client, &oh, sizeof oh) < 0 ||
                         write_exactly(client, data, bytes) < 0;
            }
            ort->ReleaseValue(out[0]);
            ort->ReleaseValue(out[1]);
            printf("frame: %.2f ms inference\n", infer);
            fflush(stdout);
            if (failed) break;
        }
        close(client);
        printf("client gone\n");
        fflush(stdout);
    }
}

static void usage(void) {
    fprintf(stderr,
            "usage: nss_server --model M [--backend libQnnHtp.so] [--cpu]\n"
            "                  (--bench INPUT [--dump PREFIX] [--runs N] | --serve [PORT])\n");
    exit(2);
}

int main(int argc, char **argv) {
    const char *model = NULL, *backend = "libQnnHtp.so", *input = NULL, *dump = NULL;
    int mode_bench = 0, port = DEFAULT_PORT, runs = 20;

    // adb shell is not a tty, so stdout would be block-buffered and lost if
    // anything aborts. Progress messages are useless in that case.
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) {
        fprintf(stderr, "ORT API version %d not available\n", ORT_API_VERSION);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else if (!strcmp(argv[i], "--cpu")) backend = NULL;
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) { mode_bench = 1; input = argv[++i]; }
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--serve")) { if (i + 1 < argc && argv[i + 1][0] != '-') port = atoi(argv[++i]); }
        else usage();
    }
    if (!model) usage();

    int rc = mode_bench ? bench(model, backend, input, dump, runs)
                        : serve(model, backend, port);

    // ORT's static destructors run at exit and touch an already-destroyed
    // mutex on this platform, aborting a run that has otherwise completed and
    // printed its results. Everything of ours is released above, so leave
    // without unwinding the C++ runtime.
    fflush(NULL);
    _exit(rc);
}
