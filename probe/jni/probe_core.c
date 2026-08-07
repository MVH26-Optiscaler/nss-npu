// Whether this process can reach the Hexagon NPU.
//
// The interesting part is not the answer for one process but the difference
// between processes: shell, runas_app and untrusted_app are separate SELinux
// domains, and only the last one is what a real app gets. Run the same code
// in each and compare.
#include "probe_core.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef uint64_t remote_handle64;
typedef int (*fn_handle_open)(const char *name, remote_handle64 *ph);
typedef int (*fn_session_control)(uint32_t req, void *data, uint32_t len);

// From qualcomm/fastrpc inc/remote.h: enum session_control_req_id. Unsigned
// process domains are what let code without OEM signing keys onto the DSP,
// and this must be enabled before the first handle open.
#define DSPRPC_CONTROL_UNSIGNED_MODULE 2
#define DOMAIN_CDSP 3

struct unsigned_module { int domain; int enable; };

static const char *const kNodes[] = {
    "/dev/fastrpc-cdsp",
    "/dev/fastrpc-cdsp-secure",
    "/dev/fastrpc-adsp-secure",
};

static void probe_node(probe_emit emit, void *ctx, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd >= 0) { emit(ctx, "  %s  O_RDONLY OK", path); close(fd); }
    else emit(ctx, "  %s  O_RDONLY FAIL errno=%d (%s)", path, errno, strerror(errno));

    fd = open(path, O_RDWR);
    if (fd >= 0) { emit(ctx, "  %s  O_RDWR   OK", path); close(fd); }
    else emit(ctx, "  %s  O_RDWR   FAIL errno=%d (%s)", path, errno, strerror(errno));
}

void probe_run(probe_emit emit, void *ctx, const char *dsp_lib_dir) {
    char domain[128] = "?";
    int fd = open("/proc/self/attr/current", O_RDONLY);
    if (fd >= 0) {
        ssize_t n = read(fd, domain, sizeof domain - 1);
        if (n > 0) domain[n] = 0;
        close(fd);
    }
    emit(ctx, "uid=%d  selinux=%s", getuid(), domain);

    emit(ctx, "[device nodes]");
    for (size_t i = 0; i < sizeof kNodes / sizeof *kNodes; i++)
        probe_node(emit, ctx, kNodes[i]);

    emit(ctx, "[libcdsprpc]");
    void *h = dlopen("libcdsprpc.so", RTLD_NOW);
    if (!h) h = dlopen("/vendor/lib64/libcdsprpc.so", RTLD_NOW);
    if (!h) { emit(ctx, "  dlopen FAIL: %s", dlerror()); return; }
    emit(ctx, "  dlopen OK");

    fn_session_control session_control = (fn_session_control)dlsym(h, "remote_session_control");
    if (session_control) {
        struct unsigned_module um = { .domain = DOMAIN_CDSP, .enable = 1 };
        int r = session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
        emit(ctx, "  remote_session_control(unsigned pd, cdsp) = 0x%x%s",
             r, r == 0 ? " (unsigned PD enabled)" : "");
    } else {
        emit(ctx, "  remote_session_control not exported");
    }

    fn_handle_open handle_open = (fn_handle_open)dlsym(h, "remote_handle64_open");
    if (!handle_open) {
        emit(ctx, "  remote_handle64_open not exported");
        return;
    }

    if (!dsp_lib_dir) return;
    setenv("ADSP_LIBRARY_PATH", dsp_lib_dir, 1);
    emit(ctx, "  ADSP_LIBRARY_PATH=%s", dsp_lib_dir);

    // URI taken verbatim from the strings of libQnnHtpV79CalculatorStub.so --
    // it is case sensitive, and "Calculator_skel_handle_invoke" is not the
    // lowercase spelling the IDL convention would suggest.
    {
        remote_handle64 handle = 0;
        int r = handle_open("file:///libCalculator_skel.so?Calculator_skel_handle_invoke"
                            "&_modver=1.0&_dom=cdsp", &handle);
        emit(ctx, "  remote_handle64_open(Calculator) = 0x%x  handle=%llu",
             r, (unsigned long long)handle);
        emit(ctx, "  %s", r == 0 ? "*** DSP SESSION OPENED -- our code is on the NPU ***"
                                 : "no session (0x80000600 = could not create one)");
    }

    // Deliberately not calling Qnn_calculatorTest from the stub: its signature
    // is not published, and calling it as int(void) segfaults inside the stub.
    // The handle open above already proves the session, so it buys nothing.
}
