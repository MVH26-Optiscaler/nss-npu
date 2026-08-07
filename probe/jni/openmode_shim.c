// Two jobs, both aimed at the same question: how far into the FastRPC stack
// does a non-system process actually get?
//
//  1. The node is mode 0664 system:system, so anyone else can only open it
//     read-only. FastRPC drives everything through ioctl(), which does not
//     need write access, so downgrade the mode and see if the rest works.
//  2. Trace every open and ioctl that touches a fastrpc path, so a failure
//     inside libcdsprpc can be attributed to the driver rather than guessed at.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>

#define MAX_TRACKED 16

static int  tracked[MAX_TRACKED];
static int  tracked_count;
static int  trace_on = -1;

static int tracing(void) {
    if (trace_on < 0) trace_on = getenv("FASTRPC_TRACE") != NULL;
    return trace_on;
}

static int is_fastrpc(const char *path) {
    return path && strstr(path, "fastrpc") != NULL;
}

static void track(int fd) {
    if (fd >= 0 && tracked_count < MAX_TRACKED) tracked[tracked_count++] = fd;
}

static int is_tracked(int fd) {
    for (int i = 0; i < tracked_count; i++)
        if (tracked[i] == fd) return 1;
    return 0;
}

// libcdsprpc looks for the DSP-side shell in a hardcoded list -- /usr/lib/dsp
// (which does not exist on Android) and /vendor/dsp (which the shell domain
// cannot read). ADSP_LIBRARY_PATH does not affect this lookup; it only tells
// the DSP loader where to find skels. So redirect the open instead: point
// FASTRPC_DSP_DIR at a directory holding a copy of fastrpc_shell_3.
static const char *redirect(const char *path, char *out, size_t out_len) {
    const char *dir = getenv("FASTRPC_DSP_DIR");
    const char *base;

    if (!dir || !path) return path;
    if (strncmp(path, "/usr/lib/dsp/", 13) != 0 && strncmp(path, "/vendor/dsp/", 12) != 0)
        return path;

    base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if ((size_t)snprintf(out, out_len, "%s/%s", dir, base) >= out_len) return path;
    if (tracing()) fprintf(stderr, "[shim] redirect %s -> %s\n", path, out);
    return out;
}

static int downgrade(const char *path, int flags) {
    if (is_fastrpc(path) && (flags & O_ACCMODE) != O_RDONLY) {
        if (tracing()) fprintf(stderr, "[shim] %s: O_RDWR -> O_RDONLY\n", path);
        flags = (flags & ~O_ACCMODE) | O_RDONLY;
    }
    return flags;
}

static void report_open(const char *fn, const char *path, int fd) {
    if (!tracing() || !is_fastrpc(path)) return;
    if (fd >= 0) fprintf(stderr, "[shim] %s(%s) = fd %d\n", fn, path, fd);
    else fprintf(stderr, "[shim] %s(%s) = FAIL errno=%d (%s)\n",
                 fn, path, errno, strerror(errno));
}

int open(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...);
    char buf[PATH_MAX];
    mode_t mode = 0;
    int fd;
    if (!real) real = dlsym(RTLD_NEXT, "open");
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); }
    fd = real(redirect(path, buf, sizeof buf), downgrade(path, flags), mode);
    if (is_fastrpc(path)) { report_open("open", path, fd); track(fd); }
    return fd;
}

int open64(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...);
    char buf[PATH_MAX];
    mode_t mode = 0;
    int fd;
    if (!real) real = dlsym(RTLD_NEXT, "open64");
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); }
    fd = real(redirect(path, buf, sizeof buf), downgrade(path, flags), mode);
    if (is_fastrpc(path)) { report_open("open64", path, fd); track(fd); }
    return fd;
}

int openat(int dirfd, const char *path, int flags, ...) {
    static int (*real)(int, const char *, int, ...);
    char buf[PATH_MAX];
    mode_t mode = 0;
    int fd;
    if (!real) real = dlsym(RTLD_NEXT, "openat");
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); }
    fd = real(dirfd, redirect(path, buf, sizeof buf), downgrade(path, flags), mode);
    if (is_fastrpc(path)) { report_open("openat", path, fd); track(fd); }
    return fd;
}

int ioctl(int fd, int request, ...) {
    static int (*real)(int, int, ...);
    va_list ap;
    void *arg;
    int r;

    if (!real) real = dlsym(RTLD_NEXT, "ioctl");
    va_start(ap, request);
    arg = va_arg(ap, void *);
    va_end(ap);

    r = real(fd, request, arg);
    if (tracing() && is_tracked(fd)) {
        if (r < 0) fprintf(stderr, "[shim] ioctl(fd %d, 0x%x) = %d errno=%d (%s)\n",
                           fd, request, r, errno, strerror(errno));
        else fprintf(stderr, "[shim] ioctl(fd %d, 0x%x) = %d\n", fd, request, r);
    }
    return r;
}
