// CLI build -- run from adb shell (domain u:r:shell:s0).
#include "probe_core.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static void emit(void *ctx, const char *fmt, ...) {
    (void)ctx;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
}

int main(void) {
    const char *dir = getenv("ADSP_LIBRARY_PATH");
    probe_run(emit, NULL, dir ? dir : "/data/local/tmp/qnn");
    return 0;
}
