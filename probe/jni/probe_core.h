#ifndef PROBE_CORE_H
#define PROBE_CORE_H

// Where a line of output goes differs by host: stdout for the CLI build,
// logcat plus a returned string for the app build.
typedef void (*probe_emit)(void *ctx, const char *fmt, ...);

// dsp_lib_dir holds the Hexagon skels; it becomes ADSP_LIBRARY_PATH so the
// DSP-side loader can find them. NULL skips the skel test.
void probe_run(probe_emit emit, void *ctx, const char *dsp_lib_dir);

#endif
