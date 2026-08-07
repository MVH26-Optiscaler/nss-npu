// App build -- runs in the untrusted_app domain, which is the one that
// actually decides whether an app can drive the NPU.
#include "probe_core.h"

#include <android/log.h>
#include <jni.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG "nssprobe"

struct sink {
    char  *buf;
    size_t len;
    size_t cap;
};

static void emit(void *ctx, const char *fmt, ...) {
    struct sink *s = ctx;
    char line[512];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);

    __android_log_write(ANDROID_LOG_INFO, TAG, line);

    size_t n = strlen(line);
    if (s->len + n + 2 < s->cap) {
        memcpy(s->buf + s->len, line, n);
        s->len += n;
        s->buf[s->len++] = '\n';
        s->buf[s->len] = 0;
    }
}

JNIEXPORT jstring JNICALL
Java_dev_ynk_nssnpu_ProbeActivity_runProbe(JNIEnv *env, jobject self, jstring dspDir) {
    (void)self;
    const char *dir = dspDir ? (*env)->GetStringUTFChars(env, dspDir, NULL) : NULL;
    char buf[16384];
    struct sink s = { .buf = buf, .len = 0, .cap = sizeof buf };
    buf[0] = 0;
    probe_run(emit, &s, dir);
    if (dir) (*env)->ReleaseStringUTFChars(env, dspDir, dir);
    return (*env)->NewStringUTF(env, buf);
}

JNIEXPORT void JNICALL
Java_dev_ynk_nssnpu_ProbeActivity_setEnv(JNIEnv *env, jobject self,
                                         jstring name, jstring value) {
    const char *n = (*env)->GetStringUTFChars(env, name, NULL);
    const char *v = (*env)->GetStringUTFChars(env, value, NULL);
    (void)self;
    setenv(n, v, 1);
    __android_log_print(ANDROID_LOG_INFO, TAG, "setenv %s=%s", n, v);
    (*env)->ReleaseStringUTFChars(env, name, n);
    (*env)->ReleaseStringUTFChars(env, value, v);
}
