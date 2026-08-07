package dev.ynk.nssnpu;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/** Runs models on the NPU and on the CPU so the numbers are comparable. */
final class OrtBenchmark {

    private static final int WARMUP = 3;
    private static final int RUNS = 20;

    private OrtBenchmark() {}

    static String run(File[] models, File nativeLibDir) {
        StringBuilder out = new StringBuilder();
        // Verbose so ORT logs its node placements -- the only way to tell whether
        // the QNN EP actually took the graph or quietly handed it back to the CPU.
        OrtEnvironment env = OrtEnvironment.getEnvironment(
                OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE, "nssprobe");

        out.append("[onnxruntime ").append(env.getVersion()).append("]\n");

        Map<String, String> qnn = new HashMap<>();
        qnn.put("backend_path", new File(nativeLibDir, "libQnnHtp.so").getAbsolutePath());
        qnn.put("enable_htp_fp16_precision", "1");
        // No skip_qnn_version_check here on purpose. ORT 1.21.x is built
        // against a QNN API minor that QAIRT 2.32.6 satisfies, so the check
        // passes honestly. Forcing a newer ORT past it gets a valid interface
        // and then dies in CreateContext with QNN_CONTEXT_ERROR_INVALID_CONFIG.

        for (File model : models) {
            if (!model.exists()) continue;
            out.append(model.getName()).append('\n');
            time(env, model, qnn, "QNN / HTP (NPU)", out);
            time(env, model, null, "CPU", out);
        }
        return out.toString();
    }

    private static void time(OrtEnvironment env, File model, Map<String, String> qnn,
                             String label, StringBuilder out) {
        long build = System.nanoTime();
        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            if (qnn != null) opts.addQnn(qnn);
            opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
            opts.setSessionLogVerbosityLevel(1);

            try (OrtSession session = env.createSession(model.getAbsolutePath(), opts)) {
                long buildMs = (System.nanoTime() - build) / 1_000_000;

                // Take the shape from the model rather than hardcoding it, so this
                // works for the super-resolution smoke test and for NSS alike.
                String inputName = session.getInputNames().iterator().next();
                NodeInfo info = session.getInputInfo().get(inputName);
                long[] shape = ((TensorInfo) info.getInfo()).getShape();

                int count = 1;
                for (long d : shape) count *= (int) d;
                FloatBuffer buffer = FloatBuffer.allocate(count);
                for (int i = 0; i < count; i++) buffer.put(i, (i % 255) / 255.0f);

                try (OnnxTensor input = OnnxTensor.createTensor(env, buffer, shape)) {
                    Map<String, OnnxTensor> feed = Collections.singletonMap(inputName, input);

                    for (int i = 0; i < WARMUP; i++) session.run(feed).close();

                    long best = Long.MAX_VALUE, total = 0;
                    for (int i = 0; i < RUNS; i++) {
                        long t = System.nanoTime();
                        session.run(feed).close();
                        long dt = System.nanoTime() - t;
                        total += dt;
                        if (dt < best) best = dt;
                    }
                    out.append(String.format(
                            "  %-16s session %5d ms   mean %7.2f ms   best %7.2f ms%n",
                            label, buildMs, total / (double) RUNS / 1e6, best / 1e6));
                }
            }
        } catch (Throwable t) {
            out.append("  ").append(label).append(" FAILED: ").append(t).append('\n');
        }
    }
}
