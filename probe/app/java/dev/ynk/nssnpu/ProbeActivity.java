package dev.ynk.nssnpu;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStream;

/** Runs the FastRPC probe in a real untrusted_app process and reports what it found. */
public class ProbeActivity extends Activity {
    static { System.loadLibrary("nssprobe"); }

    private native String runProbe(String dspLibDir);

    /** QNN reads ADSP_LIBRARY_PATH from the environment, so it must be set in-process. */
    private native void setEnv(String name, String value);

    private File stageAsset(String asset, File dir) throws Exception {
        File out = new File(dir, new File(asset).getName());
        try (InputStream in = getAssets().open(asset);
             FileOutputStream os = new FileOutputStream(out)) {
            byte[] chunk = new byte[65536];
            int n;
            while ((n = in.read(chunk)) > 0) os.write(chunk, 0, n);
        }
        return out;
    }

    /** The Hexagon skels ride along as assets; the DSP loader needs them on disk. */
    private File stageDspLibs() throws Exception {
        File dir = new File(getFilesDir(), "dsp");
        dir.mkdirs();
        for (String name : getAssets().list("dsp")) {
            File out = new File(dir, name);
            try (InputStream in = getAssets().open("dsp/" + name);
                 FileOutputStream os = new FileOutputStream(out)) {
                byte[] chunk = new byte[65536];
                int n;
                while ((n = in.read(chunk)) > 0) os.write(chunk, 0, n);
            }
            Log.i("nssprobe", "staged " + out + " (" + out.length() + " bytes)");
        }
        return dir;
    }

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);

        String report;
        try {
            File dspDir = stageDspLibs();
            setEnv("ADSP_LIBRARY_PATH", dspDir.getAbsolutePath());
            report = runProbe(dspDir.getAbsolutePath());

            File[] models = {
                stageAsset("superres_fixed.onnx", getFilesDir()),
                stageAsset("nss_v1_high_544x960.onnx", getFilesDir()),
                stageAsset("nss_v1_high_544x960_int8.onnx", getFilesDir()),
            };
            File libDir = new File(getApplicationInfo().nativeLibraryDir);
            report += "\n" + OrtBenchmark.run(models, libDir);
            for (String line : report.split("\n")) Log.i("nssprobe", line);
        } catch (Throwable t) {
            report = "probe threw: " + t;
        }

        Log.i("nssprobe", "---- probe complete ----");

        // Also drop it on shared storage so adb can read it without logcat filtering.
        try {
            File out = new File(getExternalFilesDir(null), "probe.txt");
            FileWriter w = new FileWriter(out);
            w.write(report);
            w.close();
            Log.i("nssprobe", "wrote " + out);
        } catch (Exception e) {
            Log.w("nssprobe", "could not write report: " + e);
        }

        TextView text = new TextView(this);
        text.setTextSize(11);
        text.setText(report);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(text);
        setContentView(scroll);
    }
}
