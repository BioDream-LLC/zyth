// Runs the Zyth program packed into this app (lib/arm64-v8a/libzythapp.so: a `zyth build`
// executable, which Android installs as an executable file) and shows what it printed.
package ai.biodream.zythdemo;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ScrollView;
import android.widget.TextView;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        TextView text = new TextView(this);
        text.setTypeface(android.graphics.Typeface.MONOSPACE);
        text.setPadding(32, 64, 32, 32);
        text.setContentDescription("Program output");
        ScrollView scroll = new ScrollView(this);
        scroll.addView(text);
        setContentView(scroll);
        new Thread(() -> {
            String out;
            try {
                String exe = getApplicationInfo().nativeLibraryDir + "/libzythapp.so";
                Process p = new ProcessBuilder(exe).redirectErrorStream(true).directory(getFilesDir()).start();
                InputStream in = p.getInputStream();
                ByteArrayOutputStream buf = new ByteArrayOutputStream();
                byte[] b = new byte[8192];
                int n;
                while ((n = in.read(b)) > 0) buf.write(b, 0, n);
                int code = p.waitFor();
                out = buf.toString("UTF-8") + (code == 0 ? "" : "\n(exit status " + code + ")");
            } catch (Exception e) {
                out = "cannot run the program: " + e;
            }
            for (String line : out.split("\n")) Log.i("ZythDemo", line);
            final String shown = out;
            runOnUiThread(() -> text.setText(shown));
        }).start();
    }
}
