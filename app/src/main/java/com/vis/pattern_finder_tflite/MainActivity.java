package com.vis.pattern_finder_tflite;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {


    private Interpreter interpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }
        // dataset trained to find the pattern y=2x-1
        // value entered will be the x
        //output predicted is y
        // x is feature y is label
        final TextView result = findViewById(R.id.textView);
        final EditText editText = findViewById(R.id.editText);
        Button run = findViewById(R.id.button);

        run.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                float f = doInference(editText.getText().toString());
                result.setText(f + "");
            }
        });
    }

    public float doInference(String val) {
        float[] input = new float[1];
        input[0] = Float.parseFloat(val);

        float[][] output = new float[1][1];
        interpreter.run(input, output);
        return output[0][0];
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd("linear.tflite");
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long length = assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
    }


}

