package com.irhammuch.android.facerecognition;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.Image;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.speech.tts.TextToSpeech;
import android.text.InputType;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.bumptech.glide.Glide;
import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.firebase.firestore.DocumentReference;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.storage.FileDownloadTask;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.StreamDownloadTask;
import com.google.firebase.storage.UploadTask;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import android.speech.tts.TextToSpeech;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int PERMISSION_CODE = 1001;
    private static final String CAMERA_PERMISSION = Manifest.permission.CAMERA;
    private PreviewView previewView;
    private CameraSelector cameraSelector;
    private ProcessCameraProvider cameraProvider;
    private int lensFacing = CameraSelector.LENS_FACING_BACK;
    private Preview previewUseCase;
    private ImageAnalysis analysisUseCase;
    private GraphicOverlay graphicOverlay;
    private ImageView previewImg , gifimageview;
    private TextView detectionTextView;

    private ArrayList<JSONObject> jsonlist;
    private  JSONObject jsonObject;
    private FirebaseStorage db;
    private StorageReference storageRef;
    private final HashMap<String, SimilarityClassifier.Recognition> registered = new HashMap<>(); //saved Faces
    private Interpreter tfLite;
    private boolean flipX = false;
    private boolean start = true;
    private float[][] embeddings;

    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    private static final int INPUT_SIZE = 112;
    private static final int OUTPUT_SIZE=192;

    private TextToSpeech textToSpeech;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.previewView);
        previewView.setScaleType(PreviewView.ScaleType.FIT_CENTER);
        graphicOverlay = findViewById(R.id.graphic_overlay);
        previewImg = findViewById(R.id.preview_img);
        detectionTextView = findViewById(R.id.detection_text);


        gifimageview= findViewById(R.id.gifImageView);

        gifimageview.setVisibility(View.INVISIBLE);
        Glide.with(this).asGif().load(R.drawable.talking).into(gifimageview);

        jsonlist = new ArrayList<JSONObject>();
        db = FirebaseStorage.getInstance();
        storageRef = db.getReference().child("file.json");

        synccloud2();

        ImageButton addBtn = findViewById(R.id.add_btn);
        addBtn.setOnClickListener((v -> addFace()));

        ImageButton switchCamBtn = findViewById(R.id.switch_camera);
        switchCamBtn.setOnClickListener((view -> switchCamera()));


        textToSpeech = new TextToSpeech(this, null);
        textToSpeech.setLanguage(Locale.US);


        loadModel();


    }
    private void synccloud2(){
        FirebaseStorage storage = FirebaseStorage.getInstance();
        StorageReference storageRef = storage.getReferenceFromUrl("gs://face-recognizer-humans.appspot.com");
        StorageReference fileRef = storageRef.child("file.json");

    // Create a temporary file to store the downloaded JSON file
        File localFile;
        try {
            localFile = File.createTempFile("file", "json");
        } catch (IOException e) {
            Log.i(TAG, "sync cloud2: addFace "+e);
            return; // Error creating temporary file
        }

        fileRef.getFile(localFile)
                .addOnSuccessListener(new OnSuccessListener<FileDownloadTask.TaskSnapshot>() {
                    @Override
                    public void onSuccess(FileDownloadTask.TaskSnapshot taskSnapshot) {
                        // File downloaded successfully


                        try {

                            speakText("Welcome to Robo Receptionist");

                            BufferedReader reader = new BufferedReader(new FileReader(localFile));
                            String line;
                            StringBuilder stringBuilder = new StringBuilder();

                            while ((line = reader.readLine()) != null) {
                                stringBuilder.append(line);
                            }

                            reader.close();

                            // Parse the downloaded JSON file
                            JSONArray jsonArray = new JSONArray(stringBuilder.toString());

                            for (int i = 0; i < jsonArray.length(); i++) {
                                JSONObject jsonObject = jsonArray.getJSONObject(i);
                                jsonlist.add(jsonObject);
                                Log.i(TAG, "onSuccess: addFace-"+ jsonObject);


                                String k = jsonObject.keys().next();
                                Log.i(TAG, "onSuccess: addFace 4 key"+k+" val"+ jsonObject.getString(k));

                                SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                                        "0", "", -1f);


    // Remove the brackets [ and ] from the input string
                                String cleanedString = jsonObject.getString(k).substring(1,jsonObject.getString(k).length() - 1);

    // Split the string by comma to get individual float values
                                String[] floatStrings = cleanedString.split(",");

    // Create an array to store the float values
                                float[] floatArray = new float[floatStrings.length];

    // Convert each string value to a float and store it in the float array
                                for (int j = 0; j < floatStrings.length; j++) {
                                    floatArray[j] = Float.parseFloat(floatStrings[j]);
                                    //Log.i(TAG, "onSuccess: addFaces float"+floatArray[j]);
                                }
                                Log.i(TAG, "onSuccess: addFaces float"+floatArray);


                                float[][] embeddings = {floatArray};


                                result.setExtra(embeddings);


                                registered.put( k,result);

                                Log.i(TAG, "onSuccess: addFaces==>>> \n " + embeddings[0] + "\n \n" + jsonlist);
                            }

                            // Use the jsonList as needed
                            // ...

                        } catch (IOException | JSONException e) {
                            Log.i(TAG, "synccloud2:addFace "+e);
                            // Error parsing JSON or reading the file
                        }

                        // Delete the temporary file
                        localFile.delete();
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        // Error downloading the file
                        e.printStackTrace();
                    }
                });
}
    @Override
    protected void onResume() {
        super.onResume();
        startCamera();
    }

    /** Permissions Handler */
    private void getPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{CAMERA_PERMISSION}, PERMISSION_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, int[] grantResults) {
        for (int r : grantResults) {
            if (r == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
                return;
            }
        }

        if (requestCode == PERMISSION_CODE) {
            setupCamera();
        }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    /** Setup camera & use cases */
    private void startCamera() {
        if(ContextCompat.checkSelfPermission(this, CAMERA_PERMISSION) == PackageManager.PERMISSION_GRANTED) {
            setupCamera();
        } else {
            getPermissions();
        }
    }

    private void setupCamera() {
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraSelector = new CameraSelector.Builder().requireLensFacing(lensFacing).build();

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindAllCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "cameraProviderFuture.addListener Error", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindAllCameraUseCases() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
            bindPreviewUseCase();
            bindAnalysisUseCase();
        }
    }

    private void bindPreviewUseCase() {
        if (cameraProvider == null) {
            return;
        }

        if (previewUseCase != null) {
            cameraProvider.unbind(previewUseCase);
        }

        Preview.Builder builder = new Preview.Builder();
        builder.setTargetAspectRatio(AspectRatio.RATIO_4_3);
        builder.setTargetRotation(getRotation());

        previewUseCase = builder.build();
        previewUseCase.setSurfaceProvider(previewView.getSurfaceProvider());

        try {
            cameraProvider
                    .bindToLifecycle(this, cameraSelector, previewUseCase);
        } catch (Exception e) {
            Log.e(TAG, "Error when bind preview", e);
        }
    }

    private void bindAnalysisUseCase() {
        if (cameraProvider == null) {
            return;
        }

        if (analysisUseCase != null) {
            cameraProvider.unbind(analysisUseCase);
        }

        Executor cameraExecutor = Executors.newSingleThreadExecutor();

        ImageAnalysis.Builder builder = new ImageAnalysis.Builder();
        builder.setTargetAspectRatio(AspectRatio.RATIO_4_3);
        builder.setTargetRotation(getRotation());

        analysisUseCase = builder.build();
        analysisUseCase.setAnalyzer(cameraExecutor, this::analyze);

        try {
            cameraProvider
                    .bindToLifecycle(this, cameraSelector, analysisUseCase);
        } catch (Exception e) {
            Log.e(TAG, "Error when bind analysis", e);
        }
    }

    protected int getRotation() throws NullPointerException {
        return previewView.getDisplay().getRotation();
    }

    private void switchCamera() {
        if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            lensFacing = CameraSelector.LENS_FACING_FRONT;
            flipX = true;
        } else {
            lensFacing = CameraSelector.LENS_FACING_BACK;
            flipX = false;
        }

        if(cameraProvider != null) cameraProvider.unbindAll();
        startCamera();
    }

    /** Face detection processor */
    @SuppressLint("UnsafeOptInUsageError")
    private void analyze(@NonNull ImageProxy image) {
        if (image.getImage() == null) return;

        InputImage inputImage = InputImage.fromMediaImage(
                image.getImage(),
                image.getImageInfo().getRotationDegrees()
        );

        FaceDetector faceDetector = FaceDetection.getClient();

        faceDetector.process(inputImage)
                .addOnSuccessListener(faces -> onSuccessListener(faces, inputImage))
                .addOnFailureListener(e -> Log.e(TAG, "Barcode process failure", e))
                .addOnCompleteListener(task -> image.close());
    }
    public static  String person_name;
    private Bitmap bitmaptemp;
    private void onSuccessListener(List<Face> faces, InputImage inputImage) {
        Rect boundingBox = null;
        String name = null;
        float scaleX = (float) previewView.getWidth() / (float) inputImage.getHeight();
        float scaleY = (float) previewView.getHeight() / (float) inputImage.getWidth();

        if(faces.size() > 0) {
            detectionTextView.setText(R.string.face_detected);
            // get first face detected
            Face face = faces.get(0);

            // get bounding box of face;
            boundingBox = face.getBoundingBox();


            // convert img to bitmap & crop img
            Bitmap bitmap = mediaImgToBmp(
                    inputImage.getMediaImage(),
                    inputImage.getRotationDegrees(),
                    boundingBox);

            bitmaptemp = bitmap;
            // ye bitmap bhi save ker dena chaye-------------
            //phir server side pey issey revert kerlo
            // ya extra code likhdo

            if(start) name = recognizeImage(bitmap);

            if(name != null) detectionTextView.setText(name);
        }
        else {
            detectionTextView.setText(R.string.no_face_detected);
        }

        graphicOverlay.draw(boundingBox, scaleX, scaleY, name);
    }
    private void speakText(String text){
        gifimageview.setVisibility(View.VISIBLE);

        // Hide the ImageView after 2 seconds
        Handler handler = new Handler();

        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                gifimageview.setVisibility(View.GONE);
            }
        }, 2500); // Delay in milliseconds (2 seconds)
    }

    private void saveBitmapToGallery(Bitmap bitmap, String imgname) {
        File galleryPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        String fileName = imgname + "_" + System.currentTimeMillis() + ".jpg";
        File file = new File(galleryPath, fileName);

        try {
            FileOutputStream outputStream = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
            outputStream.flush();
            outputStream.close();
            Toast.makeText(this, "Image saved to gallery.", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to save image to gallery.", Toast.LENGTH_SHORT).show();
        }
    }

    /** Recognize Processor and check face*/
    private void addFace() {
        start=false;

        AlertDialog.Builder builder = new AlertDialog.Builder(this);

        builder.setTitle("Enter Name");

        // Set up the input
        final EditText input = new EditText(this);

        input.setInputType(InputType.TYPE_CLASS_TEXT );
        input.setMaxWidth(200);
        builder.setView(input);


        // Set up the buttons

        builder.setNegativeButton("Cancel", (dialog, which) -> {
            start = true;
            dialog.cancel();
        });

        if (graphicOverlay.name == null || graphicOverlay.name == "unknown") {
            builder.setPositiveButton("ADD", (dialog, which) -> {
                //Toast.makeText(context, input.getText().toString(), Toast.LENGTH_SHORT).show();

                //Create and Initialize new object with Face embeddings and Name.
                SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                        "0", "", -1f);
                result.setExtra(embeddings);

                Log.i(TAG, "addFace: 2 embeddings"+embeddings[0][0]);
                registered.put( input.getText().toString(),result);
                start = true;

                saveBitmapToGallery(bitmaptemp, input.getText().toString());
                jsonObject = new JSONObject();
                try {

                    JSONArray jsonArray = new JSONArray(embeddings[0]);
                    jsonObject.put(input.getText().toString(), jsonArray);


                    Log.i(TAG, "addFace: json " +jsonObject);

                    jsonlist.add(jsonObject);
                    Log.i(TAG, "addFace: list> "+ jsonlist);
                } catch (JSONException e) {
                    e.printStackTrace();
                }

                // Convert the JSON to bytes
                byte[] data = jsonlist.toString().getBytes();

                // Upload the data to Firebase Storage
                UploadTask uploadTask = storageRef.putBytes(data);
                uploadTask.addOnSuccessListener(taskSnapshot -> {
                    // Handle successful upload
                    // You can retrieve the download URL or perform any additional actions here
                    Log.i(TAG, "addFace: success");
                    addFace();
                }).addOnFailureListener(e -> {
                    // Handle upload failure
                    Log.i(TAG, "addFace: failure");
                });


            });
            speakText("hi , what is your name");
            builder.show();

        }
        else{

            builder.setPositiveButton("Meet", (dialog, which) -> {
                start = true;
                person_name= input.getText().toString();
                meeting_logic(graphicOverlay.name);
            });
            speakText("hi " + graphicOverlay.name + ",  who do you want to meet");
            builder.show();

        }

    }
    private ArrayList<JSONObject>  person_list;
    private  void meeting_logic(String customer){


        FirebaseStorage storage;
        StorageReference storageRef;

        storage = FirebaseStorage.getInstance();
        storageRef = storage.getReferenceFromUrl("gs://face-recognizer-humans.appspot.com");
        Log.i(TAG, "meeting_logic: enter");

        String fileName = "user_availablity.json";
        StorageReference fileRef = storageRef.child(fileName);

        fileRef.getBytes(Long.MAX_VALUE).addOnSuccessListener(bytes -> {
            String jsonString = new String(bytes);

            try {
                JSONObject jsonObject = new JSONObject(jsonString);

                // Now you have the JSONObject, you can work with it as needed.
                Log.i(TAG, "meeting_logic: "+jsonObject.optString(person_name));
                if (jsonObject.optString(person_name).equalsIgnoreCase("unavailable")){
                    //record_message(customer);
                    speakText(person_name+" is not available . Please leave a message.");
                    Intent intent = new Intent(MainActivity.this,Recording.class);
                    intent.putExtra("person_name",person_name);
                    startActivity(intent);

                }else{
                    // if person is available action
                    speakText("You can meet "+ person_name +" Right now");
                }


            } catch (JSONException e) {
                Log.e("meeting", "Error parsing JSON: " + e.getMessage());
            }
        }).addOnFailureListener(exception -> {
            Log.e("meeting", "Error downloading file: " + exception.getMessage());
        });


    }


    public String recognizeImage(final Bitmap bitmap) {
        // set image to preview
        previewImg.setImageBitmap(bitmap);

        //Create ByteBuffer to store normalized image

        ByteBuffer imgData = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);

        imgData.order(ByteOrder.nativeOrder());

        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];

        //get pixel values from Bitmap to normalize
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        //imgData is input to our model
        Object[] inputArray = {imgData};

        Map<Integer, Object> outputMap = new HashMap<>();


        embeddings = new float[1][OUTPUT_SIZE]; //output of model will be stored in this variable

        outputMap.put(0, embeddings);

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap); //Run model



        float distance;

        //Compare new face with saved Faces.
        if (registered.size() > 0) {

            final Pair<String, Float> nearest = findNearest(embeddings[0]);//Find closest matching face

            if (nearest != null) {

                final String name = nearest.first;
                distance = nearest.second;



                // change this value to control and optimize face recognition
                if(distance<1.000f) //If distance between Closest found face is more than 1.000 ,then output UNKNOWN face.
                    return name;
                else
                    return "unknown";
            }
        }

        return null;
    }

    //Compare Faces by distance between face embeddings
    private Pair<String, Float> findNearest(float[] emb) {

        Pair<String, Float> ret = null;
        for (Map.Entry<String, SimilarityClassifier.Recognition> entry : registered.entrySet()) {

            final String name = entry.getKey();
            final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

            float distance = 0;
            for (int i = 0; i < emb.length; i++) {
                float diff = emb[i] - knownEmb[i];
                distance += diff*diff;
            }
            distance = (float) Math.sqrt(distance);
            if (ret == null || distance < ret.second) {
                ret = new Pair<>(name, distance);
            }
        }

        return ret;

    }

    /** Bitmap Converter */
    private Bitmap mediaImgToBmp(Image image, int rotation, Rect boundingBox) {
        //Convert media image to Bitmap
        Bitmap frame_bmp = toBitmap(image);

        //Adjust orientation of Face
        Bitmap frame_bmp1 = rotateBitmap(frame_bmp, rotation, flipX);

        //Crop out bounding box from whole Bitmap(image)
        float padding = 0.0f;
        RectF adjustedBoundingBox = new RectF(
                boundingBox.left - padding,
                boundingBox.top - padding,
                boundingBox.right + padding,
                boundingBox.bottom + padding);
        Bitmap cropped_face = getCropBitmapByCPU(frame_bmp1, adjustedBoundingBox);

        //Resize bitmap to 112,112
        return getResizedBitmap(cropped_face);
    }

    private Bitmap getResizedBitmap(Bitmap bm) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) 112) / width;
        float scaleHeight = ((float) 112) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        bm.recycle();
        return resizedBitmap;
    }

    private static Bitmap getCropBitmapByCPU(Bitmap source, RectF cropRectF) {
        Bitmap resultBitmap = Bitmap.createBitmap((int) cropRectF.width(),
                (int) cropRectF.height(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(resultBitmap);

        // draw background
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        paint.setColor(Color.WHITE);
        canvas.drawRect(//from  w w  w. ja v  a  2s. c  om
                new RectF(0, 0, cropRectF.width(), cropRectF.height()),
                paint);

        Matrix matrix = new Matrix();
        matrix.postTranslate(-cropRectF.left, -cropRectF.top);

        canvas.drawBitmap(source, matrix, paint);

        if (source != null && !source.isRecycled()) {
            source.recycle();
        }

        return resultBitmap;
    }

    private static Bitmap rotateBitmap(
            Bitmap bitmap, int rotationDegrees, boolean flipX) {
        Matrix matrix = new Matrix();

        // Rotate the image back to straight.
        matrix.postRotate(rotationDegrees);

        // Mirror the image along the X or Y axis.
        matrix.postScale(flipX ? -1.0f : 1.0f, 1.0f);
        Bitmap rotatedBitmap =
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // Recycle the old bitmap if it has changed.
        if (rotatedBitmap != bitmap) {
            bitmap.recycle();
        }
        return rotatedBitmap;
    }

    private static byte[] YUV_420_888toNV21(Image image) {

        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width*height;
        int uvSize = width*height/4;

        byte[] nv21 = new byte[ySize + uvSize*2];

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer(); // Y
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer(); // U
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer(); // V

        int rowStride = image.getPlanes()[0].getRowStride();
        assert(image.getPlanes()[0].getPixelStride() == 1);

        int pos = 0;

        if (rowStride == width) { // likely
            yBuffer.get(nv21, 0, ySize);
            pos += ySize;
        }
        else {
            long yBufferPos = -rowStride; // not an actual position
            for (; pos<ySize; pos+=width) {
                yBufferPos += rowStride;
                yBuffer.position((int) yBufferPos);
                yBuffer.get(nv21, pos, width);
            }
        }

        rowStride = image.getPlanes()[2].getRowStride();
        int pixelStride = image.getPlanes()[2].getPixelStride();

        assert(rowStride == image.getPlanes()[1].getRowStride());
        assert(pixelStride == image.getPlanes()[1].getPixelStride());

        if (pixelStride == 2 && rowStride == width && uBuffer.get(0) == vBuffer.get(1)) {
            // maybe V an U planes overlap as per NV21, which means vBuffer[1] is alias of uBuffer[0]
            byte savePixel = vBuffer.get(1);
            try {
                vBuffer.put(1, (byte)~savePixel);
                if (uBuffer.get(0) == (byte)~savePixel) {
                    vBuffer.put(1, savePixel);
                    vBuffer.position(0);
                    uBuffer.position(0);
                    vBuffer.get(nv21, ySize, 1);
                    uBuffer.get(nv21, ySize + 1, uBuffer.remaining());

                    return nv21; // shortcut
                }
            }
            catch (ReadOnlyBufferException ex) {
                // unfortunately, we cannot check if vBuffer and uBuffer overlap
            }

            // unfortunately, the check failed. We must save U and V pixel by pixel
            vBuffer.put(1, savePixel);
        }

        // other optimizations could check if (pixelStride == 1) or (pixelStride == 2),
        // but performance gain would be less significant

        for (int row=0; row<height/2; row++) {
            for (int col=0; col<width/2; col++) {
                int vuPos = col*pixelStride + row*rowStride;
                nv21[pos++] = vBuffer.get(vuPos);
                nv21[pos++] = uBuffer.get(vuPos);
            }
        }

        return nv21;
    }

    private Bitmap toBitmap(Image image) {

        byte[] nv21=YUV_420_888toNV21(image);


        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    /** Model loader */
    @SuppressWarnings("deprecation")
    private void loadModel() {
        try {

            //model name
            String modelFile = "mobile_face_net.tflite";
            tfLite = new Interpreter(loadModelFile(MainActivity.this, modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


}