package com.irhammuch.android.facerecognition;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;

import static android.Manifest.permission.RECORD_AUDIO;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;

public class Recording extends AppCompatActivity implements AdapterView.OnItemSelectedListener{

    private Spinner spinner;
    private List<String> dataList;
    private HashMap<String,String> audiohash;

    // Initializing all variables..
    private TextView startTV, stopTV, playTV, stopplayTV, statusTV, delete;

    // creating a variable for media recorder object class.
    private MediaRecorder mRecorder;

    private String audioFilePath;

    // creating a variable for mediaplayer class
    private MediaPlayer mPlayer;

    // string variable is created for storing a file name
    private static String mFileName = null;

    // constant for storing audio permission
    public static final int REQUEST_AUDIO_PERMISSION_CODE = 1;

    private String file_to_play="";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.recording);
        audiohash = new HashMap<String,String>();
        dataList= new ArrayList<String>();
        dataList.add(" ");
        Spinner spin = findViewById(R.id.audiospinner);


        // Create the instance of ArrayAdapter
        // having the list of courses
        ArrayAdapter ad = new ArrayAdapter(this,
                android.R.layout.simple_spinner_item, dataList);

        // set simple layout resource file
        // for each item of spinner
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spin.setAdapter(ad);


        // Set the ArrayAdapter (ad) data on the
        // Spinner which binds data to spinner

//        dataList = new ArrayList<>();
//
//        spinner =findViewById(R.id.audiolist);
//        spinner.setAdapter(new ArrayAdapter<>(Recording.this, R.layout.spinner_custom,dataList));
//        spinner.setOnItemSelectedListener (new AdapterView.OnItemSelectedListener() {
//
//            public void onItemClick(AdapterView<?> adapter, View view, int pos,
//                                    long id) {
//                // TODO Auto-generated method stub
//                Log.i("chkr", "onItemClick: ");
//            }
//
//            @Override
//            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
//                file_to_play = spinner.getItemAtPosition(i).toString();
//                Log.i("chkr", "onItem SELECTED: "+file_to_play);
//
//            }
//
//            @Override
//            public void onNothingSelected(AdapterView<?> adapterView) {
//
//            }
//
//        });

        spin.setOnItemSelectedListener(this);

        // initialize all variables with their layout items.
        statusTV = findViewById(R.id.idTVstatus);
        startTV = findViewById(R.id.btnRecord);
        stopTV = findViewById(R.id.btnStop);
        playTV = findViewById(R.id.btnPlay);
        stopplayTV = findViewById(R.id.btnStopPlay);
        stopTV.setBackgroundColor(getResources().getColor(R.color.gray));
        playTV.setBackgroundColor(getResources().getColor(R.color.gray));
        stopplayTV.setBackgroundColor(getResources().getColor(R.color.gray));

        delete = findViewById(R.id.deletebutton);

        File privateExternalDir = getExternalFilesDir(Environment.DIRECTORY_MUSIC);

        if (privateExternalDir != null && privateExternalDir.exists()) {
            // Get a list of all files in the directory
            File[] files = privateExternalDir.listFiles();

            // Check if any files are present
            if (files != null && files.length > 0) {
                // Iterate through the files and print their names

                for (File file : files) {
                    if (!dataList.contains(file.getName())){
                        dataList.add(file.getName());
                    }
                    Log.d("File List", file.getName());
                }
            } else {
                Log.d("File List", "No files found in the directory.");
            }
        } else {
            Log.d("File List", "Directory not found or doesn't exist.");
        }

        delete.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Delete recording method
                Log.i("chkr", "onClick: delete" +file_to_play);
                if (privateExternalDir != null && privateExternalDir.exists()) {
                    // Specify the file name you want to delete

                    // Create a File object representing the file to be deleted
                    File fileToDelete = new File(file_to_play);

                    // Check if the file exists before attempting to delete
                    if (fileToDelete.exists()) {
                        // Attempt to delete the file
                        boolean isDeleted = fileToDelete.delete();

                        if (isDeleted) {
                            //    file_to_play ko null kerdo ya smt
                            dataList.remove(fileToDelete.getName());
                            Log.d("File Delete", "==== File deleted successfully."+ fileToDelete.getName());
                            //toast
                        } else {
                            Log.d("File Delete", "Failed to delete the file.");
                        }
                    } else {
                        Log.d("File Delete", "File not found at the specified location.");
                    }
                } else {
                    Log.d("File Delete", "Directory not found or doesn't exist.");
                }
            }
        });

        startTV.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // start recording method will
                // start the recording of audio.
                startRecording();
            }
        });
        stopTV.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // pause Recording method will
                // pause the recording of audio.
                pauseRecording();

            }
        });
        playTV.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // play audio method will play
                // the audio which we have recorded
                playAudio();
            }
        });
        stopplayTV.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // pause play method will
                // pause the play of audio
                pausePlaying();
            }
        });

    }

    private void startRecording() {
        // check permission method is used to check
        // that the user has granted permission
        // to record and store the audio.
        if (CheckPermissions()) {

            // setbackgroundcolor method will change
            // the background color of text view.
            stopTV.setBackgroundColor(getResources().getColor(R.color.purple_200));
            startTV.setBackgroundColor(getResources().getColor(R.color.gray));
            playTV.setBackgroundColor(getResources().getColor(R.color.gray));
            stopplayTV.setBackgroundColor(getResources().getColor(R.color.gray));

            // we are here initializing our filename variable
            // with the path of the recorded audio file.

//            mFileName = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator ;
//            mFileName += "message to " + MainActivity.person_name +" _ "+ System.currentTimeMillis() + ".3gp"; // or .mp3, .m4a, etc.
//            mFileName = Environment.getExternalStorageDirectory().getAbsolutePath();
//            mFileName += "/AudioRecording.3gp";
            String fileName = MainActivity.person_name+" __ " + System.currentTimeMillis() + ".3gp"; // or .mp3, .m4a, etc.

            // Get the app's private external storage directory
            File privateExternalDir = getExternalFilesDir(Environment.DIRECTORY_MUSIC);

            // Create the output file in the app's private external storage directory
            File outputFile = new File(privateExternalDir, fileName);
            audioFilePath = outputFile.getAbsolutePath();
            Log.i("chkr", "startRecording: "+ audioFilePath);

            // below method is used to initialize
            // the media recorder class
            mRecorder = new MediaRecorder();

            // below method is used to set the audio
            // source which we are using a mic.
            mRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);

            // below method is used to set
            // the output format of the audio.
            mRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);

            // below method is used to set the
            // audio encoder for our recorded audio.
            mRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);

            // below method is used to set the
            // output file location for our recorded audio
            mRecorder.setOutputFile(audioFilePath);
            dataList.add(fileName);
            audiohash.put(fileName,audioFilePath);
            try {
                // below method will prepare
                // our audio recorder class
                mRecorder.prepare();
            } catch (IOException e) {
                Log.e("chkr", "prepare() failed 2"+audioFilePath);
            }
            // start method will start
            // the audio recording.
            mRecorder.start();
            statusTV.setText("Recording Started");
        } else {
            // if audio recording permissions are
            // not granted by user below method will
            // ask for runtime permission for mic and storage.
            RequestPermissions();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        // this method is called when user will
        // grant the permission for audio recording.
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQUEST_AUDIO_PERMISSION_CODE:
                if (grantResults.length > 0) {
                    boolean permissionToRecord = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean permissionToStore = grantResults[1] == PackageManager.PERMISSION_GRANTED;
                    if (permissionToRecord && permissionToStore) {
                        Toast.makeText(getApplicationContext(), "Permission Granted", Toast.LENGTH_LONG).show();
                    } else {
                        Toast.makeText(getApplicationContext(), "Permission Denied", Toast.LENGTH_LONG).show();
                    }
                }
                break;
        }
    }

    public boolean CheckPermissions() {
        // this method is used to check permission
        int result = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);
        int result1 = ContextCompat.checkSelfPermission(getApplicationContext(), RECORD_AUDIO);
        return result == PackageManager.PERMISSION_GRANTED && result1 == PackageManager.PERMISSION_GRANTED;
    }

    private void RequestPermissions() {
        // this method is used to request the
        // permission for audio recording and storage.
        ActivityCompat.requestPermissions(Recording.this, new String[]{RECORD_AUDIO, WRITE_EXTERNAL_STORAGE}, REQUEST_AUDIO_PERMISSION_CODE);
    }


    public void playAudio() {
        stopTV.setBackgroundColor(getResources().getColor(R.color.gray));
        startTV.setBackgroundColor(getResources().getColor(R.color.purple_200));
        playTV.setBackgroundColor(getResources().getColor(R.color.gray));
        stopplayTV.setBackgroundColor(getResources().getColor(R.color.purple_200));

        // for playing our recorded audio
        // we are using media player class.
        mPlayer = new MediaPlayer();
        try {
            // below method is used to set the
            // data source which will be our file name
            Log.i("chkr", "playAudio: 2" + file_to_play+"<<<");
        // file ni mil rahi



            mPlayer.setDataSource(file_to_play);

            Log.i("chkr", "playAudio: " + file_to_play);
            // below method will prepare our media player
            mPlayer.prepare();

            // below method will start our media player.
            mPlayer.start();
            statusTV.setText("Recording Started Playing");
        } catch (IOException e) {
            Log.e("chkr", "prepare() failed--"+e.toString());
        }
    }

    public void pauseRecording() {
        stopTV.setBackgroundColor(getResources().getColor(R.color.gray));
        startTV.setBackgroundColor(getResources().getColor(R.color.purple_200));
        playTV.setBackgroundColor(getResources().getColor(R.color.purple_200));
        stopplayTV.setBackgroundColor(getResources().getColor(R.color.purple_200));
        try {
            // below method will stop
            // the audio recording.
            mRecorder.stop();

            // below method will release
            // the media recorder class.
            mRecorder.release();
            mRecorder = null;
            statusTV.setText("Recording Stopped");


            File privateExternalDir = getExternalFilesDir(Environment.DIRECTORY_MUSIC);

            if (privateExternalDir != null && privateExternalDir.exists()) {
                // Get a list of all files in the directory
                File[] files = privateExternalDir.listFiles();

                // Check if any files are present
                if (files != null && files.length > 0) {
                    // Iterate through the files and print their names

                    for (File file : files) {
                        if (!dataList.contains(file.getName())) {
                            dataList.add(file.getName());
                        }
                        Log.d("File List", file.getName());
                    }
                } else {
                    Log.d("File List", "No files found in the directory.");
                }
            } else {
                Log.d("File List", "Directory not found or doesn't exist.");
            }
        }catch(Exception e){
            Log.i("error", "pauseRecording: "+e);
        }
    }

    public void pausePlaying() {
        // this method will release the media player
        // class and pause the playing of our recorded audio.
        mPlayer.release();
        mPlayer = null;
        stopTV.setBackgroundColor(getResources().getColor(R.color.gray));
        startTV.setBackgroundColor(getResources().getColor(R.color.purple_200));
        playTV.setBackgroundColor(getResources().getColor(R.color.purple_200));
        stopplayTV.setBackgroundColor(getResources().getColor(R.color.gray));
        statusTV.setText("Recording Play Stopped");
    }

    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {

        String fileName =  dataList.get(i);
        File privateExternalDir = getExternalFilesDir(Environment.DIRECTORY_MUSIC);

        // Create the output file in the app's private external storage directory
        File outputFile = new File(privateExternalDir, fileName);
        file_to_play = outputFile.getAbsolutePath();
        Log.i("chkr", "onItemSelected:__ "+ file_to_play);
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }
}
