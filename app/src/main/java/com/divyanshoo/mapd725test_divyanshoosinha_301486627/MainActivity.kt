package com.divyanshoo.mapd725test_divyanshoosinha_301486627

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.divyanshoo.mapd725test_divyanshoosinha_301486627.ui.theme.MAPD725Test_DivyanshooSinha_301486627Theme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

private const val TAG = "SeaAnimalClassifier"

private val FALLBACK_LABELS = listOf("shark", "dolphin", "whale", "jellyfish", "crab", "octopus", "starfish", "turtle")

fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(modelPath)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

fun loadLabelsFromAsset(context: Context): List<String> {
    return try {
        context.assets.open("class_names.txt").bufferedReader().useLines { lines ->
            lines.filter { it.isNotEmpty() }.toList()
        }
    } catch (e: Exception) {
        Log.e(TAG, "Error loading labels", e)
        FALLBACK_LABELS
    }
}

fun extractImageData(bitmap: Bitmap): ByteBuffer {
    val modelInputSize = 150 * 150 * 3 * 4
    val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
    byteBuffer.order(ByteOrder.nativeOrder())

    val pixels = IntArray(150 * 150)
    bitmap.getPixels(pixels, 0, 150, 0, 0, 150, 150)

    for (pixel in pixels) {
        val r = (pixel shr 16 and 0xFF) / 255.0f
        val g = (pixel shr 8 and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f

        byteBuffer.putFloat(r)
        byteBuffer.putFloat(g)
        byteBuffer.putFloat(b)
    }

    return byteBuffer
}

data class SeaAnimalPrediction(
    val classIndex: Int,
    val className: String,
    val confidence: Float,
    val allProbabilities: FloatArray,
    val allLabels: List<String>
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as SeaAnimalPrediction

        if (classIndex != other.classIndex) return false
        if (className != other.className) return false
        if (confidence != other.confidence) return false
        if (!allProbabilities.contentEquals(other.allProbabilities)) return false
        if (allLabels != other.allLabels) return false

        return true
    }

    override fun hashCode(): Int {
        var result = classIndex
        result = 31 * result + className.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + allProbabilities.contentHashCode()
        result = 31 * result + allLabels.hashCode()
        return result
    }
}

suspend fun predictWithModel(context: Context, bitmap: Bitmap): SeaAnimalPrediction {
    return withContext(Dispatchers.Default) {
        try {
            val seaAnimalLabels = loadLabelsFromAsset(context)
            Log.d(TAG, "Loaded labels: $seaAnimalLabels")

            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 150, 150, true)
            val inputBuffer = extractImageData(resizedBitmap)
            inputBuffer.rewind()

            val tfliteModel = loadModelFile(context, "sea_animals_model.tflite")
            val tfliteOptions = Interpreter.Options()
            val interpreter = Interpreter(tfliteModel, tfliteOptions)

            val numClasses = seaAnimalLabels.size
            val outputBuffer = ByteBuffer.allocateDirect(numClasses * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            Log.d(TAG, "Running model inference")
            interpreter.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()
            val probabilities = FloatArray(numClasses)
            for (i in probabilities.indices) {
                probabilities[i] = outputBuffer.float
            }

            var maxIndex = 0
            var maxProb = probabilities[0]

            for (i in 1 until probabilities.size) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i]
                    maxIndex = i
                }
            }

            val logBuilder = StringBuilder("Class probabilities: ")
            for (i in probabilities.indices) {
                logBuilder.append("${seaAnimalLabels[i]}: ${probabilities[i]}, ")
            }
            Log.d(TAG, logBuilder.toString())

            interpreter.close()

            SeaAnimalPrediction(
                classIndex = maxIndex,
                className = seaAnimalLabels[maxIndex],
                confidence = maxProb,
                allProbabilities = probabilities,
                allLabels = seaAnimalLabels
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error during prediction", e)
            SeaAnimalPrediction(-1, "Error: ${e.message}", 0f, FloatArray(8), FALLBACK_LABELS)
        }
    }
}

class MainActivity : ComponentActivity() {
    private var selectedBitmap: Bitmap? = null
    private var predictionResult: SeaAnimalPrediction? = null
    private var isProcessing by mutableStateOf(false)
    private var errorMessage by mutableStateOf("")

    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            setContent {
                MAPD725Test_DivyanshooSinha_301486627Theme {
                    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                        SeaAnimalClassifierScreen(this, modifier = Modifier.padding(innerPadding))
                    }
                }
            }
        } else {
            showPermissionDeniedDialog()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        if (!hasPermissions()) {
            requestPermission()
        } else {
            setContent {
                MAPD725Test_DivyanshooSinha_301486627Theme {
                    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                        SeaAnimalClassifierScreen(this, modifier = Modifier.padding(innerPadding))
                    }
                }
            }
        }
    }

    private fun requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requestPermissionLauncher.launch(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            requestPermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    private fun hasPermissions(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun showPermissionDeniedDialog() {
        val showDialog = mutableStateOf(true)

        setContent {
            MAPD725Test_DivyanshooSinha_301486627Theme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    SeaAnimalClassifierScreen(this, modifier = Modifier.padding(innerPadding))

                    if (showDialog.value) {
                        AlertDialog(
                            onDismissRequest = {
                                showDialog.value = false
                                errorMessage = "Permission denied: Cannot access images"
                            },
                            title = { Text("Permission Required") },
                            text = { Text("This app needs access to your storage to pick images.") },
                            confirmButton = {
                                TextButton(onClick = {
                                    showDialog.value = false
                                    val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                                    val uri = Uri.fromParts("package", packageName, null)
                                    intent.data = uri
                                    startActivity(intent)
                                }) {
                                    Text("Grant")
                                }
                            },
                            dismissButton = {
                                TextButton(onClick = {
                                    showDialog.value = false
                                    errorMessage = "Permission denied: Cannot access images"
                                }) {
                                    Text("Cancel")
                                }
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun SeaAnimalClassifierScreen(context: ComponentActivity, modifier: Modifier = Modifier) {
    val coroutineScope = rememberCoroutineScope()
    val imageUri = remember { mutableStateOf<Uri?>(null) }
    val selectedBitmap = remember { mutableStateOf<Bitmap?>(null) }
    val predictionResult = remember { mutableStateOf<SeaAnimalPrediction?>(null) }
    val isProcessing = remember { mutableStateOf(false) }
    val errorMessage = remember { mutableStateOf("") }

    val backgroundColor = Color(0xFFF5F5F5)
    val accentColor = Color(0xFF1976D2)

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            imageUri.value = it
            predictionResult.value = null
            errorMessage.value = ""
            isProcessing.value = true

            coroutineScope.launch {
                try {
                    val bitmap = MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                    selectedBitmap.value = bitmap

                    val result = predictWithModel(context, bitmap)
                    predictionResult.value = result
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing image", e)
                    errorMessage.value = "Error: ${e.message}"
                } finally {
                    isProcessing.value = false
                }
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(backgroundColor)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Sea Animal Identifier",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF0D47A1),
            modifier = Modifier.padding(vertical = 24.dp)
        )

        Button(
            onClick = { launcher.launch("image/*") },
            modifier = Modifier
                .padding(bottom = 20.dp)
                .height(48.dp)
                .fillMaxWidth(0.8f),
            colors = ButtonDefaults.buttonColors(
                containerColor = accentColor
            ),
            shape = RoundedCornerShape(24.dp)
        ) {
            Text(
                "Select Sea Animal Image",
                fontSize = 16.sp
            )
        }

        if (isProcessing.value) {
            Text(
                "Analyzing your image...",
                color = Color.Gray,
                modifier = Modifier.padding(vertical = 12.dp)
            )
        }

        if (errorMessage.value.isNotEmpty()) {
            Text(
                text = errorMessage.value,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 16.dp)
            )
        }

        selectedBitmap.value?.let { bitmap ->
            Card(
                modifier = Modifier
                    .padding(vertical = 16.dp)
                    .fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color.White
                )
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(16.dp)
                ) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Selected sea animal image",
                        modifier = Modifier
                            .size(220.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .padding(bottom = 16.dp)
                    )

                    predictionResult.value?.let { prediction ->
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp),
                            color = Color(0xFFE1F5FE),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Column(
                                modifier = Modifier.padding(16.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Text(
                                    text = "${prediction.className.capitalize()}",
                                    fontSize = 24.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFF0D47A1)
                                )

                                Text(
                                    text = "Confidence: ${(prediction.confidence * 100).toInt()}%",
                                    fontSize = 16.sp,
                                    color = Color.DarkGray,
                                    modifier = Modifier.padding(top = 8.dp)
                                )
                            }
                        }

                        Column(
                            modifier = Modifier
                                .padding(top = 16.dp)
                                .fillMaxWidth()
                        ) {
                            Text(
                                text = "All Predictions:",
                                fontWeight = FontWeight.Bold,
                                fontSize = 16.sp,
                                color = Color.DarkGray,
                                modifier = Modifier.padding(bottom = 12.dp)
                            )

                            prediction.allProbabilities.forEachIndexed { index, probability ->
                                if (index < prediction.allLabels.size) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 6.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = prediction.allLabels[index].capitalize(),
                                            fontSize = 16.sp,
                                            fontWeight = if (index == prediction.classIndex) FontWeight.Bold else FontWeight.Normal,
                                            color = if (index == prediction.classIndex) Color(0xFF0D47A1) else Color.Black,
                                            modifier = Modifier.weight(1f)
                                        )

                                        val percentColor = when {
                                            probability > 0.7 -> Color(0xFF2196F3)
                                            probability > 0.3 -> Color(0xFF64B5F6)
                                            else -> Color(0xFFBDBDBD)
                                        }

                                        Surface(
                                            shape = CircleShape,
                                            color = percentColor,
                                            modifier = Modifier.padding(start = 8.dp)
                                        ) {
                                            Text(
                                                text = "${(probability * 100).toInt()}%",
                                                color = Color.White,
                                                fontSize = 14.sp,
                                                fontWeight = FontWeight.Bold,
                                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                                            )
                                        }
                                    }

                                    if (index < prediction.allLabels.size - 1) {
                                        Divider(
                                            color = Color.LightGray,
                                            thickness = 0.5.dp,
                                            modifier = Modifier.padding(vertical = 4.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fun String.capitalize(): String {
    return if (this.isNotEmpty()) {
        this[0].uppercase() + this.substring(1)
    } else {
        this
    }
}