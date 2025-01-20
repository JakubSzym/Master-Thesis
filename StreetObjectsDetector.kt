package com.example.walkassistant

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class StreetObjectsDetector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: StreetObjectsDetectorListener
) {
    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private var instruction: String? = null

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    private val depthEstimator = DepthEstimator(context)

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        options.numThreads = 4
        interpreter = Interpreter(model, options)

        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2]

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun clear() {
        interpreter?.close()
        interpreter = null
    }

    fun detect(frame: Bitmap) {
        interpreter ?: return
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        instruction = null
        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1 , numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter?.run(imageBuffer, output.buffer)


        val bestBoxes = bestBox(output.floatArray)


        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        val depthMap = depthEstimator.estimate(frame)

        val labeledBoxes = getClosestObjects(bestBoxes, depthMap)

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (instruction == null)
        {
            instruction = "No crossings close to you"
        }

        detectorListener.onDetect(labeledBoxes, inferenceTime, instruction)
    }

    private fun getClosestObjects(boxes: List<BoundingBox>, depthMap: Bitmap): MutableList<BoundingBox> {
        val width = depthMap.width
        val height = depthMap.height
        val leftPivot: Int = (width/3).toInt()
        val rightPivot: Int = width - leftPivot

        var outBoxes = mutableListOf<BoundingBox>()

        for (box in boxes) {
            val left = box.x1 * width
            val top = box.y1 * height
            val right = box.x2 * width
            val bottom = box.y2 * height

            val x = ((left + right) / 2).toInt()
            val y = ((top + bottom) / 2).toInt()
            box.centerX = x
            box.centerY = y

            val pixel = depthMap.getPixel(x, y)

            val redValue = Color.red(pixel)
            val blueValue = Color.blue(pixel)
            val greenValue = Color.green(pixel)

            var depthValue = (redValue + blueValue + greenValue).toFloat() / 3f
            depthValue /= 2.55f
            box.centerDepthValue = depthValue
        }

        boxes.sortedByDescending { it.centerDepthValue }

        if (boxes.size > NUM_CLOSEST_OBJECTS) {
            for (i in 0..<NUM_CLOSEST_OBJECTS) {
                outBoxes.add(boxes[i])
            }
        }
        else {
            outBoxes = boxes.toMutableList()
        }

        for (box in outBoxes) {
            val position: String

            if (box.centerX!! <= leftPivot) {
                position = "LEFT"
            } else if (box.centerX!! <= rightPivot) {
                position = "CENTER"
            } else {
                position = "RIGHT"
            }

            val distance: String

            if (box.centerDepthValue!! > 40f) {
                distance = "VERY CLOSE"
            } else if (box.centerDepthValue!! > 30f) {
                distance = "CLOSE"
            } else if (box.centerDepthValue!! > 15f) {
                distance = "MEDIUM"
            } else if (box.centerDepthValue!! > 5f) {
                distance = "FAR"
            } else {
                distance = "VERY FAR"
            }

            box.objectLabel = ObjectLabel(position, distance)

            if (box.clsName == "crossing") {
                if (box.objectLabel!!.distance == "MEDIUM") {
                    if (box.objectLabel!!.position == "CENTER") {
                        instruction = "You are approaching a crossing in front of you"
                    }
                    else if (box.objectLabel!!.position == "LEFT") {
                        instruction = "You are approaching a crossing on your left"
                    }
                    else {
                        instruction = "You are approaching a crossing on your right"
                    }
                }
                else if (box.objectLabel!!.distance == "CLOSE" ||
                         box.objectLabel!!.distance == "VERY CLOSE") {
                    if (box.objectLabel!!.position == "CENTER") {
                        instruction = "You are close to a crossing in front of you"
                    }
                    else if (box.objectLabel!!.position == "LEFT") {
                        instruction = "You are close to a crossing on your left"
                    }
                    else {
                        instruction = "You are close to a crossing on your right"
                    }
                }
            }

        }

        return outBoxes
    }

    private fun bestBox(array: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = labels[maxIdx]
                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    interface StreetObjectsDetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long, instruction: String?)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
        private const val NUM_CLOSEST_OBJECTS = 3
    }
}