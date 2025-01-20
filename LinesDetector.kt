package com.example.walkassistant

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.sqrt

class LinesDetector(
    private val context: Context,
    private val detectorListener: LinesDetectorListener
) {

    private fun grayscale(image: Bitmap): Bitmap {
        val bitmap: Bitmap = image
        val mat = Mat(image.height, image.width, CvType.CV_8UC1)
        Utils.bitmapToMat(image, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY)
        Utils.matToBitmap(mat, bitmap)
        return bitmap
    }

    private fun blur(image: Bitmap): Bitmap {
        val mat = Mat(image.height, image.width, CvType.CV_8UC1)
        Utils.bitmapToMat(image, mat)
        Imgproc.GaussianBlur(mat, mat, Size(5.0,5.0), 0.0, 0.0)
        Utils.matToBitmap(mat, image)
        return image
    }

    private fun detectEdges(image: Bitmap): Bitmap {
        val mat = Mat(image.height, image.width, CvType.CV_8UC1)
        var result = Mat()
        Utils.bitmapToMat(image, mat)
        Imgproc.Canny(mat, result, 50.0, 150.0)
        Utils.matToBitmap(result, image)
        return image
    }

    private fun getLines(image: Bitmap): Mat {
        val mat = Mat(image.height, image.width, CvType.CV_8UC1)
        Utils.bitmapToMat(image, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY)
        val lines = Mat()
        Imgproc.HoughLinesP(mat, lines,
            1.0,
            PI/180,
            20,
            50.0,
            40.0)
        return lines
    }

    private fun getRoi(image: Bitmap): Bitmap {
        val mat = Mat(image.height, image.width, CvType.CV_8UC1)
        Utils.bitmapToMat(image, mat)
        val polygon: List<MatOfPoint> = listOf(
                MatOfPoint(
                    Point(0.0, mat.height().toDouble()),
                    Point(0.0, mat.height() * 0.6),
                    Point(mat.width() * 0.33, mat.height() * 0.4),
                    Point(mat.width() * 0.66, mat.height() * 0.4),
                    Point(mat.width().toDouble(), mat.height() * 0.6),
                    Point(mat.width().toDouble(), mat.height().toDouble())
                )
        )

        val blackImage = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)

        val mask = Mat(image.height, image.width, CvType.CV_8UC1)
        Utils.bitmapToMat(blackImage, mask)

        Imgproc.fillPoly(mask, polygon, Scalar(255.0))

        val dst = Mat()
        val retImage = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        Core.bitwise_and(mat, mask, dst)
        Utils.matToBitmap(dst, retImage)
        return retImage
    }

    private fun averageSlopeIntercept(lines: Mat): Pair<Pair<Double, Double>?, Pair<Double, Double>?> {
        var leftLines = mutableListOf<Pair<Double,Double>>()
        var leftWeights = mutableListOf<Double>()
        var rightLines = mutableListOf<Pair<Double,Double>>()
        var rightWeights = mutableListOf<Double>()

        for (i in 0..lines.rows()-1) {
            for (j in 0..lines.cols()-1) {
                val line = lines[i, j]
                val x1: Double = line[0]
                val y1: Double = line[1]
                val x2: Double = line[2]
                val y2: Double = line[3]

                if (x1 == x2) {
                    continue
                }

                val slope = (y2 - y1) / (x2 - x1)
                val intercept = y1 - (slope * x1)
                val length = sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))

                if ((slope > 0.5 && slope < 15.0) || (slope > -15.0 && slope < -0.5)) {
                    if (slope < 0) {
                        leftLines.add(Pair(slope, intercept))
                        leftWeights.add(length)
                    } else {
                        rightLines.add(Pair(slope, intercept))
                        rightWeights.add(length)
                    }
                }
            }
        }

        var leftLane: Pair<Double, Double>? = null
        var rightLane: Pair<Double, Double>? = null

        if (leftWeights.size > 0) { leftLane = averageLine(leftWeights, leftLines) }
        if (rightWeights.size > 0) { rightLane = averageLine(rightWeights, rightLines) }

        return Pair(leftLane, rightLane)
    }

    fun averageLine(weights: MutableList<Double>,
                    lines: MutableList<Pair<Double,Double>>): Pair<Double, Double> {
        var slope: Double = 0.0
        var intercept: Double = 0.0

        for (i in 0..lines.size-1) {
            slope += weights[i] * lines[i].first
            intercept += weights[i] * lines[i].second
        }

        var sumOfWeights = 0.0
        weights.forEach {
            sumOfWeights += it
        }

        slope /= sumOfWeights
        intercept /= sumOfWeights

        return Pair(slope, intercept)
    }

    fun pixelPoints(a1: Double, a2: Double, line: Pair<Double, Double>?): Line? {
        if (line == null) {
            return null
        }

        val (slope, intercept) = line

        val x1 = ((a1 - intercept) / slope).toInt()
        val x2 = ((a2 - intercept) / slope).toInt()
        val y1 = a1.toInt()
        val y2 = a2.toInt()

        return Line(x1, y1, x2, y2)
    }

    fun laneLines(bitmap: Bitmap, lines: Mat): Pair<Line?, Line?> {
        val lanes = averageSlopeIntercept(lines)
        val y1: Double = 0.0
        val y2: Double = bitmap.height.toDouble()

        val leftLine = pixelPoints(y1, y2, lanes.first)
        val rightLine = pixelPoints(y1, y2, lanes.second)

        return Pair(leftLine, rightLine)
    }

    fun perform(frame: Bitmap) {
        val grayImage = grayscale(frame)
        val blurredImage = blur(frame)
        val edgedImage = detectEdges(blurredImage)
        val roiImage = getRoi(edgedImage)
        val lines = getLines(roiImage)
        val finalLines = laneLines(frame, lines)

        if (finalLines.first == null && finalLines.second == null) {
            detectorListener.onNoResults()
        }

        detectorListener.onPerform(finalLines)
    }

    interface LinesDetectorListener {
        fun onPerform(lines: Pair<Line?, Line?>)
        fun onNoResults()
    }
}