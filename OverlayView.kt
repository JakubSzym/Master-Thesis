package com.example.walkassistant

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import org.opencv.core.Point

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var lines = mutableListOf<Line>()
    private var message = String()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var linePaint = Paint()

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        lines = mutableListOf()
        message = String()
        results = listOf()
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        linePaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.red)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE

        linePaint.color = Color.BLUE
        linePaint.strokeWidth = 8F
        linePaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            canvas.drawRect(left, top, right, bottom, boxPaint)
            val drawableText = it.clsName + ", " +
                    it.objectLabel?.distance + "," +
                    it.objectLabel?.position

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }

        lines.forEach {
            val (x1, y1, x2, y2) = it
            if (x2 != x1) {
                val slope = (y2 - y1) / (x2 - x1)
                if (slope != 0) {
                    val intercept = y1 - slope * x1

                    val startY = height
                    val endY = 0
                    val startX = (startY - intercept) / slope
                    val endX = (endY - intercept) / slope

                    canvas.drawLine(
                        startX.toFloat(), startY.toFloat(),
                        endX.toFloat(), endY.toFloat(), linePaint
                    )
                }
            }
        }

    }

    fun isOnTrack(lines: Pair<Line?, Line?>): Boolean {
        if (lines.first == null || lines.second == null) {
            return false
        }

        val (leftLine, rightLine) = lines
        val (xr1, yr1, xr2, yr2) = rightLine!!
        val (xl1, yl1, xl2, yl2) = leftLine!!

        val slopeRight = (yr2 - yr1) / (xr2 - xr1)
        val interceptRight = yr1 - slopeRight * xr1

        val slopeLeft = (yl2 - yl1) / (xl2 - xl1)
        val interceptLeft = yl1 - slopeLeft * xl1

        if (slopeRight == slopeLeft) {
            return false
        }

        if (slopeRight > -0.2 && slopeRight < 0.2) {
            return false
        }

        if (slopeLeft > -0.2 && slopeLeft < 0.2) {
            return false
        }

        val x = (interceptRight - interceptLeft) / (slopeLeft - slopeRight)
        val y = slopeLeft * x + interceptLeft

        if (x >= width * 0.25 && x <= width * 0.75 && y <= height * 0.4) {
            return true
        }

        return false
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    fun setLines(inLines: Pair<Line?, Line?>) {
        lines = mutableListOf()
        if (inLines.first != null) { lines.add(inLines.first!!) }
        if (inLines.second != null) { lines.add(inLines.second!!) }
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}