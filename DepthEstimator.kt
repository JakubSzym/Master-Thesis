package com.example.walkassistant

import android.content.Context
import android.graphics.Bitmap

class DepthEstimator (context: Context) {
    private val depthEstimationModel = MidasModel(context)

    fun estimate(bitmap: Bitmap): Bitmap {
        val output = depthEstimationModel.getDepthMap(bitmap)
        return BitmapUtils.resizeBitmap(output, bitmap.width, bitmap.height)
    }
}