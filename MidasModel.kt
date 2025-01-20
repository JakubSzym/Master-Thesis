package com.example.walkassistant

import android.content.Context
import android.graphics.Bitmap
import com.example.walkassistant.Paths.DEPTH_ESTIMATION_MODEL_PATH
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat

class MidasModel (
    context: Context
) {
    private var interpreter: Interpreter
    private val numThreads = 4

    private val inputImageDim = 256
    private val mean = floatArrayOf( 123.675f ,  116.28f ,  103.53f )
    private val std = floatArrayOf( 58.395f , 57.12f ,  57.375f )

    private val inputTensorProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputImageDim, inputImageDim, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(mean, std))
        .build()

    private val outputTensorProcessor = TensorProcessor.Builder()
        .add(MinMaxScalingOp())
        .build()

    init {
        val interpreterOptions = Interpreter.Options().setNumThreads(numThreads)

        interpreter = Interpreter(
            FileUtil.loadMappedFile(context, DEPTH_ESTIMATION_MODEL_PATH),
            interpreterOptions
        )
    }

    fun getDepthMap(inputImage: Bitmap) : Bitmap {
        return run(inputImage)
    }

    private fun run(inputImage: Bitmap) : Bitmap {
        var inputTensor = TensorImage.fromBitmap(inputImage)

        inputTensor = inputTensorProcessor.process(inputTensor)

        var outputTensor = TensorBufferFloat.createFixedSize(
            intArrayOf(inputImageDim, inputImageDim, 1), DataType.FLOAT32
        )

        interpreter.run(inputTensor.buffer, outputTensor.buffer)

        outputTensor = outputTensorProcessor.process(outputTensor)

        return BitmapUtils.byteBufferToBitmap(outputTensor.floatArray, inputImageDim)
    }

    class MinMaxScalingOp : TensorOperator {
        override fun apply( input : TensorBuffer?): TensorBuffer {
            val values = input!!.floatArray
            // Compute min and max of the output
            val max = values.maxOrNull()!!
            val min = values.minOrNull()!!
            for ( i in values.indices ) {
                // Normalize the values and scale them by a factor of 255
                var p = ((( values[ i ] - min ) / ( max - min )) * 255).toInt()
                if ( p < 0 ) {
                    p += 255
                }
                values[ i ] = p.toFloat()
            }
            // Convert the normalized values to the TensorBuffer and load the values in it.
            val output = TensorBufferFloat.createFixedSize( input.shape , DataType.FLOAT32 )
            output.loadArray( values )
            return output
        }

    }
}