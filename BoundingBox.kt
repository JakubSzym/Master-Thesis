package com.example.walkassistant

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val cnf: Float,
    val cls: Int,
    val clsName: String,
    var objectLabel: ObjectLabel? = null,
    var centerDepthValue: Float? = null,
    var centerX: Int? = null,
    var centerY: Int? = null
)
