import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

fun main(args: Array<String>) {
    initOpenCV()

    print("Enter the image path > ")

    val inputFilePath = readLine()

    if(inputFilePath.isNullOrBlank()) {
        println("ERROR: Please enter the correct path.")
        return
    }

    val inputFile = File(inputFilePath)

    if(!inputFile.exists() || !inputFile.isFile) {
        println("ERROR: The file don't exists or isn't file type.")
        return
    }

    val inputDir = inputFile.parentFile
    val outputFile = File(inputDir, "${inputFile.nameWithoutExtension}-output.${inputFile.extension}")

    if(trumpDetection(inputFile, outputFile)) {
        println("FINISH: Output success. [${outputFile.absoluteFile}]")
    } else {
        println("FINISH: Output failed. ")
    }
}

fun initOpenCV() {
    try {
        val libDir = File("libs")
        val libPath = libDir.absolutePath
        val dllPath = "$libPath\\${Core.NATIVE_LIBRARY_NAME}.dll"

        if(File(dllPath).exists()) {
            System.load(dllPath)
            return
        }

        val jarPath = System.getProperty("java.class.path")
        val jarDir = File(jarPath).parentFile

        System.load("${jarDir.absolutePath}\\libs\\${Core.NATIVE_LIBRARY_NAME}.dll")
    } catch (e: UnsatisfiedLinkError) {
        println("ERROR: Can't load OpenCV.")
        println("ERROR: Place the JAR file in the initial location and run it.")
    }
}

fun trumpDetection(inputFile: File, outputFile: File): Boolean {
    println("PROCESS: Image processing...")

    val inputMat = Imgcodecs.imread(inputFile.absolutePath)
    val resultMat = inputMat.clone()
    val outputMat = inputMat.clone()

    Imgproc.cvtColor(resultMat, resultMat, Imgproc.COLOR_RGB2GRAY)
    Core.bitwise_not(resultMat, resultMat)
    Imgproc.adaptiveThreshold(resultMat, resultMat, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 255, 2.0)
    Imgproc.medianBlur(resultMat, resultMat, 15)
    Imgproc.dilate(resultMat, resultMat, Mat(), Point(-1.0, -1.0), 8)

    println("PROCESS: Contour detection in progress...")

    val contours = arrayListOf<MatOfPoint>()
    val hierarchy = Mat()

    Imgproc.findContours(resultMat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_NONE)
    Imgproc.drawContours(outputMat, contours, -1, Scalar(0.0, 0.0, 255.0, 255.0), 10)

    val (cardDrawMat, approx) = findSquares(outputMat, contours)
    val transformedMat = projectionTransform(inputMat, approx)

    if(transformedMat == null || approx == null) {
        println("ERROR: Can't find trump card.")
        println("ERROR: Approx: $approx, TransformedMat: $transformedMat")

        return false
    }

    return Imgcodecs.imwrite(outputFile.absolutePath, transformedMat)
}

fun findSquares(inputMat: Mat, contours: List<MatOfPoint>): Pair<Mat, MatOfPoint2f?> {
    val rectInfos = contours.mapIndexed { index, point -> Triple(point, Imgproc.contourArea(point, false), index) }.sortedByDescending { it.second }
    val trumpInfo = rectInfos.elementAtOrNull(1) ?: rectInfos.elementAtOrNull(0)

    if(trumpInfo != null) {
        val (point, areaSize, index) = trumpInfo
        val curve = MatOfPoint2f(*point.toArray())
        val approx = MatOfPoint2f()

        Imgproc.approxPolyDP(curve, approx, 0.01 * Imgproc.arcLength(curve, true), true)

        if(approx.toArray().size == 4 && areaSize >= 1000) {
            Imgproc.drawContours(inputMat, contours, index, Scalar(0.0, 255.0, 0.0, 255.0), 10)
            return (inputMat to approx)
        }
    }

    return (inputMat to null)
}

fun projectionTransform(inputMat: Mat, approx: MatOfPoint2f?): Mat? {
    approx ?: return null

    val width: Double
    val height: Double
    val approxSize = approx.sizeOfSide()

    println("PROCESS: Projection transforming... [Approx: $approxSize]")

    if(approxSize.first >= approxSize.second) {
        width = 1024.0
        height = width * 1.618
    } else {
        height = 1024.0
        width = height * 1.618
    }

    val dstMat = MatOfPoint2f(
        Point(0.0, 0.0),
        Point(0.0, height),
        Point(width, height),
        Point(width, 0.0)
    )

    val resultMat = Imgproc.getPerspectiveTransform(approx, dstMat)

    Imgproc.warpPerspective(inputMat, inputMat, resultMat, Size(width, height))
    Core.flip(inputMat, inputMat, 1)

    if(approxSize.first < approxSize.second) {
        Core.rotate(inputMat, inputMat, Core.ROTATE_90_CLOCKWISE)
    }

    return inputMat
}

fun MatOfPoint2f.sizeOfSide(): Pair<Double, Double> {
    val points = this.toArray()
    val side1 = points[0].toLength(points[1])
    val side2 = points[0].toLength(points[3])
    return (side1 to side2)
}

fun Point.toLength(point: Point): Double {
    return sqrt(abs(this.x - point.x).pow(2) + abs(this.y - point.y).pow(2))
}