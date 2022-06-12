import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.*

fun main(args: Array<String>) {
    initOpenCV()

    print("Enter the image path > ")

    val inputFilePath = readLine()

    val isDebugMode = (args.elementAtOrNull(0)?.lowercase() == "--debug" || args.elementAtOrNull(0)?.lowercase() == "-d")

    if(inputFilePath.isNullOrBlank()) {
        println("ERROR: Please enter the correct path.")
        return
    }

    val inputFile = File(inputFilePath)

    if(!inputFile.exists()) {
        println("ERROR: The file don't exists or isn't file type.")
        return
    }

    val arrowExtensions = listOf("jpg", "png")
    val params = listOf(
        Parameter(15, 8),
        Parameter(7, 5),
        Parameter(5, 2),
        Parameter(1, 2),
        Parameter(1, 1),
        Parameter(0, 0)
    )

    if(inputFile.isFile) {
        if(!arrowExtensions.contains(inputFile.extension.lowercase())) {
            println("ERROR: Files with this extension are not supported.")
            return
        }

        val inputDir = inputFile.parentFile
        val outputFile = File(inputDir, "${inputFile.nameWithoutExtension}-output.${inputFile.extension}")

        if (trumpDetection(inputFile, outputFile, params, isDebugMode)) {
            println("FINISH: Output success. [${outputFile.absoluteFile}]")
        } else {
            println("FINISH: Output failed. ")
        }
    } else {
        val files = inputFile.listFiles() ?: emptyArray()
        val outputDir = File(inputFile, "output")
        val imageFiles = files.filter { arrowExtensions.contains(it.extension.lowercase()) && !it.name.contains(Regex("-output|-process")) }
        val failedImages = mutableListOf<File>()

        if(!outputDir.exists()) outputDir.mkdir()

        println("PROCESS: ${imageFiles.size} image files found in the given folder.")

        for((index, file) in imageFiles.withIndex()) {
            println("PROCESS: Start the detection... [$index/${imageFiles.size}]")

            val outputFile = File(outputDir, file.name)

            if (trumpDetection(file, outputFile, params, false)) {
                println("FINISH: Output success. [${outputFile.absoluteFile}]")
            } else {
                failedImages.add(file)
                println("FINISH: Output failed. ")
            }
        }

        println("FINISH: [${failedImages.size}/${imageFiles.size}] images failed to process.")
        println("FINISH: Process is complete.")
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

fun trumpDetection(inputFile: File, outputFile: File, params: List<Parameter>, isDebugMode: Boolean): Boolean {
    for (param in params) {
        println("PROCESS: Image processing... [$param]")

        val inputMat = Imgcodecs.imread(inputFile.absolutePath)
        val resultMat = inputMat.clone()
        val outputMat = inputMat.clone()

        Imgproc.cvtColor(resultMat, resultMat, Imgproc.COLOR_RGB2GRAY)
        Core.bitwise_not(resultMat, resultMat)
        Imgproc.adaptiveThreshold(resultMat, resultMat, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 255, 2.0)

        if (isDebugMode) {
            val middleFile = File(inputFile.parentFile, "${inputFile.nameWithoutExtension}-$param-process1.${inputFile.extension}")
            Imgcodecs.imwrite(middleFile.absolutePath, resultMat)
        }

        if(param.bluer >= 1) Imgproc.medianBlur(resultMat, resultMat, param.bluer)
        if(param.dilate >= 2) Imgproc.dilate(resultMat, resultMat, Mat(), Point(-1.0, -1.0), param.dilate)
        if(param.dilate >= 2) Imgproc.erode(resultMat, resultMat, Mat(), Point(-1.0, -1.0), param.dilate)
        Imgproc.medianBlur(resultMat, resultMat, 1)

        println("PROCESS: Contour detection in progress...")

        if (isDebugMode) {
            val middleFile = File(inputFile.parentFile, "${inputFile.nameWithoutExtension}-$param-process2.${inputFile.extension}")
            Imgcodecs.imwrite(middleFile.absolutePath, resultMat)
        }

        val contours = arrayListOf<MatOfPoint>()
        val hierarchy = Mat()

        Imgproc.findContours(resultMat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_NONE)
        Imgproc.drawContours(outputMat, contours, -1, Scalar(0.0, 0.0, 255.0, 255.0), 2)

        val (cardDrawMat, approx) = findSquares(outputMat, contours)
        val transformedMat = projectionTransform(inputMat, approx)

        if (isDebugMode) {
            val middleFile = File(inputFile.parentFile, "${inputFile.nameWithoutExtension}-$param-process3.${inputFile.extension}")
            Imgcodecs.imwrite(middleFile.absolutePath, cardDrawMat)
        }

        if (transformedMat == null || approx == null) {
            println("ERROR: Can't find trump card.")
            println("ERROR: Approx: $approx, TransformedMat: $transformedMat")

            continue
        }

        return Imgcodecs.imwrite(outputFile.absolutePath, transformedMat)
    }

    return false
}

fun findSquares(inputMat: Mat, contours: List<MatOfPoint>): Pair<Mat, MatOfPoint2f?> {
    val rectInfos = contours.mapIndexed { index, point -> Triple(point, Imgproc.contourArea(point, false), index) }.sortedByDescending { it.second }
    val trumpInfos = rectInfos.take(5)

    for(trumpInfo in trumpInfos) {
        val (point, areaSize, index) = trumpInfo
        val curve = MatOfPoint2f(*point.toArray())
        val approx = MatOfPoint2f()

        Imgproc.approxPolyDP(curve, approx, 0.01 * Imgproc.arcLength(curve, true), true)
        Imgproc.drawContours(inputMat, contours, index, Scalar(255.0, 0.0, 0.0, 255.0), 2)

        if(approx.toArray().size == 4 && areaSize >= 1000 && compareSize(approx.sizeOfSide(), Pair(inputMat.size().width, inputMat.size().height)) <= 0.9) {
            Imgproc.drawContours(inputMat, contours, index, Scalar(0.0, 255.0, 0.0, 255.0), 2)
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

    println("PROCESS: Projection transforming... [Approx: $approxSize, Mat: ${inputMat.size()}]")

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

fun compareSize(size1: Pair<Double, Double>, size2: Pair<Double, Double>): Double {
    val area1 = size1.first * size1.second
    val area2 = size2.first * size2.second
    return min(area1, area2) / max(area1, area2)
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

data class Parameter(
    val bluer: Int,
    val dilate: Int,
)