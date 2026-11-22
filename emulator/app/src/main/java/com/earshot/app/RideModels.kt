package com.earshot.app

import android.content.Context
import android.content.res.AssetManager
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp

internal fun AssetManager.readJson(path: String): JSONObject =
    open(path).bufferedReader().use { JSONObject(it.readText()) }

internal fun JSONObject.strings(key: String): List<String> {
    val array = getJSONArray(key)
    return List(array.length(), array::getString)
}

internal class SoundModel(assets: AssetManager, config: JSONObject) : AutoCloseable {
    private val audio = config.getJSONObject("audio")
    private val window = audio.getInt("window_samples")
    private val head = assets.readJson(config.getJSONObject("head").getString("file"))
    private val weights = head.getJSONArray("weights").let { values ->
        FloatArray(values.length()) { values.getDouble(it).toFloat() }
    }
    private val bias = head.getDouble("bias")
    val labels = assets.open("models/yamnet_labels.txt").bufferedReader().use { it.readLines() }
    private val model = assets.openFd("models/${audio.getString("model")}").use { fd ->
        fd.createInputStream().use {
            it.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }
    private val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(2))
    private val samples = FloatArray(window)
    private val input = floats(window)
    private val scores = floats(labels.size)
    private val embedding = floats(weights.size / 2)
    private val outputs: Map<Int, ByteBuffer>
    private var previous: FloatArray? = null

    init {
        try {
            interpreter.resizeInput(0, intArrayOf(window))
            interpreter.allocateTensors()
            val indices = (0 until interpreter.outputTensorCount).associateBy {
                interpreter.getOutputTensor(it).shape().last()
            }
            outputs = mapOf(
                indices.getValue(labels.size) to scores,
                indices.getValue(weights.size / 2) to embedding,
            )
        } catch (error: Exception) {
            interpreter.close()
            throw error
        }
    }

    data class Frame(val scores: FloatArray, val head: Double)

    fun infer(pcm: ByteArray): Frame {
        val count = pcm.size / Short.SIZE_BYTES
        samples.copyInto(samples, endIndex = window, startIndex = count)
        val chunk = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        for (i in 0 until count) samples[window - count + i] = chunk[i] / 32768f

        input.rewind()
        input.asFloatBuffer().put(samples)
        scores.rewind()
        embedding.rewind()
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)

        val current = FloatArray(weights.size / 2)
        embedding.rewind()
        embedding.asFloatBuffer().get(current)
        val last = previous ?: current
        var dot = 0f
        for (i in current.indices) {
            dot += weights[i] * current[i] + weights[i + current.size] * last[i]
        }
        previous = current
        val p = 1.0 / (1.0 + exp(-(dot + bias).coerceIn(-30.0, 30.0)))
        scores.rewind()
        return Frame(FloatArray(labels.size).also { scores.asFloatBuffer().get(it) }, p)
    }

    override fun close() = interpreter.close()

    private fun floats(size: Int) =
        ByteBuffer.allocateDirect(size * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
}

internal fun loadSpeechModel(context: Context, config: JSONObject): Model {
    val name = config.getJSONObject("asr").getString("model")
    val modelDir = File(context.noBackupFilesDir, "earshot-v${config.getInt("version")}/$name")
    val ready = File(modelDir, ".complete")
    if (!ready.isFile) {
        copyAssetTree(context.assets, "models/$name", modelDir)
        ready.writeText(name)
    }
    return Model(modelDir.absolutePath)
}

private fun copyAssetTree(assets: AssetManager, path: String, destination: File) {
    val children = assets.list(path).orEmpty()
    if (children.isEmpty()) {
        destination.parentFile?.mkdirs()
        assets.open(path).use { source ->
            FileOutputStream(destination).use { source.copyTo(it) }
        }
    } else {
        children.forEach { copyAssetTree(assets, "$path/$it", File(destination, it)) }
    }
}

internal class SpeechStream(model: Model, sampleRate: Int, config: JSONObject) : AutoCloseable {
    private data class Word(val text: String, val start: Double, val end: Double)

    private val recognizer = Recognizer(model, sampleRate.toFloat())
    private val minConfidence = config.getDouble("min_word_conf")
    private val contextSeconds = maxOf(config.getDouble("context_s"), config.getDouble("gps_context_s"))
    private val words = ArrayDeque<Word>()
    private var partial = ""

    init {
        recognizer.setWords(true)
    }

    fun feed(now: Double, pcm: ByteArray): Pair<String, Boolean> {
        val finished = recognizer.acceptWaveForm(pcm, pcm.size)
        if (!finished) {
            partial = JSONObject(recognizer.partialResult).optString("partial")
            return partial to false
        }
        val result = JSONObject(recognizer.result).optJSONArray("result")
        val accepted = mutableListOf<String>()
        if (result != null) {
            for (i in 0 until result.length()) {
                val word = result.getJSONObject(i)
                if (word.getDouble("conf") < minConfidence) continue
                val text = word.getString("word")
                words.addLast(Word(text, word.getDouble("start"), word.getDouble("end")))
                accepted += text
            }
        }
        while (words.firstOrNull()?.let { it.end <= now - contextSeconds } == true) {
            words.removeFirst()
        }
        partial = ""
        return accepted.joinToString(" ") to true
    }

    fun transcript(from: Double, to: Double): String {
        val finalized = words.asSequence().filter { it.end > from && it.start <= to }
            .joinToString(" ") { it.text }
        return "$finalized $partial".trim()
    }

    override fun close() = recognizer.close()
}
