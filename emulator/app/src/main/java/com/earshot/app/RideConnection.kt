package com.earshot.app

import android.content.Context
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.CopyOnWriteArrayList

internal enum class RidePhase {
    Ready, Loading, Connecting, Live, Comparing, Complete, Disconnected, Error;

    val active: Boolean get() = this == Loading || this == Connecting || this == Live || this == Comparing
}

internal data class RideView(
    val phase: RidePhase = RidePhase.Ready,
    val error: String? = null,
    val rideId: String = "",
    val elapsed: Double = 0.0,
    val chunks: Int = 0,
    val location: Coordinate? = null,
    val gps: GpsState = GpsState(),
    val loudScore: Double = 0.0,
    val audioRate: Int = 0,
    val chunkSeconds: Double = 0.0,
    val transcript: String = "",
    val decisions: List<RideDecision> = emptyList(),
    val baselineDecisions: List<RideDecision> = emptyList(),
)

internal class RideConnection : AutoCloseable {
    private val socket = Socket()

    suspend fun run(context: Context, update: (RideView) -> Unit) {
        val config = context.assets.readJson("config.json")
        val sampleRate = config.getJSONObject("audio").getInt("sr")
        loadSpeechModel(context, config).use { model ->
            SpeechStream(model, sampleRate, config.getJSONObject("asr")).use { speech ->
                SoundModel(context.assets, config).use { sound ->
                    update(RideView(phase = RidePhase.Connecting))
                    socket.connect(InetSocketAddress("127.0.0.1", 5005), 4_000)
                    socket.soTimeout = 20_000
                    socket.getInputStream().bufferedReader(Charsets.UTF_8).use { stream ->
                        receive(stream, config, sound, speech, update)
                    }
                }
            }
        }
    }

    private suspend fun receive(
        stream: BufferedReader,
        config: JSONObject,
        sound: SoundModel,
        speech: SpeechStream,
        update: (RideView) -> Unit,
    ) = coroutineScope {
        val audio = config.getJSONObject("audio")
        val rate = audio.getInt("sr")
        val tick = audio.getDouble("chunk_s")
        val bytesPerChunk = (rate * tick).toInt() * Short.SIZE_BYTES
        val start = stream.message(
            "No replay started. Press Play in Unity and wait for Android, or start the Python replay.",
        )
        require(start.getString("type") == "start") { "Expected a ride start message" }
        val ride = start.getString("ride_id")
        val planned = start.getJSONArray("route")
        require(ride.isNotBlank() && planned.length() >= 2) { "Missing ride ID or planned route" }
        val route = List(planned.length()) { coordinate(planned.getJSONArray(it)) }
        val judge = EventJudge()
        val pipeline = RidePipeline(config, route, sound, speech, judge)
        val baseline = BaselineAudio(config.getJSONObject("baseline"), rate, bytesPerChunk)
        val pending = Channel<AudioWindow>(capacity = 2)
        val baselineResults = CopyOnWriteArrayList<RideDecision>()
        val baselineWorker = launch {
            for (window in pending) {
                val decision = judge.evaluateAudio(window)
                baselineResults += decision
                Log.i(LOG, "$ride baseline t=${decision.time} ${decision.decision} tokens=${decision.tokens}")
            }
        }
        var ticks = 0
        var reported = 0
        var location: Coordinate? = null

        fun snapshot(phase: RidePhase = RidePhase.Live) = RideView(
            phase = phase,
            rideId = ride,
            elapsed = ticks * tick,
            chunks = ticks,
            location = location,
            gps = pipeline.gps,
            loudScore = pipeline.loudScore,
            audioRate = rate,
            chunkSeconds = tick,
            transcript = speech.transcript(ticks * tick - 10, ticks * tick),
            decisions = pipeline.decisions.toList(),
            baselineDecisions = baselineResults.toList(),
        )

        update(snapshot())
        while (true) {
            val gps = stream.message()
            if (gps.getString("type") == "stop") {
                require(gps.getString("ride_id") == ride) { "Stop message has the wrong ride ID" }
                baseline.finish()?.let { pending.send(it) }
                pending.close()
                update(snapshot(RidePhase.Comparing))
                baselineWorker.join()
                Log.i(LOG, "$ride complete chunks=$ticks decisions=${pipeline.decisions.count { !it.summary }}")
                update(snapshot(RidePhase.Complete))
                return@coroutineScope
            }
            val time = ticks * tick
            require(gps.getString("type") == "gps" && gps.getDouble("t") == time) { "Unexpected GPS at $time s" }
            val point = coordinate(JSONArray().put(gps.getDouble("lat")).put(gps.getDouble("lon")))
            val speed = gps.getDouble("speed_kmh")
            require(speed.isFinite() && speed >= 0) { "Invalid GPS speed" }
            val frame = stream.message()
            require(frame.getString("type") == "audio" && frame.getDouble("t") == time && frame.getInt("sr") == rate) {
                "Unexpected audio at $time s"
            }
            val pcm = Base64.decode(frame.getString("pcm16_b64"), Base64.DEFAULT)
            require(pcm.size == bytesPerChunk) { "Invalid audio chunk size at $time s" }
            baseline.add(pcm)?.let { pending.send(it) }
            pipeline.onGps(time, point, speed)
            pipeline.onAudio(time, pcm)
            location = point
            ticks++

            for (decision in pipeline.decisions.drop(reported)) {
                Log.i(LOG, "$ride t=${decision.time} ${decision.decision} source=${decision.source} summary=${decision.summary}")
            }
            reported = pipeline.decisions.size
            update(snapshot())
        }
    }

    private fun BufferedReader.message(eofMessage: String = "Connection closed before ride end") =
        JSONObject(readLine() ?: error(eofMessage))

    private fun coordinate(pair: JSONArray): Coordinate {
        require(pair.length() == 2) { "Expected a latitude and longitude pair" }
        return Coordinate(pair.getDouble(0), pair.getDouble(1)).also {
            require(it.lat in -90.0..90.0 && it.lon in -180.0..180.0) { "Invalid GPS coordinate" }
        }
    }

    override fun close() = socket.close()

    private companion object {
        const val LOG = "EarshotRide"
    }
}
