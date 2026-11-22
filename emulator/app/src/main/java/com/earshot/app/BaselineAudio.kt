package com.earshot.app

import org.json.JSONObject

internal data class AudioWindow(val endTime: Double, val pcm: ByteArray)

internal class BaselineAudio(config: JSONObject, rate: Int, private val packetBytes: Int) {
    private val secondsPerPacket = packetBytes / (rate * 2.0)
    private val windowPackets = packetsFor(config.getDouble("chunk_s"))
    private val hopPackets = packetsFor(config.getDouble("hop_s"))
    private val minimumTail = packetsFor(config.getDouble("min_tail_s"))
    private val packets = ArrayDeque<ByteArray>()
    private var startPacket = 0

    init {
        require(hopPackets <= windowPackets && minimumTail <= windowPackets)
    }

    fun add(pcm: ByteArray): AudioWindow? {
        packets.addLast(pcm)
        if (packets.size < windowPackets) return null
        val window = audioWindow()
        repeat(hopPackets) { packets.removeFirst() }
        startPacket += hopPackets
        return window
    }

    fun finish(): AudioWindow? = if (packets.size >= minimumTail) audioWindow() else null

    private fun audioWindow(): AudioWindow {
        val pcm = ByteArray(packets.size * packetBytes)
        packets.forEachIndexed { index, packet -> packet.copyInto(pcm, index * packetBytes) }
        return AudioWindow((startPacket + packets.size) * secondsPerPacket, pcm)
    }

    private fun packetsFor(seconds: Double): Int {
        val frames = seconds / secondsPerPacket
        require(frames > 0 && frames == frames.toInt().toDouble()) {
            "Baseline timing must use whole audio packets"
        }
        return frames.toInt()
    }
}
