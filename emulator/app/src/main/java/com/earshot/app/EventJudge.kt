package com.earshot.app

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

internal class EventJudge {
    fun evaluate(time: Double, evidence: JSONObject): RideDecision = evaluate(
        time, "judge", "application/json", evidence.toString().toByteArray(Charsets.UTF_8), 12_000,
        "check_in", "Judge backend unavailable. Event needs review.",
    )

    fun evaluateAudio(window: AudioWindow): RideDecision = evaluate(
        window.endTime, "baseline", "application/octet-stream", window.pcm, 25_000,
        "unavailable", "Raw audio backend unavailable.",
    )

    private fun evaluate(
        time: Double,
        endpoint: String,
        contentType: String,
        body: ByteArray,
        timeout: Int,
        fallback: String,
        failureReason: String,
    ): RideDecision {
        return try {
            val reply = send(endpoint, contentType, body, timeout)
            val decision = reply.getString("decision")
            require(decision in setOf("alert", "dismiss", "check_in"))
            RideDecision(
                time, decision, "llm", reply.optInt("severity", 2).coerceIn(1, 3),
                reply.optString("reason"), reply.optJSONObject("usage")?.optInt("total_tokens")?.coerceAtLeast(0) ?: 0,
            )
        } catch (_: Exception) {
            RideDecision(time, fallback, "offline", 2, failureReason, backendUnavailable = true)
        }
    }

    private fun send(endpoint: String, contentType: String, body: ByteArray, timeout: Int): JSONObject {
        val connection = URL("http://127.0.0.1:5006/$endpoint").openConnection() as HttpURLConnection
        try {
            connection.instanceFollowRedirects = false
            connection.requestMethod = "POST"
            connection.connectTimeout = 2_000
            connection.readTimeout = timeout
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", contentType)
            connection.setFixedLengthStreamingMode(body.size)
            connection.outputStream.use { it.write(body) }
            return connection.inputStream.bufferedReader().use { JSONObject(it.readText()) }
        } finally {
            connection.disconnect()
        }
    }
}
