package com.earshot.app

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

internal class EventJudge {
    fun evaluate(time: Double, evidence: JSONObject): RideDecision {
        return try {
            val reply = send(evidence)
            val decision = reply.getString("decision")
            require(decision in setOf("alert", "dismiss", "check_in"))
            RideDecision(
                time, decision, "llm", reply.optInt("severity", 2).coerceIn(1, 3),
                reply.optString("reason"), reply.optJSONObject("usage")?.optInt("total_tokens") ?: 0,
            )
        } catch (_: Exception) {
            RideDecision(time, "check_in", "offline", 2, "Judge backend unavailable. Event needs review.")
        }
    }

    private fun send(evidence: JSONObject): JSONObject {
        val connection = URL("http://127.0.0.1:5006/judge").openConnection() as HttpURLConnection
        try {
            connection.instanceFollowRedirects = false
            connection.requestMethod = "POST"
            connection.connectTimeout = 2_000
            connection.readTimeout = 12_000
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json")
            connection.outputStream.use { it.write(evidence.toString().toByteArray(Charsets.UTF_8)) }
            return connection.inputStream.bufferedReader().use { JSONObject(it.readText()) }
        } finally {
            connection.disconnect()
        }
    }
}
