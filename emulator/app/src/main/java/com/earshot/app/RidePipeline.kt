package com.earshot.app

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.round

internal data class RideDecision(
    val time: Double,
    val decision: String,
    val source: String,
    val severity: Int,
    val reason: String,
    val tokens: Int = 0,
    val summary: Boolean = false,
    val backendUnavailable: Boolean = false,
)

internal class RidePipeline(
    config: JSONObject,
    route: List<Coordinate>,
    private val sound: SoundModel,
    private val speech: SpeechStream,
    private val judge: EventJudge,
) {
    private data class Group(
        val labels: List<Int>,
        val head: Boolean,
        val trigger: Boolean,
        val sustained: Boolean,
        val threshold: Double,
        val high: Double,
        val recent: ArrayDeque<Double> = ArrayDeque(),
        var run: Int = 0,
    )
    private data class Event(
        val opened: Double,
        val trigger: String,
        val note: String,
        val classMax: FloatArray,
        val groupMax: MutableMap<String, Double> = mutableMapOf(),
        var activeWindows: Int = 0,
    )

    private val eventConfig = config.getJSONObject("event")
    private val asr = config.getJSONObject("asr")
    private val audio = config.getJSONObject("audio")
    private val gpsConfig = config.getJSONObject("gps")
    private val wordless = config.getJSONObject("wordless_rule")
    private val gpsTracker = GpsTracker(gpsConfig, route)
    private val keywords = asr.strings("keywords").sortedByDescending(String::length)
    private val heard = mutableMapOf<String, Int>()
    private val groups = config.getJSONObject("groups").let { all ->
        all.keys().asSequence().associateWith { name ->
            val group = all.getJSONObject(name)
            Group(
                group.strings("classes").map(sound.labels::indexOf).onEach { require(it >= 0) },
                group.optString("source") == "head",
                group.optBoolean("trigger", true),
                group.getString("type") == "sustained",
                group.getDouble("threshold"),
                group.getDouble("high"),
            )
        }
    }
    private val recent = ArrayDeque<FloatArray>()
    private val smoothWindows = eventConfig.getInt("smooth_windows")
    private val sustainWindows = eventConfig.getInt("sustain_windows")
    private val cooldown = eventConfig.getDouble("cooldown_s")
    private val chunkSeconds = audio.getDouble("chunk_s")
    private var event: Event? = null
    private var lastTrigger = Double.NEGATIVE_INFINITY
    private var alreadyAlerted = false

    var gps = GpsState()
        private set
    var loudScore = 0.0
        private set
    val decisions = mutableListOf<RideDecision>()

    fun onGps(time: Double, point: Coordinate, speed: Double) {
        val previous = gps.flags
        gps = gpsTracker.update(time, point, speed)
        for (flag in gps.flags - previous) {
            if (event != null || time - lastTrigger < cooldown) continue
            if (gpsConfig.optBoolean("judge_anomalies", true)) {
                open(time, "gps", flag)
            } else {
                decisions += RideDecision(time, "check_in", "gps", 1, flag)
            }
        }
    }

    fun onAudio(time: Double, pcm: ByteArray) {
        val now = time + chunkSeconds
        val frame = sound.infer(pcm)
        loudScore = frame.head
        recent.addLast(frame.scores)
        if (recent.size > smoothWindows) recent.removeFirst()

        val groupScores = groups.mapValues { (_, group) ->
            if (group.head) frame.head else group.labels.maxOf { frame.scores[it].toDouble() }
        }
        val fired = groups.keys.filter { name -> groups.getValue(name).fires(groupScores.getValue(name)) }
        val (utterance, finished) = speech.feed(now, pcm)
        var keywordTriggered = false
        if (asr.optBoolean("keyword_trigger", true)) {
            for ((keyword, count) in keywordCounts(utterance)) {
                if (count > heard.getOrDefault(keyword, 0)) {
                    heard[keyword] = count
                    keywordTriggered = true
                }
            }
        }
        if (finished) heard.clear()

        if (fired.isNotEmpty() || keywordTriggered) {
            if (event == null && now - lastTrigger >= cooldown) {
                open(now, if (fired.isNotEmpty()) "sound" else "keyword")
            }
            lastTrigger = now
            val high = fired.firstOrNull { groupScores.getValue(it) >= groups.getValue(it).high }
            if (high != null && gps.flags.isNotEmpty() && !alreadyAlerted) {
                alreadyAlerted = true
                decisions += RideDecision(
                    now, "alert", "rule", 3,
                    "$high ${formatScore(groupScores.getValue(high))} + ${gps.flags.joinToString()}",
                )
            }
        }

        event?.let { active ->
            mergeScores(active.classMax, frame.scores)
            for ((name, score) in groupScores) {
                active.groupMax[name] = maxOf(active.groupMax.getOrDefault(name, 0.0), score)
            }
            if (groups.any { (name, group) -> group.trigger && groupScores.getValue(name) >= group.threshold }) {
                active.activeWindows++
            }
            if (now >= active.opened + audio.getDouble("clip_post_s")) {
                decisions += seal(now, active)
                event = null
            }
        }
    }

    private fun Group.fires(score: Double): Boolean {
        if (!trigger) return false
        if (!sustained) return score >= threshold
        recent.addLast(score)
        if (recent.size > smoothWindows) recent.removeFirst()
        run = if (recent.average() >= threshold) run + 1 else 0
        return run >= sustainWindows
    }

    private fun open(time: Double, trigger: String, note: String = "") {
        val maxima = FloatArray(sound.labels.size)
        for (scores in recent) mergeScores(maxima, scores)
        alreadyAlerted = false
        event = Event(time, trigger, note, maxima)
        lastTrigger = time
    }

    private fun mergeScores(maxima: FloatArray, scores: FloatArray) {
        for (i in maxima.indices) maxima[i] = maxOf(maxima[i], scores[i])
    }

    private fun seal(now: Double, active: Event): RideDecision {
        if (wordless.optBoolean("enabled") && active.trigger == "sound") {
            val words = speech.transcript(active.opened - 5, now)
                .split(' ').count(String::isNotBlank)
            if (words <= wordless.getInt("max_words")) {
                val benign = wordless.strings("benign_labels").maxOf { active.classMax[sound.labels.indexOf(it)] }
                return when {
                    alreadyAlerted -> RideDecision(now, "alert", "rule+summary", 3, "Loud sound, no speech", summary = true)
                    benign >= wordless.getDouble("benign_min_score") ->
                        RideDecision(now, "dismiss", "rule", 1, "Loud sound with laughter or cough labels, no speech")
                    gps.flags.isNotEmpty() ->
                        RideDecision(now, "alert", "rule", 3, "Loud sound with no speech during a GPS anomaly")
                    else -> RideDecision(now, "check_in", "rule", 2, "Loud sound with no speech")
                }
            }
        }

        val span = if (active.trigger == "gps") asr.getDouble("gps_context_s") else asr.getDouble("context_s")
        val transcript = speech.transcript(now - span, now)
        val top = JSONObject()
        for (i in active.classMax.indices.sortedByDescending { active.classMax[it] }
            .take(eventConfig.getInt("top_labels"))) {
            val value = active.classMax[i].toDouble()
            if (value >= eventConfig.getDouble("top_label_min_score")) {
                top.put(sound.labels[i], formatScore(value))
            }
        }
        val trigger = if (active.note.isEmpty()) active.trigger else "${active.trigger}: ${active.note}"
        val evidence = JSONObject()
            .put("t", active.opened)
            .put("trigger", trigger)
            .put("groups", JSONObject(active.groupMax.filterValues { it >= 0.1 }.mapValues { formatScore(it.value) }))
            .put("duration_s", active.activeWindows * chunkSeconds)
            .put("top_labels", top)
            .put("gps", gps.toJudgeJson())
            .put("keywords", JSONArray(keywordCounts(transcript).keys.toList()))
            .put("transcript", transcript)
        val earlier = decisions.filter { !it.summary }.takeLast(3).map {
            JSONObject().put("t", it.time).put("decision", it.decision).put("reason", it.reason)
        }
        if (earlier.isNotEmpty()) evidence.put("earlier", JSONArray(earlier))
        val reply = judge.evaluate(now, evidence)
        return if (alreadyAlerted) reply.copy(decision = "alert", source = "rule+summary", severity = 3, summary = true) else reply
    }

    private fun keywordCounts(text: String): Map<String, Int> {
        var remaining = " $text "
        val matches = mutableMapOf<String, Int>()
        for (keyword in keywords) {
            val phrase = " $keyword "
            var index = remaining.indexOf(phrase)
            while (index >= 0) {
                matches[keyword] = matches.getOrDefault(keyword, 0) + 1
                remaining = remaining.replaceRange(index, index + phrase.length, " # ")
                index = remaining.indexOf(phrase)
            }
        }
        return matches
    }

    private fun formatScore(score: Double) = round(score * 100) / 100
}
