package com.earshot.app

import org.json.JSONObject
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.round

internal data class Coordinate(val lat: Double, val lon: Double)

internal data class GpsState(
    val speed: Double = 0.0,
    val offRouteMeters: Double = 0.0,
    val stoppedSeconds: Double = 0.0,
    val suddenStop: Boolean = false,
    val offRoute: Boolean = false,
    val longStop: Boolean = false,
) {
    val flags: Set<String>
        get() = buildSet {
            if (offRoute) add("left the planned route")
            if (suddenStop) add("sudden stop")
            if (longStop) add("long stop away from pickup/drop off")
        }

    fun toJudgeJson() = JSONObject()
        .put("speed_kmh", round(speed * 10) / 10)
        .put("off_route_m", round(offRouteMeters))
        .put("sudden_stop", suddenStop)
        .put("stopped_s", round(stoppedSeconds * 10) / 10)
}

internal class GpsTracker(private val config: JSONObject, route: List<Coordinate>) {
    private data class Point(val x: Double, val y: Double) {
        fun distance(other: Point) = hypot(x - other.x, y - other.y)
    }

    private val origin = route.first()
    private val longitudeScale = METRES_PER_DEGREE * cos(Math.toRadians(origin.lat))
    private val path = route.map(::project)
    private val recentSpeeds = ArrayDeque<Pair<Double, Double>>()
    private var offSince: Double? = null
    private var stoppedSince: Double? = null
    private var suddenUntil = -1.0

    fun update(time: Double, coordinate: Coordinate, speed: Double): GpsState {
        val point = project(coordinate)
        val offMeters = (0 until path.lastIndex).minOf { i ->
            val a = path[i]
            val b = path[i + 1]
            val dx = b.x - a.x
            val dy = b.y - a.y
            val u = (((point.x - a.x) * dx + (point.y - a.y) * dy) /
                maxOf(dx * dx + dy * dy, 1e-9)).coerceIn(0.0, 1.0)
            point.distance(Point(a.x + u * dx, a.y + u * dy))
        }
        offSince = if (offMeters > config.getDouble("off_route_m")) offSince ?: time else null
        val offRoute = offSince?.let { time - it >= config.getDouble("off_route_hold_s") } ?: false

        recentSpeeds.addLast(time to speed)
        while (time - recentSpeeds.first().first > config.getDouble("sudden_stop_window_s")) {
            recentSpeeds.removeFirst()
        }
        if (recentSpeeds.maxOf { it.second } - speed > config.getDouble("sudden_stop_drop_kmh")) {
            suddenUntil = time + config.getDouble("sudden_stop_latch_s")
        }

        stoppedSince = if (speed < config.getDouble("stopped_speed_kmh")) stoppedSince ?: time else null
        val stopped = stoppedSince?.let { time - it } ?: 0.0
        val nearEnd = minOf(point.distance(path.first()), point.distance(path.last())) <=
            config.getDouble("endpoint_radius_m")
        val longStop = stopped > config.getDouble("stopped_hold_s") && !nearEnd
        return GpsState(speed, offMeters, stopped, time <= suddenUntil, offRoute, longStop)
    }

    private fun project(coordinate: Coordinate) =
        Point((coordinate.lon - origin.lon) * longitudeScale, (coordinate.lat - origin.lat) * METRES_PER_DEGREE)

    private companion object {
        const val METRES_PER_DEGREE = 111_320.0
    }
}
