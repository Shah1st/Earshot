package com.earshot.app

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import java.util.Locale

@Composable
internal fun RideScreen(
    ride: RideView,
    connect: () -> Unit,
    disconnect: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
        Column(
            Modifier.widthIn(max = 560.dp).fillMaxWidth().verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically) {
                Text("Earshot", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                RideStatus(ride.phase)
            }

            if (ride.phase == RidePhase.Ready) {
                Text("Start the Python replay on your computer, then connect to monitor the ride.",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            ride.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium) }

            if (ride.rideId.isNotEmpty()) {
                RideSignals(ride)
            }
            if (ride.phase.active) {
                OutlinedButton(onClick = disconnect, modifier = Modifier.fillMaxWidth()) {
                    Text("Disconnect")
                }
            } else {
                Button(onClick = connect, modifier = Modifier.fillMaxWidth()) {
                    Text("Connect to replay")
                }
            }

            if (ride.rideId.isNotEmpty()) {
                RideDecisions(ride)
                if (ride.transcript.isNotBlank()) {
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Recent speech", style = MaterialTheme.typography.titleSmall)
                        Text(ride.transcript, style = MaterialTheme.typography.bodyMedium, maxLines = 4,
                            overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
    }
}

@Composable
private fun RideStatus(phase: RidePhase) {
    val label = when (phase) {
        RidePhase.Ready -> "Ready"
        RidePhase.Loading -> "Loading models"
        RidePhase.Connecting -> "Connecting"
        RidePhase.Live -> "Live"
        RidePhase.Complete -> "Ride complete"
        RidePhase.Disconnected -> "Disconnected"
        RidePhase.Error -> "Stream error"
    }
    val colors = MaterialTheme.colorScheme
    val (background, content) = when (phase) {
        RidePhase.Error -> colors.errorContainer to colors.onErrorContainer
        RidePhase.Live, RidePhase.Complete -> colors.tertiaryContainer to colors.onTertiaryContainer
        else -> colors.surfaceVariant to colors.onSurfaceVariant
    }
    Surface(
        modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
        shape = CircleShape, color = background, contentColor = content,
    ) {
        Row(Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(6.dp).background(content, CircleShape))
            Text(label, style = MaterialTheme.typography.labelLarge)
        }
    }
}

@Composable
private fun RideSignals(ride: RideView) {
    val typography = MaterialTheme.typography
    val colors = MaterialTheme.colorScheme
    Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.large, color = colors.surfaceContainer) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Text(rideName(ride.rideId), Modifier.weight(1f), style = typography.titleLarge)
                Text("${number(ride.elapsed, 1)} s", style = typography.titleMedium, color = colors.primary)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                SignalCount("Audio chunks", ride.chunks, Modifier.weight(1f))
                SignalCount("GPS fixes", ride.chunks, Modifier.weight(1f))
            }
            HorizontalDivider()
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                val rate = String.format(Locale.US, "%,d", ride.audioRate)
                Text("$rate Hz · PCM16 · ${number(ride.chunkSeconds, 1)} s chunks",
                    style = typography.bodySmall, color = colors.onSurfaceVariant)
                Text("Loud event score ${number(ride.loudScore, 3)}", style = typography.bodyMedium)
                LinearProgressIndicator(progress = { ride.loudScore.toFloat().coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth())
            }
            ride.location?.let { point ->
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("${ride.gps.speed.toInt()} km/h · ${ride.gps.offRouteMeters.toInt()} m from route",
                        style = typography.bodyMedium)
                    Text("${number(point.lat, 5)}, ${number(point.lon, 5)}",
                        style = typography.bodySmall, color = colors.onSurfaceVariant)
                    if (ride.gps.flags.isNotEmpty()) {
                        Text(ride.gps.flags.joinToString().replaceFirstChar(Char::uppercaseChar),
                            style = typography.labelMedium, color = colors.tertiary)
                    }
                }
            }
        }
    }
}

@Composable
private fun SignalCount(label: String, count: Int, modifier: Modifier) {
    Surface(modifier, shape = MaterialTheme.shapes.medium, color = MaterialTheme.colorScheme.surface) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
            Text(count.toString(), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
            Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun RideDecisions(ride: RideView) {
    val tokens = ride.decisions.sumOf { it.tokens }
    val colors = MaterialTheme.colorScheme
    val typography = MaterialTheme.typography
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically) {
        Text("Decisions", style = typography.titleMedium)
        Text("$tokens judge tokens", style = typography.labelMedium, color = colors.onSurfaceVariant)
    }
    if (ride.decisions.isEmpty()) {
        Text(if (ride.phase == RidePhase.Complete) "No safety decisions" else "No decisions yet",
            style = typography.bodyMedium, color = colors.onSurfaceVariant)
    }
    for (decision in ride.decisions.asReversed()) {
        val alert = decision.decision == "alert"
        Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium,
            color = if (alert) colors.errorContainer else colors.secondaryContainer,
            contentColor = if (alert) colors.onErrorContainer else colors.onSecondaryContainer) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically) {
                    val label = when {
                        decision.summary -> "Alert update"
                        decision.decision == "check_in" -> "Check-in"
                        else -> decision.decision.replaceFirstChar(Char::uppercaseChar)
                    }
                    Text(label, style = typography.titleSmall)
                    Text("${number(decision.time, 1)} s", style = typography.labelLarge)
                }
                Text(decision.reason, style = typography.bodyMedium)
            }
        }
    }
}

private fun rideName(id: String): String {
    val parts = id.split('_', limit = 3)
    return if (parts.size == 3 && parts[0] == "demo" && parts[1].toIntOrNull() != null) {
        "Demo ${parts[1]} · ${parts[2].replace('_', ' ').replaceFirstChar(Char::uppercaseChar)}"
    } else id.replace('_', ' ').replaceFirstChar(Char::uppercaseChar)
}

private fun number(value: Double, decimals: Int) = String.format(Locale.US, "%.${decimals}f", value)
