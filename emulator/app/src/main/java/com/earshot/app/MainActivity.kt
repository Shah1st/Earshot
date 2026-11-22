package com.earshot.app

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.lifecycleScope
import com.earshot.app.ui.theme.EarshotTheme
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.net.ConnectException

class MainActivity : ComponentActivity() {
    private var ride by mutableStateOf(RideView())
    @Volatile private var connection: RideConnection? = null
    private var rideJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            EarshotTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { insets ->
                    RideScreen(ride, ::connect, ::disconnect, Modifier.padding(insets))
                }
            }
        }
    }

    private fun connect() {
        if (rideJob?.isActive == true) return
        val session = RideConnection()
        connection = session
        ride = RideView(phase = RidePhase.Loading)
        rideJob = lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    session.use {
                        it.run(applicationContext) { update ->
                            if (connection === session) ride = update
                        }
                    }
                }
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                showError(session, error)
            } catch (error: LinkageError) {
                showError(session, error)
            } finally {
                if (connection === session) connection = null
            }
        }
    }

    private fun showError(session: RideConnection, error: Throwable) {
        if (connection !== session) return
        Log.w("EarshotRide", "Ride failed", error)
        val message = if (error is ConnectException) {
            "Start a replay on your computer and check adb reverse for port 5005."
        } else {
            error.message ?: "The ride ended with an error."
        }
        ride = ride.copy(phase = RidePhase.Error, error = message)
    }

    private fun disconnect() {
        val session = connection
        connection = null
        session?.close()
        rideJob?.cancel()
        ride = ride.copy(phase = RidePhase.Disconnected)
    }

    override fun onDestroy() {
        disconnect()
        super.onDestroy()
    }
}
