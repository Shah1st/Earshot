# Earshot Unity simulator

Open the project in the Unity version pinned in `ProjectSettings/ProjectVersion.txt`. Open
`Assets/Scenes/EarshotReferenceReplay.unity` for the Android integration demo. The scene shows the
original navigation route in blue, the actual GPS trail in green, and the latest GPS fix and speed.

1. Enter Play Mode and choose a named scenario in the Game view.
2. The simulator waits on `127.0.0.1:5005` and plays the scenario when a client connects.
3. For a local check, run `python simulation/tools/check_reference_stream.py --ride <ride-id>`
   from the repository root. For Android, run
   `adb reverse tcp:5005 tcp:5005` and connect the app to `127.0.0.1:5005` instead.
4. Click **Reload scene** to return to the picker while staying in Play Mode.

Each connection sends a `start` message with the planned route as `[lat, lon]` pairs.
Each 0.5-second tick sends a `gps` message and an `audio` message with the same ride-relative
timestamp. Audio contains 8,000 little-endian mono PCM16 samples at 16 kHz, encoded as base64.
The first block has timestamp `0.0` and is sent after it finishes playing. A `stop` message ends
the ride, then the simulator closes the TCP connection.

The five fixtures are in `Assets/StreamingAssets/ReferenceRides`. The replay uses the WAVs and
GPS traces exported in the ML demo pack at `1453e82`. The manifest contains the corresponding
planned route. Fetch Git LFS content before opening either scene.

For a test with the moving taxi, open `Assets/Scenes/LauzHackSimulation.unity`, press Play,
and choose a scenario. The city scene has the same **Reload scene** button.
When Unity shows **Waiting for Android**, tap **Connect to replay** in Earshot. The taxi starts
driving and playing the selected WAV at pickup. Android receives the navigator's original route,
the taxi's live GPS position and speed, and the matching audio chunks until the WAV ends.
Live GPS can produce different decisions from the reference replay.

Clear **Stream To Android** on the taxi's **Taxi Scenario Runner** to drive the city
without connecting. The `playReferenceRide` and
`cues` fields in `Assets/Scenarios/TaxiScenarios.json` select the audio and time the movement cues.

Driving noise from Cerence In-Car McVAMPIRE (Mueller and Wolff, 2024), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Routes use data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), routed with OSRM.
