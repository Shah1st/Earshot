# Test Earshot on Android without Unity

Open the Android project in Android Studio and run the debug app on the Android Emulator.
Gradle packages the models and configuration from the repository's `demo_pack` directory.

With the emulator running, open a terminal at the repository root and run:

```powershell
adb reverse tcp:5005 tcp:5005
python emulator/tools/replay_ride.py demo_4_normal
```

Tap **Connect to replay** in Earshot. The app receives the planned route, then one GPS fix
and 8,000 PCM16 audio samples per half second. During the replay, compare the audio and
GPS counts and watch the latest position, YAMNet score, and speech. The header shows
**Ride complete** after the end message. The normal ride ends at 33 seconds with 66 audio
chunks, 66 GPS fixes, and no decisions. Start the Python command again for another ride.

The replay accepts any of the five ride IDs from `demo_pack/rides`. Use `demo_2_roadworks`
to exercise the GPS route deviation.

## Run the judge backend

Set `OPENROUTER_API_KEY` in the backend terminal environment or in the ignored `.env`
file at the repository root. With the emulator running, run these commands from the
repository root.

```powershell
adb reverse tcp:5006 tcp:5006
uv run --project ml --no-sync python emulator/tools/judge_backend.py
```

The command uses the existing ML Python environment. Run `uv sync --project ml` first
if that environment has not been set up. Android sends event evidence, including a
short transcript, to the backend when a judge is needed. If the backend is unavailable,
the local models and rules still run and judge-needed events appear as offline check-ins.

To switch to Unity, close the Python server and press Play in the 3D city scene or the
reference replay scene. Tap **Connect to replay** when Unity shows **Waiting for Android**.
Both scenes use the same port and messages. The city streams GPS from the moving taxi.
The `EarshotRide` logcat tag records completion and decision timestamps.
