"""Validate one Unity reference ride as a TCP client, without Android."""

import argparse
import base64
import json
from pathlib import Path
import socket
import time
import wave


ASSETS = Path(__file__).resolve().parents[1] / "Assets" / "StreamingAssets" / "ReferenceRides"


def expect(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Unexpected {label}")


def check(host, port, ride_id, speed=1.0):
    manifest = json.loads((ASSETS / "replay.json").read_text(encoding="utf-8"))
    entry = next(ride for ride in manifest["rides"] if ride["id"] == ride_id)
    route = [[point["lat"], point["lon"]] for point in entry["route"]]
    fixes = [json.loads(line) for line in (ASSETS / f"{ride_id}.gps.jsonl").read_text(encoding="utf-8").splitlines()]
    rate = manifest["sampleRate"]
    samples_per_tick = manifest["samplesPerTick"]
    tick_seconds = samples_per_tick / rate
    with (
        wave.open(str(ASSETS / f"{ride_id}.wav"), "rb") as wav,
        socket.create_connection((host, port), timeout=5) as connection,
        connection.makefile("rb") as stream,
    ):
        def receive():
            line = stream.readline()
            if not line.endswith(b"\n"):
                raise ValueError("Missing newline-delimited JSON message")
            return json.loads(line)

        expect((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (rate, 1, 2), "WAV format")
        expect(receive(), {"type": "start", "ride_id": ride_id, "route": route}, "start message")
        start = time.monotonic()
        max_early = 0
        for index, fix in enumerate(fixes):
            expect(fix["t"], index * tick_seconds, f"fixture time at tick {index}")
            expect(receive(), {"type": "gps", **fix}, f"GPS message at tick {index}")
            packet = receive()
            actual_pcm = base64.b64decode(packet.pop("pcm16_b64"), validate=True)
            expect(packet, {"type": "audio", "t": fix["t"], "sr": rate}, f"audio metadata at tick {index}")
            expect(len(actual_pcm), samples_per_tick * 2, f"PCM byte count at tick {index}")
            expect(actual_pcm, wav.readframes(samples_per_tick), f"PCM bytes at tick {index}")
            max_early = max(max_early, (index + 1) * tick_seconds / speed - (time.monotonic() - start))

        expect(receive(), {"type": "stop", "ride_id": ride_id}, "stop message")
        expect(stream.read(1), b"", "connection close")
        expect(wav.readframes(1), b"", "WAV end")

    elapsed = time.monotonic() - start
    planned = len(fixes) * tick_seconds / speed
    if max_early > 0.15 or elapsed > planned + 2:
        raise ValueError(f"Unexpected pacing: {elapsed:.2f} s instead of about {planned:.2f} s")
    print(f"{ride_id}: verified {len(fixes)} GPS/audio pairs and stream closure in {elapsed:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ride", default="demo_2_roadworks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--speed", type=float, default=1.0, help="Clock multiplier for a test harness")
    options = parser.parse_args()
    if options.speed <= 0:
        parser.error("--speed must be positive")
    check(options.host, options.port, options.ride, options.speed)
