"""Serve a demo-pack ride on the Unity TCP protocol without starting Unity."""

import argparse
import base64
import json
from pathlib import Path
import socket
import time
import wave


DEMO_PACK = Path(__file__).resolve().parents[2] / "demo_pack"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stream_ride(ride, port=5005, speed=1.0):
    config = read_json(DEMO_PACK / "config.json")["audio"]
    rate, interval = config["sr"], config["chunk_s"]
    samples = int(rate * interval)
    scenario = read_json(DEMO_PACK / "scenarios" / f"{ride}.json")
    geometry = read_json(DEMO_PACK / scenario["route"])["geometry"]
    route = [[lat, lon] for lon, lat in geometry["coordinates"]]
    gps = [
        json.loads(line)
        for line in (DEMO_PACK / "rides" / f"{ride}.gps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with (
        wave.open(str(DEMO_PACK / "rides" / f"{ride}.wav"), "rb") as wav,
        socket.socket() as server,
    ):
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, rate):
            raise ValueError("Expected mono PCM16 at the configured sample rate")
        if wav.getnframes() != len(gps) * samples or any(fix["t"] != i * interval for i, fix in enumerate(gps)):
            raise ValueError("Expected one timestamped GPS fix per audio chunk")
        server.bind(("127.0.0.1", port))
        server.listen(1)
        print(f"Waiting for Android on 127.0.0.1:{port} ({ride})", flush=True)
        client, _ = server.accept()
        with client:
            def send(message):
                return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")

            client.sendall(send({"type": "start", "ride_id": ride, "route": route}))
            start = time.monotonic()
            for index, fix in enumerate(gps):
                target = start + (index + 1) * interval / speed
                time.sleep(max(0, target - time.monotonic()))
                pcm = base64.b64encode(wav.readframes(samples)).decode("ascii")
                client.sendall(send({"type": "gps", **fix}) + send({
                    "type": "audio", "t": fix["t"], "sr": rate, "pcm16_b64": pcm,
                }))
            client.sendall(send({"type": "stop", "ride_id": ride}))
    print(f"Sent {len(gps)} GPS/audio pairs and the stop message", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ride", nargs="?", default="demo_4_normal",
                        choices=sorted(path.stem for path in (DEMO_PACK / "rides").glob("*.wav")))
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--speed", type=float, default=1.0)
    options = parser.parse_args()
    if not 0 < options.speed < float("inf"):
        parser.error("--speed must be finite and positive")
    stream_ride(options.ride, options.port, options.speed)
