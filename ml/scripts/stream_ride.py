"""Stand-in for Unity: serve one ride over the demo socket protocol (plan 1.1), in real time.

Lets the Android app be developed and tested without Unity:
    uv run python scripts/stream_ride.py ../shared/scenarios/demo/demo_03_threat.json [--speed 1.0] [--play]
    adb reverse tcp:5005 tcp:5005        # then the app connects to localhost:5005

Newline delimited JSON: start, then every 0.5 s one gps + one audio message with the same t, then stop.
Ground truth (`expect`) is never sent. --play also plays the audio on the laptop speakers.
"""
from __future__ import annotations

import argparse
import base64
import json
import socket
import time

from earshot.common import SR, to_pcm16
from earshot.rides import load_ride


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--speed", type=float, default=1.0, help="2.0 = twice as fast as real time")
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()
    ride = load_ride(args.scenario)
    scn, audio, gps = ride["scn"], ride["audio"], ride["gps"]
    hop = SR // 2

    srv = socket.create_server(("0.0.0.0", args.port))
    print(f"{scn['id']}: waiting for the app on port {args.port} ...")
    conn, addr = srv.accept()
    print(f"connected: {addr}")

    def send(msg: dict) -> None:
        conn.sendall((json.dumps(msg, separators=(",", ":")) + "\n").encode())

    send({"type": "start", "ride_id": scn["id"], "route": ride["route"]})
    if args.play:
        import sounddevice as sd
        sd.play(audio, int(SR * args.speed))
    t0 = time.monotonic()
    for i in range(len(audio) // hop):
        t = i * 0.5
        time.sleep(max(0.0, t0 + (t + 0.5) / args.speed - time.monotonic()))  # a chunk is sent once it has "happened"
        if i < len(gps):
            send({"type": "gps", **gps[i]})
        send({"type": "audio", "t": t, "sr": SR,
              "pcm16_b64": base64.b64encode(to_pcm16(audio[i * hop:(i + 1) * hop]).tobytes()).decode()})
    send({"type": "stop", "ride_id": scn["id"]})
    conn.close()
    print("ride finished")


if __name__ == "__main__":
    main()
