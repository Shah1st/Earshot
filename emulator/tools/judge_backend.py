"""Loopback-only judge endpoint for the Android demo."""

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import wave


ROOT = Path(__file__).resolve().parents[2]
MAX_BODY = 32 * 1024


def make_server(evaluate, port=5006, *, evaluate_audio=None, sample_rate=16000, audio_limits=None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/baseline" and evaluate_audio is not None:
                self.handle_baseline()
                return
            if self.path != "/judge":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            if self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "Expected JSON"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self.reply(413, {"error": "Invalid event length"})
                    return
                event = json.loads(self.rfile.read(length))
                if not isinstance(event, dict) or not isinstance(event.get("transcript"), str):
                    raise ValueError("Invalid event")
                if not isinstance(event.get("trigger"), str):
                    raise ValueError("Invalid trigger")
            except ValueError:
                self.reply(400, {"error": "Invalid event"})
                return

            try:
                result = evaluate(event)
            except Exception:
                self.reply(502, {"error": "Judge unavailable"})
                return
            self.reply(200, result)

        def handle_baseline(self):
            if self.headers.get_content_type() != "application/octet-stream":
                self.reply(415, {"error": "Expected raw PCM16 audio"})
                return
            minimum, maximum = audio_limits
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not minimum <= length <= maximum or length % 2:
                    raise ValueError("Invalid PCM length")
                pcm = self.rfile.read(length)
                if len(pcm) != length:
                    raise ValueError("Incomplete PCM audio")
            except ValueError:
                self.reply(400, {"error": "Invalid PCM audio"})
                return
            wav = io.BytesIO()
            with wave.open(wav, "wb") as audio:
                audio.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
                audio.writeframes(pcm)
            try:
                result = evaluate_audio(wav.getvalue())
            except Exception:
                self.reply(502, {"error": "Raw audio judge unavailable"})
                return
            self.reply(200, result)

        def setup(self):
            self.request.settimeout(5)
            super().setup()

        def reply(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    sys.path.insert(0, str(ROOT / "ml"))
    from earshot.common import load_config, openrouter_key
    from earshot.judge import Judge, judge_audio

    try:
        openrouter_key()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None

    config = load_config(ROOT / "demo_pack" / "config.json")
    judge = Judge(config, use_cache=False)
    baseline = config["baseline"]
    sample_rate = config["audio"]["sr"]
    limits = tuple(int(baseline[key] * sample_rate) * 2 for key in ("min_tail_s", "chunk_s"))

    def evaluate_audio(wav):
        return judge_audio(baseline, wav, use_cache=False)

    with make_server(judge, evaluate_audio=evaluate_audio, sample_rate=sample_rate, audio_limits=limits) as server:
        print("Earshot judge listening on http://127.0.0.1:5006", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
