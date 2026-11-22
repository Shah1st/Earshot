"""Loopback-only judge endpoint for the Android demo."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MAX_BODY = 32 * 1024


def make_server(evaluate, port=5006):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
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

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    sys.path.insert(0, str(ROOT / "ml"))
    from earshot.common import load_config, openrouter_key
    from earshot.judge import Judge

    try:
        openrouter_key()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None

    judge = Judge(load_config(ROOT / "demo_pack" / "config.json"))
    with make_server(judge) as server:
        print("Earshot judge listening on http://127.0.0.1:5006", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
