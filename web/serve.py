"""Local page server for the WebNN runs: static files with the cross-origin isolation headers (threads and
SharedArrayBuffer in ONNX Runtime Web), plus two POST endpoints the page uses to hand its results back to the disk:
  POST /log/<tag>      appends the body (one text line) to <results>/<tag>.log
  POST /result/<tag>   writes the body to <results>/<tag>.json and an empty <results>/<tag>.done
Binds to 127.0.0.1 only.

    python serve.py --root <dir with index.html, ort/, models/> --results <dir> --port 8780
"""

import argparse
import functools
import os
import re
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8")
SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class Handler(SimpleHTTPRequestHandler):
    results = "."
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".wasm": "application/wasm",
                      ".onnx": "application/octet-stream", ".bin": "application/octet-stream", ".json": "application/json"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) != 2 or parts[0] not in ("log", "result") or not SAFE.match(parts[1]):
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        tag = parts[1]
        if parts[0] == "log":
            with open(os.path.join(self.results, tag + ".log"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + body.decode("utf-8", "replace").rstrip("\n") + "\n")
        else:
            with open(os.path.join(self.results, tag + ".json"), "wb") as f:
                f.write(body)
            open(os.path.join(self.results, tag + ".done"), "w").close()
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):  # quiet: the page logs through /log
        pass


class Server(ThreadingHTTPServer):
    # http.server sets SO_REUSEADDR, and on Windows that lets this server bind a port another server already listens
    # on (the page then talks to whichever answers). Without it the bind fails when the port is taken.
    allow_reuse_address = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    os.makedirs(args.results, exist_ok=True)
    Handler.results = os.path.abspath(args.results)
    if not os.path.isfile(os.path.join(args.root, "index.html")):
        sys.exit(f"no index.html in --root {args.root}")
    try:
        server = Server(("127.0.0.1", args.port), functools.partial(Handler, directory=os.path.abspath(args.root)))
    except OSError as error:
        sys.exit(f"cannot listen on 127.0.0.1:{args.port} ({error}); is another server running there?")
    print(f"serving {args.root} on http://127.0.0.1:{args.port}/, results -> {args.results}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
