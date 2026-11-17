"""
Local dashboard for the thesis project. It only shows what has been built so far
(currently: Phase 0 = config + rate encoding) and grows one phase at a time.

Run:   .venv/bin/python src/app_server.py      then open http://127.0.0.1:8000
No training happens here; the page asks this server for real values computed by
the project's own code (config_loader.py, data.py, encoding.py).
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from config_loader import load_config
from data import load_mnist
from encoding import rate_encode

WEB_DIR = Path(__file__).resolve().parent / "web"
_encode_lock = threading.Lock()

CONFIG = load_config()
IMAGES, LABELS = load_mnist(train=False)


def info():
    return {
        "config": CONFIG,
        "config_file": "configs/experiment_config.yaml",
        "num_images": len(IMAGES),
    }


def sample(index, resample):
    index = int(index) % len(IMAGES)
    T = CONFIG["simulation"]["num_steps"]
    n_in = CONFIG["architecture"]["num_inputs"]
    with _encode_lock:
        torch.manual_seed(index * 1000 + int(resample))
        spikes = rate_encode(IMAGES[index], T)          # (T, 784), the project's own function
    pairs = torch.nonzero(spikes).tolist()              # [[t, pixel], ...]
    per_step = spikes.sum(dim=1).int().tolist()
    return {
        "index": index,
        "label": int(LABELS[index]),
        "T": T,
        "image": (IMAGES[index] * 255).round().int().tolist(),
        "spikes": pairs,
        "spikes_per_step": per_step,
        "total_spikes": int(spikes.sum()),
        "active_pixels": int((spikes.sum(dim=0) > 0).sum()),
        "encoder_comparisons": T * n_in,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/info":
            return self._json(info())
        name = "index.html" if path == "/" else path.lstrip("/")
        f = (WEB_DIR / name).resolve()
        if WEB_DIR.resolve() not in f.parents or not f.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}.get(f.suffix, "text/plain")
        self._send(200, f.read_bytes(), ctype + "; charset=utf-8")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/sample":
                return self._json(sample(body.get("index", 0), body.get("resample", 0)))
            self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Dashboard running at http://127.0.0.1:{port}   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
