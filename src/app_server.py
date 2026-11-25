"""
Local dashboard for the thesis project. It only shows what has been built so far
(Phase 0 = config + rate encoding, Phase 1 = plaintext ANN baseline,
Phase 2 = LIF hidden layer, Phase 3 = STDP learning, Phase 4 = readout + full SNN pipeline) and grows one
phase at a time.

Run:   .venv/bin/python src/app_server.py      then open http://127.0.0.1:8000
The page asks this server for real values computed by the project's own code
(config_loader.py, data.py, encoding.py, ann_model.py). Training only starts when
you press a Train button (Phase 1, 3 and 4 tabs).
"""

import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from ann_model import train_ann
from config_loader import load_config
from data import load_mnist, make_splits
from encoding import rate_encode
from lif_model import LIFLayer
import mpc_lif
import mpc_real
import mpc_sim
import mpc_stdp
from readout import LinearReadout, classification_metrics
from stdp import STDPTrainer

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
ANN_DIR = ROOT / "results" / "ann"
STDP_DIR = ROOT / "results" / "stdp"
READOUT_DIR = ROOT / "results" / "readout"
SLIF_DIR = ROOT / "results" / "mpc"
_encode_lock = threading.Lock()

CONFIG = load_config()
IMAGES, LABELS = load_mnist(train=False)
LIF = LIFLayer(CONFIG["architecture"]["num_inputs"], CONFIG["architecture"]["num_hidden"],
               CONFIG["neuron"], CONFIG["seed"])

# ---------- Phase 1: ANN baseline job ----------
ANN_JOB = {"running": False, "epoch": 0, "total": CONFIG["ann"]["epochs"], "history": [], "error": None}
ANN_RESULT = None
_ann_stop = threading.Event()
_splits = None

if (ANN_DIR / "metrics.json").exists():
    ANN_RESULT = json.loads((ANN_DIR / "metrics.json").read_text())
    ANN_JOB["history"] = ANN_RESULT["history"]
    ANN_JOB["epoch"] = ANN_RESULT["epochs_run"]


def _ann_run():
    global ANN_RESULT, _splits
    try:
        if _splits is None:
            _splits = make_splits(CONFIG)

        def on_epoch(rec):
            ANN_JOB["history"].append(rec)
            ANN_JOB["epoch"] = rec["epoch"]

        model, metrics = train_ann(CONFIG, _splits, on_epoch, _ann_stop.is_set)
        ANN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ANN_DIR / "model.pt")
        (ANN_DIR / "metrics.json").write_text(json.dumps(metrics))
        ANN_RESULT = metrics
    except Exception as e:
        ANN_JOB["error"] = f"{type(e).__name__}: {e}"
    finally:
        ANN_JOB["running"] = False


def ann_start():
    global ANN_RESULT
    if ANN_JOB["running"]:
        return False
    _ann_stop.clear()
    ANN_RESULT = None
    ANN_JOB.update(running=True, epoch=0, total=CONFIG["ann"]["epochs"], history=[], error=None)
    threading.Thread(target=_ann_run, daemon=True).start()
    return True


def ann_status():
    return {"job": {**ANN_JOB, "history": list(ANN_JOB["history"])}, "result": ANN_RESULT}


# ---------- Phase 3: STDP learning job ----------
# A separate layer from the Phase 2 one, so the Phase 2 tab keeps showing random weights.
STDP_LAYER = LIFLayer(CONFIG["architecture"]["num_inputs"], CONFIG["architecture"]["num_hidden"],
                      CONFIG["neuron"], CONFIG["seed"])
STDP_JOB = {"running": False, "sample": 0, "total": 0, "history": [], "error": None}
STDP_RESULT = None
_stdp_stop = threading.Event()
_stdp_lock = threading.Lock()
_stdp_version = 0

if (STDP_DIR / "model.pt").exists() and (STDP_DIR / "result.json").exists():
    _ck = torch.load(STDP_DIR / "model.pt")
    STDP_LAYER.W, STDP_LAYER.theta = _ck["W"], _ck["theta"]
    STDP_RESULT = json.loads((STDP_DIR / "result.json").read_text())
    STDP_JOB["history"] = STDP_RESULT["history"]
    STDP_JOB["sample"] = STDP_JOB["total"] = STDP_RESULT["samples"]
    _stdp_version = 1


def _stdp_run():
    global STDP_RESULT, _splits, _stdp_version
    try:
        if _splits is None:
            _splits = make_splits(CONFIG)
        X = _splits["train"][0]                      # the labels are never used here
        learn, T = CONFIG["learning"], CONFIG["simulation"]["num_steps"]
        with _stdp_lock:
            STDP_LAYER.reset(CONFIG["seed"])
        trainer = STDPTrainer(STDP_LAYER, learn)
        STDP_JOB["total"] = len(X) * learn["epochs"]
        _stdp_version += 1
        order_gen = torch.Generator().manual_seed(CONFIG["seed"])
        t0, k, recent = time.time(), 0, 0
        for _ in range(learn["epochs"]):
            for idx in torch.randperm(len(X), generator=order_gen).tolist():
                if _stdp_stop.is_set():
                    break
                with _encode_lock:
                    torch.manual_seed(CONFIG["seed"] * 100003 + k)
                    spikes = rate_encode(X[idx], T)
                with _stdp_lock:
                    recent += trainer.train_sample(spikes)
                k += 1
                STDP_JOB["sample"] = k
                if k % 100 == 0:
                    STDP_JOB["history"].append({"samples": k, "spikes": recent / 100,
                                                "theta_mean": float(STDP_LAYER.theta.mean())})
                    recent = 0
                    _stdp_version += 1
            if _stdp_stop.is_set():
                break
        STDP_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"W": STDP_LAYER.W, "theta": STDP_LAYER.theta}, STDP_DIR / "model.pt")
        result = {"samples": k, "seconds": round(time.time() - t0, 1),
                  "history": list(STDP_JOB["history"]), "stopped_early": _stdp_stop.is_set()}
        (STDP_DIR / "result.json").write_text(json.dumps(result))
        STDP_RESULT = result
        _stdp_version += 1
    except Exception as e:
        STDP_JOB["error"] = f"{type(e).__name__}: {e}"
    finally:
        STDP_JOB["running"] = False


def stdp_start():
    global STDP_RESULT
    if STDP_JOB["running"]:
        return False
    _stdp_stop.clear()
    STDP_RESULT = None
    STDP_JOB.update(running=True, sample=0, total=0, history=[], error=None)
    threading.Thread(target=_stdp_run, daemon=True).start()
    return True


def stdp_status():
    return {"job": {**STDP_JOB, "history": list(STDP_JOB["history"])}, "result": STDP_RESULT,
            "weights_version": _stdp_version}


def stdp_weights():
    """Each hidden neuron's 784 incoming weights, scaled per neuron to 0..255 for display."""
    with _stdp_lock:
        W, theta, version = STDP_LAYER.W.t().clone(), STDP_LAYER.theta.clone(), _stdp_version
    scaled = (W / W.max(dim=1, keepdim=True).values.clamp_min(1e-8) * 255).round().to(torch.uint8)
    return {"w": base64.b64encode(scaled.numpy().tobytes()).decode(),
            "theta": theta.tolist(), "version": version}


# ---------- Phase 4: readout + full SNN pipeline ----------
RD_JOB = {"running": False, "phase": "", "progress": 0.0, "error": None}
RD_RESULT = None
RD_MODEL = None            # the trained LinearReadout (SNN features)

if (READOUT_DIR / "metrics.json").exists() and (READOUT_DIR / "model.pt").exists():
    RD_RESULT = json.loads((READOUT_DIR / "metrics.json").read_text())
    _rk = torch.load(READOUT_DIR / "model.pt")
    RD_MODEL = LinearReadout(CONFIG["architecture"]["num_hidden"], CONFIG["architecture"]["num_outputs"])
    RD_MODEL.W, RD_MODEL.b = _rk["W"], _rk["b"]


def _encode_batch(X, seed):
    """(B, 784) images -> (T, B, 784) spike trains, reproducible for a given seed."""
    T = CONFIG["simulation"]["num_steps"]
    with _encode_lock:
        torch.manual_seed(seed)
        return torch.stack([rate_encode(x, T) for x in X], dim=1)


def _features(layer, X, seed):
    """Hidden spike counts / T for every image, plus activity totals."""
    T, H = CONFIG["simulation"]["num_steps"], CONFIG["architecture"]["num_hidden"]
    out, in_total, hid_total, ever = [], 0.0, 0.0, torch.zeros(H, dtype=torch.bool)
    for i in range(0, len(X), 500):
        spikes = _encode_batch(X[i:i + 500], seed + i)
        counts = layer.spike_counts(spikes, threshold=CONFIG["learning"]["threshold"])
        in_total += float(spikes.sum())
        hid_total += float(counts.sum())
        ever |= counts.sum(dim=0) > 0
        out.append(counts / T)
    return torch.cat(out), in_total, hid_total, ever


def _fit_and_test(layer, splits):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["val"], splits["test"]
    H, n_out, r = CONFIG["architecture"]["num_hidden"], CONFIG["architecture"]["num_outputs"], CONFIG["readout"]
    f_tr, _, _, _ = _features(layer, Xtr, 1_000_000)
    f_va, _, _, _ = _features(layer, Xva, 2_000_000)
    f_te, in_total, hid_total, ever = _features(layer, Xte, 3_000_000)
    ro = LinearReadout(H, n_out)
    history = ro.fit(f_tr, ytr, f_va, yva, r["epochs"], r["lr"], r["record_every"])
    metrics = classification_metrics(yte, ro.predict(f_te), n_out)
    return ro, history, metrics, (in_total, hid_total, ever)


def _readout_run():
    global RD_RESULT, RD_MODEL, _splits
    try:
        if STDP_RESULT is None:
            raise RuntimeError("Train the STDP layer first (Phase 3 tab).")
        if STDP_JOB["running"]:
            raise RuntimeError("STDP training is still running.")
        if _splits is None:
            _splits = make_splits(CONFIG)
        t0 = time.time()
        T, H, n_in = CONFIG["simulation"]["num_steps"], CONFIG["architecture"]["num_hidden"], CONFIG["architecture"]["num_inputs"]
        n_test = len(_splits["test"][1])

        RD_JOB.update(phase="SNN features and readout", progress=0.05)
        with _stdp_lock:
            layer = LIFLayer(n_in, H, CONFIG["neuron"], CONFIG["seed"])
            layer.W, layer.theta = STDP_LAYER.W.clone(), STDP_LAYER.theta.clone()
        ro, history, metrics, (in_total, hid_total, ever) = _fit_and_test(layer, _splits)

        RD_JOB.update(phase="Control: untrained hidden layer", progress=0.55)
        control = LIFLayer(n_in, H, CONFIG["neuron"], CONFIG["seed"])
        _, _, control_metrics, _ = _fit_and_test(control, _splits)

        avg_in, avg_hid = in_total / n_test, hid_total / n_test
        result = {
            "n_test": n_test, "seconds": round(time.time() - t0, 1),
            "history": history, "test": metrics,
            "last": history[-1],
            "control": {"accuracy": control_metrics["accuracy"]},
            "activity": {"avg_input_spikes": avg_in, "avg_hidden_spikes": avg_hid,
                         "firing_rate": hid_total / (n_test * T * H), "neurons_ever_fired": int(ever.sum()),
                         "T": T, "hidden": H},
            "ops_per_image": {"encoder_comparisons": T * n_in, "threshold_checks": T * H, "membrane_updates": T * H,
                              "dense_synaptic_adds": T * n_in * H, "event_driven_synaptic_adds": round(avg_in * H),
                              "readout_multiplications": H * CONFIG["architecture"]["num_outputs"]},
        }
        READOUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"W": ro.W.detach(), "b": ro.b.detach()}, READOUT_DIR / "model.pt")
        (READOUT_DIR / "metrics.json").write_text(json.dumps(result))
        RD_MODEL, RD_RESULT = ro, result
        RD_JOB["progress"] = 1.0
    except Exception as e:
        RD_JOB["error"] = f"{type(e).__name__}: {e}"
    finally:
        RD_JOB["running"] = False


def readout_start():
    global RD_RESULT
    if RD_JOB["running"]:
        return False
    RD_RESULT = None
    RD_JOB.update(running=True, phase="", progress=0.0, error=None)
    threading.Thread(target=_readout_run, daemon=True).start()
    return True


def readout_status():
    return {"job": dict(RD_JOB), "result": RD_RESULT, "stdp_ready": STDP_RESULT is not None}


def readout_predict(index, resample):
    """The full pipeline on one image: encode -> hidden LIF layer (STDP weights) -> readout."""
    if RD_MODEL is None:
        raise RuntimeError("Train the readout first (Phase 4 tab).")
    index, spikes = _encode(index, resample)
    with _stdp_lock:
        counts = STDP_LAYER.spike_counts(spikes[:, None, :], threshold=CONFIG["learning"]["threshold"])[0]
    probs = torch.softmax(RD_MODEL.scores(counts[None, :] / spikes.shape[0])[0], dim=0)
    return {"index": index, "label": int(LABELS[index]), "pred": int(probs.argmax()),
            "probs": [round(v, 4) for v in probs.tolist()],
            "image": (IMAGES[index] * 255).round().int().tolist(),
            "input_spikes": int(spikes.sum()), "hidden_spikes": int(counts.sum()),
            "neurons_fired": int((counts > 0).sum())}


def stdp_run(index, resample):
    """Which hidden neurons answer to this image (winner-take-all forward pass)."""
    index, spikes = _encode(index, resample)
    with _stdp_lock:
        out = STDP_LAYER.run(spikes, threshold=CONFIG["learning"]["threshold"], wta=True)
    counts = out["spikes"].sum(dim=0)
    top = [{"neuron": int(j), "spikes": int(counts[j])}
           for j in counts.argsort(descending=True)[:5].tolist() if counts[j] > 0]
    return {"index": index, "label": int(LABELS[index]), "T": spikes.shape[0],
            "image": (IMAGES[index] * 255).round().int().tolist(),
            "total_spikes": int(counts.sum()), "neurons_fired": int((counts > 0).sum()),
            "top": top}


# ---------- Phase 6: the Phase 2 LIF layer on secret shares ----------
SLIF_JOB = {"running": False, "phase": "", "progress": 0.0, "error": None}
SLIF_RESULT = None
if (SLIF_DIR / "secure_lif.json").exists():
    SLIF_RESULT = json.loads((SLIF_DIR / "secure_lif.json").read_text())


def _slif_run(index, resample):
    global SLIF_RESULT
    try:
        index, spikes = _encode(index, resample)
        res = mpc_lif.run_secure_lif(LIF, spikes, lambda phase, frac: SLIF_JOB.update(phase=phase, progress=frac))
        res.update(image_index=index, resample=int(resample), label=int(LABELS[index]))
        SLIF_DIR.mkdir(parents=True, exist_ok=True)
        (SLIF_DIR / "secure_lif.json").write_text(json.dumps(res))
        SLIF_RESULT = res
        SLIF_JOB["progress"] = 1.0
    except Exception as e:
        SLIF_JOB["error"] = f"{type(e).__name__}: {e}"
    finally:
        SLIF_JOB["running"] = False
        mpc_real.RUN_LOCK.release()


def slif_start(index, resample):
    """Start the secure run in the background. False if another MP-SPDZ job is running."""
    global SLIF_RESULT
    if not mpc_real.RUN_LOCK.acquire(blocking=False):
        return False
    SLIF_RESULT = None
    SLIF_JOB.update(running=True, phase="starting", progress=0.0, error=None)
    threading.Thread(target=_slif_run, args=(index, resample), daemon=True).start()
    return True


def slif_status():
    ok, why = mpc_real.available()
    return {"available": ok, "reason": why, "job": dict(SLIF_JOB), "result": SLIF_RESULT}


# ---------- Phase 7: one STDP training image on secret shares ----------
SSTDP_JOB = {"running": False, "phase": "", "progress": 0.0, "error": None}
SSTDP_RESULT = None
if (SLIF_DIR / "secure_stdp.json").exists():
    SSTDP_RESULT = json.loads((SLIF_DIR / "secure_stdp.json").read_text())


def _sstdp_run(index, resample):
    global SSTDP_RESULT
    try:
        index, spikes = _encode(index, resample)
        n_in, n_hid = CONFIG["architecture"]["num_inputs"], CONFIG["architecture"]["num_hidden"]
        layer = LIFLayer(n_in, n_hid, CONFIG["neuron"], CONFIG["seed"])           # a COPY: the saved model is never touched
        trained = STDP_RESULT is not None
        if trained:
            with _stdp_lock:
                layer.W, layer.theta = STDP_LAYER.W.clone(), STDP_LAYER.theta.clone()
        res = mpc_stdp.run_secure_stdp(layer, CONFIG["learning"], spikes, lambda phase, frac: SSTDP_JOB.update(phase=phase, progress=frac))
        res.update(image_index=index, resample=int(resample), label=int(LABELS[index]), start_state="trained STDP layer" if trained else "untrained layer")
        SLIF_DIR.mkdir(parents=True, exist_ok=True)
        (SLIF_DIR / "secure_stdp.json").write_text(json.dumps(res))
        SSTDP_RESULT = res
        SSTDP_JOB["progress"] = 1.0
    except Exception as e:
        SSTDP_JOB["error"] = f"{type(e).__name__}: {e}"
    finally:
        SSTDP_JOB["running"] = False
        mpc_real.RUN_LOCK.release()


def sstdp_start(index, resample):
    global SSTDP_RESULT
    if not mpc_real.RUN_LOCK.acquire(blocking=False):
        return False
    SSTDP_RESULT = None
    SSTDP_JOB.update(running=True, phase="starting", progress=0.0, error=None)
    threading.Thread(target=_sstdp_run, args=(index, resample), daemon=True).start()
    return True


def sstdp_status():
    ok, why = mpc_real.available()
    return {"available": ok, "reason": why, "job": dict(SSTDP_JOB), "result": SSTDP_RESULT, "trained": STDP_RESULT is not None}


def info():
    return {
        "config": CONFIG,
        "config_file": "configs/experiment_config.yaml",
        "num_images": len(IMAGES),
    }


def _encode(index, resample):
    """Same image + same resample number always gives the same spike train."""
    index = int(index) % len(IMAGES)
    T = CONFIG["simulation"]["num_steps"]
    with _encode_lock:
        torch.manual_seed(index * 1000 + int(resample))
        spikes = rate_encode(IMAGES[index], T)          # (T, 784), the project's own function
    return index, spikes


def lif_run(index, resample, beta=None, threshold=None):
    """Phase 2: run the input spike train through the hidden LIF layer."""
    index, spikes = _encode(index, resample)
    T, n_in = spikes.shape
    n_hid = CONFIG["architecture"]["num_hidden"]
    out = LIF.run(spikes, beta, threshold)
    hs = out["spikes"]
    n_input_spikes = int(spikes.sum())
    total = int(hs.sum())
    return {
        "index": index, "label": int(LABELS[index]), "T": T,
        "beta": LIF.beta if beta is None else beta,
        "threshold": LIF.threshold if threshold is None else threshold,
        "image": (IMAGES[index] * 255).round().int().tolist(),
        "input_spikes": torch.nonzero(spikes).tolist(),
        "hidden_spikes": torch.nonzero(hs).tolist(),                       # [[t, neuron], ...]
        "membrane": [[round(v, 2) for v in row] for row in out["membrane"].tolist()],
        "hidden_per_step": hs.sum(dim=1).int().tolist(),
        "total_hidden_spikes": total,
        "firing_rate": total / (T * n_hid),                                # spikes per neuron per step
        "neurons_fired": int((hs.sum(dim=0) > 0).sum()),
        "input_spike_count": n_input_spikes,
        "mean_current": round(float(out["current"].mean()), 2),               # avg input per neuron per step
        "ops": {                                                           # counted analytically, plaintext
            "threshold_checks": T * n_hid,
            "membrane_updates": T * n_hid,
            "dense_synaptic_adds": T * n_in * n_hid,
            "event_driven_synaptic_adds": n_input_spikes * n_hid,
        },
    }


def sample(index, resample):
    index, spikes = _encode(index, resample)
    T = CONFIG["simulation"]["num_steps"]
    n_in = CONFIG["architecture"]["num_inputs"]
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
        if path == "/api/ann/status":
            return self._json(ann_status())
        if path == "/api/readout/status":
            return self._json(readout_status())
        if path == "/api/stdp/status":
            return self._json(stdp_status())
        if path == "/api/stdp/weights":
            return self._json(stdp_weights())
        if path == "/api/slif/status":
            return self._json(slif_status())
        if path == "/api/sstdp/status":
            return self._json(sstdp_status())
        if path == "/api/mpc/real/status":
            return self._json(mpc_real.status())
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
            if self.path == "/api/lif/run":
                return self._json(lif_run(body.get("index", 0), body.get("resample", 0),
                                          body.get("beta"), body.get("threshold")))
            if self.path == "/api/readout/train":
                return self._json({"started": readout_start()})
            if self.path == "/api/readout/predict":
                return self._json(readout_predict(body.get("index", 0), body.get("resample", 0)))
            if self.path == "/api/stdp/train":
                return self._json({"started": stdp_start()})
            if self.path == "/api/stdp/stop":
                _stdp_stop.set()
                return self._json({"ok": True})
            if self.path == "/api/stdp/run":
                return self._json(stdp_run(body.get("index", 0), body.get("resample", 0)))
            if self.path == "/api/mpc/run":
                return self._json(mpc_sim.run(body.get("op", "add"), body.get("a", 0), body.get("b", 0),
                                              body.get("bits", 8), body.get("seed", 0)))
            if self.path == "/api/mpc/secrecy":
                return self._json(mpc_sim.secrecy_demo(int(body.get("bits", 8)), int(body.get("a", 0)),
                                                       int(body.get("b", 1)), seed=int(body.get("seed", 0))))
            if self.path == "/api/mpc/snn":
                return self._json(mpc_sim.snn_preview(CONFIG["simulation"]["num_steps"], CONFIG["architecture"]["num_hidden"],
                                                      CONFIG["architecture"]["num_inputs"], int(body.get("bits", 64))))
            if self.path == "/api/sstdp/start":
                return self._json({"started": sstdp_start(body.get("index", 0), body.get("resample", 0))})
            if self.path == "/api/slif/start":
                return self._json({"started": slif_start(body.get("index", 0), body.get("resample", 0))})
            if self.path == "/api/mpc/real/start":
                return self._json({"started": mpc_real.start(int(body.get("a", 0)), int(body.get("b", 0)))})
            if self.path == "/api/ann/train":
                return self._json({"started": ann_start()})
            if self.path == "/api/ann/stop":
                _ann_stop.set()
                return self._json({"ok": True})
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
