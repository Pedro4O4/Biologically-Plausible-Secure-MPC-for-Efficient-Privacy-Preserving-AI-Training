"""
Phase 5: run the SAME operations for real in MP-SPDZ (3 parties on this machine, replicated ring,
honest majority, semi-honest, 64-bit ring) and read back what MP-SPDZ itself reports.

mpc/ops_demo.mpc is compiled once per mode (base, add, mul, cmp). "base" only shares the two inputs
and opens one value, so  cost(op) = cost(op program) - cost(base program).

All parties run as processes on localhost, so nothing here says anything about network latency:
we compare ROUNDS and DATA SENT, which do not depend on the machine.
"""

import re
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPDZ = ROOT / "MP-SPDZ"
PROGRAM = ROOT / "mpc" / "ops_demo.mpc"
MODES = ("base", "add", "mul", "cmp")
_NUM = r"([-+0-9.eE]+)"

JOB = {"running": False, "phase": "", "progress": 0.0, "error": None}
RESULT = None
# One MP-SPDZ job at a time: every job writes the same Player-Data/Input-P*-0 files and uses local ports.
RUN_LOCK = threading.Lock()


def available():
    """(ok, reason). Everything the real run needs must exist."""
    for rel in ("compile.py", "replicated-ring-party.x", "Scripts/ring.sh", "Scripts/setup-ssl.sh"):
        if not (SPDZ / rel).exists():
            return False, f"MP-SPDZ is missing {rel}. Build it first (see README)."
    if not PROGRAM.exists():
        return False, "mpc/ops_demo.mpc is missing."
    return True, ""


def parse_costs(text):
    """Rounds, bytes and time that MP-SPDZ prints at the end of a run (None if the run did not finish)."""
    rounds = re.search(r"Data sent = " + _NUM + r" MB in ~(\d+) rounds", text)
    glob = re.search(r"Global data sent = " + _NUM + r" MB", text)
    secs = re.search(r"Time = " + _NUM + r" seconds", text)
    if not (rounds and glob):
        return None
    return {"rounds": int(rounds.group(2)),
            "party0_bytes": round(float(rounds.group(1)) * 1e6),
            "global_bytes": round(float(glob.group(1)) * 1e6),
            "seconds": float(secs.group(1)) if secs else None}


def parse_output(text):
    """Extract result, rounds and data sent from the text MP-SPDZ prints."""
    res = re.search(r"RESULT\s+(-?\d+)", text)
    costs = parse_costs(text)
    if not (res and costs):
        return None
    return {"result": int(res.group(1)), **costs}


def sh(cmd, timeout):
    p = subprocess.run(cmd, cwd=SPDZ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return p.returncode, p.stdout


_sh = sh


def ensure_ssl():
    """The parties talk over TLS even on localhost; create the local certificates once."""
    pd = SPDZ / "Player-Data"
    pd.mkdir(exist_ok=True)
    if not (pd / "P0.pem").exists():
        _, out = sh(["bash", "Scripts/setup-ssl.sh", "3"], 120)
        if not (pd / "P0.pem").exists():
            raise RuntimeError("could not create the local SSL certificates: " + out[-300:])


def _compile(mode):
    sch = SPDZ / "Programs" / "Schedules" / f"ops_demo-{mode}.sch"
    if sch.exists() and sch.stat().st_mtime >= PROGRAM.stat().st_mtime:
        return
    code, out = _sh(["python3", "compile.py", "-R", "64", str(PROGRAM), mode], 300)
    if code != 0 or not sch.exists():
        raise RuntimeError("MP-SPDZ compile failed: " + out[-400:])


def _run_mode(mode):
    last = ""
    for _ in range(3):                                  # ring.sh picks a random port; retry if it is taken
        code, out = _sh(["bash", "Scripts/ring.sh", f"ops_demo-{mode}"], 120)
        parsed = parse_output(out)
        if parsed:
            return parsed
        last = out[-400:]
        time.sleep(1)
    raise RuntimeError(f"MP-SPDZ run '{mode}' failed: {last}")


def run_all(a, b, progress):
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    ensure_ssl()
    pd = SPDZ / "Player-Data"
    (pd / "Input-P0-0").write_text(f"{int(a)}\n")
    (pd / "Input-P1-0").write_text(f"{int(b)}\n")
    (pd / "Input-P2-0").write_text("")
    runs = {}
    for k, mode in enumerate(MODES):
        progress(f"{mode}: compiling", (k + 0.0) / len(MODES))
        _compile(mode)
        progress(f"{mode}: running 3 parties", (k + 0.5) / len(MODES))
        runs[mode] = _run_mode(mode)
    base = runs["base"]
    ops = {m: {"rounds": runs[m]["rounds"] - base["rounds"],
               "global_bytes": runs[m]["global_bytes"] - base["global_bytes"],
               "party0_bytes": runs[m]["party0_bytes"] - base["party0_bytes"]}
           for m in ("add", "mul", "cmp")}
    return {"a": int(a), "b": int(b), "ring_bits": 64, "runs": runs, "ops": ops,
            "protocol": "replicated-ring-party.x (3 parties, honest majority, semi-honest)"}


def _job(a, b):
    global RESULT
    try:
        def progress(phase, frac):
            JOB.update(phase=phase, progress=frac)
        RESULT = run_all(a, b, progress)
        JOB["progress"] = 1.0
    except Exception as e:
        JOB["error"] = f"{type(e).__name__}: {e}"
    finally:
        JOB["running"] = False
        RUN_LOCK.release()


def start(a, b):
    """Start the background run. Returns False if an MP-SPDZ job (this or Phase 6) is already running."""
    global RESULT
    if not RUN_LOCK.acquire(blocking=False):
        return False
    RESULT = None
    JOB.update(running=True, phase="starting", progress=0.0, error=None)
    threading.Thread(target=_job, args=(a, b), daemon=True).start()
    return True


def status():
    ok, why = available()
    return {"available": ok, "reason": why, "job": dict(JOB), "result": RESULT}
