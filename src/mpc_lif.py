"""
Phase 6: run the Phase 2 LIF layer on secret shares in MP-SPDZ and check it against the plaintext layer.

Roles (3 parties, replicated ring, 64-bit, fixed point with 16 fractional bits):
    party 0  data owner   : the spike trains of one image (and a few helper bits)
    party 1  model owner  : the weights W and a membrane vector
    party 2  no input, but takes part in every computation

Two kinds of runs (see mpc/secure_lif.mpc):
    full      opens every spike, membrane value and current. It exists only to VALIDATE the secure result
              against the plaintext layer. A real deployment would never open them.
    the rest  measure the cost: "run" is the whole forward pass; current / leak / thresh / reset are one
              timestep of one operation. cost(operation) = cost(mode) - cost("base"), where "base" only
              shares the same inputs and opens one value.

All parties run on this machine, so ROUNDS and DATA SENT are meaningful, time on a real network is not.
"""

import re
import time

import torch

import mpc_real

SPDZ, PROGRAM = mpc_real.SPDZ, mpc_real.ROOT / "mpc" / "secure_lif.mpc"
FRAC_BITS = 16
LIMIT = 2 ** 15                       # sfix(16, 31): magnitudes must stay below 2^15
PARTS = (("current", "Synaptic current  I = W·s", "matrix product (secret spikes × secret weights) + truncation"),
         ("leak", "Leak  β·u", "multiplication by a public constant + truncation"),
         ("thresh", "Threshold  u ≥ θ", "secure comparison (one per neuron)"),
         ("reset", "Reset  u − θ·spike", "secret bit × public constant: free when θ is a whole number, otherwise a truncation is needed"))


def schedule_name(mode, T, N, H, beta, thr):
    return f"secure_lif-{mode}-{T}-{N}-{H}-{beta}-{thr}"


def compile_mode(mode, T, N, H, beta, thr):
    name = schedule_name(mode, T, N, H, beta, thr)
    sch = SPDZ / "Programs" / "Schedules" / f"{name}.sch"
    if sch.exists() and sch.stat().st_mtime >= PROGRAM.stat().st_mtime:
        return name
    code, out = mpc_real.sh(["python3", "compile.py", "-R", "64", str(PROGRAM), mode, str(T), str(N), str(H), str(beta), str(thr)], 600)
    if code != 0 or not sch.exists():
        raise RuntimeError("MP-SPDZ compile failed: " + out[-500:])
    return name


def write_inputs(spikes, W, u0, bits):
    """spikes (T, N) 0/1, W (N, H), u0 (H), bits (H) 0/1  ->  the input files of party 0 and party 1."""
    mpc_real.ensure_ssl()
    pd = SPDZ / "Player-Data"
    flat = lambda t, fmt: " ".join(fmt % v for v in t.reshape(-1).tolist())
    (pd / "Input-P0-0").write_text(flat(spikes, "%d") + " " + flat(bits, "%d") + "\n")
    (pd / "Input-P1-0").write_text(flat(W, "%.7f") + " " + flat(u0, "%.7f") + "\n")
    (pd / "Input-P2-0").write_text("")


def run_program(name):
    last = ""
    for _ in range(3):                                   # ring.sh picks a random port; retry if it is taken
        code, out = mpc_real.sh(["bash", "Scripts/ring.sh", name], 900)
        if mpc_real.parse_costs(out):
            return out
        last = out[-400:]
        time.sleep(1)
    raise RuntimeError(f"MP-SPDZ run '{name}' failed: {last}")


def parse_full(text, T, H):
    """SPK / MEM / CUR lines -> tensors. Membrane and current are raw fixed-point integers (value * 2^16)."""
    rows = {"SPK": {}, "MEM": {}, "CUR": {}}
    for m in re.finditer(r"^(SPK|MEM|CUR) (\d+):\s*(.*)$", text, re.M):
        rows[m.group(1)][int(m.group(2))] = [int(v) for v in m.group(3).split()]
    for k, d in rows.items():
        if sorted(d) != list(range(T)) or any(len(v) != H for v in d.values()):
            raise RuntimeError(f"secure run printed an incomplete {k} table")
    spikes = torch.tensor([rows["SPK"][t] for t in range(T)], dtype=torch.float32)
    mem = torch.tensor([rows["MEM"][t] for t in range(T)], dtype=torch.float64) / 2 ** FRAC_BITS
    cur = torch.tensor([rows["CUR"][t] for t in range(T)], dtype=torch.float64) / 2 ** FRAC_BITS
    return spikes, mem.float(), cur.float()


def compare(plain, secure):
    """How far the secure run is from the plaintext layer."""
    ps, ss = plain["spikes"], secure["spikes"]
    bad = (ps != ss).nonzero().tolist()
    em = (plain["membrane"] - secure["membrane"]).abs()
    ec = (plain["current"] - secure["current"]).abs()
    return {"spike_mismatches": len(bad), "first_mismatch": bad[0] if bad else None,
            "spikes_plain": int(ps.sum()), "spikes_secure": int(ss.sum()),
            "cells": int(ps.numel()),
            "max_membrane_error": float(em.max()), "mean_membrane_error": float(em.mean()),
            "max_current_error": float(ec.max()), "lsb": 2.0 ** -FRAC_BITS}


def _matrix(t, nd=5):
    return [[round(v, nd) for v in row] for row in t.tolist()]


def run_secure_lif(layer, spikes, progress=lambda phase, frac: None):
    """
    layer: LIFLayer (Phase 2). spikes: (T, n_in) 0/1 tensor. The caller must hold mpc_real.RUN_LOCK.
    """
    ok, why = mpc_real.available()
    if not ok:
        raise RuntimeError(why)
    if not PROGRAM.exists():
        raise RuntimeError("mpc/secure_lif.mpc is missing")
    if layer.reset_mode != "subtract":
        raise RuntimeError("the secure program implements reset = 'subtract' only; set neuron.reset to subtract")
    T, N = spikes.shape
    H = layer.n_hid
    beta, thr = float(layer.beta), float(layer.threshold)

    plain = layer.run(spikes, beta, thr)
    peak = max(float(plain["membrane"].abs().max()), float(plain["current"].abs().max()), float(layer.W.abs().max()))
    if peak >= LIMIT:
        raise RuntimeError(f"values up to {peak:.0f} do not fit in the fixed-point format (limit {LIMIT})")

    ref = T // 2                                          # a realistic membrane state for the one-step benchmarks
    u0 = plain["membrane"][ref].clamp(0, LIMIT - 1)
    write_inputs(spikes, layer.W, u0, plain["spikes"][ref])

    modes = ["full", "base", "current", "leak", "thresh", "reset", "run"]
    names = {}
    for k, m in enumerate(modes):
        progress(f"compiling {m}", 0.02 + 0.2 * k / len(modes))
        names[m] = compile_mode(m, T, N, H, beta, thr)

    progress("running the whole forward pass on shares (validation run)", 0.25)
    out = run_program(names["full"])
    s_spk, s_mem, s_cur = parse_full(out, T, H)
    secure = {"spikes": s_spk, "membrane": s_mem, "current": s_cur}
    check = compare(plain, secure)

    costs = {"full": mpc_real.parse_costs(out)}
    for k, m in enumerate(["base", "current", "leak", "thresh", "reset", "run"]):
        progress(f"measuring cost: {m}", 0.45 + 0.5 * k / 6)
        text = run_program(names[m])
        costs[m] = mpc_real.parse_costs(text)
    b = costs["base"]
    delta = lambda m: {"rounds": costs[m]["rounds"] - b["rounds"], "global_bytes": costs[m]["global_bytes"] - b["global_bytes"],
                       "party0_bytes": costs[m]["party0_bytes"] - b["party0_bytes"]}
    parts = []
    for mode, label, how in PARTS:
        d = delta(mode)
        parts.append({"mode": mode, "name": label, "how": how, "per_step": d,
                      "per_image_bytes": d["global_bytes"] * T, "per_image_rounds": d["rounds"] * T})
    total = delta("run")
    return {
        "T": T, "n_in": N, "H": H, "beta": beta, "threshold": thr,
        "fixed_point": {"fractional_bits": FRAC_BITS, "ring_bits": 64, "max_abs": LIMIT, "lsb": 2.0 ** -FRAC_BITS},
        "reference_step": ref,
        "plain": {"spikes": plain["spikes"].int().tolist(), "membrane": _matrix(plain["membrane"]), "current": _matrix(plain["current"])},
        "secure": {"spikes": s_spk.int().tolist(), "membrane": _matrix(s_mem), "current": _matrix(s_cur)},
        "check": check, "costs": costs, "parts": parts, "total": total,
        "sum_of_parts_bytes": sum(p["per_image_bytes"] for p in parts),
        "inputs_bytes": b["global_bytes"],
        "protocol": "replicated-ring-party.x (3 parties, honest majority, semi-honest), 64-bit ring",
    }
