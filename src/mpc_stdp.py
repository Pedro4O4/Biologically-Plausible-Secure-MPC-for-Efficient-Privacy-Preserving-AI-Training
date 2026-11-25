"""
Phase 7: ONE STDP training image on secret shares: winner-take-all plus the trace-based weight update.

Roles (3 parties, replicated ring, 64-bit, fixed point with 16 fractional bits), as in Phase 6:
    party 0  data owner  : the spike trains of one image
    party 1  model owner : the weights W and the adaptive thresholds theta
    party 2  no input, takes part in every computation

The secure program (mpc/secure_stdp.mpc) mirrors src/stdp.py::STDPTrainer.train_sample. It is checked
against a plaintext copy of that loop. The "full" run OPENS the final weights, the thresholds and the winner
of every step so the two can be compared; a real deployment would never open them. Costs come from
separate runs that do not: cost(operation) = cost(mode) - cost("base").

The model that is passed in is NEVER modified: everything works on copies.
"""

import re
import time

import torch

import mpc_lif
import mpc_real

SPDZ, PROGRAM = mpc_real.SPDZ, mpc_real.ROOT / "mpc" / "secure_stdp.mpc"
SCALE = 2 ** 16
LIMIT = 2 ** 15

# (mode, label, how it is done on shares). "per" says how often it happens: step = every timestep, image = once.
PARTS = (
    ("trace", "Input trace  x = decay·x + s", "step", "multiplication by a public constant + truncation"),
    ("current", "Synaptic current  I = W·s", "step", "spikes are integers: integer × fixed-point matrix product, no truncation"),
    ("leak", "Leak  β·u", "step", "multiplication by a public constant + truncation"),
    ("thresh", "Threshold  u ≥ θ_base + θ_i", "step", "secure comparison, one per neuron (θ is secret)"),
    ("wta", "Winner-take-all", "step", "maximum by halving (comparisons), then who equals it, first one wins, nobody if no neuron fired"),
    ("gather", "Winner's weight column", "step", "sum over neurons of W · one-hot: secret bit × secret weight, summed"),
    ("delta", "STDP change  lr·(x - x_target)·(w_max - w)", "step", "two fixed-point products, each truncated"),
    ("clamp", "Keep weights in [0, w_max]", "step", "two comparisons per input, on the winner's column only"),
    ("scatter", "Write the column back", "step", "outer product (change) × (one-hot): one secret multiplication per weight"),
    ("theta", "Winner's θ += θ_plus", "step", "secret bit × public integer constant: local"),
    ("reset", "Reset every membrane if someone spiked", "step", "secret bit × secret membrane"),
    ("normalize", "Rescale each neuron's weights to w_norm", "image", "one secure division per neuron, then every weight times its factor"),
)


def schedule_name(mode, T, N, H, args, rn=False):
    parts = [mode, T, N, H, *args] + (["rn"] if rn else [])
    return "secure_stdp-" + "-".join(str(p) for p in parts)


def _args(layer, learn):
    return [float(layer.beta), float(learn["threshold"]), float(learn["trace_decay"]), float(learn["lr"]),
            float(learn["x_target"]), float(learn["w_max"]), float(learn["theta_plus"]), float(layer.w_norm)]


def compile_mode(mode, T, N, H, args, rn=False):
    name = schedule_name(mode, T, N, H, args, rn)
    sch = SPDZ / "Programs" / "Schedules" / f"{name}.sch"
    if sch.exists() and sch.stat().st_mtime >= PROGRAM.stat().st_mtime:
        return name
    cmd = ["python3", "compile.py", "-M", "-R", "64", str(PROGRAM), mode, str(T), str(N), str(H), *[str(a) for a in args]] + (["rn"] if rn else [])
    code, out = mpc_real.sh(cmd, 900)
    if code != 0 or not sch.exists():
        raise RuntimeError("MP-SPDZ compile failed: " + out[-500:])
    return name


def plain_image(layer, learn, spikes):
    """
    The loop of STDPTrainer.train_sample, step by step, on COPIES. Returns the trajectory and the final state.
    tests/test_mpc_stdp.py checks that it equals STDPTrainer itself.
    """
    W, theta = layer.W.clone(), layer.theta.clone()
    T, N = spikes.shape
    H = layer.n_hid
    u, trace = torch.zeros(H), torch.zeros(N)
    steps = []
    for t in range(T):
        s_in = spikes[t]
        trace = learn["trace_decay"] * trace + s_in
        u = layer.beta * u + s_in @ W
        thr = learn["threshold"] + theta
        fired = u >= thr
        info = {"u": u.clone(), "trace": trace.clone(), "winner": -1, "col": None}
        if fired.any():
            j = int(torch.where(fired, u - thr, torch.full_like(u, -1e9)).argmax())
            col = W[:, j].clone()
            dw = learn["lr"] * (trace - learn["x_target"]) * (learn["w_max"] - col) ** learn["mu"]
            W[:, j] = (col + dw).clamp(0.0, learn["w_max"])
            theta[j] += learn["theta_plus"]
            u = torch.zeros_like(u)
            info.update(winner=j, col=col)
        steps.append(info)
    W = W * (layer.w_norm / W.sum(0, keepdim=True).clamp_min(1e-8))
    return {"W": W, "theta": theta, "steps": steps, "winners": [s["winner"] for s in steps]}


def _raw(t):
    return [str(int(round(v * SCALE))) for v in t.reshape(-1).tolist()]


def write_inputs(spikes, W, theta, u0, tr0, col0, oh0):
    mpc_real.ensure_ssl()
    pd = SPDZ / "Player-Data"
    (pd / "Input-P0-0").write_text(" ".join(str(int(v)) for v in spikes.reshape(-1).tolist()) + "\n")
    (pd / "Input-P1-0").write_text(" ".join(_raw(W) + _raw(theta) + _raw(u0) + _raw(tr0) + _raw(col0) + [str(int(v)) for v in oh0.tolist()]) + "\n")
    (pd / "Input-P2-0").write_text("")


def parse_full(text, T, N, H):
    """NWIN / WIN t / THETA / W i lines -> (winners per step, theta, W). Raw fixed-point integers are divided by 2^16."""
    nwin = re.search(r"^NWIN (\d+)", text, re.M)
    th = re.search(r"^THETA:(.*)$", text, re.M)
    if not (nwin and th):
        raise RuntimeError("secure run printed an incomplete result")
    winners = []
    for t in range(T):
        m = re.search(rf"^WIN {t}:(.*)$", text, re.M)
        if not m:
            raise RuntimeError("secure run printed an incomplete winner table")
        row = [int(v) for v in m.group(1).split()]
        if len(row) != H or sum(row) > 1:
            raise RuntimeError("secure run printed a malformed winner row")
        winners.append(row.index(1) if 1 in row else -1)
    rows = []
    for i in range(N):
        m = re.search(rf"^W {i}:(.*)$", text, re.M)
        if not m:
            raise RuntimeError("secure run printed an incomplete weight table")
        rows.append([int(v) / SCALE for v in m.group(1).split()])
    return winners, torch.tensor([int(v) / SCALE for v in th.group(1).split()]), torch.tensor(rows)


def compare(plain, winners, theta, W):
    pw = plain["winners"]
    bad = [t for t in range(len(pw)) if pw[t] != winners[t]]
    ew, et = (plain["W"] - W).abs(), (plain["theta"] - theta).abs()
    return {"winner_mismatches": len(bad), "first_mismatch": bad[0] if bad else None, "steps": len(pw),
            "winning_steps_plain": sum(1 for x in pw if x >= 0), "winning_steps_secure": sum(1 for x in winners if x >= 0),
            "max_weight_error": float(ew.max()), "mean_weight_error": float(ew.mean()),
            "max_theta_error": float(et.max()), "lsb": 1.0 / SCALE}


def _round(t, nd=4):
    return [[round(v, nd) for v in row] for row in t.tolist()]


def run_secure_stdp(layer, learn, spikes, progress=lambda phase, frac: None):
    """
    layer: LIFLayer holding the STDP state (W, theta) to start from; it is not modified.
    learn: the `learning` section of the config. spikes: (T, n_in) 0/1 tensor.
    The caller must hold mpc_real.RUN_LOCK.
    """
    ok, why = mpc_real.available()
    if not ok:
        raise RuntimeError(why)
    if not PROGRAM.exists():
        raise RuntimeError("mpc/secure_stdp.mpc is missing")
    if float(learn["mu"]) != 1.0:
        raise RuntimeError("the secure program implements the soft bound with mu = 1 only")
    T, N = spikes.shape
    H = layer.n_hid
    args = _args(layer, learn)

    plain = plain_image(layer, learn, spikes)
    peak = max(float(layer.W.abs().max()), float((layer.theta + learn["threshold"]).abs().max()),
               max(float(s["u"].abs().max()) for s in plain["steps"]), float(plain["theta"].max() + learn["threshold"]),
               float(layer.w_norm) * 2)
    if peak >= LIMIT:
        raise RuntimeError(f"values up to {peak:.0f} do not fit in the fixed-point format (limit {LIMIT})")

    ref = next((t for t, s in enumerate(plain["steps"]) if s["winner"] >= 0), T // 2)      # a step where somebody wins
    st = plain["steps"][ref]
    oh0 = torch.zeros(H)
    oh0[max(st["winner"], 0)] = 1.0
    col0 = st["col"] if st["col"] is not None else layer.W[:, 0]
    write_inputs(spikes, layer.W, layer.theta, st["u"], st["trace"], col0, oh0)

    modes = ["full", "base", *[p[0] for p in PARTS], "current_trunc", "clamp_all", "run"]
    names = {}
    for k, m in enumerate(modes):
        progress(f"compiling {m}", 0.02 + 0.2 * k / len(modes))
        names[m] = compile_mode(m, T, N, H, args)
    names["run_rn"] = compile_mode("run", T, N, H, args, rn=True)
    names["leak_rn"] = compile_mode("leak", T, N, H, args, rn=True)
    names["base_rn"] = compile_mode("base", T, N, H, args, rn=True)

    progress("running one whole STDP image on shares (validation run)", 0.25)
    out = mpc_lif.run_program(names["full"])
    winners, th, Ws = parse_full(out, T, N, H)
    check = compare(plain, winners, th, Ws)

    costs = {"full": mpc_real.parse_costs(out)}
    order = ["base"] + [p[0] for p in PARTS] + ["current_trunc", "clamp_all", "run", "base_rn", "leak_rn", "run_rn"]
    for k, m in enumerate(order):
        progress(f"measuring cost: {m}", 0.35 + 0.6 * k / len(order))
        costs[m] = mpc_real.parse_costs(mpc_lif.run_program(names[m]))
    b, brn = costs["base"], costs["base_rn"]

    def delta(m, base=b):
        return {"rounds": costs[m]["rounds"] - base["rounds"], "global_bytes": costs[m]["global_bytes"] - base["global_bytes"],
                "party0_bytes": costs[m]["party0_bytes"] - base["party0_bytes"]}

    parts = []
    for mode, label, per, how in PARTS:
        d = delta(mode)
        times = T if per == "step" else 1
        parts.append({"mode": mode, "name": label, "per": per, "how": how, "per_unit": d,
                      "per_image_bytes": d["global_bytes"] * times, "per_image_rounds": d["rounds"] * times})
    won = sorted({w for w in plain["winners"] if w >= 0})[:8]                        # weight images of up to 8 winners
    neurons = [{"neuron": j, "before": [round(v, 4) for v in layer.W[:, j].tolist()],
                "plain": [round(v, 4) for v in plain["W"][:, j].tolist()],
                "secure": [round(v, 4) for v in Ws[:, j].tolist()]} for j in won]
    return {
        "T": T, "n_in": N, "H": H, "learning": {k: float(learn[k]) for k in ("threshold", "lr", "trace_decay", "x_target", "w_max", "theta_plus")},
        "beta": float(layer.beta), "w_norm": float(layer.w_norm),
        "fixed_point": {"fractional_bits": 16, "ring_bits": 64, "max_abs": LIMIT, "lsb": 1.0 / SCALE},
        "reference_step": ref,
        "plain": {"winners": plain["winners"], "theta": _round(plain["theta"][None, :])[0], "theta_before": _round(layer.theta[None, :])[0]},
        "secure": {"winners": winners, "theta": _round(th[None, :])[0]},
        "neurons": neurons, "check": check, "costs": costs, "parts": parts,
        "sum_of_parts_bytes": sum(p["per_image_bytes"] for p in parts),
        "total": delta("run"), "total_nearest": delta("run_rn", brn), "leak_nearest": delta("leak_rn", brn),
        "current_trunc": delta("current_trunc"), "naive_clamp": delta("clamp_all"), "inputs_bytes": b["global_bytes"],
        "protocol": "replicated-ring-party.x (3 parties, honest majority, semi-honest), 64-bit ring",
    }
