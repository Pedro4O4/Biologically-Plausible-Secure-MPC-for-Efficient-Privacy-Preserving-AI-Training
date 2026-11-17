"""
Hidden layer of Leaky Integrate-and-Fire (LIF) neurons, plaintext.

Written by hand (not snnTorch) so every operation is explicit. Later, each of these
operations is what has to be done securely under MPC.

Per timestep t, for every hidden neuron i:
    I_i[t] = sum_j W[j, i] * s_in_j[t]                synaptic input (a sum over input spikes)
    u_i[t] = beta * u_i[t-1] + I_i[t]                 leak + integrate
    s_i[t] = 1 if u_i[t] >= threshold + theta_i       threshold comparison
    reset  : u_i = u_i - (threshold + theta_i)        (or u_i = 0 when reset = "zero")

theta_i is an extra per-neuron threshold. It is 0 in Phase 2 and only grows during
STDP training (Phase 3).

With wta=True (winner-take-all, used in Phase 3) at most one neuron spikes per
timestep, the one furthest above its threshold, and every membrane is reset
afterwards. This is a simple form of lateral inhibition.
"""

import torch


class LIFLayer:
    def __init__(self, num_inputs, num_hidden, neuron_cfg, seed=0):
        self.n_in, self.n_hid = num_inputs, num_hidden
        self.beta = neuron_cfg["beta"]
        self.threshold = neuron_cfg["threshold"]
        self.reset_mode = neuron_cfg["reset"]
        self.w_norm = neuron_cfg["w_norm"]
        self.reset(seed)

    def reset(self, seed=0):
        """Random positive weights, each neuron's incoming weights summing to w_norm."""
        g = torch.Generator().manual_seed(seed)
        w = torch.rand(self.n_in, self.n_hid, generator=g)
        self.W = w * (self.w_norm / w.sum(0, keepdim=True))
        self.theta = torch.zeros(self.n_hid)

    def step(self, u, s_in, beta, threshold, wta=False):
        """
        One timestep. Returns (u before reset, spikes, input current, effective thresholds).
        """
        current = s_in @ self.W
        u = beta * u + current
        thr = threshold + self.theta
        fired = u >= thr
        if wta:
            s = torch.zeros_like(u)
            if fired.any():
                winner = torch.where(fired, u - thr, torch.full_like(u, -1e9)).argmax()
                s[winner] = 1.0
        else:
            s = fired.float()
        return u, s, current, thr

    @torch.no_grad()
    def run(self, spikes_in, beta=None, threshold=None, wta=False):
        """
        spikes_in: (T, num_inputs) of 0/1. Always runs exactly T steps.
        Returns spikes (T, hidden), membrane (T, hidden) before the reset,
        and current (T, hidden).
        """
        beta = self.beta if beta is None else beta
        threshold = self.threshold if threshold is None else threshold
        u = torch.zeros(self.n_hid)
        spikes, membrane, current = [], [], []
        for t in range(spikes_in.shape[0]):
            u, s, i_t, thr = self.step(u, spikes_in[t], beta, threshold, wta)
            spikes.append(s)
            membrane.append(u)
            current.append(i_t)
            if wta:
                u = torch.zeros_like(u) if s.any() else u
            elif self.reset_mode == "subtract":
                u = u - s * thr
            else:
                u = u * (1 - s)
        return {"spikes": torch.stack(spikes), "membrane": torch.stack(membrane),
                "current": torch.stack(current)}
