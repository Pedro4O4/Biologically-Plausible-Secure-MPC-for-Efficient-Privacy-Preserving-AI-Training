"""
Phase 2: hidden layer of Leaky Integrate-and-Fire (LIF) neurons, plaintext, forward pass only.

Written by hand (not snnTorch) so every operation is explicit. Later, each of these
operations is what has to be done securely under MPC.

Per timestep t, for every hidden neuron i:
    I_i[t] = sum_j W[j, i] * s_in_j[t]          synaptic input (a sum over input spikes)
    u_i[t] = beta * u_i[t-1] + I_i[t]           leak + integrate
    s_i[t] = 1 if u_i[t] >= threshold else 0    threshold comparison
    reset  : u_i = u_i - threshold              (or u_i = 0 when reset = "zero")
"""

import torch


class LIFLayer:
    def __init__(self, num_inputs, num_hidden, neuron_cfg, seed=0):
        self.n_in, self.n_hid = num_inputs, num_hidden
        self.beta = neuron_cfg["beta"]
        self.threshold = neuron_cfg["threshold"]
        self.reset = neuron_cfg["reset"]
        # No learning yet: random positive weights, scaled so each neuron's incoming
        # weights sum to w_norm. Phase 3 (STDP) will change these.
        g = torch.Generator().manual_seed(seed)
        w = torch.rand(num_inputs, num_hidden, generator=g)
        self.W = w * (neuron_cfg["w_norm"] / w.sum(0, keepdim=True))

    @torch.no_grad()
    def run(self, spikes_in, beta=None, threshold=None):
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
            i_t = spikes_in[t] @ self.W
            u = beta * u + i_t
            s = (u >= threshold).float()
            spikes.append(s)
            membrane.append(u)
            current.append(i_t)
            u = u - s * threshold if self.reset == "subtract" else u * (1 - s)
        return {"spikes": torch.stack(spikes), "membrane": torch.stack(membrane),
                "current": torch.stack(current)}
