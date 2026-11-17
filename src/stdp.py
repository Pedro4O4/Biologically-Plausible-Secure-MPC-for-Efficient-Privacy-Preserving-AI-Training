"""
Phase 3: trace-based STDP, unsupervised and local.

Every input pixel keeps a "trace": a number that jumps up when the pixel spikes and
decays each timestep (so it says "how recently did this input fire").

When a hidden neuron j spikes, each of its incoming weights moves according to the
trace of its input pixel:

    W[i, j] += lr * (trace_i - x_target) * (w_max - W[i, j]) ** mu

  trace_i > x_target  (input fired recently)  -> weight grows
  trace_i < x_target  (input silent)          -> weight shrinks

Only the winning neuron learns, and it uses nothing but its own spike and its own
inputs' traces: no labels, no gradients, no backward pass.

Two stabilisers keep neurons from all learning the same thing:
  * winner-take-all: one hidden spike per timestep at most
  * adaptive threshold: a neuron that wins gets theta += theta_plus (harder to win again)
After every training image each neuron's incoming weights are rescaled to sum to w_norm.
"""

import torch


class STDPTrainer:
    def __init__(self, layer, learning_cfg):
        self.layer = layer
        self.threshold = learning_cfg["threshold"]
        self.lr = learning_cfg["lr"]
        self.trace_decay = learning_cfg["trace_decay"]
        self.x_target = learning_cfg["x_target"]
        self.w_max = learning_cfg["w_max"]
        self.mu = learning_cfg["mu"]
        self.theta_plus = learning_cfg["theta_plus"]

    @torch.no_grad()
    def train_sample(self, spikes_in):
        """One image: spikes_in is (T, num_inputs). Returns the number of hidden spikes."""
        layer = self.layer
        u = torch.zeros(layer.n_hid)
        trace = torch.zeros(layer.n_in)
        n_spikes = 0
        for t in range(spikes_in.shape[0]):
            s_in = spikes_in[t]
            trace = self.trace_decay * trace + s_in
            u, s, _, _ = layer.step(u, s_in, layer.beta, self.threshold, wta=True)
            if s.any():
                j = int(s.argmax())
                n_spikes += 1
                col = layer.W[:, j]
                dw = self.lr * (trace - self.x_target) * (self.w_max - col) ** self.mu
                layer.W[:, j] = (col + dw).clamp(0.0, self.w_max)
                layer.theta[j] += self.theta_plus
                u = torch.zeros_like(u)
        layer.W *= layer.w_norm / layer.W.sum(0, keepdim=True).clamp_min(1e-8)
        return n_spikes
