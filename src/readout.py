"""
Phase 4: supervised linear readout on the hidden spike counts.

The STDP layer is frozen. For every image we count how many times each hidden neuron
spiked (divided by T), and a softmax classifier learns which neurons mean which digit.
This is the only place in the SNN pipeline that uses the labels.
"""

import torch
import torch.nn.functional as F


class LinearReadout:
    def __init__(self, num_inputs, num_outputs):
        # Zero start: softmax regression is a convex problem, so no random init is needed.
        self.W = torch.zeros(num_inputs, num_outputs, requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def scores(self, X):
        return X @ self.W + self.b

    def fit(self, X, y, X_val, y_val, epochs, lr, record_every=10):
        """Full-batch Adam. Returns a history (for the convergence plot)."""
        opt = torch.optim.Adam([self.W, self.b], lr=lr)
        history = []
        for step in range(1, epochs + 1):
            opt.zero_grad()
            loss = F.cross_entropy(self.scores(X), y)
            loss.backward()
            opt.step()
            if step % record_every == 0 or step == epochs:
                with torch.no_grad():
                    history.append({
                        "step": step, "train_loss": loss.item(),
                        "train_acc": (self.scores(X).argmax(1) == y).float().mean().item(),
                        "val_loss": F.cross_entropy(self.scores(X_val), y_val).item(),
                        "val_acc": (self.scores(X_val).argmax(1) == y_val).float().mean().item(),
                    })
        return history

    @torch.no_grad()
    def predict(self, X):
        return self.scores(X).argmax(1)


def classification_metrics(y_true, y_pred, num_classes=10):
    conf = torch.bincount(y_true * num_classes + y_pred, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = conf.diag().float()
    recall = tp / conf.sum(1).clamp_min(1)          # per-digit accuracy
    precision = tp / conf.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"accuracy": float(tp.sum() / conf.sum()), "confusion": conf.tolist(),
            "per_class": recall.tolist(), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean())}
