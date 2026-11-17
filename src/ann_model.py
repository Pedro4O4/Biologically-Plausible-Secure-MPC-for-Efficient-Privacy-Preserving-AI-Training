"""
Phase 1: plaintext ANN baseline trained with backpropagation.

    784 inputs -> Linear -> ReLU -> Linear -> 10 outputs

Same hidden size as the SNN so the later comparison is fair.
"""

import time

import torch
from torch import nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    pred = logits.argmax(1)
    return F.cross_entropy(logits, y).item(), (pred == y).float().mean().item(), pred


def train_ann(cfg, splits, on_epoch=None, should_stop=lambda: False):
    """
    Returns (model, metrics). on_epoch(record) is called after every epoch
    so a UI can show progress; should_stop() lets a UI stop training early.
    """
    arch, hp = cfg["architecture"], cfg["ann"]
    torch.manual_seed(cfg["seed"])
    model = ANN(arch["num_inputs"], arch["num_hidden"], arch["num_outputs"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    X_train, y_train = splits["train"]
    shuffler = torch.Generator().manual_seed(cfg["seed"])
    history, t0 = [], time.time()

    for epoch in range(1, hp["epochs"] + 1):
        if should_stop():
            break
        model.train()
        order = torch.randperm(len(X_train), generator=shuffler)
        for i in range(0, len(X_train), hp["batch_size"]):
            idx = order[i:i + hp["batch_size"]]
            opt.zero_grad()
            F.cross_entropy(model(X_train[idx]), y_train[idx]).backward()   # forward + backprop
            opt.step()                                                      # weight update
        tr_loss, tr_acc, _ = evaluate(model, X_train, y_train)
        va_loss, va_acc, _ = evaluate(model, *splits["val"])
        record = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                  "val_loss": va_loss, "val_acc": va_acc,
                  "seconds": round(time.time() - t0, 2)}
        history.append(record)
        if on_epoch:
            on_epoch(record)
    train_seconds = round(time.time() - t0, 2)

    X_test, y_test = splits["test"]
    te_loss, te_acc, pred = evaluate(model, X_test, y_test)
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(y_test.tolist(), pred.tolist()):
        confusion[t, p] += 1
    per_class = (confusion.diag().float() / confusion.sum(1).clamp_min(1)).tolist()

    n_in, n_hid, n_out = arch["num_inputs"], arch["num_hidden"], arch["num_outputs"]
    metrics = {
        "history": history,
        "epochs_run": len(history),
        "train_seconds": train_seconds,
        "test": {"n": len(y_test), "accuracy": te_acc, "loss": te_loss,
                 "per_class": per_class, "confusion": confusion.tolist()},
        "cost": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "multiplications_per_sample": n_in * n_hid + n_hid * n_out,
            "relu_comparisons_per_sample": n_hid,
        },
        "setup": {"seed": cfg["seed"], "train": len(X_train), "val": len(splits["val"][1]),
                  "test": len(y_test), **hp},
    }
    return model, metrics
