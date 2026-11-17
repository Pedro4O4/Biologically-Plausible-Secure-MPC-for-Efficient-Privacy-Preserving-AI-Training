"""MNIST loading as flat float tensors in [0, 1]."""

from pathlib import Path

from torchvision import datasets

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist(train: bool):
    ds = datasets.MNIST(root=str(_DATA_DIR), train=train, download=False)
    images = ds.data.reshape(len(ds), -1).float() / 255.0
    return images, ds.targets


def make_splits(cfg: dict):
    """
    Fixed, seeded train/val/test split shared by every model (ANN now, SNN later).
    Returns {"train": (X, y), "val": (X, y), "test": (X, y)}.
    """
    import torch

    X, y = load_mnist(train=True)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train, n_val = cfg["data"]["num_train"], cfg["data"]["num_val"]
    tr, va = perm[:n_train], perm[n_train:n_train + n_val]
    return {"train": (X[tr], y[tr]), "val": (X[va], y[va]), "test": load_mnist(train=False)}
