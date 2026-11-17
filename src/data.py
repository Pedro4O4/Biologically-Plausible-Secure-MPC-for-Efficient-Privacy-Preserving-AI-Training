"""MNIST loading as flat float tensors in [0, 1]."""

from pathlib import Path

from torchvision import datasets

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist(train: bool):
    ds = datasets.MNIST(root=str(_DATA_DIR), train=train, download=False)
    images = ds.data.reshape(len(ds), -1).float() / 255.0
    return images, ds.targets
