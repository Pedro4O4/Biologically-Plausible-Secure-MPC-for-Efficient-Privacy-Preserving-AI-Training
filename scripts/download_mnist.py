"""Download MNIST into data/ (the training code reads it with download=False)."""

from pathlib import Path

from torchvision import datasets

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    for train in (True, False):
        datasets.MNIST(root=str(DATA_DIR), train=train, download=True)
    print(f"MNIST ready in {DATA_DIR / 'MNIST'}")
