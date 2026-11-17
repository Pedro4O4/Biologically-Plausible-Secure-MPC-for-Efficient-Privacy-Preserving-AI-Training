"""
Loads one real MNIST digit, converts it to a spike train using
rate_encode(), and saves two plots to results/:
  1. The original digit image.
  2. A spike raster: which pixels fired at which timestep.

This is a visual sanity check only — not a thesis experiment.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in WSL; write straight to a file.
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from config_loader import load_config
from encoding import rate_encode

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_RESULTS_DIR = _PROJECT_ROOT / "results"


def main():
    config = load_config()
    num_steps = config["simulation"]["num_steps"]

    # Load the MNIST test set (already downloaded under data/MNIST).
    # ToTensor() scales pixel values from [0, 255] to [0.0, 1.0],
    # which is exactly the range rate_encode expects.
    dataset = datasets.MNIST(root=str(_DATA_DIR), train=False, download=False,
                              transform=transforms.ToTensor())

    image, label = dataset[0]          # shape: (1, 28, 28)
    flat_image = image.view(-1)        # shape: (784,)

    spike_train = rate_encode(flat_image, num_steps)   # shape: (num_steps, 784)

    print(f"Digit label: {label}")
    print(f"Image shape: {tuple(image.shape)} -> flattened: {tuple(flat_image.shape)}")
    print(f"Spike train shape: {tuple(spike_train.shape)}")
    print(f"Total spikes across all pixels and steps: {int(spike_train.sum())}")

    # --- Plot 1: the original digit ---
    fig1, ax1 = plt.subplots(figsize=(3, 3))
    ax1.imshow(image.squeeze(), cmap="gray")
    ax1.set_title(f"Original digit: {label}")
    ax1.axis("off")
    fig1.savefig(_RESULTS_DIR / "encoding_demo_original.png", bbox_inches="tight")

    # --- Plot 2: spike raster ---
    # x-axis = timestep, y-axis = pixel index. A dot means "this pixel
    # fired a spike at this timestep". We plot all 784 pixels.
    fig2, ax2 = plt.subplots(figsize=(6, 8))
    timesteps, pixel_ids = torch.where(spike_train == 1)
    ax2.scatter(timesteps, pixel_ids, s=2, color="black")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Pixel index (0-783)")
    ax2.set_title(f"Spike raster (rate coding), digit={label}, T={num_steps}")
    fig2.savefig(_RESULTS_DIR / "encoding_demo_raster.png", bbox_inches="tight")

    print(f"Saved: {_RESULTS_DIR / 'encoding_demo_original.png'}")
    print(f"Saved: {_RESULTS_DIR / 'encoding_demo_raster.png'}")


if __name__ == "__main__":
    main()
