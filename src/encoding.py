
import torch

from config_loader import load_config
"""
    مسؤولة انها تقرا ال .yamal 
"""

def rate_encode(image: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
     هترجع tensor و هيبقي اسمه spikes
    """
    """
    Args:
        image: tensor of shape (num_inputs,), pixel values in [0, 1].
        num_steps: T, the fixed number of timesteps (from config).

    Returns:
        spikes: tensor of shape (num_steps, num_inputs) with values
                in {0.0, 1.0}. spikes[t, i] = 1 means pixel i fired
                a spike at timestep t.
    """
    # One random number per (timestep, pixel), drawn independently.
    random_values = torch.rand(num_steps, *image.shape)

    # Core rate-coding rule: a spike happens at step t for pixel i
    # if the random draw at that (t, i) falls below the pixel's
    # intensity. Brighter pixel -> higher chance of spiking at any
    # given step -> more spikes across the T steps on average.
    spikes = (random_values < image).float()
    #.float عشان تطلع true or false
    
    return spikes


if __name__ == "__main__":
    # num_steps comes from the single source of truth (the YAML config),
    # never hard-coded here.
    config = load_config()
    num_steps = config["simulation"]["num_steps"]

    # Smoke check with a tiny fake "image": 4 pixels, dark to bright.
    fake_pixels = torch.tensor([0.0, 0.3, 0.7, 1.0])

    spike_train = rate_encode(fake_pixels, num_steps)

    print("Pixel intensities:", fake_pixels.tolist())
    print("Spike train shape:", tuple(spike_train.shape))  # (25, 4)
    print("Spikes fired per pixel (out of 25 steps):")
    print(spike_train.sum(dim=0).tolist())
