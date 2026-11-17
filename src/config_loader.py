"""
Single entry point for reading configs/experiment_config.yaml.

Every script in src/ should call load_config() instead of opening
the YAML file itself or hard-coding any of its values. This keeps
all experiment parameters in one reproducible place.
"""

from pathlib import Path
import yaml

# Resolve the path relative to this file, so it works no matter
# which directory the script is launched from.
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiment_config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    print("Loaded config from:", _CONFIG_PATH)
    print(config)
