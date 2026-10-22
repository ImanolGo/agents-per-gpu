"""Load configs/*.yaml. All tunables live there, not in code."""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path("configs")


def load(name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    with open(Path(config_dir) / f"{name}.yaml") as f:
        return yaml.safe_load(f)
