import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Приводим пути к абсолютным
    base = Path.cwd()
    for key in cfg["paths"]:
        cfg["paths"][key] = (base / cfg["paths"][key]).resolve().as_posix()
    return cfg
