from datetime import datetime
from pathlib import Path
import re


SLUG_PATTERN = re.compile(r"[^a-z0-9._-]+")


def slugify(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    normalized = SLUG_PATTERN.sub("-", normalized)
    normalized = normalized.strip("-._")
    return normalized or "experiment"


def create_run_dir(output_root: str, experiment_title: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_{slugify(experiment_title)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
