import json
from pathlib import Path

from planner.models import Hardware

CATALOG_PATH = Path(__file__).parent / "data/hardware.v1.json"


def load_catalog(path: Path = CATALOG_PATH) -> list[Hardware]:
    data = json.loads(path.read_text())
    if data["schema_version"] != 1:
        raise ValueError("Unsupported hardware catalog schema")
    return [Hardware.model_validate(row) for row in data["hardware"]]
