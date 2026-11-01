import json
from pathlib import Path

from planner.hf_metadata import normalize
from planner.models import Provenance


def example_model():
    config = json.loads((Path(__file__).parent.parent / "examples/llama31-8b.config.json").read_text())
    model = normalize("meta-llama/Llama-3.1-8B-Instruct", None, config)
    for key in model.provenance:
        if model.provenance[key].kind == "Repository":
            model.provenance[key] = Provenance(
                kind="Illustrative", source="Offline example config; verify with Hub lookup"
            )
    model.notes.insert(
        0,
        "Offline illustrative Llama 3.1 8B configuration, not a live lookup or benchmark. No commit SHA is pinned.",
    )
    return model
