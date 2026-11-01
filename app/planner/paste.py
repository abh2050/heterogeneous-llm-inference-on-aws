"""Parse pasted `vllm bench serve` output (text or JSON) and build a context from a short form.

Results go through the existing parsers (`import_vllm_result`, `import_normalized`); this module only
turns text into the flat result dict they already accept and fills the configuration sidecar from
model, hardware and runtime answers instead of requiring a sidecar file.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from planner.benchmark import build_run_context
from planner.models import Candidate
from planner.service import PlanningRequest

MAX_PASTE_CHARS = 2_000_000

_SCALARS = {
    "successful requests": "completed",
    "failed requests": "failed",
    "benchmark duration (s)": "duration",
    "total input tokens": "total_input_tokens",
    "total generated tokens": "total_output_tokens",
    "request throughput (req/s)": "request_throughput",
    "output token throughput (tok/s)": "output_throughput",
    "maximum request concurrency": "max_concurrency",
}
_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 ()/.\-]*?)\s*:\s*(-?\d+(?:\.\d+)?)\s*$")
_METRIC = re.compile(r"^(mean|median|p\d+(?:\.\d+)?)\s+(ttft|tpot|itl|e2el)\s*\(ms\)$", re.I)


def parse_vllm_text(text: str) -> dict[str, Any]:
    """Flat result dict (the keys of `vllm bench serve --save-result`) from the printed summary."""
    raw: dict[str, Any] = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        label, number = m.group(1).strip().lower(), float(m.group(2))
        if label in _SCALARS:
            raw[_SCALARS[label]] = int(number) if number.is_integer() and label != "benchmark duration (s)" else number
            continue
        metric = _METRIC.match(m.group(1).strip())
        if metric:
            stat, name = metric.group(1).lower(), metric.group(2).lower()
            raw[f"{stat}_{name}_ms"] = number
    if "failed" in raw and "completed" in raw:
        raw["num_prompts"] = raw["completed"] + raw["failed"]
    if not any(k.endswith("_ms") for k in raw):
        raise ValueError(
            "No latency results found. Paste the full `vllm bench serve` summary or its --save-result JSON."
        )
    return raw


def classify(text: str) -> tuple[str, Any]:
    """('normalized', payload) for Atlas-format JSON, otherwise ('vllm', flat result dict)."""
    if not text or not text.strip():
        raise ValueError("Paste benchmark output first")
    if len(text) > MAX_PASTE_CHARS:
        raise ValueError("Pasted text is too large (2 MB limit)")
    stripped = text.strip()
    if stripped[0] in "{[":
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            raise ValueError("The pasted JSON is not valid. Check for truncated or extra text.") from None
        rows = data if isinstance(data, list) else [data]
        if rows and all(isinstance(r, dict) and "config" in r and "metrics" in r for r in rows):
            return "normalized", data
        if isinstance(data, dict):
            if not any(k.endswith("_ms") for k in data):
                raise ValueError("JSON has no latency fields (for example mean_ttft_ms) and is not Atlas format")
            return "vllm", data
        raise ValueError("Paste one result object, or a list of Atlas-format records")
    return "vllm", parse_vllm_text(text)


def context_defaults(plan: PlanningRequest, raw: dict[str, Any]) -> dict[str, Any]:
    concurrency = raw.get("max_concurrency")
    return {
        "model_id": plan.model.model_id,
        "hardware_id": plan.hardware[0].id,
        "tp": 1,
        "replicas": 1,
        "concurrency": int(concurrency) if concurrency else None,
        "engine_version": plan.assumptions.engine_version,
    }


def short_context(plan: PlanningRequest, form: dict[str, Any]) -> dict[str, Any]:
    """Full sidecar from the short form; run_id/timestamp are the paste time and are labeled as such."""
    h = next((x for x in plan.hardware if x.id == form.get("hardware_id")), None)
    if h is None:
        raise ValueError("Choose the hardware this benchmark ran on")
    tp, replicas, concurrency = (int(form.get(k) or 0) for k in ("tp", "replicas", "concurrency"))
    if min(tp, replicas, concurrency) < 1:
        raise ValueError("Devices per replica, replicas and concurrency must be at least 1")
    model = plan.model
    model_id = str(form.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("Enter the model that was served")
    if model_id != model.model_id:
        model = model.model_copy(update={"model_id": model_id, "revision": None})
    assumptions = plan.assumptions.model_copy(
        update={"engine_version": str(form.get("engine_version") or plan.assumptions.engine_version)}
    )
    per_replica = -(-concurrency // replicas)
    candidate = Candidate(
        id=f"pasted/{h.id}/tp{tp}/c{per_replica}",
        hardware_id=h.id,
        tp=tp,
        concurrency=per_replica,
        replicas=replicas,
        device_count=tp * replicas,
    )
    context = build_run_context(model, h, candidate, plan.workload, assumptions)
    context.update(
        run_id="pasted-" + uuid.uuid4().hex[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="Pasted by user; timestamp is the import time, not the measurement time",
        load_mode="unknown",
    )
    context["request_rate_rps"] = None
    return context
