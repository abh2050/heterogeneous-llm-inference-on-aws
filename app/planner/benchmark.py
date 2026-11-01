"""Small service boundary. No service endpoints or authentication scheme are assumed."""

from __future__ import annotations

import importlib
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import Field

from planner.exports import vllm_command
from planner.models import (
    Assumptions,
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkRecord,
    Candidate,
    Hardware,
    ModelSpec,
    Percentiles,
    Record,
    Workload,
)
from planner.security import scrub


class Capabilities(Record):
    configured: bool = False
    service_name: str = "Unconfigured · JSON import available"
    health: bool = False
    submit: bool = False
    poll: bool = False
    results: bool = False
    cancel: bool = False
    notes: list[str] = Field(default_factory=list)


class JobState(Record):
    job_id: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown"]
    message: str | None = None


class BenchmarkAdapter(ABC):
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Return real service capabilities; do not probe guessed endpoints."""

    def health(self) -> dict:
        raise NotImplementedError("Health check is not supported")

    def submit(self, payload: dict) -> JobState:
        raise NotImplementedError("Job submission is not supported")

    def poll(self, job_id: str) -> JobState:
        raise NotImplementedError("Polling is not supported")

    def results(self, job_id: str) -> list[BenchmarkRecord]:
        raise NotImplementedError("Result fetching is not supported")

    def cancel(self, job_id: str) -> JobState:
        raise NotImplementedError("Cancellation is not supported")


class UnconfiguredAdapter(BenchmarkAdapter):
    def capabilities(self):
        return Capabilities(
            notes=[
                "Provide the service base URL, authentication method and API spec/sample result.",
                "No benchmark service implementation or contract exists in the supplied workspace.",
            ]
        )


def load_adapter(path: str | None = None) -> BenchmarkAdapter:
    """Factory lives in trusted local code; all secrets stay in its process environment."""
    path = path or os.getenv("BENCHMARK_ADAPTER", "")
    if not path:
        return UnconfiguredAdapter()
    try:
        module, factory = path.split(":", 1)
        adapter = getattr(importlib.import_module(module), factory)()
        if not isinstance(adapter, BenchmarkAdapter):
            raise TypeError
        return adapter
    except Exception:
        raise ValueError(
            "Cannot load BENCHMARK_ADAPTER. Use an importable module:factory returning BenchmarkAdapter; check local configuration."
        ) from None


def build_run_context(
    model: ModelSpec, h: Hardware, candidate: Candidate, w: Workload, a: Assumptions
) -> dict[str, Any]:
    config = BenchmarkConfig(
        model_id=model.model_id,
        revision=model.revision,
        architecture=model.architecture,
        hardware_id=h.id,
        hardware_count=candidate.device_count,
        topology=h.topology,
        engine=a.engine,
        engine_version=a.engine_version,
        weight_dtype=a.activation_dtype if not model.quantization else model.weight_dtype,
        launch_config={
            "argv": vllm_command(model, h, candidate, w, a).argv,
            "environment": vllm_command(model, h, candidate, w, a).environment,
        },
        weight_bytes=model.effective_weight_bytes or a.weight_bytes,
        quantization=model.quantization,
        activation_dtype=a.activation_dtype,
        kv_dtype=a.kv_dtype,
        tp=candidate.tp,
        replicas=candidate.replicas,
        max_sequences=candidate.concurrency,
        max_model_len=w.input_max + w.output_max,
        max_batched_tokens=a.batched_tokens,
        memory_utilization=a.memory_utilization,
        chunked_prefill=a.chunked_prefill,
        prefix_caching=a.prefix_caching,
    )
    return scrub(
        {
            "config": config.model_dump(mode="json"),
            "workload": w.model_dump(mode="json"),
            "concurrency": candidate.concurrency * candidate.replicas,
            "request_rate_rps": w.planning_rps,
            "load_mode": "open_loop",
        }
    )


def import_normalized(payload: str | bytes | dict | list) -> list[BenchmarkRecord]:
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    rows = payload if isinstance(payload, list) else [payload]
    if len(rows) > 1000:
        raise ValueError("Import at most 1000 benchmark records at a time")
    return [BenchmarkRecord.model_validate(scrub(row)) for row in rows]


def import_vllm_result(raw: dict, context: dict) -> BenchmarkRecord:
    """vLLM 0.10.2 `bench serve` flat JSON + explicit configuration sidecar.

    Missing metrics are retained as None. No inferred TPOT from ITL, and no
    inferred timeouts/warmup/client arrival rate from unrelated fields.
    """
    metrics: dict[str, Any] = {}
    for name, dest in (("ttft", "ttft_s"), ("tpot", "tpot_s"), ("itl", "itl_s"), ("e2el", "e2e_s")):
        values = {}
        for pct in ("mean", "p50", "p90", "p95", "p99"):
            field = f"{pct}_{name}_ms"
            if raw.get(field) is not None:
                values[pct] = float(raw[field]) / 1000
        # vLLM uses median_* rather than p50_* in standard output.
        if raw.get(f"median_{name}_ms") is not None and "p50" not in values:
            values["p50"] = float(raw[f"median_{name}_ms"]) / 1000
        metrics[dest] = Percentiles(**values)
    for source, dest in (
        ("request_throughput", "request_throughput_rps"),
        ("output_throughput", "output_throughput_tps"),
    ):
        if raw.get(source) is not None:
            metrics[dest] = raw[source]
    if raw.get("total_input_tokens") is not None and raw.get("duration", 0) > 0:
        metrics["input_throughput_tps"] = raw["total_input_tokens"] / raw["duration"]
    if raw.get("num_prompts") is not None and raw.get("completed") is not None:
        metrics["errors"] = int(raw["num_prompts"]) - int(raw["completed"])
    # A flat vLLM result does not distinguish timeouts. Retain a known count
    # only when the user explicitly supplies one from their measurement records.
    if context.get("metrics", {}).get("timeouts") is not None:
        metrics["timeouts"] = context["metrics"]["timeouts"]
    result = dict(context)
    result["metrics"] = BenchmarkMetrics(**metrics).model_dump(mode="json")
    result["raw_result"] = scrub(raw)
    if raw.get("duration") is not None:
        result["duration_s"] = raw["duration"]
    if raw.get("completed"):
        result["sample_count"] = raw["completed"]
    # run_id and timezone-aware timestamp come from sidecar; never manufacture a run.
    return BenchmarkRecord.model_validate(scrub(result))
