"""Validated domain records. Performance units are seconds, tokens, bytes and USD."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CANONICAL_METRIC_DEFINITIONS = {
    "ttft_s": "Client request to first token, including queueing and transport",
    "tpot_s": "Per-request (E2E - TTFT) / (output tokens - 1)",
    "itl_s": "Individual gaps between successive streamed tokens; distinct from TPOT",
}


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Provenance(Record):
    source: str
    kind: Literal["Repository", "Derived estimate", "Override", "Illustrative"]
    original: Any = None


class ModelSpec(Record):
    model_id: str
    revision: str | None = None
    requested_revision: str = "main"
    architecture: str = "unknown"
    model_type: str = "unknown"
    parameters: int | None = Field(default=None, gt=0)
    active_parameters: int | None = Field(default=None, gt=0)
    num_experts: int | None = Field(default=None, gt=0)
    num_experts_per_tok: int | None = Field(default=None, gt=0)
    expert_intermediate_size: int | None = Field(default=None, gt=0)
    layers: int | None = Field(default=None, gt=0)
    attention_heads: int | None = Field(default=None, gt=0)
    kv_heads: int | None = Field(default=None, gt=0)
    head_dim: int | None = Field(default=None, gt=0)
    hidden_size: int | None = Field(default=None, gt=0)
    context_limit: int | None = Field(default=None, gt=0)
    weight_dtype: str = "unknown"
    quantization: dict[str, Any] = Field(default_factory=dict)
    tensor_metadata: dict[str, Any] = Field(default_factory=dict)
    effective_weight_bytes: float | None = Field(default=None, gt=0)
    provenance: dict[str, Provenance] = Field(default_factory=dict)
    unsupported: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    raw_config: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return bool(self.num_experts and self.num_experts_per_tok)

    @property
    def missing(self) -> list[str]:
        return [
            x
            for x in (
                "parameters",
                "layers",
                "attention_heads",
                "kv_heads",
                "head_dim",
                "hidden_size",
                "context_limit",
            )
            if getattr(self, x) is None
        ]


class Workload(Record):
    input_avg: int = Field(default=2048, ge=1)
    input_max: int = Field(default=4096, ge=1)
    output_avg: int = Field(default=256, ge=2)
    output_max: int = Field(default=512, ge=2)
    # Optional summaries are retained, never silently substituted for means/maxima.
    input_percentiles: dict[str, float] = Field(default_factory=dict)
    output_percentiles: dict[str, float] = Field(default_factory=dict)
    distribution_notes: str = "Fixed synthetic lengths; production distribution unknown"
    peak_rps: float = Field(default=2.0, gt=0)
    burst_factor: float = Field(default=1.2, ge=1)
    headroom: float = Field(default=0.25, ge=0, le=2)
    prefix_fraction: float = Field(default=0, ge=0, le=1)
    prefix_length: int = Field(default=0, ge=0)
    prefix_hit_rate: float = Field(default=0, ge=0, le=1)
    retained_prefixes: int = Field(default=1, ge=1)
    ttft_p95_s: float = Field(default=1.0, gt=0)
    tpot_s: float = Field(default=0.05, gt=0)
    tpot_percentile: Literal["p50", "p90", "p95", "p99"] = "p95"
    operating_hours: float = Field(default=730, gt=0, le=8784)

    @model_validator(mode="after")
    def lengths(self):
        if self.input_avg > self.input_max or self.output_avg > self.output_max:
            raise ValueError("Average lengths must not exceed maxima")
        if self.prefix_length > self.input_avg:
            raise ValueError("Shared prefix must fit the average input length")
        for values, maximum in (
            (self.input_percentiles, self.input_max),
            (self.output_percentiles, self.output_max),
        ):
            if any(k not in {"p50", "p90", "p95", "p99"} or not 0 < v <= maximum for k, v in values.items()):
                raise ValueError("Length percentiles must use p50/p90/p95/p99 and fit the maximum")
            ordered = [values[k] for k in ("p50", "p90", "p95", "p99") if k in values]
            if ordered != sorted(ordered):
                raise ValueError("Length percentiles must be monotonic")
        return self

    @property
    def planning_rps(self) -> float:
        return self.peak_rps * self.burst_factor * (1 + self.headroom)


class Assumptions(Record):
    activation_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    weight_bytes: float = Field(default=2, gt=0, le=8)
    weight_overhead: float = Field(default=0.03, ge=0, le=1)
    kv_dtype: Literal["bfloat16", "float16", "fp8_e4m3"] = "bfloat16"
    kv_bytes: float = Field(default=2, gt=0, le=4)
    gpu_mfu: float = Field(default=0.40, gt=0, le=1)
    gpu_mbu: float = Field(default=0.65, gt=0, le=1)
    cpu_mfu: float = Field(default=0.25, gt=0, le=1)
    cpu_mbu: float = Field(default=0.65, gt=0, le=1)
    runtime_gib: float = Field(default=2, ge=0)
    activation_gib: float = Field(default=1, ge=0)
    communication_gib: float = Field(default=0.5, ge=0)
    safety_fraction: float = Field(default=0.10, ge=0, lt=0.5)
    memory_utilization: float = Field(default=0.9, gt=0, le=1)
    utilization_ceiling: float = Field(default=0.7, gt=0, lt=1)
    block_size: Literal[16] = 16
    tp_efficiency: float = Field(default=0.85, gt=0, le=1)
    collective_latency_us: float = Field(default=5, ge=0)
    engine: str = "vllm"
    engine_version: str = "0.10.2"
    batched_tokens: int = Field(default=8192, ge=16)
    prefix_caching: bool = False
    chunked_prefill: bool = True

    @model_validator(mode="after")
    def precision(self):
        expected = 1 if self.kv_dtype == "fp8_e4m3" else 2
        if self.kv_bytes != expected:
            raise ValueError("KV element bytes must match the selected KV dtype")
        return self


class Hardware(Record):
    id: str
    name: str
    kind: Literal["GPU", "CPU"]
    memory_bytes: int = Field(gt=0)
    bandwidth_gbs: float = Field(gt=0)
    dense_bf16_tflops: float = Field(gt=0)
    allowed_tp: list[int]
    interconnect_gbs: float = Field(ge=0)
    topology: str
    hourly_usd: float = Field(ge=0)
    provider: str
    region: str
    price_date: str
    pricing_kind: Literal["Illustrative", "Supplied"]
    price_scope: str
    source_urls: list[str]
    notes: list[str] = Field(default_factory=list)
    fp8_kv: bool = False


class Candidate(Record):
    id: str
    hardware_id: str
    tp: int
    concurrency: int
    replicas: int = 0
    device_count: int = 0
    feasible: bool = False
    reasons: list[str] = Field(default_factory=list)
    evidence: Literal["Estimated", "Calibrated", "Measured"] = "Estimated"
    validation: str = "Tail latency unvalidated"
    memory: dict[str, float] = Field(default_factory=dict)
    ttft_service_s: float | None = None
    prefill_service_s: float | None = None
    tpot_service_s: float | None = None
    request_capacity_per_replica: float | None = None
    total_request_capacity: float | None = None
    output_capacity_tps: float | None = None
    hourly_usd: float | None = None
    period_usd: float | None = None
    cost_per_million_output: float | None = None
    cost_per_million_total: float | None = None
    cost_denominator_output_tps: float | None = None
    measured_ttft_p95_s: float | None = None
    measured_tpot_s: float | None = None
    run_ids: list[str] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)


class Percentiles(Record):
    mean: float | None = Field(default=None, ge=0)
    p50: float | None = Field(default=None, ge=0)
    p90: float | None = Field(default=None, ge=0)
    p95: float | None = Field(default=None, ge=0)
    p99: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def monotonic(self):
        vals = [getattr(self, k) for k in ("p50", "p90", "p95", "p99") if getattr(self, k) is not None]
        if vals != sorted(vals):
            raise ValueError("Latency percentiles must be monotonic")
        return self


class BenchmarkMetrics(Record):
    ttft_s: Percentiles = Field(default_factory=Percentiles)
    tpot_s: Percentiles = Field(default_factory=Percentiles)
    itl_s: Percentiles = Field(default_factory=Percentiles)
    e2e_s: Percentiles = Field(default_factory=Percentiles)
    request_throughput_rps: float | None = Field(default=None, ge=0)
    input_throughput_tps: float | None = Field(default=None, ge=0)
    output_throughput_tps: float | None = Field(default=None, ge=0)
    errors: int | None = Field(default=None, ge=0)
    timeouts: int | None = Field(default=None, ge=0)


class BenchmarkConfig(Record):
    model_id: str
    revision: str | None = None
    architecture: str | None = None
    hardware_id: str
    hardware_count: int = Field(ge=1)
    topology: str | None = None
    engine: str
    engine_version: str
    launch_config: dict[str, Any] = Field(default_factory=dict)
    weight_dtype: str
    weight_bytes: float = Field(gt=0)
    quantization: dict[str, Any] = Field(default_factory=dict)
    activation_dtype: str
    kv_dtype: str
    tp: int = Field(default=1, ge=1)
    replicas: int = Field(default=1, ge=1)
    max_sequences: int = Field(ge=1)
    max_model_len: int = Field(ge=1)
    max_batched_tokens: int = Field(ge=1)
    memory_utilization: float = Field(gt=0, le=1)
    chunked_prefill: bool
    prefix_caching: bool
    observed_prefix_hit_rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def devices(self):
        if self.hardware_count != self.tp * self.replicas:
            raise ValueError("hardware_count must equal tp × replicas for supported dense serving")
        return self


class BenchmarkRecord(Record):
    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    timestamp: datetime
    source: str
    raw_result_reference: str | None = None
    config: BenchmarkConfig
    workload: Workload
    concurrency: int = Field(ge=1)
    request_rate_rps: float | None = Field(default=None, gt=0)
    load_mode: Literal["open_loop", "closed_loop", "unknown"] = "unknown"
    warmup_s: float | None = Field(default=None, ge=0)
    duration_s: float | None = Field(default=None, gt=0)
    sample_count: int | None = Field(default=None, ge=1)
    metrics: BenchmarkMetrics
    metric_definitions: dict[str, str] = Field(default_factory=lambda: CANONICAL_METRIC_DEFINITIONS.copy())
    raw_result: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def context(self):
        if self.timestamp.tzinfo is None:
            raise ValueError("Benchmark timestamp must include a timezone")
        if self.concurrency > self.config.max_sequences * self.config.replicas:
            raise ValueError("Concurrency exceeds the recorded deployment's scheduler slots")
        if self.workload.input_max + self.workload.output_max > self.config.max_model_len:
            raise ValueError("Benchmark workload exceeds recorded context limit")
        return self
