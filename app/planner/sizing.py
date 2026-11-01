"""Conservative dense decoder heuristics. No analytical output is a p95 measurement.

All arithmetic uses seconds, bytes, FLOPs and decimal bandwidth GB/s.
See docs/methodology.md for limitations and assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from planner.models import Assumptions, Candidate, Hardware, ModelSpec, Workload

GIB = 2**30
GB = 10**9


def kv_bytes_per_token(model: ModelSpec, kv_element_bytes: float) -> float:
    if any(getattr(model, key) is None for key in ("layers", "kv_heads", "head_dim")):
        raise ValueError("KV sizing needs layers, KV heads and head dimension")
    return 2 * model.layers * model.kv_heads * model.head_dim * kv_element_bytes


def cost_per_million(hourly_usd: float, achieved_tokens_per_second: float) -> float | None:
    if achieved_tokens_per_second <= 0:
        return None
    return hourly_usd * 1_000_000 / (achieved_tokens_per_second * 3600)


def compatibility(model: ModelSpec, hardware: Hardware, a: Assumptions, tp: int, w: Workload) -> list[str]:
    reasons = list(model.unsupported)
    if model.missing:
        reasons.append("Missing metadata: " + ", ".join(model.missing))
    if tp not in hardware.allowed_tp:
        reasons.append("TP topology is not cataloged for this hardware")
    if model.attention_heads and model.attention_heads % tp:
        reasons.append("Attention heads must divide tensor-parallel devices")
    if model.attention_heads and model.kv_heads and model.attention_heads % model.kv_heads:
        reasons.append("Dense grouped-query attention requires query heads to divide evenly across KV heads")
    if model.kv_heads and model.kv_heads % tp and tp % model.kv_heads:
        reasons.append("KV heads must shard evenly or replicate evenly across TP ranks")
    if model.hidden_size and model.hidden_size % tp:
        reasons.append("Hidden dimension does not divide tensor parallelism")
    if tp > 1 and hardware.interconnect_gbs <= 0:
        reasons.append("No supported interconnect for multi-device tensor parallelism")
    if model.context_limit and w.input_max + w.output_max > model.context_limit:
        reasons.append("Maximum input + output exceeds the model context limit")
    if a.kv_dtype != "fp8_e4m3" and a.kv_dtype != a.activation_dtype:
        reasons.append("Non-FP8 KV dtype must match the vLLM serving dtype")
    if not model.quantization and a.weight_bytes != (4 if a.activation_dtype == "float32" else 2):
        reasons.append("Dense weight bytes must match the serving weight/activation dtype")
    if not a.chunked_prefill and a.batched_tokens < w.input_max + w.output_max:
        reasons.append("Without chunked prefill, batched-token budget must cover max model length")
    if a.kv_dtype == "fp8_e4m3" and (not hardware.fp8_kv or a.engine_version != "0.10.2"):
        reasons.append("FP8 KV support is not verified for this hardware/runtime profile")
    if hardware.kind == "CPU" and a.activation_dtype != "bfloat16":
        reasons.append("CPU compute catalog only covers the AMX BF16 serving path")
    if hardware.kind == "GPU" and a.activation_dtype == "float32":
        reasons.append("Catalog compute ceilings cover BF16/FP16, not FP32")
    if model.quantization and model.effective_weight_bytes is None:
        reasons.append(
            "Quantized weight storage is unknown: override effective bytes/parameter including format packing assumptions"
        )
    if model.quantization:
        reasons.append(
            "Quantized kernel/format is not in the verified runtime profiles; memory inspection only"
        )
    if a.engine != "vllm" or a.engine_version != "0.10.2":
        reasons.append(
            "Runtime compatibility profile unavailable; choose vLLM 0.10.2 or add a verified profile"
        )
    return reasons


def memory_breakdown(
    model: ModelSpec, h: Hardware, w: Workload, a: Assumptions, tp: int, concurrency: int
) -> dict[str, float]:
    weight_bytes = model.effective_weight_bytes or a.weight_bytes
    weights = model.parameters * weight_bytes * (1 + a.weight_overhead) / tp
    kv_shards = min(tp, model.kv_heads)
    tokens_per_sequence = math.ceil((w.input_max + w.output_max) / a.block_size) * a.block_size
    # Do not assume memory savings from shared blocks; account for retained cached prefixes separately.
    retained = (
        math.ceil(w.prefix_length / a.block_size) * a.block_size * w.retained_prefixes
        if a.prefix_caching and w.prefix_fraction > 0
        else 0
    )
    kv = kv_bytes_per_token(model, a.kv_bytes) * (concurrency * tokens_per_sequence + retained) / kv_shards
    kv_raw = kv
    if h.kind == "CPU":
        kv = math.ceil(kv / GIB) * GIB  # CPU runtime allocates KV in whole GiB
    runtime = a.runtime_gib * GIB
    activations = a.activation_gib * GIB
    comm = a.communication_gib * GIB if tp > 1 else 0
    safety = h.memory_bytes * a.safety_fraction
    used = weights + kv + runtime + activations + comm + safety
    return {
        "weights": weights,
        "kv": kv,
        "kv_unrounded": kv_raw,
        "activations": activations,
        "runtime": runtime,
        "communication": comm,
        "safety": safety,
        "total": used,
        "budget": h.memory_bytes * a.memory_utilization,
        "physical": h.memory_bytes,
        "resident_tokens_per_sequence": tokens_per_sequence,
        "kv_shards": kv_shards,
    }


@dataclass(frozen=True)
class ServiceTimes:
    prefill_s: float
    ttft_s: float
    tpot_s: float
    capacity_rps: float
    e2e_s: float
    prefill_fraction: float


def service_times(
    model: ModelSpec, h: Hardware, w: Workload, a: Assumptions, tp: int, concurrency: int
) -> ServiceTimes:
    p = model.active_parameters or model.parameters
    mfu = a.cpu_mfu if h.kind == "CPU" else a.gpu_mfu
    mbu = a.cpu_mbu if h.kind == "CPU" else a.gpu_mbu
    scale = a.tp_efficiency ** math.log2(tp)
    compute = h.dense_bf16_tflops * 1e12 * tp * mfu * scale
    bandwidth = h.bandwidth_gbs * GB * tp * mbu * scale
    saved = w.prefix_fraction * w.prefix_length * w.prefix_hit_rate if a.prefix_caching else 0
    uncached = max(1, w.input_avg - saved)
    activation_bytes = 4 if a.activation_dtype == "float32" else 2

    def communication(tokens: float) -> float:
        if tp == 1:
            return 0
        # Two all-reduces per transformer layer; ring factor plus per-collective latency.
        traffic = 2 * model.layers * 2 * (tp - 1) / tp * tokens * model.hidden_size * activation_bytes
        return traffic / (h.interconnect_gbs * GB) + 2 * model.layers * a.collective_latency_us / 1e6

    # Include an attention FLOP term, but not kernel scheduling/queueing costs.
    prefill_flops = 2 * p * uncached + 4 * model.layers * model.hidden_size * uncached**2
    prefill = prefill_flops / compute + communication(uncached)
    context = w.input_avg + (w.output_avg - 1) / 2
    # KV heads replicate when tp exceeds their count; aggregate traffic is scaled accordingly.
    kv_replication = tp / min(tp, model.kv_heads)
    traffic = (
        p * (model.effective_weight_bytes or a.weight_bytes) * (1 + a.weight_overhead)
        + kv_bytes_per_token(model, a.kv_bytes) * context * concurrency * kv_replication
    )
    decode_flops = concurrency * (2 * p + 4 * model.layers * model.hidden_size * context)
    step = max(decode_flops / compute, traffic / bandwidth) + communication(concurrency)
    ttft = prefill + step  # first generated token; service estimate, not client p95
    e2e = ttft + (w.output_avg - 1) * step
    service_demand = prefill + w.output_avg * step / concurrency
    capacity = min(a.utilization_ceiling / service_demand, a.utilization_ceiling * concurrency / e2e)
    return ServiceTimes(prefill, ttft, step, capacity, e2e, prefill / service_demand)


def price_candidate(candidate: Candidate, h: Hardware, w: Workload):
    capacity = candidate.request_capacity_per_replica or 0
    if capacity <= 0:
        return
    candidate.replicas = max(1, math.ceil(w.planning_rps / capacity))
    candidate.device_count = candidate.replicas * candidate.tp
    candidate.total_request_capacity = capacity * candidate.replicas
    candidate.output_capacity_tps = candidate.total_request_capacity * w.output_avg
    candidate.hourly_usd = h.hourly_usd * candidate.device_count
    candidate.period_usd = candidate.hourly_usd * w.operating_hours
    # Delivered workload, not idle theoretical maximum, is the price denominator.
    delivered_rps = min(w.peak_rps, candidate.total_request_capacity)
    candidate.cost_denominator_output_tps = delivered_rps * w.output_avg
    candidate.cost_per_million_output = cost_per_million(candidate.hourly_usd, delivered_rps * w.output_avg)
    candidate.cost_per_million_total = cost_per_million(
        candidate.hourly_usd, delivered_rps * (w.input_avg + w.output_avg)
    )


def estimate(
    model: ModelSpec, h: Hardware, w: Workload, a: Assumptions, tp: int, concurrency: int
) -> Candidate:
    candidate = Candidate(
        id=f"{h.id}/tp{tp}/c{concurrency}", hardware_id=h.id, tp=tp, concurrency=concurrency
    )
    candidate.reasons = compatibility(model, h, a, tp, w)
    if a.batched_tokens < concurrency:
        candidate.reasons.append("Batched-token limit must cover at least one token per active sequence")
    if model.missing:
        return candidate
    candidate.memory = memory_breakdown(model, h, w, a, tp, concurrency)
    if candidate.memory["total"] > candidate.memory["budget"]:
        candidate.reasons.append("Per-device memory exceeds utilization budget after safety reserve")
    if candidate.reasons:
        return candidate
    t = service_times(model, h, w, a, tp, concurrency)
    candidate.prefill_service_s = t.prefill_s
    candidate.ttft_service_s = t.ttft_s
    candidate.tpot_service_s = t.tpot_s
    candidate.request_capacity_per_replica = t.capacity_rps
    candidate.coverage = {
        "prefill_service_fraction": t.prefill_fraction,
        "little_law_concurrency_at_capacity": t.capacity_rps * t.e2e_s,
        "latency_basis": "Analytical mean-length service time; no queueing model",
    }
    if t.ttft_s > w.ttft_p95_s:
        candidate.reasons.append("Estimated TTFT service time already exceeds p95 target")
    if t.tpot_s > w.tpot_s:
        candidate.reasons.append(
            "Estimated decode step exceeds TPOT target; request TPOT tails need measurements"
        )
    candidate.feasible = not candidate.reasons
    candidate.validation = (
        "Analytically feasible · tail latency unvalidated"
        if candidate.feasible
        else "Analytical screen failed"
    )
    price_candidate(candidate, h, w)
    return candidate


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (
            not c.feasible,
            c.hourly_usd if c.hourly_usd is not None else float("inf"),
            c.device_count,
            c.id,
        ),
    )


@dataclass(frozen=True)
class DisaggregationInputs:
    network_gbps: float
    network_efficiency: float
    routing_ms: float
    router_hourly_usd: float
    transfer_stack: str


def compare_disaggregated(
    model: ModelSpec,
    prefill_h: Hardware,
    decode_h: Hardware,
    w: Workload,
    a: Assumptions,
    decode: Candidate,
    inputs: DisaggregationInputs,
) -> dict:
    """Exploratory phase pools; never declares disaggregated runtime compatibility."""
    if (
        inputs.network_gbps <= 0
        or not 0 < inputs.network_efficiency <= 1
        or inputs.routing_ms < 0
        or inputs.router_hourly_usd < 0
        or not inputs.transfer_stack.strip()
    ):
        raise ValueError(
            "Network, efficiency, routing, cost and an actual transfer/router stack are required"
        )
    pre = estimate(model, prefill_h, w, a, 1, 1)
    if not pre.prefill_service_s or not decode.tpot_service_s:
        return {"status": "Insufficient compatible memory/performance evidence"}
    # Full input KV transfer is conservative when shared blocks could be reused remotely.
    transfer_bytes = kv_bytes_per_token(model, a.kv_bytes) * w.input_avg
    link_bytes_s = inputs.network_gbps * 1e9 / 8 * inputs.network_efficiency
    transfer_s = transfer_bytes / link_bytes_s
    pre_capacity = a.utilization_ceiling / (pre.prefill_service_s + transfer_s)
    decode_capacity = a.utilization_ceiling * decode.concurrency / (w.output_avg * decode.tpot_service_s)
    n_pre = max(1, math.ceil(w.planning_rps / pre_capacity))
    n_decode = max(1, math.ceil(w.planning_rps / decode_capacity))
    aggregate_gbps = w.planning_rps * transfer_bytes * 8 / 1e9
    cost = (
        n_pre * prefill_h.hourly_usd + n_decode * decode.tp * decode_h.hourly_usd + inputs.router_hourly_usd
    )
    ttft = pre.prefill_service_s + transfer_s + inputs.routing_ms / 1000 + decode.tpot_service_s
    screen = (
        aggregate_gbps <= inputs.network_gbps * inputs.network_efficiency
        and ttft <= w.ttft_p95_s
        and decode.tpot_service_s <= w.tpot_s
    )
    return {
        "status": "Estimated phase-pool comparison · runtime and tails unvalidated",
        "analytical_screen_passes": screen,
        "prefill_replicas": n_pre,
        "prefill_devices": n_pre,
        "decode_replicas": n_decode,
        "decode_devices": n_decode * decode.tp,
        "hourly_usd": cost,
        "transfer_gib_per_request": transfer_bytes / GIB,
        "transfer_service_s": transfer_s,
        "ttft_service_s": ttft,
        "aggregate_network_gbps": aggregate_gbps,
        "network_budget_gbps": inputs.network_gbps * inputs.network_efficiency,
        "transfer_stack": inputs.transfer_stack,
        "cost_per_million_output": cost_per_million(cost, w.peak_rps * w.output_avg),
        "limitations": [
            "No queueing/tail model or validated router/connector compatibility",
            "Prefill concurrency fixed at one; decode reserves full resident KV",
            "One shared network budget assumed; verify bidirectional fabric and routing",
            "Phase isolation benefit and cache behavior require end-to-end benchmarks",
        ],
    }
