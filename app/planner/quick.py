"""Quick estimate (simplified): the plain closed-form formulas, kept apart from the roofline model.

Weights GB = parameters (billions) x bytes per parameter
KV GB      = 2 x layers x KV heads x head_dim x KV bytes x context x users / 1e9
Fits       = (weights + KV) x 1.10 <= memory
Decode ceiling (tokens/s per user) = bandwidth / bytes read per step x 0.7
Cost per million tokens = hourly cost / (total tokens/s x 3600) x 1,000,000

Bytes read per step = active-parameter weights plus every user's KV cache (each decode step reads both).
Every value here is labeled "calculated, simplified"; the advanced roofline results are separate.
"""

from __future__ import annotations

from planner.hardware_catalog import estimate_prefill_tps
from planner.models import Hardware, ModelSpec

SIMPLIFIED = "calculated, simplified"
ADVANCED = "calculated, advanced"
FIT_MARGIN = 1.10
DECODE_EFFICIENCY = 0.7


def weights_gb(parameters: float, bytes_per_parameter: float) -> float:
    return parameters / 1e9 * bytes_per_parameter


def kv_gb(model: ModelSpec, kv_bytes: float, context: int, users: int) -> float:
    return 2 * model.layers * model.kv_heads * model.head_dim * kv_bytes * context * users / 1e9


def quick_estimate(
    model: ModelSpec,
    hardware: Hardware,
    *,
    bytes_per_parameter: float,
    kv_bytes: float,
    context: int,
    users: int,
    input_tokens: int | None = None,
    devices: int = 1,
    decode_efficiency: float = DECODE_EFFICIENCY,
) -> dict:
    if model.missing:
        raise ValueError("Quick estimate needs: " + ", ".join(model.missing))
    if min(context, users, devices) < 1 or not 0 < decode_efficiency <= 1:
        raise ValueError("Context, users and devices must be at least 1; efficiency must be in (0, 1]")
    w = weights_gb(model.parameters, bytes_per_parameter)
    kv = kv_gb(model, kv_bytes, context, users)
    memory = hardware.memory_bytes * devices / 1e9
    needed = (w + kv) * FIT_MARGIN
    active = model.active_parameters or model.parameters
    step_bytes = active * bytes_per_parameter + kv * 1e9
    per_user = hardware.bandwidth_gbs * 1e9 * devices / step_bytes * decode_efficiency
    total = per_user * users
    hourly = hardware.hourly_usd * devices
    prefill = hardware.prefill_tokens_per_second
    prefill_label = "supplied"
    if prefill is None:
        prefill, prefill_label = estimate_prefill_tps(hardware.dense_bf16_tflops * devices, active), "estimated"
    return {
        "hardware_id": hardware.id,
        "devices": devices,
        "label": SIMPLIFIED,
        "weights_gb": w,
        "kv_gb": kv,
        "memory_gb": memory,
        "needed_gb": needed,
        "fits": needed <= memory,
        "decode_tokens_per_second_per_user": per_user,
        "decode_tokens_per_second_total": total,
        "decode_efficiency": decode_efficiency,
        "prefill_tokens_per_second": prefill,
        "prefill_label": prefill_label,
        "ttft_s": input_tokens / prefill if prefill and input_tokens else None,
        "hourly_usd": hourly,
        "cost_per_million_tokens": hourly / (total * 3600) * 1_000_000,
    }


def quick_table(model, hardware_list, **kwargs) -> list[dict]:
    """One row per hardware: the smallest allowed device count that fits, else the largest tried."""
    rows = []
    for h in hardware_list:
        row = None
        for n in sorted(h.allowed_tp):
            row = quick_estimate(model, h, devices=n, **kwargs)
            if row["fits"]:
                break
        rows.append(row)
    return rows
