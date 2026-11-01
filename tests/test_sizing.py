import math

import pytest
from pydantic import ValidationError

from planner.catalog import load_catalog
from planner.models import Assumptions, Workload
from planner.sizing import (
    GB,
    GIB,
    DisaggregationInputs,
    compare_disaggregated,
    cost_per_million,
    estimate,
    kv_bytes_per_token,
    memory_breakdown,
    service_times,
)


def test_units_and_llama_kv(model):
    assert GIB != GB
    assert kv_bytes_per_token(model, 2) == 131072
    assert kv_bytes_per_token(model, 1) == 65536
    assert kv_bytes_per_token(model, 2) * 8192 / GIB == 1


def test_block_rounding_and_maximum_residency(model, hardware, assumptions):
    w = Workload(input_avg=17, input_max=17, output_avg=2, output_max=2)
    m = memory_breakdown(model, hardware, w, assumptions, 1, 3)
    assert m["resident_tokens_per_sequence"] == 32
    assert m["kv"] == 3 * 32 * 131072


def test_precision_changes_only_kv_not_weights(model, hardware, workload, assumptions):
    m = memory_breakdown(model, hardware, workload, assumptions, 1, 32)
    fp8 = assumptions.model_copy(update={"kv_bytes": 1, "kv_dtype": "fp8_e4m3"})
    m8 = memory_breakdown(model, hardware, workload, fp8, 1, 32)
    assert m8["kv"] == m["kv"] / 2
    assert m8["weights"] == m["weights"]


def test_memory_exhaustion_is_not_serving_capacity(model, hardware, workload, assumptions):
    c = estimate(model, hardware, workload, assumptions, 1, 1024)
    assert not c.feasible
    assert c.request_capacity_per_replica is None
    assert any("memory" in r for r in c.reasons)


def test_cost_denominator():
    assert cost_per_million(3.6, 1000) == 1
    assert cost_per_million(1, 0) is None
    assert cost_per_million(0, 100) == 0


def test_replica_counts_and_cost_at_requested_workload(model, hardware, assumptions):
    w = Workload(peak_rps=10)
    c = estimate(model, hardware, w, assumptions, 1, 32)
    assert c.feasible
    assert c.replicas == math.ceil(w.planning_rps / c.request_capacity_per_replica)
    assert c.device_count == c.tp * c.replicas
    assert c.total_request_capacity >= w.planning_rps
    assert c.hourly_usd == c.device_count * hardware.hourly_usd
    assert c.cost_per_million_output == cost_per_million(c.hourly_usd, w.peak_rps * w.output_avg)
    assert c.cost_per_million_total == cost_per_million(
        c.hourly_usd, w.peak_rps * (w.input_avg + w.output_avg)
    )


def test_parallelism_sharding_and_collectives(model, hardware, workload, assumptions):
    one = service_times(model, hardware, workload, assumptions, 1, 32)
    two = service_times(model, hardware, workload, assumptions, 2, 32)
    assert two.tpot_s > one.tpot_s / 2  # scaling is not ideal
    assert memory_breakdown(model, hardware, workload, assumptions, 2, 32)["kv_shards"] == 2
    gqa = model.model_copy(update={"kv_heads": 2})
    assert memory_breakdown(gqa, hardware, workload, assumptions, 4, 32)["kv_shards"] == 2


def test_incompatible_heads_and_topology(model, hardware, workload, assumptions):
    bad = model.model_copy(update={"attention_heads": 30})
    assert any("Attention heads" in r for r in estimate(bad, hardware, workload, assumptions, 4, 1).reasons)
    l4 = load_catalog()[2]
    assert not estimate(model, l4, workload, assumptions, 2, 1).feasible


def test_missing_metadata_rejected(model, hardware, workload, assumptions):
    m = model.model_copy(update={"layers": None})
    c = estimate(m, hardware, workload, assumptions, 1, 1)
    assert not c.feasible
    assert not c.memory
    with pytest.raises(ValueError):
        kv_bytes_per_token(m, 2)


def test_context_overflow(model, hardware, assumptions):
    w = Workload(input_max=131072)
    assert any("context" in r for r in estimate(model, hardware, w, assumptions, 1, 1).reasons)


def test_prefix_fraction_without_length_does_not_save_compute(model, hardware, assumptions):
    w = Workload(prefix_fraction=0.8, prefix_hit_rate=1)
    off = service_times(model, hardware, w, assumptions, 1, 8)
    a = assumptions.model_copy(update={"prefix_caching": True})
    on = service_times(model, hardware, w, a, 1, 8)
    assert off.prefill_s == on.prefill_s
    cached = w.model_copy(update={"prefix_length": 1024})
    assert service_times(model, hardware, cached, a, 1, 8).prefill_s < on.prefill_s
    assert (
        memory_breakdown(model, hardware, cached, a, 1, 8)["kv"]
        > memory_breakdown(model, hardware, w, a, 1, 8)["kv"]
    )


def test_littles_law_is_consistent(model, hardware, workload, assumptions):
    t = service_times(model, hardware, workload, assumptions, 1, 32)
    assert t.capacity_rps * t.e2e_s <= 32 * assumptions.utilization_ceiling


def test_cpu_specific_assumptions_and_cache_rounding(model, workload, assumptions):
    cpu = load_catalog()[3]
    c = estimate(model, cpu, workload, assumptions, 1, 1)
    assert c.memory["kv"] % GIB == 0
    changed = assumptions.model_copy(update={"gpu_mfu": 0.01, "gpu_mbu": 0.01})
    assert service_times(model, cpu, workload, assumptions, 1, 1) == service_times(
        model, cpu, workload, changed, 1, 1
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(input_avg=9000),
        dict(output_avg=1),
        dict(prefix_length=5000),
        dict(input_percentiles={"p95": 1000, "p50": 2000}),
    ],
)
def test_invalid_workloads(changes):
    with pytest.raises(ValidationError):
        Workload(**changes)


def test_invalid_precision_and_runtime(model, hardware, workload):
    with pytest.raises(ValidationError):
        Assumptions(kv_dtype="fp8_e4m3", kv_bytes=2)
    bad = Assumptions(engine_version="99.0")
    assert not estimate(model, hardware, workload, bad, 1, 1).feasible
    assert not estimate(model, hardware, workload, Assumptions(weight_bytes=0.5), 1, 1).feasible
    a = Assumptions(activation_dtype="float16", kv_dtype="bfloat16")
    assert not estimate(model, hardware, workload, a, 1, 1).feasible


def test_disaggregation_accounts_for_all_pools_and_network(model, hardware, workload, assumptions):
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    inputs = DisaggregationInputs(100, 0.7, 1, 0.25, "test-router / test-connector")
    result = compare_disaggregated(model, hardware, hardware, workload, assumptions, c, inputs)
    assert (
        result["hourly_usd"]
        == (result["prefill_devices"] + result["decode_devices"]) * hardware.hourly_usd + 0.25
    )
    assert result["aggregate_network_gbps"] > 0
    assert result["transfer_service_s"] > 0
    saturated = compare_disaggregated(
        model,
        hardware,
        hardware,
        workload,
        assumptions,
        c,
        DisaggregationInputs(0.01, 0.7, 1, 0, "known-stack"),
    )
    assert not saturated["analytical_screen_passes"]


def test_scheduler_budget_and_gqa_geometry(model, hardware, workload, assumptions):
    small_budget = assumptions.model_copy(update={"batched_tokens": 16})
    assert not estimate(model, hardware, workload, small_budget, 1, 32).feasible
    invalid_gqa = model.model_copy(update={"kv_heads": 5})
    assert not estimate(invalid_gqa, hardware, workload, assumptions, 1, 1).feasible
