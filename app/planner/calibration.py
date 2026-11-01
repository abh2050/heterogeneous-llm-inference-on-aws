"""Context-matched empirical corrections; no attempt to identify MFU/MBU from aggregates."""

from __future__ import annotations

import math
import statistics

from planner.exports import vllm_command
from planner.models import (
    CANONICAL_METRIC_DEFINITIONS,
    Assumptions,
    BenchmarkRecord,
    Candidate,
    Hardware,
    ModelSpec,
    Workload,
)
from planner.sizing import estimate, price_candidate


WORKLOAD_KEYS = (
    "input_avg",
    "input_max",
    "output_avg",
    "output_max",
    "prefix_fraction",
    "prefix_length",
    "prefix_hit_rate",
    "retained_prefixes",
    "input_percentiles",
    "output_percentiles",
    "distribution_notes",
)


def context_matches(
    run: BenchmarkRecord,
    model: ModelSpec,
    h: Hardware,
    a: Assumptions,
    candidate: Candidate,
    w: Workload,
    exact_workload: bool = True,
) -> bool:
    cfg = run.config
    if not model.revision or not cfg.revision or model.revision != cfg.revision:
        return False
    equal = (
        cfg.model_id == model.model_id
        and cfg.hardware_id == h.id
        and cfg.topology == h.topology
        and cfg.engine == a.engine
        and cfg.engine_version == a.engine_version
        and cfg.tp == candidate.tp
        and cfg.weight_dtype == (a.activation_dtype if not model.quantization else model.weight_dtype)
        and cfg.weight_bytes == (model.effective_weight_bytes or a.weight_bytes)
        and cfg.quantization == model.quantization
        and cfg.activation_dtype == a.activation_dtype
        and cfg.kv_dtype == a.kv_dtype
        and cfg.prefix_caching == a.prefix_caching
        and cfg.chunked_prefill == a.chunked_prefill
        and cfg.max_sequences == candidate.concurrency
        and cfg.max_model_len == w.input_max + w.output_max
        and cfg.max_batched_tokens == a.batched_tokens
        and math.isclose(cfg.memory_utilization, a.memory_utilization)
    )
    command = vllm_command(model, h, candidate, w, a)
    # Per-run scheduler concurrency is part of argv, while observed active concurrency may vary.
    equal = equal and cfg.launch_config == {"argv": command.argv, "environment": command.environment}
    if a.prefix_caching and w.prefix_fraction > 0:
        equal = equal and (
            cfg.observed_prefix_hit_rate is not None
            and math.isclose(cfg.observed_prefix_hit_rate, w.prefix_hit_rate)
        )
    if exact_workload:
        equal = equal and run.concurrency == candidate.concurrency * cfg.replicas
        equal = equal and all(getattr(run.workload, key) == getattr(w, key) for key in WORKLOAD_KEYS)
    return bool(equal)


def has_quality(run: BenchmarkRecord) -> bool:
    # Explicit evidence floor, not a statistical power/confidence guarantee.
    return (
        run.load_mode == "open_loop"
        and run.request_rate_rps is not None
        and run.sample_count is not None
        and run.sample_count >= 200
        and run.duration_s is not None
        and run.duration_s >= 30
        and run.warmup_s is not None
        and run.warmup_s >= 5
        and run.metrics.errors == 0
        and run.metrics.timeouts == 0
        and all(
            run.metric_definitions.get(key) == CANONICAL_METRIC_DEFINITIONS[key]
            for key in ("ttft_s", "tpot_s")
        )
    )


def apply_evidence(
    candidate: Candidate,
    model: ModelSpec,
    h: Hardware,
    w: Workload,
    a: Assumptions,
    runs: list[BenchmarkRecord],
) -> Candidate:
    c = candidate.model_copy(deep=True)
    # Real measurements may rescue an analytical latency screen, never an incompatible/OOM config.
    if c.ttft_service_s is None or c.request_capacity_per_replica is None:
        return c
    exact = [r for r in runs if context_matches(r, model, h, a, c, w)]
    qualified = [
        r
        for r in exact
        if has_quality(r)
        and r.metrics.ttft_s.p95 is not None
        and r.metrics.ttft_s.p95 > 0
        and getattr(r.metrics.tpot_s, w.tpot_percentile) is not None
        and getattr(r.metrics.tpot_s, w.tpot_percentile) > 0
        and r.metrics.request_throughput_rps is not None
        and r.metrics.request_throughput_rps > 0
    ]
    if qualified:
        # At comparable loads, retain conservative latency/capacity across repeat runs.
        c.measured_ttft_p95_s = max(r.metrics.ttft_s.p95 for r in qualified)
        c.measured_tpot_s = max(getattr(r.metrics.tpot_s, w.tpot_percentile) for r in qualified)
        c.request_capacity_per_replica = min(
            r.metrics.request_throughput_rps / r.config.replicas for r in qualified
        )
        c.evidence = "Measured"
        c.run_ids = [r.run_id for r in qualified]
        c.reasons = []
        if c.measured_ttft_p95_s > w.ttft_p95_s:
            c.reasons.append("Matching measured p95 TTFT exceeds target")
        if c.measured_tpot_s > w.tpot_s:
            c.reasons.append(f"Matching measured {w.tpot_percentile} TPOT exceeds target")
        c.feasible = not c.reasons
        price_candidate(c, h, w)
        deployment_matches = [
            r
            for r in qualified
            if r.config.replicas == c.replicas
            and r.request_rate_rps >= w.planning_rps
            and r.metrics.request_throughput_rps >= w.planning_rps * 0.98
        ]
        c.validation = (
            "Benchmark validated · matching deployment/load"
            if c.feasible and deployment_matches
            else "Measured per-replica evidence · planned deployment tails unvalidated"
        )
        c.coverage = {
            "matching_runs": len(qualified),
            "exact_workload": True,
            "quality_floor": "≥200 samples, ≥30s duration, ≥5s warmup, open-loop, zero errors/timeouts",
            "deployment_match": bool(deployment_matches),
            "latency_basis": "Measured client latency percentiles, kept distinct from analytical service time",
        }
        return c

    # Calibrate only single-replica runs; config/precision/runtime must match exactly.
    pool = [
        r
        for r in runs
        if context_matches(r, model, h, a, c, w, exact_workload=False)
        and r.config.replicas == 1
        and has_quality(r)
        and all(
            getattr(r.workload, key) == getattr(w, key)
            for key in (
                "prefix_fraction",
                "prefix_length",
                "prefix_hit_rate",
                "retained_prefixes",
                "input_percentiles",
                "output_percentiles",
                "distribution_notes",
            )
        )
    ]
    ratios = []
    for r in pool:
        baseline = estimate(model, h, r.workload, a, c.tp, r.concurrency)
        if (
            baseline.ttft_service_s
            and baseline.tpot_service_s
            and baseline.request_capacity_per_replica
            and r.metrics.ttft_s.mean
            and r.metrics.tpot_s.mean
            and r.metrics.request_throughput_rps
        ):
            ratios.append(
                (
                    r,
                    r.metrics.ttft_s.mean / baseline.ttft_service_s,
                    r.metrics.tpot_s.mean / baseline.tpot_service_s,
                    r.metrics.request_throughput_rps / baseline.request_capacity_per_replica,
                )
            )
    if len(ratios) < 3:
        if exact:
            c.coverage["unusable_matching_runs"] = len(exact)
            c.coverage["reason"] = "Missing required metrics or representative open-loop quality context"
        return c
    correction = [statistics.median(row[k] for row in ratios) for k in (1, 2, 3)]
    c.ttft_service_s *= correction[0]
    c.tpot_service_s *= correction[1]
    c.request_capacity_per_replica *= correction[2]
    # Check Little's law after correcting capacity; do not allow a fit to bypass scheduler occupancy.
    e2e = c.ttft_service_s + (w.output_avg - 1) * c.tpot_service_s
    c.request_capacity_per_replica = min(
        c.request_capacity_per_replica, a.utilization_ceiling * c.concurrency / e2e
    )
    c.evidence = "Calibrated"
    c.run_ids = [row[0].run_id for row in ratios]
    ranges = {
        key: [
            min(getattr(row[0].workload, key) for row in ratios),
            max(getattr(row[0].workload, key) for row in ratios),
        ]
        for key in ("input_avg", "output_avg")
    }
    concs = [row[0].concurrency for row in ratios]
    ranges["concurrency"] = [min(concs), max(concs)]
    extrapolates = any(
        not lo <= (c.concurrency if key == "concurrency" else getattr(w, key)) <= hi
        for key, (lo, hi) in ranges.items()
    )
    residuals = [
        statistics.mean(abs(row[k] - correction[k - 1]) / row[k] for row in ratios) for k in (1, 2, 3)
    ]
    c.coverage = {
        "runs": len(ratios),
        "ranges": ranges,
        "extrapolates": extrapolates,
        "correction_factors_ttft_tpot_capacity": correction,
        "in_sample_relative_residual_ttft_tpot_capacity": residuals,
        "caveat": "Empirical means, not identifiable MFU/MBU or confidence intervals; tail SLO remains unvalidated",
    }
    c.reasons = []
    if c.ttft_service_s > w.ttft_p95_s:
        c.reasons.append("Calibrated mean TTFT exceeds target")
    if c.tpot_service_s > w.tpot_s:
        c.reasons.append("Calibrated mean TPOT exceeds target")
    c.feasible = not c.reasons
    c.validation = "Calibrated means · tail latency unvalidated" + (
        " · extrapolation" if extrapolates else ""
    )
    price_candidate(c, h, w)
    return c
