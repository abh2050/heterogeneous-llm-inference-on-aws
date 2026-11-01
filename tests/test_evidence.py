import pytest

from planner.benchmark import import_normalized, import_vllm_result, load_adapter
from planner.calibration import apply_evidence, context_matches
from planner.sizing import estimate
from planner.storage import Store


def test_missing_metrics_preserved_tpot_is_not_itl(run_factory):
    r = run_factory()
    context = r.model_dump(mode="json", exclude={"metrics", "raw_result"})
    parsed = import_vllm_result(
        {
            "p95_ttft_ms": 200,
            "p95_itl_ms": 40,
            "duration": 60,
            "completed": 300,
            "num_prompts": 300,
            "request_throughput": 5,
        },
        context,
    )
    assert parsed.metrics.ttft_s.p95 == 0.2
    assert parsed.metrics.itl_s.p95 == 0.04
    assert parsed.metrics.tpot_s.p95 is None
    assert parsed.metrics.timeouts is None
    assert parsed.metrics.output_throughput_tps is None


def test_record_requires_provenance_and_rejects_nonfinite(run_factory):
    r = run_factory().model_dump(mode="json")
    del r["config"]["engine_version"]
    with pytest.raises(ValueError):
        import_normalized(r)
    r = run_factory().model_dump(mode="json")
    r["metrics"]["tpot_s"]["mean"] = float("nan")
    with pytest.raises(ValueError):
        import_normalized(r)
    r = run_factory().model_dump(mode="json")
    r["timestamp"] = "2026-10-31T12:00:00"
    with pytest.raises(ValueError):
        import_normalized(r)


def test_unconfigured_adapter_has_no_invented_endpoints(monkeypatch):
    monkeypatch.delenv("BENCHMARK_ADAPTER", raising=False)
    a = load_adapter()
    assert not a.capabilities().configured
    assert not a.capabilities().submit
    with pytest.raises(NotImplementedError):
        a.submit({})


def test_measurement_precedence_and_deployment_validation(
    model, hardware, workload, assumptions, run_factory
):
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    r = run_factory(c)
    result = apply_evidence(c, model, hardware, workload, assumptions, [r])
    assert result.evidence == "Measured"
    assert result.measured_tpot_s == 0.02
    assert result.tpot_service_s == c.tpot_service_s  # baseline stays separately labeled
    assert "Benchmark validated" in result.validation
    assert c.evidence == "Estimated"


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "b" * 40),
        ("engine_version", "0.9.0"),
        ("kv_dtype", "fp8_e4m3"),
        ("topology", "different fabric"),
        ("launch_config", {"unknown_flag": True}),
    ],
)
def test_unrelated_context_cannot_calibrate(
    model, hardware, workload, assumptions, run_factory, field, value
):
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    r = run_factory().model_copy(deep=True)
    setattr(r.config, field, value)
    assert not context_matches(r, model, hardware, assumptions, c, workload)
    result = apply_evidence(c, model, hardware, workload, assumptions, [r])
    assert result.evidence == "Estimated"


def test_no_tail_validation_from_mean_only(model, hardware, workload, assumptions, run_factory):
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    r = run_factory()
    r.metrics.tpot_s.p95 = None
    result = apply_evidence(c, model, hardware, workload, assumptions, [r])
    assert result.evidence == "Estimated"
    assert "unvalidated" in result.validation


def test_horizontal_scaling_is_not_tail_validation(model, hardware, workload, assumptions, run_factory):
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    r = run_factory()
    r.metrics.request_throughput_rps = 1
    result = apply_evidence(c, model, hardware, workload, assumptions, [r])
    assert result.replicas == 3
    assert result.evidence == "Measured"
    assert "unvalidated" in result.validation


def test_empirical_calibration_coverage_and_extrapolation(
    model, hardware, workload, assumptions, run_factory
):
    runs = []
    for i, length in enumerate([1024, 1536, 2048]):
        w = workload.model_copy(update={"input_avg": length})
        c = estimate(model, hardware, w, assumptions, 1, 32)
        r = run_factory(c, w, run_id=f"fit-{i}")
        r.metrics.ttft_s.mean = c.ttft_service_s * 1.2
        r.metrics.tpot_s.mean = c.tpot_service_s * 1.2
        r.metrics.request_throughput_rps = c.request_capacity_per_replica * 0.8
        runs.append(r)
    target = workload.model_copy(update={"input_avg": 2560})
    c = estimate(model, hardware, target, assumptions, 1, 32)
    result = apply_evidence(c, model, hardware, target, assumptions, runs)
    assert result.evidence == "Calibrated"
    assert result.coverage["runs"] == 3
    assert result.coverage["extrapolates"]
    assert result.coverage["correction_factors_ttft_tpot_capacity"] == pytest.approx([1.2, 1.2, 0.8])
    assert result.coverage["in_sample_relative_residual_ttft_tpot_capacity"] == pytest.approx([0, 0, 0])
    assert "unvalidated" in result.validation


def test_storage_history_is_immutable_and_sanitized(tmp_path, run_factory, monkeypatch):
    s = Store(tmp_path / "planner.sqlite3")
    r = run_factory()
    # Synthetic credential-shaped sentinel, never a real credential.
    monkeypatch.setenv("BENCHMARK_API_KEY", "synthetic-secret-value")
    r.raw_result = {
        "Authorization": "Bearer synthetic-secret-value",
        "input_tokens": 123,
        "note": "synthetic-secret-value",
        "resident_tokens_per_sequence": 42,
    }
    s.add_benchmark(r)
    saved = s.benchmarks()[0]
    assert saved.raw_result["Authorization"] == "[REDACTED]"
    assert saved.raw_result["note"] == "[REDACTED]"
    assert saved.raw_result["input_tokens"] == 123
    assert saved.raw_result["resident_tokens_per_sequence"] == 42
    s.add_benchmark(r)  # idempotent
    r.metrics.errors = 1
    with pytest.raises(ValueError, match="different content"):
        s.add_benchmark(r)
    assert len(s.benchmarks()) == 1


def test_observed_prefix_behavior_required(model, hardware, workload, assumptions, run_factory):
    assumptions.prefix_caching = True
    workload.prefix_fraction = 0.5
    workload.prefix_length = 512
    workload.prefix_hit_rate = 0.8
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    r = run_factory(c)
    assert not context_matches(r, model, hardware, assumptions, c, workload)
    r.config.observed_prefix_hit_rate = 0.8
    assert context_matches(r, model, hardware, assumptions, c, workload)


def test_changed_metric_definition_cannot_validate(model, hardware, workload, assumptions, run_factory):
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    r = run_factory()
    r.metric_definitions["tpot_s"] = "Individual streamed token gaps, not request-average TPOT"
    result = apply_evidence(c, model, hardware, workload, assumptions, [r])
    assert result.evidence == "Estimated"


def test_import_batch_rolls_back_on_conflict(tmp_path, run_factory):
    s = Store(tmp_path / "atomic.sqlite3")
    original = run_factory(run_id="existing")
    s.add_benchmark(original)
    first = run_factory(run_id="new-run")
    conflict = run_factory(run_id="existing")
    conflict.metrics.errors = 1
    with pytest.raises(ValueError):
        s.add_benchmarks([first, conflict])
    assert [r.run_id for r in s.benchmarks()] == ["existing"]


def test_explicit_timeout_count_from_sidecar(run_factory):
    context = run_factory().model_dump(mode="json", exclude={"raw_result"})
    parsed = import_vllm_result({"p95_ttft_ms": 200, "p95_tpot_ms": 20}, context)
    assert parsed.metrics.timeouts == 0
    context["metrics"]["timeouts"] = None
    parsed = import_vllm_result({}, context)
    assert parsed.metrics.timeouts is None


def test_credentials_in_launch_arguments_are_scrubbed():
    from planner.security import scrub

    result = scrub(
        {
            "argv": ["vllm", "--api-key", "synthetic-opaque-value", "--max-num-seqs", "32"],
            "command": "serve --api-key synthetic-opaque-value",
            "url": "https://user:synthetic-pass@example.test",
            "header": "Authorization: Basic synthetic-basic-value",
        }
    )
    assert result["argv"][2] == "[REDACTED]"
    assert result["argv"][-1] == "32"
    assert "synthetic" not in str(result)
