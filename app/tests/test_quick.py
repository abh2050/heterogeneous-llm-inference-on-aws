import pytest

from planner.quick import ADVANCED, SIMPLIFIED, kv_gb, quick_estimate, quick_table, weights_gb


def test_formulas_match_hand_calculation(model, hardware):
    # Llama 3.1 8B example: 32 layers, 8 KV heads, head_dim 128 -> 131072 B/token at 2-byte KV.
    assert weights_gb(8e9, 2) == 16
    assert kv_gb(model, 2, 1000, 10) == pytest.approx(2 * 32 * 8 * 128 * 2 * 1000 * 10 / 1e9)
    r = quick_estimate(model, hardware, bytes_per_parameter=2, kv_bytes=2, context=1000, users=10)
    assert r["label"] == SIMPLIFIED != ADVANCED
    assert r["fits"] == ((r["weights_gb"] + r["kv_gb"]) * 1.10 <= hardware.memory_bytes / 1e9)
    step = model.parameters * 2 + r["kv_gb"] * 1e9
    assert r["decode_tokens_per_second_per_user"] == pytest.approx(hardware.bandwidth_gbs * 1e9 / step * 0.7)
    total = r["decode_tokens_per_second_total"]
    assert r["cost_per_million_tokens"] == pytest.approx(hardware.hourly_usd / (total * 3600) * 1e6)
    assert r["prefill_label"] == "estimated" and r["ttft_s"] is None


def test_efficiency_and_prefill_override_are_editable(model, hardware):
    base = quick_estimate(model, hardware, bytes_per_parameter=2, kv_bytes=2, context=1000, users=1)
    half = quick_estimate(
        model, hardware, bytes_per_parameter=2, kv_bytes=2, context=1000, users=1, decode_efficiency=0.35
    )
    assert half["decode_tokens_per_second_per_user"] == pytest.approx(
        base["decode_tokens_per_second_per_user"] / 2
    )
    custom = hardware.model_copy(update={"prefill_tokens_per_second": 5000})
    r = quick_estimate(
        model, custom, bytes_per_parameter=2, kv_bytes=2, context=1000, users=1, input_tokens=1000
    )
    assert r["prefill_label"] == "supplied" and r["ttft_s"] == pytest.approx(0.2)


def test_quick_table_uses_smallest_fitting_device_count(model, hardware):
    rows = quick_table(model, [hardware], bytes_per_parameter=2, kv_bytes=2, context=1000, users=1)
    assert rows[0]["devices"] == 1 and rows[0]["fits"]


def test_quick_estimate_endpoint(model):
    from fastapi.testclient import TestClient

    from app import app

    with TestClient(app) as client:
        boot = client.get("/api/bootstrap").json()
        plan = {k: boot[k] for k in ("model", "workload", "assumptions", "hardware")}
        out = client.post("/api/quick-estimate", json={"plan": plan, "users": 4}).json()
    assert out["users"] == 4 and len(out["rows"]) == len(boot["hardware"])
    assert out["label"] == "calculated, simplified" and out["advanced_label"] == "calculated, advanced"
