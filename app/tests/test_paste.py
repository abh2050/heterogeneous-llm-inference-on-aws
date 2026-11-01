import json

import pytest
from fastapi.testclient import TestClient

from app import app
from planner.paste import classify, parse_vllm_text

TEXT = """
============ Serving Benchmark Result ============
Successful requests:                     98
Failed requests:                         2
Maximum request concurrency:             16
Benchmark duration (s):                  20.00
Total input tokens:                      200704
Total generated tokens:                  25088
Request throughput (req/s):              4.90
Output token throughput (tok/s):         1254.40
---------------Time to First Token----------------
Mean TTFT (ms):                          120.50
Median TTFT (ms):                        110.00
P99 TTFT (ms):                           300.25
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.00
Median TPOT (ms):                        14.00
P99 TPOT (ms):                           25.00
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.50
Median ITL (ms):                         13.00
P99 ITL (ms):                            40.00
==================================================
"""


def test_text_summary_parses_to_the_flat_keys_the_existing_parser_reads():
    raw = parse_vllm_text(TEXT)
    assert raw["mean_ttft_ms"] == 120.5 and raw["median_tpot_ms"] == 14 and raw["p99_itl_ms"] == 40
    assert raw["completed"] == 98 and raw["num_prompts"] == 100 and raw["duration"] == 20
    assert raw["max_concurrency"] == 16


@pytest.mark.parametrize("bad", ["", "   ", "hello world", '{"a": 1}', "{not json", "[1, 2]"])
def test_unusable_paste_is_rejected(bad):
    with pytest.raises(ValueError):
        classify(bad)


def test_flat_json_and_text_are_both_vllm_kind():
    assert classify(json.dumps({"mean_ttft_ms": 1.0, "completed": 3}))[0] == "vllm"
    assert classify(TEXT)[0] == "vllm"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PLANNER_DB", str(tmp_path / "t.sqlite3"))
    with TestClient(app) as c:
        yield c


def plan_of(client):
    boot = client.get("/api/bootstrap").json()
    return {k: boot[k] for k in ("model", "workload", "assumptions", "hardware")}


def test_missing_context_asks_for_short_form_then_imports(client):
    plan = plan_of(client)
    first = client.post("/api/benchmarks/paste", json={"text": TEXT, "plan": plan}).json()
    assert first["needs_context"] and first["defaults"]["concurrency"] == 16
    assert client.get("/api/benchmarks").json() == []  # nothing stored until the form is answered
    form = {**first["defaults"], "hardware_id": plan["hardware"][0]["id"]}
    done = client.post("/api/benchmarks/paste", json={"text": TEXT, "plan": plan, "context": form})
    assert done.status_code == 200 and done.json()["imported"] == 1
    runs = client.get("/api/benchmarks").json()
    assert runs[0]["metrics"]["ttft_s"]["mean"] == pytest.approx(0.1205)
    assert runs[0]["metrics"]["errors"] == 2
    assert runs[0]["config"]["hardware_id"] == plan["hardware"][0]["id"]
    assert "import time" in runs[0]["source"]


def test_bad_form_values_are_rejected(client):
    plan = plan_of(client)
    form = {"model_id": "", "hardware_id": "nope", "tp": 1, "replicas": 1, "concurrency": 4}
    assert client.post("/api/benchmarks/paste", json={"text": TEXT, "plan": plan, "context": form}).status_code == 422


def test_atlas_format_json_needs_no_form(client):
    plan = plan_of(client)
    # Round-trip: export a record via the short form, then paste it back as Atlas-format JSON.
    form = {"model_id": plan["model"]["model_id"], "hardware_id": plan["hardware"][0]["id"], "tp": 1,
            "replicas": 1, "concurrency": 16, "engine_version": "0.10.2"}
    client.post("/api/benchmarks/paste", json={"text": TEXT, "plan": plan, "context": form})
    record = client.get("/api/benchmarks").json()[0] | {"run_id": "second-run"}
    again = client.post("/api/benchmarks/paste", json={"text": json.dumps(record), "plan": plan})
    assert again.json()["imported"] == 1 and again.json()["needs_context"] is False
