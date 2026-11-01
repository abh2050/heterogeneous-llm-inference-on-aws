import pytest
from fastapi.testclient import TestClient

from app import app
from planner.benchmark import BenchmarkAdapter, Capabilities, JobState


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PLANNER_DB", str(tmp_path / "test.sqlite3"))
    monkeypatch.delenv("BENCHMARK_ADAPTER", raising=False)
    with TestClient(app) as c:
        yield c


def get_plan(client):
    d = client.get("/api/bootstrap").json()
    return {key: d[key] for key in ("model", "workload", "assumptions", "hardware")}


def test_browser_shell_bootstrap_compare_and_export(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/vendor/plotly.min.js").status_code == 200
    p = get_plan(client)
    result = client.post("/api/compare", json=p)
    assert result.status_code == 200
    data = result.json()
    assert len(data["candidates"]) == 80
    assert data["candidates"][0]["feasible"]
    assert data["candidates"][0]["evidence"] == "Estimated"
    assert len(data["chart"]["data"]) > 0
    ex = client.post("/api/export", json={"plan": p, "candidate_id": data["candidates"][0]["id"]})
    assert ex.status_code == 200
    assert not ex.json()["serving"]["runnable"]
    assert client.get("/api/benchmarks").json() == []


def test_scenario_roundtrip(client):
    p = get_plan(client)
    r = client.post("/api/scenarios", json={"name": "test plan", "plan": p, "candidate_id": None})
    assert r.status_code == 200
    assert client.get("/api/scenarios").json()["test plan"]["plan"] == p | {
        "concurrencies": [1, 2, 4, 8, 16, 32, 64, 128]
    }


def test_api_validation_and_unconfigured_service(client):
    p = get_plan(client)
    p["workload"]["input_avg"] = 99999
    assert client.post("/api/compare", json=p).status_code == 422
    assert client.post("/api/benchmark/submit", json={}).status_code == 422
    assert not client.get("/api/benchmark/capabilities").json()["configured"]
    assert client.post("/api/metadata", json={"model_id": "../bad"}).status_code == 422
    assert (
        client.post("/api/scenarios", headers={"Origin": "https://foreign.example"}, json={}).status_code
        == 403
    )


def test_import_history_and_metrics_preservation(client, run_factory):
    record = run_factory().model_dump(mode="json")
    response = client.post("/api/benchmarks/import", json={"data": record})
    assert response.status_code == 200
    saved = client.get("/api/benchmarks").json()[0]
    assert saved["metrics"]["tpot_s"]["p95"] == 0.02
    assert saved["metrics"]["itl_s"]["p95"] == 0.035
    assert saved["config"]["revision"] == "a" * 40


def test_adapter_submission_poll_results_cancel(client, monkeypatch, run_factory):
    class ExistingService(BenchmarkAdapter):
        def capabilities(self):
            return Capabilities(
                configured=True,
                service_name="test contract",
                submit=True,
                poll=True,
                results=True,
                cancel=True,
            )

        def submit(self, payload):
            assert payload["config"]["launch_config"]["argv"][0] == "vllm"
            return JobState(job_id="test-job", state="queued")

        def poll(self, job_id):
            return JobState(job_id=job_id, state="succeeded")

        def results(self, job_id):
            return [run_factory()]

        def cancel(self, job_id):
            return JobState(job_id=job_id, state="cancelled")

    monkeypatch.setattr("app.load_adapter", lambda: ExistingService())
    p = get_plan(client)
    c = client.post("/api/compare", json=p).json()["candidates"][0]
    job = client.post("/api/benchmark/submit", json={"plan": p, "candidate_id": c["id"]})
    assert job.status_code == 200
    assert client.post("/api/benchmark/job/poll", json={"job_id": "test-job"}).json()["state"] == "succeeded"
    assert client.post("/api/benchmark/job/results", json={"job_id": "test-job"}).json()["imported"] == 1
    assert (
        client.post("/api/benchmark/job/cancel", json={"job_id": "test-job"}).json()["state"] == "cancelled"
    )
    assert "test-job" in client.get("/api/benchmark/jobs").json()


def test_service_error_does_not_expose_credentials(client, monkeypatch):
    class Broken(BenchmarkAdapter):
        def capabilities(self):
            return Capabilities(configured=True, health=True)

        def health(self):
            raise RuntimeError("synthetic-secret-value")

    monkeypatch.setattr("app.load_adapter", lambda: Broken())
    response = client.post("/api/benchmark/health", json={})
    assert response.status_code == 500
    assert "synthetic-secret" not in response.text


def test_presets_served_in_bootstrap_and_endpoint(client):
    boot = client.get("/api/bootstrap").json()
    assert [p["id"] for p in boot["presets"]] == ["chat", "rag", "coding", "agents"]
    assert client.get("/api/presets").json() == boot["presets"]
    # A preset applied to the existing workload still runs through the unchanged comparison.
    plan = get_plan(client)
    plan["workload"].update(boot["presets"][0]["workload"])
    assert client.post("/api/compare", json=plan).status_code == 200
