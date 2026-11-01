"""Vercel mode: stateless server; browser owns scenarios, benchmarks and hardware edits."""

import json

import pytest
from fastapi.testclient import TestClient

from app import app, store
from planner.mode import VERCEL_CACHE, storage_mode
from tests.test_paste import TEXT


@pytest.fixture
def vercel(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_MODE", "vercel")
    monkeypatch.setenv("PLANNER_DB", str(tmp_path / "must-not-be-used.sqlite3"))
    monkeypatch.setenv("BENCHMARK_ADAPTER", "does.not:exist")  # would raise if the adapter were loaded
    with TestClient(app) as c:
        yield c
    assert not (tmp_path / "must-not-be-used.sqlite3").exists()


def plan_of(client):
    boot = client.get("/api/bootstrap").json()
    return {k: boot[k] for k in ("model", "workload", "assumptions", "hardware")}


def test_mode_selection(monkeypatch):
    monkeypatch.delenv("STORAGE_MODE", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    assert storage_mode() == "local"
    monkeypatch.setenv("VERCEL", "1")
    assert storage_mode() == "vercel"
    monkeypatch.setenv("STORAGE_MODE", "local")
    assert storage_mode() == "local"
    monkeypatch.setenv("STORAGE_MODE", "nonsense")
    with pytest.raises(ValueError):
        storage_mode()


def test_local_mode_keeps_sqlite(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("PLANNER_DB", str(tmp_path / "local.sqlite3"))
    assert store().path == tmp_path / "local.sqlite3"
    with TestClient(app) as c:
        assert c.get("/api/bootstrap").json()["storage_mode"] == "local"


def test_vercel_cache_is_tmp_only(vercel):
    assert store().path == VERCEL_CACHE and str(VERCEL_CACHE).startswith("/tmp/")
    boot = vercel.get("/api/bootstrap").json()
    assert boot["storage_mode"] == "vercel" and boot["scenarios"] == [] and boot["benchmark_count"] == 0
    assert boot["capabilities"]["configured"] is False and "Vercel" in boot["capabilities"]["service_name"]


def test_external_benchmark_adapter_is_disabled(vercel):
    plan = plan_of(vercel)
    assert vercel.get("/api/benchmark/capabilities").json()["configured"] is False
    for path, body in (
        ("/api/benchmark/health", {}),
        ("/api/benchmark/submit", {"plan": plan, "candidate_id": "x"}),
        ("/api/benchmark/job/poll", {"job_id": "j"}),
    ):
        response = vercel.post(path, json=body)
        assert response.status_code == 422 and "Vercel" in response.json()["detail"]
    assert vercel.get("/api/benchmark/jobs").json() == {}


def test_imports_and_scenarios_return_data_for_the_browser_and_store_nothing(vercel):
    plan = plan_of(vercel)
    form = {"model_id": plan["model"]["model_id"], "hardware_id": plan["hardware"][0]["id"], "tp": 1,
            "replicas": 1, "concurrency": 16, "engine_version": "0.10.2"}
    pasted = vercel.post("/api/benchmarks/paste", json={"text": TEXT, "plan": plan, "context": form}).json()
    assert pasted["imported"] == 1 and len(pasted["records"]) == 1
    assert vercel.get("/api/benchmarks").json() == []
    saved = vercel.post("/api/scenarios", json={"plan": plan, "name": "mine", "candidate_id": None}).json()
    assert saved["scenario"]["plan"]["model"]["model_id"] == plan["model"]["model_id"]
    assert "benchmarks" not in saved["scenario"]["plan"]
    assert vercel.get("/api/scenarios").json() == {}
    imported = vercel.post("/api/benchmarks/import", json={"data": pasted["records"][0]}).json()
    assert imported["records"][0]["run_id"] == pasted["records"][0]["run_id"]


def test_client_held_benchmarks_drive_calibration_and_export(vercel):
    plan = plan_of(vercel)
    plain = vercel.post("/api/compare", json=plan).json()
    assert plain["benchmark_count"] == 0
    first = plain["candidates"][0]
    form = {"model_id": plan["model"]["model_id"], "hardware_id": first["hardware_id"], "tp": first["tp"],
            "replicas": 1, "concurrency": first["concurrency"], "engine_version": "0.10.2"}
    record = vercel.post("/api/benchmarks/paste", json={"text": TEXT, "plan": plan, "context": form}).json()["records"][0]
    with_runs = vercel.post("/api/compare", json={**plan, "benchmarks": [record]}).json()
    assert with_runs["benchmark_count"] == 1
    export = vercel.post(
        "/api/export", json={"plan": {**plan, "benchmarks": [record]}, "candidate_id": first["id"]}
    )
    assert export.status_code == 200
    assert json.loads(export.json()["report"])  # report builds from client-supplied runs


def test_invalid_client_benchmark_is_rejected(vercel):
    plan = plan_of(vercel)
    assert vercel.post("/api/compare", json={**plan, "benchmarks": [{"nope": 1}]}).status_code == 422


def test_vercel_config_serves_web_statically_and_app_py_is_the_function():
    from pathlib import Path

    root = Path(__file__).parent.parent
    config = json.loads((root / "vercel.json").read_text())
    assert config["outputDirectory"] == "web" and (root / "web/index.html").exists()
    assert {"src": "/static/(.*)", "dest": "/$1"} in config["routes"]
    assert not (root / "api").exists()  # root app.py is Vercel's FastAPI entrypoint; an api/ file would shadow it
    assert (root / "app.py").exists() and "fastapi" in (root / "requirements.txt").read_text().lower()
