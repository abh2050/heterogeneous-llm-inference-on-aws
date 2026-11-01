"""Local browser app: uv run uvicorn app:app --host 127.0.0.1 --port 8000."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from planner.benchmark import build_run_context, import_normalized, import_vllm_result, load_adapter
from planner.charts import memory_chart, tpot_chart
from planner.demo import example_model
from planner.hardware_catalog import CatalogEntry, build_catalog, default_catalog
from planner.exports import DeploymentDetails, kubernetes_yaml, sizing_report, vllm_command
from planner.hf_metadata import MetadataError, apply_overrides, fetch_model, normalize, parse_model_id
from planner.models import Assumptions, Candidate, ModelSpec, Workload
from planner.presets import presets
from planner.quick import ADVANCED, DECODE_EFFICIENCY, SIMPLIFIED, quick_table
from planner.security import scrub
from planner.service import PlanningRequest, compare
from planner.sizing import DisaggregationInputs, compare_disaggregated
from planner.storage import Store

ROOT = Path(__file__).parent
app = FastAPI(title="Inference Atlas", version="0.1.0", docs_url="/api/docs", redoc_url=None)


def store():
    return Store()


def hf_token():
    """HF_TOKEN from the server environment (local shell or a Vercel environment variable)."""
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    file = os.getenv("PLANNER_SECRETS_FILE")
    if not token and file:
        with open(file, "rb") as f:
            token = tomllib.load(f).get("HF_TOKEN")
    return token.strip() if isinstance(token, str) and token.strip() else None


@app.middleware("http")
async def local_origin(request: Request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse(status_code=403, content={"detail": "Cross-origin mutations are disabled"})
    try:
        return await call_next(request)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Operation failed. Check local service configuration and connectivity; raw exceptions are suppressed."
            },
        )


@app.exception_handler(ValueError)
async def invalid_value(request, exc):
    if isinstance(exc, ValidationError):
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    else:
        details = str(exc)
    return JSONResponse(status_code=422, content={"detail": scrub(details)})


@app.exception_handler(KeyError)
async def missing_key(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Required request field is missing; check the API schema or import format."},
    )


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    # Never return a provider exception, auth headers, request content or credentials.
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Operation failed. Check local service configuration and connectivity; credentials and raw exceptions are suppressed."
        },
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/bootstrap")
def bootstrap():
    db = store()
    catalog = default_catalog()
    try:
        caps = load_adapter().capabilities().model_dump()
    except ValueError:
        caps = {
            "configured": False,
            "service_name": "Adapter configuration error",
            "notes": ["Check BENCHMARK_ADAPTER"],
        }
    return {
        "model": example_model(),
        "workload": Workload(),
        "assumptions": Assumptions(),
        "hardware": catalog["hardware"],
        "hardware_catalog": catalog["records"],
        "builtin_kept": catalog["kept_builtin"],
        "scenarios": list(db.scenarios()),
        "benchmark_count": len(db.benchmarks()),
        "capabilities": caps,
        "hf_token_configured": bool(hf_token()),
        "catalog_version": "2026-10-31.1",
        "presets": presets(),
    }


@app.post("/api/catalog/convert")
def convert_catalog(payload: dict):
    """Validate user-edited catalog records and convert them to planner hardware."""
    rows = payload.get("entries")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 60:
        raise ValueError("entries must be a list of 1 to 60 catalog records")
    entries = [CatalogEntry.model_validate({k: v for k, v in r.items() if k not in {"usable", "missing"}}) for r in rows]
    return build_catalog(entries, builtin=[])


@app.post("/api/quick-estimate")
def quick_estimate_route(payload: dict):
    request = PlanningRequest.model_validate(payload["plan"])
    w, a = request.workload, request.assumptions
    users = int(payload.get("users") or w.effective_peak_users or 1)
    context = int(payload.get("context") or w.input_max + w.output_max)
    rows = quick_table(
        request.model,
        request.hardware,
        bytes_per_parameter=request.model.effective_weight_bytes or a.weight_bytes,
        kv_bytes=a.kv_bytes,
        context=context,
        users=users,
        input_tokens=w.input_avg,
        decode_efficiency=float(payload.get("decode_efficiency", DECODE_EFFICIENCY)),
    )
    return {"rows": rows, "users": users, "context": context, "label": SIMPLIFIED, "advanced_label": ADVANCED}


@app.get("/api/presets")
def workload_presets():
    return presets()


@app.post("/api/metadata")
def metadata(payload: dict):
    if not isinstance(payload.get("model_id"), str):
        raise ValueError("Model ID is required")
    try:
        model = fetch_model(
            payload["model_id"],
            payload.get("revision", "main"),
            token=hf_token(),
            store=store(),
            refresh=bool(payload.get("refresh")),
        )
        return model
    except MetadataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/api/model/inspect")
def inspect_config(payload: dict):
    model = normalize(parse_model_id(payload["model_id"]), None, payload["config"])
    model.notes.insert(0, "User-imported config; not verified against a resolved repository commit")
    for field, prov in model.provenance.items():
        if prov.kind == "Repository":
            model.provenance[field] = prov.model_copy(
                update={"source": "Imported " + prov.source, "kind": "Override"}
            )
    return model


@app.post("/api/model/override")
def override(payload: dict):
    return apply_overrides(ModelSpec.model_validate(payload["model"]), payload["overrides"])


@app.post("/api/compare")
def comparison(request: PlanningRequest):
    runs = store().benchmarks()
    candidates = compare(request, runs)
    return {
        "candidates": candidates,
        "chart": json.loads(
            tpot_chart(
                candidates, runs, request.model, request.hardware, request.workload, request.assumptions
            ).to_json()
        ),
        "benchmark_count": len(runs),
    }


@app.post("/api/memory")
def memory(payload: dict):
    return json.loads(memory_chart(Candidate.model_validate(payload)).to_json())


@app.get("/api/benchmarks")
def history():
    return store().benchmarks()


@app.post("/api/benchmarks/import")
def import_results(payload: dict):
    if payload.get("format") == "vllm":
        records = [import_vllm_result(payload["data"], payload["context"])]
    else:
        records = import_normalized(payload["data"])
    db = store()
    db.add_benchmarks(records)
    return {"imported": len(records), "run_ids": [r.run_id for r in records]}


def candidate_from(payload: dict):
    request = PlanningRequest.model_validate(payload["plan"])
    candidates = compare(request, store().benchmarks())
    c = next((c for c in candidates if c.id == payload["candidate_id"]), None)
    if c is None:
        raise ValueError("Candidate is no longer in the current comparison")
    h = next(h for h in request.hardware if h.id == c.hardware_id)
    return request, c, h


@app.post("/api/benchmark/context")
def run_context(payload: dict):
    request, c, h = candidate_from(payload)
    if not c.feasible:
        raise ValueError("Select a feasible configuration before preparing a benchmark")
    return build_run_context(request.model, h, c, request.workload, request.assumptions)


@app.get("/api/benchmark/capabilities")
def capabilities():
    return load_adapter().capabilities()


@app.post("/api/benchmark/health")
def benchmark_health():
    adapter = load_adapter()
    if not adapter.capabilities().health:
        raise ValueError("Health endpoint is not configured by a service adapter")
    return scrub(adapter.health())


@app.post("/api/benchmark/submit")
def submit(payload: dict):
    adapter = load_adapter()
    if not adapter.capabilities().submit:
        raise ValueError(
            "Benchmark submission is unconfigured; use JSON import or provide a real service adapter"
        )
    context = run_context(payload)
    job = adapter.submit(context)
    store().save_job(job.job_id, {"state": job.model_dump(), "context": context})
    return job


@app.get("/api/benchmark/jobs")
def jobs():
    return store().jobs()


@app.post("/api/benchmark/job/{action}")
def job_action(action: str, payload: dict):
    adapter = load_adapter()
    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or job_id not in store().jobs():
        raise ValueError("Select a recorded service job")
    caps = adapter.capabilities()
    if action not in {"poll", "results", "cancel"} or not getattr(caps, action):
        raise ValueError("Requested job operation is not supported by this service")
    result = getattr(adapter, action)(job_id)
    if action == "results":
        records = [r if hasattr(r, "metrics") else import_normalized(r)[0] for r in result]
        store().add_benchmarks(records)
        return {"imported": len(records)}
    previous = store().jobs()[job_id]
    previous["state"] = result.model_dump()
    store().save_job(job_id, previous)
    return result


@app.get("/api/scenarios")
def scenarios():
    return store().scenarios()


@app.post("/api/scenarios")
def save_scenario(payload: dict):
    plan = PlanningRequest.model_validate(payload["plan"])
    store().save_scenario(
        payload["name"], {"plan": plan.model_dump(mode="json"), "candidate_id": payload.get("candidate_id")}
    )
    return {"saved": True}


@app.post("/api/export")
def export(payload: dict):
    request, c, h = candidate_from(payload)
    command = vllm_command(request.model, h, c, request.workload, request.assumptions)
    report = sizing_report(
        request.model, h, c, request.workload, request.assumptions, store().benchmarks(), command
    )
    return {"serving": command, "report": report}


@app.post("/api/export/kubernetes")
def export_kubernetes(payload: dict):
    request, c, h = candidate_from(payload)
    details = DeploymentDetails.model_validate(payload["details"])
    return {"yaml": kubernetes_yaml(request.model, h, c, request.workload, request.assumptions, details)}


@app.post("/api/disaggregation")
def disaggregation(payload: dict):
    request, c, h = candidate_from(payload)
    prefill = next((x for x in request.hardware if x.id == payload["prefill_hardware_id"]), None)
    if prefill is None or not c.feasible:
        raise ValueError("Select compatible prefill hardware and a feasible decode candidate")
    data = DisaggregationInputs(**payload["inputs"])
    return compare_disaggregated(request.model, prefill, h, request.workload, request.assumptions, c, data)


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
