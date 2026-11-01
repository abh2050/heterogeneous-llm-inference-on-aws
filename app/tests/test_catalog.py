import json

import pytest
from pydantic import ValidationError

from planner.hardware_catalog import (
    CATALOG_FILE,
    SCHEMA_FILE,
    CatalogEntry,
    build_catalog,
    default_catalog,
    estimate_prefill_tps,
    json_schema,
    load_entries,
    local_lab_entry,
    to_hardware,
)
from planner.sizing import estimate


def test_catalog_file_validates_and_schema_file_is_current():
    entries = load_entries()
    assert len(entries) == 24
    assert json.loads(SCHEMA_FILE.read_text()) == json.loads(json.dumps(json_schema(), sort_keys=True))
    bad = json.loads(CATALOG_FILE.read_text())[0] | {"surprise": 1}
    with pytest.raises(ValidationError):
        CatalogEntry.model_validate(bad)


def test_conversion_keeps_nulls_and_marks_missing_values():
    catalog = default_catalog()
    by_id = {r["id"]: r for r in catalog["records"]}
    spark = by_id["nvidia-dgx-spark"]
    assert not spark["usable"] and spark["bf16_tflops_dense"] is None  # null stays null
    assert "bf16_tflops_dense" in spark["missing"] and "price" in spark["missing"]
    ids = {h.id for h in catalog["hardware"]}
    assert "nvidia-dgx-spark" not in ids and "intel-arc-pro-b60" in ids


def test_builtin_entries_kept_only_when_catalog_lacks_them():
    catalog = default_catalog()
    ids = [h.id for h in catalog["hardware"]]
    assert "a100-sxm-80" in ids and "c7i-4xlarge" in ids  # not in the new file
    assert "h100-sxm-80" not in ids and "l4-24" not in ids  # superseded by nvidia-h100-sxm / nvidia-l4
    assert catalog["kept_builtin"] == ["a100-sxm-80", "c7i-4xlarge"]


def test_multi_gpu_system_is_split_into_devices_and_sources_survive():
    entries = {e.id: e for e in load_entries()}
    h, _ = to_hardware(entries["intel-arc-pro-b60-battlematrix-8gpu"])
    assert h.memory_bytes == 24_000_000_000 and h.bandwidth_gbs == 456 and h.dense_bf16_tflops == 49.25
    assert h.allowed_tp == [1, 2, 4, 8]
    h100, _ = to_hardware(entries["nvidia-h100-sxm"])
    assert h100.interconnect_gbs == 450 and h100.field_sources["memory_gb"].startswith("https://")
    b60, _ = to_hardware(entries["intel-arc-pro-b60"])
    assert "amortized" in b60.provider and "estimated" in b60.price_scope


def test_prefill_estimate_formula_and_missing_inputs():
    assert estimate_prefill_tps(989.5, 8e9) == pytest.approx(989.5e12 * 0.4 / 16e9)
    assert estimate_prefill_tps(None, 8e9) is None


def test_local_lab_entry_is_editable_via_convert_endpoint():
    from fastapi.testclient import TestClient

    from app import app

    lab = local_lab_entry().model_dump()
    assert not build_catalog([local_lab_entry()], builtin=[])["hardware"]
    lab.update(memory_gb=64, memory_bandwidth_gb_s=273, bf16_tflops_dense=60, cloud_hourly_usd=0.5)
    with TestClient(app) as client:
        result = client.post("/api/catalog/convert", json={"entries": [lab]}).json()
    assert [h["id"] for h in result["hardware"]] == ["local-lab"]


def test_converted_hardware_runs_through_unchanged_estimator(model, workload, assumptions):
    h = next(h for h in default_catalog()["hardware"] if h.id == "nvidia-h100-sxm")
    assert estimate(model, h, workload, assumptions, 1, 16).feasible
