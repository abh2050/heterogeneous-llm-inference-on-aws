"""Import hardware_catalog.json, validate it, and convert it to the planner's Hardware format.

Catalog values are per system. Multi-GPU systems are divided into per-device entries because the
planner sizes one tensor-parallel group from per-device memory, bandwidth and compute.
Nulls stay null ("not published"); an entry only becomes a planning candidate once every value the
roofline needs (memory, bandwidth, BF16 TFLOPS, and a price) is present or supplied by the user.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from planner.catalog import load_catalog
from planner.models import Hardware, Record

ROOT = Path(__file__).parent.parent
CATALOG_FILE = ROOT / "hardware_catalog.json"
SCHEMA_FILE = Path(__file__).parent / "data/hardware_catalog.schema.json"

AMORTIZATION_YEARS = 3
PREFILL_MFU = 0.4
NVLINK_CLASS_GBS = 300  # aggregate interconnect at or above this allows tensor parallelism across devices

# Built-in entries that the catalog file supersedes; built-ins the file lacks are kept.
SUPERSEDED = {"h100-sxm-80": "nvidia-h100-sxm", "l4-24": "nvidia-l4"}


class CatalogEntry(Record):
    id: str
    vendor: str
    name: str
    category: Literal[
        "datacenter_gpu", "workstation_gpu", "desktop_appliance", "multi_gpu_system", "cpu", "accelerator"
    ]
    memory_gb: float | None = Field(gt=0)
    memory_type: str
    memory_bandwidth_gb_s: float | None = Field(gt=0)
    bf16_tflops_dense: float | None = Field(gt=0)
    fp8_tflops_dense: float | None = Field(gt=0)
    fp4_tflops_dense: float | None = Field(gt=0)
    gpu_count: int = Field(ge=1)
    interconnect: str
    interconnect_gb_s: float | None = Field(gt=0)
    power_w: float = Field(gt=0)
    cooling: Literal["air", "liquid", "either"]
    form_factor: str
    list_price_usd: float | None = Field(ge=0)
    price_type: Literal["MSRP", "street", "estimate"]
    cloud_hourly_usd: float | None = Field(ge=0)
    compute_platform: Literal["CUDA", "ROCm", "oneAPI", "Metal"]
    serving_engines: list[str]
    status: Literal["shipping", "announced"]
    release_year: int = Field(ge=2000)
    sources: dict[str, str]
    confidence: Literal["high", "medium", "low"]
    notes: str


def load_entries(path: Path = CATALOG_FILE) -> list[CatalogEntry]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("hardware_catalog.json must be a JSON array of hardware entries")
    entries = [CatalogEntry.model_validate(row) for row in data]
    if len({e.id for e in entries}) != len(entries):
        raise ValueError("hardware_catalog.json contains duplicate IDs")
    return entries


def json_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "hardware_catalog.json",
        "type": "array",
        "items": CatalogEntry.model_json_schema(),
    }


def local_lab_entry() -> CatalogEntry:
    """Editable placeholder: specifications are unknown until the user supplies them."""
    return CatalogEntry(
        id="local-lab",
        vendor="Local",
        name="My local lab machine (edit me)",
        category="desktop_appliance",
        memory_gb=None,
        memory_type="unknown",
        memory_bandwidth_gb_s=None,
        bf16_tflops_dense=None,
        fp8_tflops_dense=None,
        fp4_tflops_dense=None,
        gpu_count=1,
        interconnect="none",
        interconnect_gb_s=None,
        power_w=1,
        cooling="either",
        form_factor="lab machine",
        list_price_usd=None,
        price_type="estimate",
        cloud_hourly_usd=None,
        compute_platform="CUDA",
        serving_engines=["vLLM", "llama.cpp", "Ollama"],
        status="shipping",
        release_year=2026,
        sources={},
        confidence="low",
        notes="Enter memory, bandwidth, BF16 TFLOPS and a price for this machine to include it.",
    )


def estimate_prefill_tps(bf16_tflops: float | None, active_parameters: int | float | None) -> float | None:
    """Estimated prefill tokens/s = bf16_tflops_dense x 1e12 x 0.4 / (2 x active parameters)."""
    if not bf16_tflops or not active_parameters:
        return None
    return bf16_tflops * 1e12 * PREFILL_MFU / (2 * active_parameters)


def to_hardware(entry: CatalogEntry) -> tuple[Hardware | None, list[str]]:
    """Convert to the planner format, or return None plus the names of values that are not published."""
    n = entry.gpu_count
    missing = [
        label
        for label, value in (
            ("memory_gb", entry.memory_gb),
            ("memory_bandwidth_gb_s", entry.memory_bandwidth_gb_s),
            ("bf16_tflops_dense", entry.bf16_tflops_dense),
        )
        if value is None
    ]
    if entry.cloud_hourly_usd is not None:
        hourly = entry.cloud_hourly_usd / n
        provider, scope = (
            "Catalog cloud price",
            "catalog cloud_hourly_usd, divided across the devices in the system",
        )
    elif entry.list_price_usd is not None:
        hourly = entry.list_price_usd / n / (AMORTIZATION_YEARS * 8760)
        provider = "Owned (list price amortized)"
        scope = (
            f"estimated: list price over {AMORTIZATION_YEARS} years of continuous use; excludes host, power "
            "and hosting"
        )
    else:
        hourly = None
        missing.append("price")
    if missing:
        return None, missing
    fabric = entry.interconnect_gb_s is not None and entry.interconnect_gb_s >= NVLINK_CLASS_GBS
    if n > 1:
        allowed_tp = [t for t in (1, 2, 4, 8) if t <= n]
    elif fabric and entry.category in {"datacenter_gpu", "accelerator"}:
        allowed_tp = [1, 2, 4, 8]  # scale-up fabric: assume a single node of up to 8 devices
    else:
        allowed_tp = [1]
    multi = len(allowed_tp) > 1
    urls = list(dict.fromkeys(entry.sources.values()))
    return (
        Hardware(
            id=entry.id,
            name=entry.name,
            kind="CPU" if entry.category == "cpu" else "GPU",
            memory_bytes=round(entry.memory_gb * 1e9 / n),
            bandwidth_gbs=entry.memory_bandwidth_gb_s / n,
            dense_bf16_tflops=entry.bf16_tflops_dense / n,
            allowed_tp=allowed_tp,
            # The catalog lists aggregate (bidirectional) link bandwidth; the planner uses one-way.
            interconnect_gbs=(entry.interconnect_gb_s or 0) / 2 if multi else 0,
            topology=f"{entry.interconnect}; {entry.form_factor}" + (f"; {n} devices per system" if n > 1 else ""),
            hourly_usd=hourly,
            provider=provider,
            region="unspecified",
            price_date="not published",
            pricing_kind="Illustrative",
            price_scope=scope,
            source_urls=urls,
            notes=[entry.notes, f"Catalog confidence: {entry.confidence}; status: {entry.status}"],
            fp8_kv=entry.compute_platform == "CUDA" and entry.fp8_tflops_dense is not None,
            vendor=entry.vendor,
            category=entry.category,
            compute_platform=entry.compute_platform,
            serving_engines=entry.serving_engines,
            confidence=entry.confidence,
            list_price_usd=entry.list_price_usd,
            field_sources=entry.sources,
        ),
        [],
    )


def build_catalog(entries: list[CatalogEntry], builtin: list[Hardware] | None = None) -> dict:
    """Records for display (nulls intact) plus the Hardware list the planner uses."""
    builtin = load_catalog() if builtin is None else builtin
    present = {e.id for e in entries}
    records, hardware = [], []
    for entry in entries:
        h, missing = to_hardware(entry)
        records.append({**entry.model_dump(), "usable": h is not None, "missing": missing})
        if h:
            hardware.append(h)
    kept = [h for h in builtin if SUPERSEDED.get(h.id) not in present]
    return {"records": records, "hardware": kept + hardware, "kept_builtin": [h.id for h in kept]}


def default_catalog() -> dict:
    return build_catalog([*load_entries(), local_lab_entry()])
