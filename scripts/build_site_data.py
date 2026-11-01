#!/usr/bin/env python3
"""Build docs/data/results.json from benchmark artifacts and RUNLOG.md.

The site deliberately consumes one generated file. Missing input files or fields are
fatal so presentation numbers cannot silently drift away from the experiment.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "results.json"


def load(path: str):
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"required benchmark is missing: {p}")
    return json.loads(p.read_text())


def require(row: dict, *fields: str) -> dict:
    missing = [f for f in fields if f not in row or row[f] is None]
    if missing:
        raise ValueError(f"missing {missing} in benchmark row: {row.get('date', row)}")
    return {f: row[f] for f in fields}


def bench(path: str) -> dict:
    row = load(path)
    return require(
        row,
        "request_rate",
        "num_prompts",
        "duration",
        "output_throughput",
        "p50_ttft_ms",
        "p99_ttft_ms",
        "p50_itl_ms",
        "p99_itl_ms",
    )


def series(paths: list[str]) -> list[dict]:
    return [bench(path) for path in paths]


runlog = (ROOT / "RUNLOG.md").read_text()


def runlog_number(pattern: str, label: str) -> float:
    match = re.search(pattern, runlog, re.I | re.M)
    if not match:
        raise ValueError(f"could not extract {label} from RUNLOG.md")
    return float(match.group(1).replace(",", ""))


hf = load("results/gpu/engine_hf.json")
vllm = load("results/gpu/engine_vllm.json")
if [x.get("batch") for x in hf] != [1, 8, 32] or [x.get("batch") for x in vllm] != [1, 8, 32]:
    raise ValueError("engine benchmark must contain batch sizes 1, 8, and 32")
for row in hf + vllm:
    require(row, "engine", "batch", "tok_per_s", "seconds")

sustained = series([f"results/gpu/sustained/gpu_chat_r{rate}.json" for rate in (16, 20, 24, 28, 32, 34, 36, 38)])
rag = series([f"results/gpu/gpu_rag_r{rate}.json" for rate in (1, 2, 4, 8, 16, "inf")])
cpu_chat = series([f"results/cpu/cpu_chat_r{rate}.json" for rate in ("0.25", "0.5", 1, 2, 4, 8, 16, 32)])

tuning_paths = {
    "Baseline": "results/gpu/gpu_chat_rinf.json",
    "50% memory": "results/gpu/tuning/memA/gpu_chat_rinf.json",
    "16 sequences": "results/gpu/tuning/seqB/gpu_chat_rinf.json",
    "256 sequences": "results/gpu/tuning/seqC/gpu_chat_rinf.json",
}
tuning = [{"name": name, **bench(path)} for name, path in tuning_paths.items()]

router = {
    "direct": bench("results/gpu/router_direct.json"),
    "litellm": bench("results/gpu/router_overhead.json"),
}
mixed = {
    "alone": bench("results/gpu/mix_0_alone.json"),
    "shared": bench("results/gpu/mix_A_allgpu.json"),
    "offloaded": bench("results/gpu/mix_B_offload.json"),
    "background_gpu": bench("results/gpu/bg_A_gpu.json"),
    "background_cpu": bench("results/gpu/bg_B_cpu.json"),
}

speedups = [v["tok_per_s"] / h["tok_per_s"] for h, v in zip(hf, vllm)]
router_delta = router["litellm"]["p50_ttft_ms"] - router["direct"]["p50_ttft_ms"]
improvement = mixed["shared"]["p99_ttft_ms"] / mixed["offloaded"]["p99_ttft_ms"]

kv_tokens = runlog_number(r"KV cache \(Step 2\.6\):\*\* ([\d,]+) tokens", "GPU KV token pool")
max_concurrency = runlog_number(r"max concurrency ([\d.]+)x at 4,096", "maximum concurrency")
single_cpu = runlog_number(r"about ([\d.]+) tok/s single-stream", "CPU single-stream throughput")

data = {
    "meta": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "runtime": "vLLM 0.29.0",
        "region": "us-east-2",
        "slo": {"p99_ttft_ms": 1000, "p99_itl_ms": 100},
        "source": "Generated from results/ JSON and RUNLOG.md by scripts/build_site_data.py",
    },
    "scoreboard": {
        "shared_p99_ttft_ms": mixed["shared"]["p99_ttft_ms"],
        "offloaded_p99_ttft_ms": mixed["offloaded"]["p99_ttft_ms"],
        "improvement_x": improvement,
        "gpu_capacity_req_s": 34,
        "vllm_speedup_min": min(speedups),
        "vllm_speedup_max": max(speedups),
        "router_p50_added_ms": router_delta,
        "gpu_saturated_tok_s": max(x["output_throughput"] for x in sustained),
        "cpu_saturated_tok_s": max(x["output_throughput"] for x in cpu_chat),
        "gpu_cost_per_million": 0.0546,
        "cpu_cost_per_million": 0.781,
    },
    "engine": {"hf": hf, "vllm": vllm},
    "gpu_sustained": sustained,
    "gpu_rag": rag,
    "tuning": tuning,
    "cpu_chat": cpu_chat,
    "router": router,
    "mixed": mixed,
    "kv": {
        "tokens": kv_tokens,
        "max_concurrency_4096": max_concurrency,
        "bytes_per_token": 28672,
    },
    "cpu_single_stream_tok_s": single_cpu,
}

for value in (improvement, router_delta, *speedups):
    if not math.isfinite(value):
        raise ValueError("derived result is not finite")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=2) + "\n")
print(f"wrote {OUT.relative_to(ROOT)} from required benchmark inputs")
