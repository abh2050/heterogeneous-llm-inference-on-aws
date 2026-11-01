# Repository audit

Audit date: 2026-10-31

This document records the repository as found before implementing the requested target features. No application code was changed as part of this audit.

## 1. Framework, structure, and local operation

### Stack

- **Backend:** FastAPI served by Uvicorn (`app.py`).
- **Frontend:** framework-free HTML, CSS, and browser JavaScript (`web/`). Plotly is vendored locally for charts.
- **Domain and calculation layer:** Python modules under `planner/`, with Pydantic request/response models.
- **Persistence:** local SQLite database at `.local/planner.sqlite3`, managed by `planner/storage.py`.
- **Model metadata:** Hugging Face Hub Python client, executed by the FastAPI server (`planner/hf_metadata.py`).
- **Testing and tooling:** pytest and Ruff, configured in `pyproject.toml`; dependencies are locked in `uv.lock`.

### Repository structure

| Path | Purpose |
|---|---|
| `app.py` | FastAPI routes, error handling, security headers, and static web serving |
| `planner/models.py` | Pydantic domain, API, candidate, benchmark, and export models |
| `planner/sizing.py` | Memory, roofline performance, latency, replica, and cost calculations |
| `planner/service.py` | Candidate enumeration, comparison, ranking, and disaggregation orchestration |
| `planner/hf_metadata.py` | Hugging Face model-ID parsing, metadata retrieval, config parsing, provenance, and overrides |
| `planner/catalog.py` | Hardware JSON loading and validation |
| `planner/data/hardware.v1.json` | Built-in hardware catalog |
| `planner/benchmark.py` | Normalized and vLLM benchmark parsing plus external-service adapter |
| `planner/calibration.py` | Benchmark matching, measured-value precedence, and calibrated estimates |
| `planner/charts.py` | Plotly-compatible memory and TPOT chart specifications |
| `planner/exports.py` | vLLM command, Kubernetes manifest, and Markdown report generation |
| `planner/storage.py` | SQLite schema and persistence for metadata, benchmarks, scenarios, and jobs |
| `planner/security.py` | SSRF/path/token protections and redaction helpers |
| `planner/demo.py` | Default model, workload, and assumptions |
| `web/index.html` | Single-page application shell |
| `web/app.js` | UI state, forms, API calls, tables, charts, scenario flow, benchmarks, and exports |
| `web/styles.css` | Existing visual design and responsive styling |
| `web/benchmark.schema.json` | Benchmark import schema |
| `web/vendor/plotly.min.js` | Vendored third-party chart library |
| `tests/` | API, sizing, metadata, evidence, and export tests |
| `docs/` | Methodology, validation, benchmark integration, study guide, and preview |
| `examples/` | Example model config and benchmark payload/context files |
| `README.md` | Setup, workflow, architecture, security, and limitations |
| `run.sh` | Local launcher |

Generated/minified artifacts (`uv.lock`, `web/vendor/plotly.min.js`, the generated benchmark schema, bytecode caches, and the preview image) were inspected for role and provenance; application source, tests, data, examples, and documentation were reviewed directly.

### Running locally

The documented path is:

```bash
uv sync --dev
./run.sh
```

`run.sh` starts `uvicorn app:app` on `127.0.0.1:8765` by default. `PLANNER_PORT` can override the port. The browser UI is served by the same FastAPI process. Direct development startup is also possible with `uv run uvicorn app:app --reload`.

### Baseline verification

- `uv run pytest -q`: **66 passed**, with one dependency deprecation warning from FastAPI/Starlette's test client.
- `uv run ruff check .`: **passed**.
- Browser JavaScript syntax check: **passed** when read from stdin. Node cannot address the file normally because the repository directory itself contains a literal trailing backslash; this is an environment/path quirk.
- This directory is **not a Git repository**. There is no `.git` directory or commit history, so diff summaries against Git and commits cannot be performed until Git is initialized here or the actual repository root is supplied.

## 2. Features that currently work

| Working feature | Implementation |
|---|---|
| Five-step browser workflow for model, workload, comparison, benchmarks, and export | `web/index.html`, `web/app.js`, `web/styles.css` |
| Direct Hugging Face model ID/URL entry with optional revision | `web/app.js`, `planner/hf_metadata.py`, `app.py` |
| Safe `config.json` retrieval without remote-code execution, pinned commit resolution, cache, provenance, and override support | `planner/hf_metadata.py`, `planner/storage.py`, `planner/security.py` |
| Dense model parsing for parameters, layers, attention/KV heads, head dimension, context, dtype, and quantization metadata | `planner/hf_metadata.py`, `planner/models.py` |
| Manual/local config inspection and explicit field overrides | `planner/hf_metadata.py`, `app.py`, `web/app.js` |
| Editable workload token lengths, distributions, peak request rate, burst/headroom, prefix reuse, latency targets, and operating hours | `planner/models.py`, `planner/demo.py`, `web/app.js` |
| Advanced editable runtime assumptions for element sizes, utilization, reserves, safety, tensor parallelism, and scheduling | `planner/models.py`, `web/app.js` |
| Hardware catalog with H100, A100, L4, and CPU entries | `planner/data/hardware.v1.json`, `planner/catalog.py` |
| Enumeration of hardware, tensor-parallel, and concurrency candidates | `planner/service.py`, `planner/models.py` |
| Weight, KV-cache, activation, runtime, communication, and safety memory estimates | `planner/sizing.py` |
| Compute/bandwidth roofline service estimates, TPOT/TTFT estimates, replicas, throughput, and deployment cost | `planner/sizing.py` |
| Feasibility reasons and lowest-cost ranking under the current memory and user-entered SLO model | `planner/service.py`, `planner/sizing.py`, `web/app.js` |
| Memory-breakdown and TPOT/concurrency charts | `planner/charts.py`, `web/app.js`, `web/vendor/plotly.min.js` |
| Upload of normalized benchmark JSON and vLLM benchmark JSON with a context sidecar | `planner/benchmark.py`, `app.py`, `web/app.js`, `web/benchmark.schema.json` |
| Benchmark validation, deduplication, strict model/hardware/runtime matching, and evidence provenance | `planner/benchmark.py`, `planner/calibration.py`, `planner/storage.py` |
| Measured latency/throughput precedence and calibrated estimates when sufficient matching samples exist | `planner/calibration.py`, `planner/service.py`, `planner/charts.py`, `web/app.js` |
| External benchmark-service adapter lifecycle (health, submit, poll, results, cancel), currently unconfigured | `planner/benchmark.py`, `app.py`, `web/app.js` |
| Scenario save/load and persisted benchmark/job history | `planner/storage.py`, `app.py`, `web/app.js` |
| vLLM command, Kubernetes manifest, and Markdown report export | `planner/exports.py`, `app.py`, `web/app.js` |
| Prefill/decode disaggregation comparison | `planner/service.py`, `planner/sizing.py`, `app.py`, `web/app.js` |
| Security headers, request limits, token redaction, safe model identifiers, and controlled outbound Hugging Face access | `app.py`, `planner/security.py`, `planner/hf_metadata.py` |
| API, formula behavior, evidence, metadata, and export regression tests | `tests/` |

## 3. Target-feature gap analysis

Legend: **Exists** means the requested behavior is present; **Partial** means related behavior exists but does not meet the requested contract; **Missing** means it is absent.

| # | Target | Status | Audit finding |
|---:|---|---|---|
| 1 | Searchable Hugging Face dropdown; fetch `config.json`; parse architecture and total/active parameters | **Partial** | Direct model-ID/URL fetch, config parsing, total parameters, layers, KV heads, and head-dimension derivation exist in `planner/hf_metadata.py`. There is no search/dropdown. MoE is detected but marked unsupported, and active parameters per token are not calculated or read. Fetching is server-side rather than browser/Vercel-function oriented. |
| 2 | Chat, RAG, coding assistant, and agents workload presets with editable values | **Missing** | Workload values are editable, but there is only one demo default and no named preset system. |
| 3 | Total users and peak concurrent users | **Missing** | The model uses peak requests/second and candidate concurrency. It has no total-user or peak-concurrent-user inputs. |
| 4 | Separate BF16/FP8/INT4 selectors for weights and KV cache | **Partial** | KV dtype/bytes and activation/weight bytes are editable. Weight precision is not a dedicated BF16/FP8/INT4 selector, FP16/FP32 are offered instead, and INT4 is not an end-to-end supported option. |
| 5 | Hardware JSON with name, GB, bandwidth, price, power, and editable local lab machine | **Partial** | JSON catalog and the first four fields conceptually exist, although memory is stored as bytes. Power and a local-lab entry are absent. The UI edits limited pricing metadata, not complete hardware entries. |
| 6a | `weights GB = parameters in billions × bytes` | **Partial** | The primitive is present, but current results add a weight-overhead factor, divide per TP device, and report binary GiB. This differs from the requested decimal-GB formula. |
| 6b | Exact KV bytes/token and KV GB formulas | **Partial** | The bytes/token primitive is correct. Resident KV instead uses max input plus output, block-size rounding, prefix-retention behavior, concurrency, and TP sharding, reported in GiB. There is no direct context-length × peak-users result in decimal GB. |
| 6c | Fit with exactly 10% overhead | **Partial** | A 10% safety default exists, but fit also includes 3% weight overhead, runtime/activation/communication reserves, and a 90% memory-utilization budget. It does not implement the requested single fit equation. |
| 6d | Decode ceiling from bandwidth / active weights+KV, with editable 0.7 efficiency | **Partial** | A more elaborate compute-and-memory roofline exists. It uses separate MFU/MBU values (default GPU bandwidth utilization 0.65), TP efficiency, communication, full weights, and contextual KV traffic—not the requested explicit ceiling or a single default 0.7 factor. Active MoE weights are unavailable. |
| 6e | TPOT = 1000 / tokens per second per user | **Partial** | TPOT is estimated from modeled batch-step service time. The requested transparent per-user equation is not exposed as such. |
| 6f | TTFT = input tokens / editable per-hardware prefill throughput | **Missing** | TTFT is currently FLOP/roofline based. Hardware entries have no editable prefill-throughput field. |
| 6g | Cost per million tokens formula | **Partial** | The same algebra exists, but the denominator is workload-requested output throughput after replica sizing, rather than an explicitly displayed total-token/s value from the requested model. |
| 7 | Paste or upload vLLM output; side-by-side measured/calculated labels | **Partial** | Upload works, but vLLM imports require contextual sidecar data and there is no paste box. Strictly matching measurements can appear beside estimates with evidence badges, but the requested consistent `measured`/`calculated` labeling for every displayed value is not implemented. |
| 8 | Cheapest option meeting ≥20 tok/s/user and TTFT <2s; matching vLLM flags | **Partial** | Lowest-cost feasible ranking and command export exist, but use user-entered current SLOs rather than the fixed criteria. The command includes model length, concurrency, KV dtype, and prefix caching, but omits `--quantization`; quantized candidates are currently rejected by export compatibility checks. |
| 9 | Unit tests for every requested formula and Llama 3.1 70B check | **Missing** | Existing calculations are tested, but not every requested formula and not the specified 70B/20-user/32K case. Using the requested formula with 80 layers, 8 KV heads, head dim 128, and BF16 gives 140 GB weights and 214.75 GB KV, which is reasonably “about 200 GB.” |
| 10 | Vercel deployment with no server secrets | **Missing** | No Vercel configuration exists. The app assumes a long-running FastAPI process, writable local SQLite, and optionally a server-side HF token/secret file. The frontend depends on same-origin `/api` routes. |
| 11 | Every assumption shown in the UI with a short computation note | **Partial** | Many assumptions and methodology notes are visible, but several values remain catalog-only or hardcoded: compute throughput, bandwidth, allowed TP set, concurrency sweep, block size default, version profile, and some ranking/derivation details. |

## 4. Broken behavior, formula mismatches, and hardcoded values

### Defects or operational blockers

1. **The requested commit workflow is unavailable.** This directory is not a Git repository. Before implementation commits, Git must be initialized here or the correct repository root must be provided.
2. **Vercel is not currently deployable as requested.** Local SQLite persistence and the optional server-side Hugging Face credential model are not suitable for stateless Vercel deployment with no server secrets. There is also no `vercel.json` or equivalent serverless entry configuration.
3. **The repository directory name ends in a literal backslash.** Node's file-based module resolver rejects paths containing that character. Browser JavaScript itself passes syntax checking through stdin, but some tooling and deployment scripts may fail until the directory is renamed or handled specially.
4. **The `app.py` module docstring says port 8000 while `run.sh` and the README use 8765.** This is stale documentation, not a runtime failure.
5. **Quantized vLLM export is incomplete.** `planner/exports.py` does not emit `--quantization` and rejects quantized candidates under its compatibility policy, conflicting with the requested weight-precision recommendation flow.

### Formula differences that must not be silently reused for the target

The current formulas are internally documented and tested, but they are not the exact requested formulas:

- Memory is generally expressed in binary GiB, while the target equations divide by `1e9` and therefore specify decimal GB.
- Current per-device weight memory includes configurable overhead and tensor-parallel sharding.
- Current KV residency uses `input_tokens_max + output_tokens_max`, block rounding, prefix-retention behavior, and sharding rather than a single editable context length multiplied by peak concurrent users.
- Current fit includes runtime, activation, communication, weight overhead, safety reserve, and `gpu_memory_utilization`; it is not the requested `(weights + KV) × 1.10 <= memory` test.
- Decode performance uses a compute/bandwidth roofline and communication model, not only bandwidth divided by active bytes with a 0.7 efficiency factor.
- TTFT is compute-derived rather than based on a hardware prefill-throughput field.
- Cost uses requested workload output throughput and replica count rather than a plainly surfaced total-token throughput input to the requested formula.

These are not necessarily invalid engineering models, but they would produce different answers. Implementation should preserve useful advanced estimates while adding clearly named target calculations instead of relabeling the existing outputs.

### Hardcoded or fixed assumptions

- Candidate concurrency sweep: `1, 2, 4, 8, 16, 32, 64, 128` in `planner/models.py`/`planner/service.py`.
- Tensor-parallel values are catalog-fixed and service validation assumes `1, 2, 4, 8`.
- vLLM export compatibility profile is fixed to version `0.10.2` in `planner/models.py` and `planner/exports.py`.
- Default KV block size is 16 tokens.
- Default GPU bandwidth utilization is 0.65, not the target 0.7.
- Hardware prices include explicitly illustrative values; they are not a live price feed.
- Hardware catalog version is duplicated as a literal in `app.py` rather than surfaced from the JSON catalog metadata.
- The hardware catalog has no power values, no editable prefill-throughput values, and no local-lab machine.
- Supported architecture families are explicitly whitelisted; detected MoE architectures are deliberately rejected.

### Test warning

The suite emits a Starlette deprecation warning because `fastapi.testclient` currently uses the deprecated `httpx` integration path. It does not fail tests today, but dependency upgrades may require changing the test-client dependency/import path.

## Implementation boundary after approval

Only items marked **Partial** or **Missing** above should be changed. Existing advanced calculations, evidence handling, exports, styling, and storage behavior should be retained where they do not conflict with the target. Before any commit, a diff summary must be shown; because Git is absent, that commit requirement needs resolution before the first commit.
