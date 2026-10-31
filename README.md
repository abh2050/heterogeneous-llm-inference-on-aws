# Inference Atlas

A working browser application for inference capacity planning: a responsive HTML/CSS/JavaScript frontend, a Python/FastAPI backend, Plotly charts, Pydantic validation, and SQLite history. There is no Streamlit dependency and no model inference or replacement benchmark engine.

The original project contained only a heterogeneous-inference study guide. It is preserved at [docs/existing-study-guide.md](docs/existing-study-guide.md). No existing benchmark-service implementation, API contract, or sample response was present.

## Run

Requires Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/). Python 3.12 was used for validation.

```sh
uv sync --frozen
uv run uvicorn app:app --host 127.0.0.1 --port 8765 --no-access-log
```

Open **http://127.0.0.1:8765**. Alternatively run `./run.sh` from any directory. API documentation is at `/api/docs`. Port 8765 avoids the common inference-server port 8000; change the startup argument if necessary.

This is a local, single-user workspace. Keep it bound to loopback. It does not implement a multi-user authentication/authorization boundary. The frontend and vendored Plotly assets are served by the same Python process; Node and a frontend build are unnecessary.

## Five-step workflow

1. **Model:** inspect a Hugging Face ID or URL and optional revision. Successful lookups pin a commit SHA, retrieve `config.json` plus available safetensors parameter metadata, and cache the result. Review normalized fields, per-field sources, compatibility notes, and explicit overrides. Local `config.json` inspection is available offline.
2. **Workload and targets:** set mean/max lengths, optional length percentiles and distribution description, peak RPS, burst multiplier, additional headroom, prefix fraction/length/hit assumptions, p95 TTFT, an explicit TPOT percentile, operating hours, and pricing. Expand serving assumptions to edit efficiencies and reserves.
3. **Compare configurations:** examine candidates ranked by full deployment hourly cost, select a configuration, inspect per-device memory, and compare TPOT with concurrency. Analytical lines are distinct from measured percentile points. An exploratory disaggregation calculator includes phase pools, KV transfer, a shared network budget, routing latency, and router cost.
4. **Validate with benchmarks:** import normalized JSON or a vLLM result plus sidecar. Measurements retain launch/workload context and separate TTFT/TPOT/ITL/E2E metrics. An installed service adapter can advertise health, submission, polling, results, and cancellation. Active jobs poll every five seconds while the benchmark view is open; results are fetched explicitly.
5. **Export configuration:** download a version-scoped vLLM command and JSON sizing report. Unresolved commits or unsupported configurations produce clearly labeled templates. Kubernetes export requires sufficient real cluster details; the app does not deploy anything.

The model, applied workload, and selected configuration remain visible while navigating. Save and reopen scenarios in the sidebar. Inputs apply on form submission.

The initial Llama 3.1 8B example is **illustrative configuration data**, not a live lookup or a benchmark. It has no resolved SHA. There are **no preloaded measurements**. Inspect a live repository before exporting a complete command.

## Credentials

Use a process environment variable named `HF_TOKEN`, or a local secret file selected by `PLANNER_SECRETS_FILE`. Enter credentials locally through your shell/secret manager; do not paste them into chat or source files.

For example, in a local zsh session, `read -s 'HF_TOKEN?HF read token: '` prompts without echoing the value; then `export HF_TOKEN` before starting the server. Restart the server after changing its environment.

Alternatively, create an ignored local `.local/secrets.toml` containing an `HF_TOKEN` setting and set `PLANNER_SECRETS_FILE` to that file's absolute path. Restrict its permissions to the current user. An environment `HF_TOKEN` takes precedence. The official Hub client can also use its normal locally saved login.

The server never sends a credential to the frontend or embeds it in exports. Imported nested authorization/credential fields and recognizable token strings are redacted before storage/export. Raw provider exceptions are suppressed. Benchmarks use their adapter's environment-based authentication, independent of HF credentials.

A token does not grant access to a gated model: request approval on the repository page and use a token with approved read access. Missing access, invalid IDs/revisions, absent configuration, rate limits, and network failures produce actionable messages. Successful lookups are cached by model ID and requested revision as well as SHA; use **Refresh cached metadata** to re-resolve a moving branch.

## Benchmark integration

**Current status: unconfigured.** Direct service integration requires its base URL, authentication method, and actual API specification or sample result. The app does not invent endpoints or infer an authentication scheme.

Use JSON import immediately. Download the normalized JSON schema from the benchmark view. A downloadable run-context sidecar captures the current configuration; fill in actual run metadata before importing. The example templates deliberately contain missing values and are not measurements.

See [docs/benchmark-integration.md](docs/benchmark-integration.md) for the adapter contract, loadable factory, import formats, metric definitions, matching rules, and sidecar details.

## Model/runtime support

The conservative analytical path covers conventional dense decoder `LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, and full-attention `MistralForCausalLM` configurations with sufficient metadata. Disabled sliding-window settings do not activate sliding-window behavior. MoE, active sliding-window attention, hybrid/state-space, multimodal, custom architectures, and alternative KV representations are explicitly unsupported by the basic estimator. Field overrides do not remove these warnings.

Parameter counts prefer Hub safetensors metadata. When unavailable for a supported dense gated-MLP model, a tensor-shape formula is labeled **Derived estimate**. Counts are never inferred from the repository name. Quantization metadata is retained, but unknown storage and unverified kernels are not assigned plausible performance.

The serving profile is intentionally scoped to **vLLM 0.10.2**, checked against its [versioned CLI](https://docs.vllm.ai/en/v0.10.2/cli/serve.html), [supported models](https://docs.vllm.ai/en/v0.10.2/models/supported_models.html), and [CPU documentation](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/cpu.html). Other entered versions remain unsupported until a verified profile is added. vLLM itself is not installed by this planner; it belongs on the target serving hardware.

Dense BF16/FP16 serving storage is modeled separately from repository dtype; vLLM's ordinary serving dtype couples dense weights and activations. Non-FP8 KV must follow that dtype. FP8 KV is opt-in for the cataloged H100/L4 profile, includes dynamic scale calculation, and requires target CUDA/kernel and model-quality validation. The planner does not claim FP8 accuracy is unchanged.

## Hardware catalog and cost

[planner/data/hardware.v1.json](planner/data/hardware.v1.json) is versioned and contains per-field units, vendor source URLs, topology assumptions, notes, and price provenance. It includes H100 SXM 80 GB, A100 SXM 80 GB, L4 24 GB, and a c7i.4xlarge CPU profile. GPU compute uses **dense** BF16 ceilings, without a sparsity multiplier. CPU bandwidth/compute are study-guide assumptions, not measurements.

**All default prices are illustrative.** H100/A100 rates are budget placeholders. L4/CPU rates were transcribed from the existing guide, not verified as current cloud quotes. The catalog date records when these assumptions were assembled. Edit price, provider, region and date in Workload → Hardware pricing; changed rows are labeled Supplied. Include all allocated host, network, storage and router costs in the supplied unit price. No live provider-pricing API is used.

Memory calculations use bytes; charts show GiB. Decimal GB = 10^9 bytes, GiB = 2^30 bytes. Catalog GPU capacities are decimal GB.

See [docs/methodology.md](docs/methodology.md) for formulas, assumptions, disaggregation limits, and the recomputed Llama/H100 sanity check.

## Validation

```sh
uv run pytest -q
uv run ruff check app.py planner tests
node --check < web/app.js  # optional JavaScript syntax check
```

Tests cover units/KV geometry, block rounding, precision, exhaustion, context and TP compatibility, CPU allocation, prefix assumptions, Little's law, replica/cost denominators, special/missing metadata, immutable sanitized imports, metric definitions, calibration context and extrapolation, service-adapter lifecycle, API workflows, and export prerequisites.

Browser checks exercised live Qwen metadata retrieval, workload recalculation, scenario persistence, comparison/charts, unconfigured benchmark behavior, and versioned exports. The public Hub smoke lookup resolved `Qwen/Qwen2.5-1.5B-Instruct` to `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` with 1,543,714,304 parameters from repository tensor metadata.

[docs/validation.md](docs/validation.md) records the verification scope. No GPU/CPU serving benchmark or cluster deployment was performed. Initial efficiencies and all analytical latency/capacity figures remain estimates. Tail SLO compliance requires context-matched representative measurements. Calibration residuals are not statistical confidence intervals.

## Project layout

```text
app.py                     FastAPI endpoints and local asset serving
web/                       Browser UI, CSS, vendored Plotly, JSON schema
planner/models.py          Typed units/context/validation records
planner/hf_metadata.py     Official Hub client and architecture normalization
planner/catalog.py         Hardware catalog loader
planner/sizing.py          Pure memory, roofline, replica, cost and pool calculations
planner/benchmark.py       Service adapter interface and result import
planner/calibration.py     Context matching and empirical corrections
planner/storage.py         SQLite metadata/scenario/benchmark/job persistence
planner/exports.py         Version-scoped vLLM and Kubernetes exports
planner/charts.py          Evidence-aware Plotly figures
examples/                  Illustrative config and incomplete import templates
tests/                     Calculation and integration boundaries
```

Default data path: `.local/planner.sqlite3` (ignored by Git). Set `PLANNER_DB` to change it. Credentials are not stored there. Back up SQLite with a consistent SQLite backup/snapshot if retaining history across installations.
