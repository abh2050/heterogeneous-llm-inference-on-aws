# Inference Atlas

**Find the cheapest hardware that runs your AI model fast enough for your users.**

Live: https://inference-atlas-one.vercel.app · API docs: `/api/docs`

Inference Atlas is a capacity planner for LLM serving. Pick a model, a job, and a number of users; it sizes memory,
estimates speed and wait time, prices every hardware option in the catalog, and exports a `vllm serve` command.
It is a FastAPI backend with a framework-free HTML/CSS/JavaScript frontend, Pydantic validation, and Plotly charts.
It does not run models or benchmarks itself; measurements come from your own runs.

It sits next to the heterogeneous-inference study in this repository (see the root [README](../README.md)).

| | |
|---|---|
| **Guided page** | Four plain-language steps, live results, a stack diagram, explainers, and a glossary |
| **Advanced section** | Roofline model, quick estimate, benchmarks, exact precision, assumptions, catalog, exports |
| **Honest labels** | Every number is tagged *measured*, *calculated*, or *estimated* with a one-line explanation |
| **Runs two ways** | Local with SQLite, or stateless on Vercel with browser storage |

## Quick start

Requires Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/) (3.12 was used for validation).

```sh
uv sync --frozen
./run.sh                      # http://127.0.0.1:8765
```

`./run.sh` runs `uvicorn app:app` on port 8765 (override with `PLANNER_PORT`). Keep it bound to loopback: local mode
is single-user and has no authentication boundary. Node and a frontend build are not needed.

Click **Start with an example** to load Qwen3 30B A3B, Agents, 40 users, Balanced, on a DGX Spark. The catalog
leaves DGX Spark's BF16 TFLOPS and price blank ("not published"), so the page asks you to enter them rather than
guessing.

## The guided page

1. **Pick a model.** Search Hugging Face (text-generation, most downloaded, popular models pinned) or paste an ID or
   URL. A card shows size, memory needed at each precision, and whether it is a mixture-of-experts model.
2. **Pick the job.** Chat, Document Q&A (RAG), Coding assistant, or Agents. Each fills typical input, output, and
   context lengths; every value stays editable in Advanced.
3. **How many people.** Total users and the share active together (10% by default) give "about N people typing at
   the same moment", which is mapped onto the concurrency sweep.
4. **Quality versus size.** Best quality (BF16), Balanced (FP8, default), Smallest (INT4).

Results update as you change anything: a one-sentence verdict, a memory bar (weights, KV cache, reserve, free), a
speed gauge in tokens per second per person with a marker at 20, the wait before the first word, monthly cost and
cost per million tokens beside an editable cloud API price, and a table of every option that fits (green meets both
targets, yellow is close, red is under 10 tokens/s or over 5 s). Below are your stack (users, gateway, scheduler,
serving engine, compute platform, hardware), short explainers that follow one analogy, and a glossary linked from
every tooltip. It works on a phone.

## Advanced

Everything from the original planner lives in the **Advanced** section at the bottom of the page.

| Page | What it does |
|---|---|
| **Model** | Inspect a Hugging Face repo at a pinned commit, or a local `config.json`; per-field sources and explicit overrides |
| **Workload & targets** | Token lengths, percentiles, request rate, burst, headroom, prefix caching, p95 TTFT, TPOT percentile, people, presets, exact weight/KV precision, efficiencies and reserves, hardware catalog and pricing |
| **Compare configurations** | Candidates ranked by deployment cost, per-device memory, TPOT versus concurrency, a *Quick estimate (simplified)* panel, and an exploratory prefill/decode disaggregation calculator |
| **Validate with benchmarks** | Import normalized JSON or a vLLM result, or paste `vllm bench serve` output (a short model/hardware/runtime form replaces the sidecar file) |
| **Export** | Version-scoped `vllm serve` command (with `--quantization` and `--kv-cache-dtype`), JSON sizing report, Kubernetes manifest |

Save and reopen scenarios from the sidebar. Inputs apply on form submission.

### Two sets of calculations, labeled apart

- **Calculated, advanced:** the roofline model in `planner/sizing.py` (compute and bandwidth ceilings, TP
  communication, queueing-free service times, replica counts). Existing outputs are unchanged.
- **Calculated, simplified:** the plain formulas in `planner/quick.py`:

  ```text
  Weights GB   = parameters (billions) × bytes per parameter
  KV GB        = 2 × layers × KV heads × head_dim × KV bytes × context × users / 1e9
  Fits         = (weights + KV) × 1.10 ≤ memory
  Decode tok/s = bandwidth / bytes read per step × 0.7      (efficiency is editable)
  $ / M tokens = hourly cost / (total tok/s × 3600) × 1,000,000
  ```

  Llama 3.1 70B, BF16 weights and KV, 20 users, 32,768 context gives 140 GB of weights and about 215 GB of KV
  cache (tested from `examples/llama31-70b.config.json`, no network).
- **Measured:** imported benchmark results keep their own label and take precedence in matching contexts.

No analytical output is a p95 measurement. Tail-latency compliance needs context-matched benchmarks.

## Model and runtime support

- **Dense and mixture-of-experts decoders.** `LlamaForCausalLM`, `Qwen2`, `Qwen3`, `Qwen3Moe`, and full-attention
  `Mistral` configs with enough metadata. For MoE models, memory uses **total** parameters and decode bandwidth
  uses **active** parameters per token (expert counts and sizes are read from `config.json`). Tested with
  Qwen/Qwen3-30B-A3B.
- **Unsupported, and flagged:** active sliding-window attention, hybrid/state-space, multimodal, custom
  architectures, and alternative KV layouts. Field overrides do not remove these warnings.
- **Parameter counts** prefer Hub safetensors metadata. Otherwise a tensor-shape formula is labeled *Derived
  estimate*. Counts are never inferred from a model name.
- **Precision.** Weights BF16, FP8 (1 byte), or INT4 (0.5 byte); KV cache BF16 or FP8. FP8 weights export as
  `--quantization fp8`. INT4 sizes correctly but exports as a template unless the model ID is a pre-quantized AWQ
  checkpoint, because vLLM cannot quantize BF16 weights to INT4 at load time. Pre-quantized AWQ, GPTQ, FP8 and
  compressed-tensors checkpoints are recognized. FP8 KV is verified only for the cataloged CUDA profiles and needs
  quality validation on your hardware.
- **Runtime profile** is scoped to **vLLM 0.10.2**, checked against its
  [CLI](https://docs.vllm.ai/en/v0.10.2/cli/serve.html),
  [supported models](https://docs.vllm.ai/en/v0.10.2/models/supported_models.html), and
  [CPU docs](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/cpu.html). Other versions stay
  unsupported until a verified profile is added. vLLM is not installed by this app.

## Hardware catalog and cost

`hardware_catalog.json` (validated by [`planner/data/hardware_catalog.schema.json`](planner/data/hardware_catalog.schema.json))
lists 24 systems across NVIDIA, AMD, and Intel, plus Apple, with memory, bandwidth, BF16/FP8/FP4 TFLOPS,
interconnect, power, price, compute platform, serving engines, and a per-value source link. It is converted into the
planner's `Hardware` format by `planner/hardware_catalog.py`.

- **Nulls stay null.** Unpublished values show as "not published", and each published value has its source in a
  tooltip. An entry joins the comparison only once memory, bandwidth, BF16 TFLOPS, and a price are all present.
  Fill gaps under Advanced → Workload → Hardware catalog. An editable "My local lab machine" entry starts blank.
- **Multi-GPU systems** are divided into per-device entries; one-way link bandwidth is half the catalog's aggregate
  figure.
- **Prices.** Where there is no cloud hourly price, cost is the list price amortized over 3 years of continuous
  use (labeled *estimated*; it excludes host, power, and hosting). All prices are illustrative, not quotes. Include
  every allocated cost in the unit price you supply.
- **Prefill speed** is estimated as `bf16_tflops_dense × 1e12 × 0.4 / (2 × active parameters)`, labeled
  *estimated*, and editable.
- **Built-ins kept.** The original A100 SXM 80 GB and c7i.4xlarge CPU entries (in
  [`planner/data/hardware.v1.json`](planner/data/hardware.v1.json)) stay because the catalog lacks them; the H100
  and L4 come from the catalog.

Memory is computed in bytes and charted in GiB. Decimal GB = 10^9 bytes; catalog capacities are decimal GB. See
[docs/methodology.md](docs/methodology.md) for formulas, assumptions, and disaggregation limits.

## Hugging Face token

Set `HF_TOKEN` as a **server environment variable**: a local shell export, a Vercel environment variable, or a
local secret file selected by `PLANNER_SECRETS_FILE` (an `HF_TOKEN` setting in an ignored `.local/secrets.toml`;
the environment variable wins).

```sh
read -s 'HF_TOKEN?HF read token: '   # zsh: prompts without echoing
export HF_TOKEN && ./run.sh
```

The token is never sent to the browser, logged, exported, or included in error messages (the page only learns
whether one is configured; tests assert this). Imported credentials and recognizable token strings are redacted
before storage. A token does not grant access to a gated model such as Llama: your account must also be approved on
the model's page. Lookups are cached by model ID, revision, and SHA; use **Refresh cached metadata** to re-resolve a
moving branch.

## Benchmarks

Measurements enter three ways, and none are preloaded:

- **Paste** `vllm bench serve` output or its `--save-result` JSON in Advanced → Validate. If it does not say what was
  benchmarked, a short form asks for model, hardware, and runtime.
- **Import** normalized JSON, or a vLLM result plus a context sidecar (download the sidecar template and schema from
  the same page).
- **Adapter** (local mode only): an installed service adapter can advertise health, submit, poll, results, and
  cancel. It is unconfigured by default; the app does not invent endpoints or infer an authentication scheme.

Measurements keep their launch and workload context and separate TTFT, TPOT, ITL, and E2E. Matching runs take
precedence over heuristics. See [docs/benchmark-integration.md](docs/benchmark-integration.md) for the adapter
contract, formats, metric definitions, and matching rules.

## Deploying to Vercel

The same app runs on Vercel's Python runtime. Set `STORAGE_MODE=vercel` (or deploy on Vercel, which is detected
automatically): the server becomes stateless, and scenarios, benchmarks, and hardware edits live in the browser's
IndexedDB with **Export my data** and **Import data** buttons. The external benchmark adapter is disabled, and the
only server-side file is a disposable metadata cache under `/tmp`. The local path is unchanged. Full steps for the
CLI and dashboard are in [DEPLOY.md](DEPLOY.md).

## Development

```sh
uv run pytest -q          # 114 tests: sizing, MoE, precision, catalog, quick estimate, paste, token, both storage modes
uv run ruff check .
```

Tests cover KV geometry and block rounding, precision, context and TP compatibility, CPU allocation, prefix
assumptions, Little's law, replica and cost denominators, catalog conversion, the quick-estimate formulas,
benchmark parsing and calibration, token handling, exports, and Vercel-mode behavior. Browser behavior was checked in
headless Chrome (guided page rendering, IndexedDB persistence across reload). See
[docs/validation.md](docs/validation.md) for the verification scope. No GPU or CPU serving benchmark or cluster
deployment was run for this app, so analytical latency and capacity figures remain estimates.

### Project layout

```text
app.py                      FastAPI endpoints; serves web/ locally
vercel.json                 Static web/ plus /static mapping (root app.py is the Vercel function)
hardware_catalog.json       Hardware catalog (validated, converted at startup)
web/                        index.html, guided.js/css, app.js (Advanced), storage.js, vendored Plotly
planner/models.py           Typed records: model, workload, assumptions, hardware, benchmarks
planner/sizing.py           Roofline memory, speed, replica, cost, and pool calculations
planner/quick.py            Quick estimate (simplified) formulas
planner/presets.py          Chat, RAG, coding, agents workload presets
planner/hardware_catalog.py Catalog schema, conversion, prefill estimate
planner/hf_metadata.py      Hub client, architecture normalization (dense and MoE), model search
planner/paste.py            vllm bench serve text/JSON parsing and short context form
planner/benchmark.py        Service adapter interface and result import
planner/calibration.py      Context matching and empirical corrections
planner/exports.py          Version-scoped vLLM and Kubernetes exports
planner/storage.py          SQLite metadata, scenarios, benchmarks, jobs (local mode)
planner/mode.py             local vs vercel storage mode
planner/security.py         Redaction and identifier protections
examples/                   Example configs (Llama 8B and 70B) and import templates
tests/                      Calculation and integration tests
```

Local data lives in `.local/planner.sqlite3` (ignored by Git; override with `PLANNER_DB`). Credentials are not
stored there. See [CHANGELOG.md](CHANGELOG.md) for what each build step added.
