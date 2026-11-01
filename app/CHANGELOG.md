# Changelog

Extensions to the existing planner, one commit per step. Existing outputs, formulas, models and tests were kept;
new calculations sit beside them with their own labels.

- **Step 0 · Setup.** Repository under git with a "baseline before extensions" commit.
- **Step 1 · Mixture of experts.** MoE configs are no longer rejected. Expert counts and sizes are read from
  `config.json`; total and active parameters per token are computed. Memory uses total parameters, decode uses
  active parameters. Tested with Qwen/Qwen3-30B-A3B.
- **Step 2 · Workload presets.** Chat, RAG, Coding assistant and Agents fill the existing token fields; every
  value stays editable (`planner/presets.py`).
- **Step 3 · Users.** Total and peak concurrent users (default 10% of total). Peak users are added to the
  concurrency sweep, and each candidate reports the users it can serve at once.
- **Step 4 · Precision.** Weight precision BF16/FP8/INT4 and a separate KV-cache precision. Quantized candidates
  export with `--quantization`; `--kv-cache-dtype` follows the KV selector. INT4 needs a pre-quantized AWQ
  checkpoint and stays a template otherwise.
- **Step 5 · Hardware catalog.** `hardware_catalog.json` is validated against a schema
  (`planner/data/hardware_catalog.schema.json`) and converted to the existing format. Nulls show as "not
  published" with per-value source tooltips. Missing prefill speed is estimated as
  `bf16_tflops_dense × 1e12 × 0.4 / (2 × active parameters)` and is editable. Built-in A100 and CPU entries are
  kept because the file lacks them. An editable local lab machine entry is included.
- **Step 6 · Quick estimate.** A "Quick estimate (simplified)" panel using the specified formulas, labeled
  *calculated, simplified*; existing values are labeled *calculated, advanced*.
- **Step 7 · Formula test.** Llama 3.1 70B, BF16, 20 users, 32,768 context: 140 GB weights, about 215 GB KV,
  from `examples/llama31-70b.config.json` with no network access.
- **Step 8 · Hugging Face token.** `HF_TOKEN` is read server-side only; tests assert it never reaches the
  browser, errors, or exports, and that gated loads use it.
- **Step 9 · Model search.** A server route queries the Hub (text-generation, by downloads, limit 20) with
  term and ID validation; popular models are pinned. Selecting a result runs the existing metadata fetch.
- **Step 10 · Benchmark paste.** Paste `vllm bench serve` output or JSON; a short model/hardware/runtime form
  replaces the sidecar file when context is missing.
- **Step 11 · Guided one-page design.** Plain-language flow, live results, stack diagram, explainers and
  glossary; the original planner lives in the Advanced section.
- **Step 12 · Vercel.** `vercel.json`, `api/index.py`, `requirements.txt`, a stateless storage mode (browser
  IndexedDB with JSON export/import, `/tmp`-only cache, benchmark adapter disabled), and `DEPLOY.md`.
