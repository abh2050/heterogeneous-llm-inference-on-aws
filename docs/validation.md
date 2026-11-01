# Validation record

Validated locally on 2026-10-31 using Python 3.12, FastAPI, Pydantic, SQLite and a Chrome browser.

## Automated checks

Final run: **66 tests passed**. Python lint and JavaScript syntax checks passed. The installed Starlette test client emits one dependency deprecation warning about its httpx compatibility; it does not affect test results.

- Pure sizing tests cover decimal/binary units, dense KV geometry, cache-block rounding, precision changes, per-device memory exhaustion, missing fields, context overflow, TP head/fabric compatibility, imperfect scaling, CPU-specific efficiencies and GiB cache allocation.
- Traffic/cost checks cover combined-phase capacity, active-sequence/Little's-law consistency, burst/headroom replica counts, all-device cost, delivered-workload output and total-token denominators, and transfer/network/all-pool costs for exploratory disaggregation.
- Metadata tests cover IDs/URLs, commit-pinned config-only downloads, caching, tensor-metadata precedence, derived parameter provenance, overrides, custom-code declarations, multimodal/MoE/window/hybrid/alternative-KV exclusion and actionable access/rate-limit errors.
- Evidence tests cover distinct TPOT/ITL, missing/nonfinite metrics, timezone metadata, immutable sanitized history, atomic batches, exact launch/revision/precision/cache matching, measurement precedence, tail-validation restrictions and empirical correction/extrapolation behavior.
- Export tests cover safe version-scoped shell argv, unresolved/unsupported templates, FP8 restrictions/scales, GPU resource requests, secret references, health/readiness probes, storage and missing Kubernetes prerequisites.
- API tests use temporary databases and a synthetic **test-only** service adapter to verify submission, polling, fetching, cancellation, history, scenario roundtrip, static asset serving and raw-exception suppression. The test adapter does not call or replace the user's benchmark service.

Run `uv run pytest -q` for the exact current count and results. `uv run ruff check app.py planner tests` and `node --check < web/app.js` validate Python imports/style and JavaScript syntax.

## Browser and live external checks

The webpage was opened in Chrome and exercised through its visible UI:

- Loaded the clearly labeled offline Llama example, with empty benchmark history.
- Inspected public `Qwen/Qwen2.5-1.5B-Instruct` through the official Hub client. The live lookup resolved SHA `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and 1,543,714,304 parameters from safetensors metadata; config normalization found 28 layers, 12 attention heads, 2 KV heads, head dimension 128 and context 32,768.
- Changed peak traffic from 2 to 4 req/s and applied it. The persistent context and provisioned replica count updated.
- Saved the estimated Qwen scenario in SQLite and verified its sidebar entry.
- Viewed ranked configurations, TPOT/concurrency plot, target line and per-device memory breakdown.
- Verified zero stored measurements, disabled submission/health for the unconfigured service, and available JSON import/context/schema controls.
- Generated a complete vLLM 0.10.2 command for the resolved Qwen revision, with explicit model/context/concurrency/TP/cache/batching settings and a downloadable sizing report.

The bundled Plotly file is generated from the locked installed Plotly package; it is served locally, without a CDN dependency. No weights were downloaded or remote model code executed. No credential supplied in chat was used or written into source.

## Remaining external validation

No benchmark-service API contract was supplied. The adapter is intentionally unconfigured; the actual service's authentication, response mapping and end-to-end job lifecycle remain to be verified after its contract is available.

No inference engine ran on H100/A100/L4 or CPU, no model-quality evaluation was performed, and no Kubernetes cluster was deployed or validated with a live API server. Export checks parse YAML and inspect required structure; they are not a live Kubernetes conformance check. Single-node TP fabric, CPU NUMA/SMT allocation, memory reserves and image compatibility need verification on the target.

Default MFU/MBU, scheduler capacity, prefill/decode interference, queueing and tail latency remain analytical assumptions. All default prices are illustrative. Calibration residuals describe the imported sample fit, not confidence intervals or out-of-sample accuracy. Disaggregation is an exploratory estimate pending an actual supported router/connector deployment and representative benchmarks.
