# Existing benchmark service adapter

## Discovered project contract

The original workspace contained a study guide discussing `vllm bench serve`, LiteLLM, Prometheus, an L4 and a CPU instance. It contained no service implementation, endpoint definitions, OpenAPI file, service authentication scheme, or actual result JSON. The preserved guide is `docs/existing-study-guide.md`.

Accordingly, direct integration is **unconfigured**. No endpoint names are guessed, no requests are sent to a fabricated base URL, and no benchmark engine is implemented. HF credentials are unrelated to benchmark-service credentials.

To complete direct integration, supply the actual base URL, authentication method, API documentation or OpenAPI contract, and one representative result (secrets omitted). This is the only missing external contract. Local planning/import works without it.

## Adapter boundary

`planner.benchmark.BenchmarkAdapter` defines:

| Method | Returns | Behavior |
|---|---|---|
| `capabilities()` | `Capabilities` | Required; advertise only actual supported operations |
| `health()` | JSON dict | Real connection/capability check, when supported |
| `submit(payload)` | `JobState` | Translate normalized run context into the real service job request |
| `poll(job_id)` | `JobState` | Map real job state to queued/running/succeeded/failed/cancelled/unknown |
| `results(job_id)` | list of `BenchmarkRecord` | Fetch and normalize real measurements |
| `cancel(job_id)` | `JobState` | Cancel only when service supports it |

Implement a factory in trusted, importable local Python code and set `BENCHMARK_ADAPTER=your_module:create_adapter` before starting the app. The factory must return a `BenchmarkAdapter` subclass. There are deliberately no HTTP example endpoints in this repository because the real contract is unknown.

The adapter owns its base URL, HTTP client timeouts, authentication from environment/secret manager, idempotency semantics, and mapping of the real service's schema. Never put credentials in `Capabilities`, `JobState`, launch configuration, imported results, logs, or source. Raise a short nonsecret error where possible; the API also suppresses raw unexpected exceptions.

`submit()` receives `config`, `workload`, `concurrency`, `request_rate_rps`, and `load_mode`. The full generated launch argv and nonsecret runtime environment are in `config.launch_config`. It is a requested workload, not evidence that the service ran it. Return normalized actual configuration on result ingestion.

Capabilities gate UI actions. Submitted job contexts/states persist in SQLite; pending/running jobs poll every five seconds while the benchmark page is open. Results and cancellation are explicit actions. A failed submission is not automatically retried, since the service's idempotency contract is unknown.

## Normalized import

Upload one `BenchmarkRecord` object or an array of up to 1000 records (UI file limit 20 MB). Download the JSON schema from the page, or inspect `web/benchmark.schema.json` / `planner.models.BenchmarkRecord`.

Every record contains:

- Run ID, timezone-aware timestamp, source and optional raw-result reference.
- Model ID/revision, hardware ID/count/topology, engine/version, launch configuration, serving/quantization/KV precisions, TP, replicas, scheduler and cache configuration.
- Mean/max lengths, optional percentile/distribution descriptions, request rate, active concurrency and load mode.
- Prefix settings and observed hit rate when known, warmup, duration and sample count.
- Separate TTFT, request TPOT, streamed ITL and E2E metric objects; each supports mean/p50/p90/p95/p99 in **seconds**.
- Request and input/output-token throughput, error and timeout counts when known.
- Metric definitions and sanitized raw result.

Optional unknown values use JSON `null`, not zero. For this supported dense-serving schema, hardware_count must equal TP × replicas, active concurrency must fit the recorded scheduler slots, and the recorded context limit must cover maximum input+output.

Run IDs are immutable: an identical import is idempotent; a conflicting record is rejected. A batch import is transactional. Raw credential fields and recognizable credential strings are sanitized before persistence. Synthetic test records live only in temporary test databases; the application ships no benchmark history.

`examples/normalized-benchmark.template.json` is an explicitly incomplete template, not an import-ready or measured record. Fill actual required metadata and metrics. Never replace missing measurements with example numbers.

## vLLM flat result + sidecar

Select **vLLM bench serve + sidecar**. Upload the actual vLLM 0.10.2 flat result and a separate context JSON. The parser is based on the versioned [vLLM source](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/benchmarks/serve.py).

Mapping:

| vLLM result | Normalized value |
|---|---|
| `mean/median/pXX_ttft_ms` | `metrics.ttft_s.mean/p50/pXX`, divided by 1000 |
| `mean/median/pXX_tpot_ms` | `metrics.tpot_s...`, divided by 1000 |
| `mean/median/pXX_itl_ms` | `metrics.itl_s...`, divided by 1000 |
| `mean/median/pXX_e2el_ms` | `metrics.e2e_s...`, divided by 1000 |
| `request_throughput` | request throughput, req/s |
| `output_throughput` | output throughput, tok/s |
| `total_input_tokens / duration` | input throughput, tok/s, when both exist |
| `num_prompts - completed` | errors, only when both counts exist |
| `duration`, `completed` | measurement duration, successful sample count |

Request all desired percentiles in the existing benchmark tool. Absent percentiles stay absent. In particular, p95 ITL does not substitute for p95 TPOT. Zero-valued TTFT/TPOT, sometimes emitted for missing streaming observations, cannot establish validation.

The sidecar supplies the full normalized context plus real `run_id`, timezone-aware `timestamp`, `source`, request rate, load mode, warmup, and any fields not present in the flat result. **Download run context** on the webpage captures the selected plan, but its null run ID/timestamp/warmup must be replaced with actual run information. A planned context is not proof of what ran; verify it against the real launch.

The ordinary flat vLLM result does not distinguish timeout counts. If a real known count is available from the benchmark client/service, supply `"metrics": {"timeouts": <actual count>}` in the sidecar; otherwise leave it absent/null. It is not inferred from successful samples. The adapter can normalize richer service-specific counts separately. The flat result's date is timezone-ambiguous; the parser does not guess a timezone.

`examples/benchmark-context.template.json` is an incomplete sidecar example. The schema and full context definitions remain in `planner/models.py`.

## Metric definitions and evidence eligibility

Normalized defaults mean:

- TTFT: client request to first token, including queueing and transport.
- TPOT: per-request `(E2E - TTFT) / (output tokens - 1)`.
- ITL: individual successive streamed-token gaps, a different distribution from TPOT.

If the source uses different definitions, preserve those definitions in `metric_definitions`; such records remain visible in history but do not automatically validate/calibrate canonical targets. Only declare canonical definitions after confirming the source measures them.

For direct measured evidence the planner requires matching resolved revision, hardware/topology, engine/version, precision, launch argv/environment, TP, scheduler/context/batched-token limits, cache behavior, exact workload lengths/distributions and concurrency. It also requires representative open-loop metadata: >=200 samples, >=30 seconds, >=5 seconds warmup, known offered request rate, explicit zero errors/timeouts and the needed nonzero latency percentiles. See `docs/methodology.md` for limits of this floor.

Matching measured performance is preferred to calibrated estimates; calibrated estimates are preferred to uncalibrated heuristics. Unrelated models or engine versions never calibrate a whole hardware family. Multi-replica tail validation additionally requires the planned deployment and traffic to have been measured. Empirical corrections require at least three matching single-replica runs and show length/concurrency coverage, residuals and extrapolation.
