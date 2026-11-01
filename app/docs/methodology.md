# Estimation method and limits

All computation is separate from UI and HTTP transport. `planner/sizing.py` is independently testable. Internal units are bytes, seconds, tokens, FLOPs and USD; bandwidth specifications are decimal GB/s.

## Eligibility

A candidate requires conventional supported dense decoder attention, parameters, layer count, attention/KV head count, head dimension, hidden dimension, context limit, and a cataloged topology. Unknowns remain unknown. A missing field or unsupported architecture excludes the candidate rather than triggering a guess from its name.

The total/active parameter distinction is explicit. Dense active parameters equal total parameters; MoE active parameters are not guessed and MoE remains unsupported. Sliding-window, hybrid, multimodal and alternative KV models need specialized estimators that are not included in this release.

## Memory

Per-device weights:

```
parameters × effective weight bytes/parameter × (1 + format overhead) / TP
```

Parameter count comes from safetensors metadata where available. The fallback Llama-style gated-MLP formula is:

```
layers × (2 × hidden × query_heads × head_dim
        + 2 × hidden × kv_heads × head_dim
        + 3 × hidden × intermediate + 2 × hidden)
+ hidden × vocabulary × (1 if tied_embeddings else 2) + hidden
```

This excludes unknown biases/auxiliary tensors and is labeled derived. Quantized tensor-entry counts need not equal logical parameter counts; quantized repositories remain excluded from performance recommendations until their format/kernel profile is verified.

KV bytes per resident token:

```
2 × layers × KV_heads × head_dim × bytes_per_KV_element
```

For each candidate, full maximum input+output length is rounded up to a 16-token cache block and multiplied by active sequences. This deliberately reserves maxima instead of claiming that all requests always reside at the mean. Effective KV shards = min(TP, KV heads), with only evenly divisible sharding or replication permitted. Per-device cache is divided by those shards, not blindly by TP. CPU cache reservations are rounded up to whole GiB for `VLLM_CPU_KVCACHE_SPACE`.

Runtime, activation and TP communication allocations are separately configurable reserves. The physical-memory safety reserve is added to used memory, then compared with the utilization budget. These reserves are conservative user assumptions, not kernel-derived allocation predictions. Runtime profiling is still required.

Prefix caching requires a request fraction, shared length, assumed hit rate and distinct retained-prefix count. Saved prefill tokens = fraction × length × hit rate, only when caching is enabled. Active sequences conservatively retain their full KV, and retained prefix blocks add cache residency; the estimator does not silently assume all blocks share physical memory. An observed hit rate must match before cached measurements qualify.

## Service time and sustainable capacity

Effective compute = dense BF16/FP16 TFLOP/s × TP × MFU heuristic × TP scaling efficiency. Effective bandwidth is analogous with the MBU heuristic. GPU and CPU efficiencies are distinct. Each TP doubling applies an explicit efficiency penalty. Two collectives per layer add traffic and per-collective latency; the fabric must be cataloged, not merely inferred from a power of two.

Prefill FLOPs start with `2 × active parameters × uncached input tokens`, with an additional approximate quadratic attention term. Decode uses the maximum of compute time and bandwidth time for the batched decode step. Weight traffic is amortized over the active batch, while KV attention traffic grows with context and concurrency. KV replication beyond the KV-head count increases aggregate traffic.

The decode step is a **service-time proxy**, not a measured request TPOT percentile or ITL distribution. TTFT service estimate = prefill service + one decode step. Neither estimate includes a queue, client transport, runtime launch overhead, prefill interference tails, or a general scheduler simulation.

Replica capacity is the smaller of:

```
utilization ceiling / (prefill seconds + output tokens × decode step seconds / concurrency)
utilization ceiling × concurrency / estimated request service residence time
```

This accounts for combined phase demand and active-sequence occupancy. Little's law checks consistency; memory capacity alone never determines serving capacity. Replica count = ceil(peak RPS × burst × (1+headroom) / replica capacity). Horizontal scaling assumes balanced routing and independent replicas; it does not establish tail behavior.

A candidate passes the initial screen only if memory/compatibility fit and service estimates do not already exceed targets. Such a candidate is **analytically feasible; tail latency unvalidated**. It is not advertised as meeting a measured p95 SLO.

Length percentiles and distribution notes are preserved and matched, but not expanded into a synthetic tail model. No arbitrary uncertainty band is called a confidence interval.

## Evidence and calibration

Direct matching checks commit, architecture family through the pinned model, hardware ID and topology, engine/version, serving/quantization/KV precision, TP, launch argv/environment, scheduler/memory/cache configuration, exact lengths/distributions, concurrency and observed cache-hit behavior.

The quality floor for automated evidence use is an open-loop run with a known arrival rate, >=200 samples, >=30 seconds of measurement, >=5 seconds of warmup, explicit zero errors/timeouts, standard metric definitions, and nonzero relevant latency metrics. This is a screening policy, not a statistical power calculation.

For matching qualified repeated runs, conservative maximum latency percentiles and minimum observed sustainable throughput are used. Observed throughput is a demonstrated operating point, not proof of peak capacity. It may require more replicas than a saturation measurement would suggest. A whole-deployment **benchmark validated** label additionally requires the planned replica count and a tested offered/achieved request rate covering the planning rate. Scaling a measured replica to more replicas keeps deployment tails unvalidated.

Other qualifying single-replica runs can produce median empirical correction factors for mean TTFT, mean TPOT, and observed throughput, with at least three runs. The launch/scheduler/precision/model context must remain matched. Mean length and observed concurrency ranges are reported, along with in-sample relative residual errors and extrapolation. Corrections never identify MFU/MBU uniquely. Corrected occupancy is capped again with Little's law. The result is **Calibrated**, not measured tail compliance.

Missing or differently defined metrics do not qualify silently. TPOT and ITL are separate fields; ITL is never substituted for TPOT.

## Disaggregation

The exploratory calculator needs an actual named router/transfer stack plus link capacity, network efficiency, routing delay and router/network cost. It uses a memory-compatible single-device prefill replica and the selected decode TP/concurrency. Prefill and decode pools are sized separately, and all devices are costed.

Transferred KV = full average input tokens × global KV bytes/token. Transfer service uses the supplied effective link bandwidth. Prefill capacity includes transfer, TTFT includes transfer and routing, and aggregate planning-rate traffic must fit the shared fabric budget. Decode retains the full conservative resident cache allocation.

No automatic prefill-utilization threshold selects a winner. The current estimate does not validate connector/layout compatibility, routing balance, bidirectional/fabric contention, scheduler overlap or phase-isolation tail benefits. It generates no disaggregated deployment commands/manifests. Test a supported runtime/router/connector deployment before choosing this layout.

## Cost

Total hourly deployment cost = all devices/nodes × their complete allocated unit price. Disaggregation additionally includes both pools plus router/network cost. Operating-period cost = hourly cost × entered operating hours.

```
USD per million output tokens = hourly cost × 1e6 / (requested delivered output tok/s × 3600)
```

The ordinary denominator is peak RPS × average output length, limited to available capacity. The total-token denominator additionally includes average input length. Headroom is provisioned in the numerator and is not counted as delivered traffic. This makes idle capacity visible in the cost. Supplied prices must include host/network/storage costs; defaults are illustrative assumptions.

## Recomputed Llama 3.1 8B / H100 sanity check

No numeric benchmark example was supplied beyond the requested model/hardware pairing. The included offline config uses 32 layers, 32 query heads, 8 KV heads, head dimension 128, hidden size 4096, intermediate size 14336 and vocabulary 128256. Its shape-derived count is 8,030,261,248 parameters, explicitly an estimate.

At BF16 KV, 2 × 32 × 8 × 128 × 2 = 131,072 bytes/token. An 8,192-token resident request uses exactly 1 GiB of KV. With the default maximum lengths 4096+512 and concurrency 32, KV is 18 GiB. BF16 weights plus 3% overhead are about 15.41 GiB. Runtime 2 GiB, activations 1 GiB and physical-memory safety of 8 decimal GB bring total reserved memory to about 43.86 GiB against a 67.06 GiB utilization budget on one 80 decimal GB H100.

Default average lengths 2048/256, compute efficiency 40%, bandwidth efficiency 65%, and utilization ceiling 70% yield approximately 100.49 ms TTFT service time, 11.79 ms decode-step time and 3.83 requests/s sustainable replica capacity at concurrency 32. A 2 req/s peak ×1.2 burst ×1.25 headroom requires 3 req/s, so one device is analytically sufficient under these assumptions. It is not forced to two devices. These are recomputed heuristics, not measured latency percentiles. At the illustrative $3/hour rate and 512 requested output tokens/s, cost is about $1.63 per million output tokens, including idle headroom.
