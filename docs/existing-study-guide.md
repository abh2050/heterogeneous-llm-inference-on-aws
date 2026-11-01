# Heterogeneous LLM inference: study guide

This guide goes with the `vllm-hetero` runbook. That experiment serves `Qwen/Qwen2.5-1.5B-Instruct` from an NVIDIA L4 GPU node and an Intel Sapphire Rapids CPU node behind one router, and measures where each node is worth using.

Each section explains one idea, names the phase that measures it, and ends with a **Predict** box. Write your prediction in the table at the bottom *before* you run that phase. Comparing your guesses with the measured numbers is where most of the learning happens.

The back-of-envelope numbers here are estimates from spec sheets. Expect real numbers to land somewhere between 30% and 80% of the theoretical ceilings.

---

## 1. How one request is served

An LLM generates text one token at a time. Each request goes through two phases.

**Prefill.** The model reads the whole prompt in one forward pass. Every prompt token is processed in parallel, so this phase does many matrix multiplications per byte of weights it reads. Prefill is **compute-bound**: its speed depends on FLOPs.

**Decode.** The model produces one new token per forward pass and feeds it back in. Each step reads *all* the weights from memory to do a small amount of math for one token per sequence. Decode is **memory-bandwidth-bound**: its speed depends on how fast the chip can stream the weights.

That split explains most of what you will see:

- **Time to first token (TTFT)** is mostly queueing time plus prefill time.
- **Inter-token latency (ITL)** is the time of one decode step.
- **Batching** helps decode a lot. One read of the weights serves every sequence in the batch, so 32 sequences cost about the same bandwidth as 1.

### The roofline estimate

A forward pass costs about **2 × parameters FLOPs per token**. A decode step has to read about **all of the weight bytes** once.

| | L4 GPU | c7i.4xlarge CPU (8 physical cores) |
|---|---|---|
| Weights (1.54B params × 2 bytes bf16) | ~3.1 GB | ~3.1 GB |
| Memory bandwidth | ~300 GB/s (GDDR6) | roughly 30–60 GB/s usable (your share of a DDR5 socket) |
| Batch-1 decode ceiling = bandwidth ÷ weight bytes | **~95 tok/s** | **~10–20 tok/s** |
| bf16 matrix throughput (peak) | ~120 TFLOPS (tensor cores, dense) | ~25 TFLOPS (AMX: 8 cores × ~1,024 bf16 FLOP/cycle × ~3 GHz) |
| Prefill of a 2,048-token prompt (~6.3 TFLOP) at ~50% of peak | ~0.1 s | ~0.5–1 s |

> **Predict.** What is the GPU's single-stream decode speed in tok/s? Will the CPU node meet a 1 s p99 TTFT on the 2,048-token `rag` shape even at a very low request rate?

---

## 2. The KV cache

During decode, attention needs the keys and values of every earlier token. Recomputing them at every step would be wasteful, so the engine stores them in a **KV cache**. The KV cache, not the weights, is what limits how many requests fit at once.

**Size per token** = 2 (K and V) × layers × KV heads × head dim × bytes per value.

Qwen2.5-1.5B has 28 layers, 2 KV heads, and a head dimension of 128. It uses **grouped-query attention (GQA)**: 12 query heads share 2 KV heads, which makes the cache 6× smaller than it would otherwise be.

```
2 × 28 × 2 × 128 × 2 bytes = 28,672 bytes ≈ 28 KB per token
4,096-token request       ≈ 117 MB
chat request (256 + 128)  ≈ 11 MB
rag request (2,048 + 256) ≈ 66 MB
```

**Pool size on the L4.** vLLM claims `gpu_memory_utilization × total memory`. It subtracts the weights and the activation and CUDA-graph workspace, and turns the remainder into the KV pool.

```
0.90 × ~22.5 GB ≈ 20 GB − 3.1 GB weights − ~1.5 GB workspace ≈ 15–16 GB → ~550k tokens → ~130× 4,096-token requests
0.50 × ~22.5 GB ≈ 11 GB − 3.1 GB − ~1.5 GB                  ≈ 6–7 GB   → ~230k tokens → ~55× 4,096-token requests
```

On the CPU node, the pool is simply `VLLM_CPU_KVCACHE_SPACE` (12 GB), which is about 440k tokens.

> **Predict.** What will the Step 2.6 log line say for KV cache tokens and for "maximum concurrency for 4,096 tokens per request"? How far off is the Config A pool (0.50) in 5.3?

---

## 3. What vLLM does differently from plain HuggingFace

| Idea | What it does | Where you see it |
|---|---|---|
| **PagedAttention** | Stores the KV cache in fixed-size blocks, like OS memory pages. A request takes only the blocks it uses, so there is almost no fragmentation and no up-front reservation for `max_model_len`. | KV log line (2.6), Config A (5.3) |
| **Continuous batching** | The scheduler adds and removes requests at every decode step. A finished request frees its slot immediately, and a new one joins on the next step. HF `generate()` uses a static batch that runs until the whole batch finishes. | Engine compare (5.1), sweeps (5.2) |
| **CUDA graphs and fused kernels** | Records the decode step once and replays it, which removes Python and kernel-launch overhead. That overhead dominates for a small model at batch 1. | Engine compare at batch 1 |
| **Prefix caching** | Reuses KV blocks for a shared prompt prefix. Step 5.1 turns it off because every prompt in the batch is identical, and HF has no such cache. | 5.1 |
| **Chunked prefill** | Splits a long prefill into chunks and mixes those chunks with ongoing decode steps, so one huge prompt does not freeze every other stream. It is enabled by default in current vLLM. | Offload experiment (5.4) |
| **Scheduler caps** | `--max-num-seqs` limits running sequences. `--max-num-batched-tokens` limits tokens per step. | Config B and C (5.3) |

**The lab correction.** `--gpu-memory-utilization` sets the size of the pool. `--max-model-len` only caps the length of a single request, and with paging it does not shrink the pool.

**A note on Step 5.1.** All prompts in that batch are identical, so HF does no padding there. Any vLLM advantage you see comes from per-step overhead (CUDA graphs, kernels, the scheduler), not from padding. Padding waste appears only when prompt lengths differ.

> **Predict.** What is vLLM's speedup over HF at batch 1, and at batch 32? The runbook expects the two engines to be close at batch 1. A 1.5B model in eager HF is often launch-overhead-bound, so vLLM may be well ahead even there. Decide which you believe.

---

## 4. Latency and throughput metrics

| Metric | Meaning | What makes it grow |
|---|---|---|
| **TTFT** (time to first token) | Time from sending a request to receiving its first token | Queueing, long prompts |
| **ITL** (inter-token latency) | Gap between consecutive streamed tokens | Large batches, and other requests' prefills sharing the step |
| **TPOT** (time per output token) | Average decode time per token for one request, excluding the first token | Same causes as ITL, averaged per request |
| **E2EL** (end-to-end latency) | Total request time ≈ TTFT + (output tokens − 1) × TPOT | Both of the above |
| **Output throughput** | Total generated tok/s across all users | Batch size, until the node saturates |

**p50 versus p99.** p50 is the typical user. p99 is the worst 1 in 100, and a service level objective (SLO) is written against it. Averages hide queueing spikes.

**SLO in this experiment:** p99 TTFT ≤ 1 s and p99 ITL ≤ 100 ms.

**Capacity at SLO**, also called **goodput**, is the highest load that still meets the SLO. Throughput measured beyond that point does not count, because users are already getting a slow service.

---

## 5. Load testing and queueing

**Open-loop load.** `--request-rate R` sends requests as a Poisson process averaging R per second, whether or not earlier requests have finished. That is how real traffic behaves. `inf` sends every request at once and measures raw maximum throughput.

**The saturation knee.** As R climbs, throughput rises roughly linearly and latency stays flat. Then the node runs out of KV space, scheduler slots, or compute. From that point, throughput plateaus, `vllm:num_requests_waiting` rises above 0, and p99 TTFT shoots up because new requests are waiting in a queue.

**Little's law:** requests in flight = arrival rate × time in system (L = λ × W). For example, 8 req/s × 2 s end-to-end means about 16 requests running at once. Use it to sanity-check the "running" panel in Grafana.

**Making runs comparable.**
- `--ignore-eos` forces exactly the requested output length.
- `--seed 42` generates the same random prompts on every run.
- The benchmark client runs on the server's machine, so network delay stays out of the engine numbers.

> **Predict.** At roughly what chat request rate does the GPU hit the knee: 8, 16, 32, or 64 req/s? Where is the CPU's knee for chat?

---

## 6. The hardware

**NVIDIA L4 (g6.xlarge).** 24 GB of GDDR6 at about 300 GB/s, with tensor cores for bf16. It is a small inference GPU: it has plenty of compute for a 1.5B model, while memory bandwidth limits decode.

**Intel Xeon Sapphire Rapids (c7i.4xlarge).** 16 vCPUs on 8 physical cores.
- **AVX-512** provides wide vector math, and the **`avx512_bf16`** flag adds bf16 dot products.
- **AMX** (Advanced Matrix Extensions; flags `amx_tile`, `amx_bf16`) adds a tile matrix-multiply unit inside each core. AMX is what makes CPU prefill workable at all.
- **Hyperthreads.** Two vCPUs on one physical core share that core's AMX unit and caches, so the runbook pins one compute thread per physical core with `VLLM_CPU_OMP_THREADS_BIND`. The API server and the benchmark client need CPU time too, which is why leaving a core free can help.
- **Memory bandwidth** is the CPU's decode bottleneck, just as on the GPU, but it is about 5–10× lower.

**What CPU inference is good for.** Latency-tolerant, low-volume or bursty work: batch summarization, embeddings, offline evaluation, background RAG preprocessing. It also uses capacity you may already be paying for. CPU inference is a poor fit for interactive chat at scale.

---

## 7. Heterogeneous serving: the reason for this project

**Workload classes.** Interactive traffic cares about TTFT and ITL. Background or batch traffic cares about cost and eventual completion. Serving both from one pool means they interfere with each other.

**Prefill interference.** When a 2,048-token RAG prompt lands in the same scheduler step as 30 streaming chats, that step takes longer, and every chat user sees a stall. Chunked prefill reduces the stall but does not remove it. The tail of the ITL distribution (p99) is where this shows up.

**Routing.** LiteLLM exposes one OpenAI-compatible endpoint and maps a model *name* to a backend: `qwen-fast` → GPU, `qwen-batch` → CPU. Real systems route by API key, tenant, priority header, prompt length, or queue depth. Every router adds one network hop and some CPU work (Step 5.4.3 measures it).

**Related designs worth knowing:**
- **Prefill/decode disaggregation** (DistServe, Splitwise). Run prefill and decode on different machines, sometimes on different hardware types, and ship the KV cache between them.
- **Priority scheduling** within a single pool, as the alternative to running a separate tier.
- **Autoscaling** on queue depth or KV usage, versus a fixed CPU tier.

> **Predict.** Will moving background RAG traffic to the CPU node visibly cut the GPU's p99 ITL and TTFT for chat users? One caveat: background load is capped at whatever the CPU node can handle, which may be too light to trouble the GPU in Run A. If Run B shows no gain, check whether Run A was under real pressure before concluding anything.

---

## 8. Cost

```
$ per 1M output tokens = hourly price ÷ (output tok/s × 3,600) × 1,000,000
```

| Node | Price/hr (us-east-2) | Example: if it sustains… | $/1M output tokens |
|---|---|---|---|
| g6.xlarge | $0.8048 | 2,000 tok/s at SLO | ~$0.11 |
| c7i.4xlarge | $0.7140 | 50 tok/s at SLO | ~$3.97 |

Only rows that meet the SLO count toward capacity. Two consequences follow. A GPU can be *cheaper per token* than a CPU even though it costs more per hour, because it produces far more tokens. And a CPU tier is justified by *isolation* (Question 5) or by already-paid-for capacity, rarely by raw $/token.

> **Predict.** What is the ratio of CPU to GPU $/1M tokens on the chat shape: 5×, 20×, or 50×?

---

## 9. Observability

- **Prometheus** pulls ("scrapes") the `/metrics` endpoint of each server every 5 s and stores time series.
- **Metric types:**
  - A **counter** only goes up (`vllm:generation_tokens_total`). Use `rate(...[1m])` to turn it into a per-second value.
  - A **gauge** goes up and down (`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`).
  - A **histogram** stores counts in buckets (`vllm:time_to_first_token_seconds_bucket`). `histogram_quantile(0.99, sum by (le, node) (rate(..._bucket[1m])))` estimates p99 from those buckets. The estimate is only as precise as the bucket edges.
- **What to watch during a sweep:**
  - `waiting` leaving 0 means saturation.
  - KV usage near 100% means you are memory-limited.
  - `waiting` > 0 while KV usage is low means a scheduler cap is the limit (Config B).

Metric names change between vLLM versions. Check the live list with `curl …/metrics | grep '^vllm:'`.

---

## 10. Experiment hygiene

- **Hold everything constant except the one thing you are testing:** model, dtype, vLLM version, dataset seed, and prompt and output lengths.
- **Warm up before timing.** The first call compiles kernels, captures CUDA graphs, and loads weights.
- **Run one engine per process.** HF and vLLM must not share GPU memory.
- **Keep the client next to the server** when you measure the engine. Measure the network or router hop separately.
- **Save the raw JSON for every run**, with a descriptive filename, so you can re-analyze later without paying to re-run.
- **Stop a sweep early** once p99 TTFT passes 10 s. Past that point you are only measuring how long the queue is.

---

## 11. The AWS pieces, and why each is there

| Piece | Why |
|---|---|
| **vCPU quotas** ("G and VT" vs. "Standard") | AWS limits instances by vCPU count per family. New accounts often have 0 GPU quota. |
| **Security group with an SG-to-SG rule** | Port 22 is open to your IP only. Port 8000 is open only to members of the same group, so the nodes can reach each other and nobody else can. |
| **SSH `LocalForward` tunnels** | Your Mac reaches the servers' ports without exposing those ports to the internet. |
| **`Project=vllm-hetero` tag** | Lets one filter find everything for audit and teardown. |
| **`DeleteOnTermination`** | Prevents orphaned EBS volumes from billing you after termination. |
| **Stop vs. terminate** | A stopped instance stops compute billing but keeps disk billing and loses its public IP. Terminate when you are done. |
| **Budget alert** | A safety net for the whole account, not just this project. |

---

## 12. Further reading

- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP 2023), the vLLM paper.
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022), which introduced continuous batching.
- Agrawal et al., *Sarathi-Serve* (OSDI 2024), on chunked prefill and the throughput–latency tradeoff.
- Zhong et al., *DistServe* (OSDI 2024), and Patel et al., *Splitwise* (ISCA 2024), on prefill/decode disaggregation, including on heterogeneous hardware.
- The vLLM docs on CPU installation, benchmarking (`vllm bench serve`), and the metrics reference.

---

## Learning checklist and results

Fill in the Predict column before each phase and the Measured column at its gate.

| # | Question | Phase | Predict | Measured | Why the gap? |
|---|---|---|---|---|---|
| 1 | GPU KV pool tokens / max concurrency at 0.90 | 2.6 | | | |
| 2 | GPU single-stream decode tok/s (vLLM, batch 1) | 5.1 | | | |
| 3 | vLLM ÷ HF speedup, batch 1 | 5.1 | | | |
| 4 | vLLM ÷ HF speedup, batch 32 | 5.1 | | | |
| 5 | GPU chat capacity at SLO (req/s, tok/s) | 5.2 | | | |
| 6 | GPU rag capacity at SLO | 5.2 | | | |
| 7 | CPU chat capacity at SLO | 5.2 | | | |
| 8 | CPU rag capacity at SLO (or "never meets SLO") | 5.2 | | | |
| 9 | $/1M output tokens: GPU chat, CPU chat | 6 | | | |
| 10 | Config A (0.50) KV pool and capacity change | 5.3 | | | |
| 11 | Config B (max-num-seqs 16): what limits it | 5.3 | | | |
| 12 | Router hop: added p50 TTFT (ms) | 5.4.3 | | | |
| 13 | Offload: p99 TTFT and ITL, Run A vs. Run B | 5.4.4 | | | |

**Concepts you should be able to explain afterward:**
- [ ] Why prefill is compute-bound and decode is bandwidth-bound
- [ ] How to compute KV cache size per token from a model config
- [ ] Why continuous batching and PagedAttention raise throughput
- [ ] What `gpu_memory_utilization`, `max_model_len`, and `max_num_seqs` each control
- [ ] TTFT vs. ITL vs. TPOT, and why SLOs use p99
- [ ] How to find a node's capacity from a rate sweep
- [ ] How to turn hourly price and throughput into $/1M tokens
- [ ] When a CPU tier earns its place in a heterogeneous backend, and when it doesn't
