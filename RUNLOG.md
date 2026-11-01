# Run log: vllm-hetero

Chronological record of what was done, what was created, and what was found. Update after every step. Raw data lives in `results/`, raw logs in `logs/`.

Times are UTC. Region: us-east-2. Model: `Qwen/Qwen2.5-1.5B-Instruct`, bfloat16, `--max-model-len 4096`. vLLM `0.29.0` on both nodes.

## Repo layout

| Path | What it is |
|---|---|
| `README.md` | Study guide: concepts, predictions table to fill in |
| `PLAN.md` | Execution plan and the runbook fixes applied |
| `RUNLOG.md` | This file |
| `env.sh` | Real environment values (account-specific; do not publish). `env.example.sh` is the shareable version |
| `docker-compose.yml`, `prometheus/prometheus.yml` | Local Prometheus + Grafana (not yet started against live nodes) |
| `scripts/engine_compare.py` | 5.1 HuggingFace vs vLLM benchmark (run on GPU node) |
| `scripts/sweep.sh` | 5.2 rate sweep wrapper around `vllm bench serve`. Edited: GPU result dir overridable with `RESDIR` |
| `analyze.py` | Capacity-at-SLO and $/1M-token table. Run with `.venv/bin/python analyze.py` (system python's pandas is broken) |
| `results/gpu/`, `results/cpu/` | Raw benchmark JSON per run |
| `results/analysis_*.txt` | Saved analysis output |
| `logs/` | Server logs, benchmark logs, KV cache line |

Never commit: `.venv/`, any `.pem` (kept in `~/.ssh/`, ignored by `.gitignore`).

## Decisions and deviations from the runbook

1. **Account.** Work moved from member account `054772656600` to the org management account `331384580339`, because the GPU (G and VT) quota there is 16 vCPUs while the member account had 0 (a request for 8 is still pending there). Profile `mgmt` via `aws login`.
2. **Budgets.** Runbook's `create-budget` step was skipped. Existing budgets: `331384580339` has "Inference economics" at $60/month. `054772656600` has $20 and $10 budgets. No $30 budget was found in either.
3. **vLLM version.** Runbook's 0.24.0 was a placeholder. Newest CPU image tag is `v0.29.0`, and `vllm==0.29.0` exists on PyPI, so both nodes use 0.29.0.
4. **Key pair file.** Named `~/.ssh/vllm-hetero-mgmt.pem` (the old account has its own `vllm-hetero.pem`).
5. **Extra sweep rates.** Added chat rates 40, 48, 56 after the standard sweep showed the knee between 32 and 64.
6. **Unused leftovers in `054772656600`** (cost nothing, not yet deleted): SG `sg-043693c119d30a672`, key pair `vllm-hetero`, pending quota request `ea0f14374d4c4dfeb6a766774e4e254eibpim7cD`.

## Timeline

### Phase 0: Mac rehearsal (free)
- Created project files, a pandas venv, and ran the Prometheus/Grafana stack.
- **Gate 0 passed:** both Prometheus targets DOWN as expected, Grafana answered HTTP 200. Stack stopped.

### Phase 1: AWS prep (no compute cost)
- Account `331384580339`: G and VT quota 16, standard quota 40, no existing key pairs, SGs or instances.
- Created SG `sg-08b9f60fddef4efe5` in default VPC `vpc-0e36dc3a086737782`, tagged `Project=vllm-hetero`. Rules: SSH 22 from `50.47.150.84/32`; port 8000 from the SG itself.
- Created key pair `vllm-hetero`.
- **Gate 1 passed:** exactly two inbound rules, key file present, quotas sufficient.

### Phase 2: GPU node
- AMI `ami-0a042a099dfcb7057` (Deep Learning Base OSS Nvidia Driver GPU, Ubuntu 22.04), root device `/dev/sda1`.
- Launched `g6.xlarge` **`i-059b382e5ff82b2fc`**, public IP `3.137.203.157` (changes on stop/start), 100 GB gp3, DeleteOnTermination. Billing from about 21:47 UTC at $0.8048/hr.
- **Gate 2a passed:** one NVIDIA L4, 23,034 MiB, driver 595.112.07, CUDA 13.2.
- Installed `uv`, Python 3.12 venv, `vllm==0.29.0` (torch 2.13.0+cu132), pandas, tmux, jq.
- Started `vllm serve` in tmux session `server` with `--gpu-memory-utilization 0.90`.
- **KV cache (Step 2.6):** 575,440 tokens; max concurrency 140.49x at 4,096 tokens/request; 15.37 GiB available. Startup (compile + graphs + warmup) about 130 s. (`logs/gpu_kv_cache_0.90.txt`)
- **Gate 2b passed:** completion returned text; `/metrics` has `vllm:` lines. Metric name check still needed for Grafana (Phase 4).

### Phase 5.1: HuggingFace vs vLLM (GPU, 128 new tokens, identical prompts, prefix caching off)

| Batch | HF tok/s | vLLM tok/s | Speedup |
|---|---|---|---|
| 1 | 43.2 | 75.6 | 1.75x |
| 8 | 333.0 | 588.0 | 1.77x |
| 32 | 1286.6 | 2115.4 | 1.64x |

Files: `results/gpu/engine_hf.json`, `engine_vllm.json`, `logs/gpu_engine_*.log`.
Notes: vLLM batch-1 speed is about 80% of the roughly 95 tok/s bandwidth ceiling. The gap did not widen at batch 32 because the prompts are identical (no padding for HF to waste). `torch_dtype` deprecation warning in HF is harmless.

### Phase 5.2: GPU chat sweep (256 in / 128 out, 300 prompts per rate, seed 42)

| Rate | Out tok/s | p99 TTFT ms | p99 ITL ms | Meets SLO | $/1M out |
|---|---|---|---|---|---|
| 1 | 126.8 | 60 | 20.3 | yes | 1.763 |
| 2 | 253.0 | 48 | 15.3 | yes | 0.884 |
| 4 | 499.8 | 49 | 16.2 | yes | 0.447 |
| 8 | 974.0 | 54 | 17.9 | yes | 0.230 |
| 16 | 1847.7 | 64 | 20.9 | yes | 0.121 |
| 32 | 3244.9 | 110 | 38.3 | yes | 0.069 |
| 40 | 3694.9 | 144 | 47.8 | yes | 0.061 |
| 48 | 3986.5 | 175 | 56.4 | yes | 0.056 |
| 56 | 4223.9 | 207 | 64.8 | yes | 0.053 |
| 64 | 4110.6 | 1294 | 71.1 | **no** | 0.054 |
| inf | 5010.8 | 5452 | 82.4 | no | 0.045 |

**GPU chat capacity at SLO: 56 req/s, 4,224 output tok/s, $0.053 per 1M output tokens.** The knee is between 56 and 64 req/s, and the failing metric is p99 TTFT.

Caveat: at 56 req/s, 300 prompts last only about 5 s, so these high-rate runs are short and include start-up transient. The capacity figure is probably optimistic. A longer run (more prompts) near 56–64 would firm it up.

Files: `results/gpu/gpu_chat_r*.json`, `logs/sweep_gpu_chat.log`, `logs/sweep_gpu_chat2.log`, `results/analysis_gpu_chat.txt`.

### Phase 5.2: GPU rag sweep (2048 in / 256 out, 300 prompts per rate, seed 42)

| Rate | Out tok/s | p99 TTFT ms | p99 ITL ms | Meets SLO | $/1M out |
|---|---|---|---|---|---|
| 1 | 252.6 | 299 | 117.1 | no (ITL) | 0.885 |
| 2 | 496.3 | 449 | 134.1 | no (ITL) | 0.450 |
| 4 | 662.1 | 27503 | 462.4 | no | 0.338 |
| 8 | 640.6 | 68701 | 476.7 | no | 0.349 |
| 16 | 639.7 | 87603 | 474.7 | no | 0.349 |
| inf | 639.3 | 106233 | 478.7 | no | 0.350 |

**GPU rag: no tested rate meets the strict SLO.** p99 ITL is already 117 ms at 1 req/s, caused by long prefills stalling other streams (prefill interference). On TTFT alone the knee is between 2 and 4 req/s, and throughput saturates near 650 tok/s (about 2.5 req/s). Rates below 1 req/s were not tested, so the exact rate where ITL drops under 100 ms is unknown.

Files: `results/gpu/gpu_rag_r*.json`, `logs/sweep_gpu_rag.log`, `results/analysis_gpu.txt`.

### Phase 5.3: memory and concurrency tuning (GPU, chat shape)

KV cache pools from the server logs (`logs/tuning/*_server.log`):

| Config | Flags | KV pool (tokens) | Max concurrency at 4096 |
|---|---|---|---|
| baseline | `--gpu-memory-utilization 0.90` | 575,440 | 140.5x |
| A | `--gpu-memory-utilization 0.50` | 272,832 | 66.6x |
| B | 0.90, `--max-num-seqs 16` | 590,800 | 144.2x |
| C | 0.90, `--max-num-seqs 256` | 602,928 | 147.2x |

Saturated throughput (rate `inf`, second run on each server, so warm):

| Config | Out tok/s | p99 TTFT ms | p99 ITL ms |
|---|---|---|---|
| baseline | 5011 | 5452 | 82.4 |
| A (0.50) | 5000 | 5458 | 87.1 |
| B (16 seqs) | 1063 | 34272 | 16.9 |
| C (256 seqs) | 5071 | 5358 | 84.6 |

Rate 56 req/s, **warm** servers (`results/gpu/tuning/*_warm/`):

| Config | Out tok/s | p99 TTFT ms | p99 ITL ms |
|---|---|---|---|
| baseline | 4224 | 207 | 64.8 |
| A (0.50) | 4238 | 199 | 62.8 |
| C (256 seqs) | 4246 | 206 | 63.4 |

Findings:
- **Memory (A):** `--gpu-memory-utilization` sets the pool size. Halving it cut the pool by 53% (575k to 273k tokens). Capacity did not change, because a chat request needs only about 11 MB of KV cache, so even 273k tokens holds hundreds of them. A smaller pool would matter only for long-context or very high-concurrency loads.
- **Concurrency cap (B):** `--max-num-seqs 16` cut saturated throughput by 4.7x (5011 to 1063 tok/s) while the KV pool was nearly empty. The scheduler, not memory, was the limit. p99 ITL was very low (16.9 ms) because only 16 streams share each step: a latency-for-throughput trade.
- **Cap of 256 (C):** no measurable effect on this workload.
- **Methodology lesson:** the first run after a server restart includes warm-up. The original cold rate-56 runs for A and C showed p99 TTFT of about 4300 ms against 207 ms warm. Always discard or pre-run a warm-up before measuring. `scripts/tuning_warm.sh` is the corrected re-check; cold rows are kept in `results/gpu/tuning/memA|seqB|seqC/` for reference. Config B's rate-56 row was also cold and was not rerun, but its `inf` row is the clear result.

### GPU instance stopped
`i-059b382e5ff82b2fc` stopped after 5.3 (about 21:47 to about 23:00 UTC, roughly 1.3 h of compute, about $1.05). Disk keeps billing at about $0.01/hr. Public IP changes on start: update `HostName` in `~/.ssh/config`, and restart the server by hand (tmux session is gone).

### Phase 3: CPU node
- Launched `c7i.4xlarge` **`i-00a898ce223f6d69e`** (AMI `ami-0ee52710c51e09069`, Ubuntu 22.04), public IP `77.113.207.41`, private IP `172.31.37.221`, 100 GB gp3. Running from about 22:40 UTC at $0.714/hr.
- **Gate 3a passed:** flags `amx_bf16 amx_tile avx512_bf16 avx512f`; vCPUs 0-7 are distinct physical cores, 8-15 are their hyperthread siblings.
- Docker 29.1.3; image `public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.29.0` (7.8 GB).
- Server: `VLLM_CPU_KVCACHE_SPACE=12`, `VLLM_CPU_OMP_THREADS_BIND=0-6` (deviation from the runbook's 0-7: core 7 left free for the API server and benchmark client). KV pool 449,280 tokens (109.7x concurrency at 4096). Startup about 2 min (`logs/cpu_server_startup.log`).
- **Gate 3b, part 1 passed:** completions and `/metrics` work. Part 2 (GPU reaching `172.31.37.221:8000`) is pending until the GPU restarts.
- **Cold start:** the first request took 85 s (one-time kernel warm-up); the second took 4.3 s for 64 tokens, so about 15 tok/s single-stream. `scripts/cpu_sweeps.sh` sends throwaway warm-up requests before measuring.

### Phase 5.2: CPU sweeps (60 prompts per rate, seed 42)

Chat (256 in / 128 out):

| Rate | Out tok/s | p99 TTFT ms | p99 ITL ms | $/1M out |
|---|---|---|---|---|
| 0.25 | 30.8 | 420 | 244 | 6.438 |
| 0.5 | 59.4 | 288 | 164 | 3.341 |
| 1 | 110.4 | 963 | 191 | 1.796 |
| 2 | 178.2 | 476 | 298 | 1.113 |
| 4 | 254.0 | 709 | 419 | 0.781 |
| 8 | 326.1 | 1733 | 822 | 0.608 |
| 16 | 366.3 | 2781 | 1254 | 0.541 |
| 32 | 366.9 | 3839 | 1263 | 0.541 |

Rag (2048 in / 256 out):

| Rate | Out tok/s | p99 TTFT ms | p99 ITL ms | $/1M out |
|---|---|---|---|---|
| 0.1 | 24.7 | 2198 | 83 | 8.019 |
| 0.25 | 59.0 | 307 | 174 | 3.360 |
| 0.5 | 108.5 | 406 | 211 | 1.829 |
| 1 | 165.4 | 579 | 268 | 1.199 |
| 2 | 215.6 | 754 | 367 | 0.920 |
| 4 | 243.2 | 1060 | 530 | 0.816 |

Findings:
- **No CPU row meets the strict SLO** (p99 TTFT <= 1 s and p99 ITL <= 100 ms). p99 ITL is above 160 ms even at 0.25 req/s on chat.
- **TTFT-only capacity** (p99 TTFT <= 1 s): chat 4 req/s (254 tok/s, $0.78/1M); rag 2 req/s (216 tok/s, $0.92/1M).
- **Chat saturates** near 367 tok/s (about 2.9 req/s): throughput is flat from rate 16 to 32. Rag had not saturated at rate 4.
- **Versus GPU:** saturated chat throughput 367 vs 5,011 tok/s (14x); $/1M at capacity $0.78 vs $0.053 (about 15x).
- The 2198 ms TTFT at rag rate 0.1 is probably warm-up for the new prompt length (the next rates are all under 600 ms). Not rerun.
- Extra rates (chat 8/16/32, rag 2/4) were added after the runbook's rates did not reach saturation. A harmless `bind [127.0.0.1]:28000: Address already in use` warning appeared (an existing SSH session holds the tunnel port).
- Files: `results/cpu/cpu_*_r*.json`, `logs/cpu_sweeps.log`, `logs/cpu_sweeps2.log`, `results/analysis_all.txt`.

**Offload experiment rates (5.4.4):** R_CHAT = 70% of GPU chat capacity = about 39 req/s. R_BG = CPU rag capacity on the TTFT-only yardstick = 2 req/s. That is close to the GPU's own rag saturation (about 2.5 req/s), so Run A should put the GPU under real pressure.

### Phase 5.4: heterogeneous routing

**Setup.** GPU server restarted (new IP `18.221.128.70`; IP changes on every stop/start). **Gate 3b passed:** the GPU node reached `http://172.31.37.221:8000` over the private network. LiteLLM router in its own venv (`~/router`) on the GPU node, port 4000, `scripts/router.yaml` (`qwen-fast` to GPU, `qwen-batch` to CPU private IP). LiteLLM refused to start without a master key, so a random key lives in `~/router.env` on the node (not in the repo). Both routes tested and answered.

**Router hop cost (5.4.3)**: chat shape, 200 requests at 4 req/s, warmed (`scripts/router_overhead.sh`):

| Path | p50 TTFT | p99 TTFT | p50 ITL | p99 ITL | Out tok/s |
|---|---|---|---|---|---|
| direct to vLLM :8000 | 46.6 ms | 68.0 ms | 14.00 ms | 22.2 ms | 494 |
| via LiteLLM :4000 | 63.2 ms | 352.3 ms | 13.95 ms | 28.1 ms | 489 |

The router adds about 17 ms at p50 TTFT and nothing measurable to ITL or throughput. The p99 TTFT rise (+284 ms) reflects the 4-vCPU GPU node sharing CPU between router, vLLM and the benchmark client; a real deployment would give the router its own instance.

**Correction to the GPU chat capacity.** The 56 req/s figure from the 300-request sweep was an overestimate: 300 requests at 56 req/s last about 5 s, so no queue builds. The first offload attempt at 39 req/s (`results/gpu/offload_overloaded/`, `scripts/offload_overloaded_R39_BG2.sh`) showed the GPU already overloaded with chat alone (p99 TTFT 19.5 s). Sustained sweep, 1,500 requests per rate (`results/gpu/sustained/`, `logs/sustained_gpu_chat*.log`):

| Rate | Out tok/s | p50 TTFT ms | p99 TTFT ms | p99 ITL ms | Meets SLO |
|---|---|---|---|---|---|
| 16 | 2005 | 52 | 67 | 22.1 | yes |
| 20 | 2491 | 58 | 77 | 26.0 | yes |
| 24 | 2971 | 65 | 92 | 30.6 | yes |
| 28 | 3441 | 77 | 113 | 37.4 | yes |
| 32 | 3888 | 99 | 142 | 48.0 | yes |
| 34 | 4098 | 114 | 173 | 56.6 | yes |
| 36 | 4148 | 144 | 1533 | 70.5 | **no** |
| 38 | 4167 | 218 | 3666 | 72.9 | no |

**Sustained GPU chat capacity: 34 req/s, 4,098 output tok/s, $0.0546 per 1M output tokens.** Saturated throughput is about 4,150 tok/s. The short-run table in Phase 5.2 above overstates capacity near the knee; treat this table as the real one. The same short-run bias applies to the CPU numbers (60 requests per rate): Run B's background work sustained only about 0.76 rag req/s (240 requests took 314 s) against the 2 req/s read from the 60-request sweep. CPU capacity figures above are likewise upper bounds.

**Offload experiment (5.4.4)**, final run (`scripts/offload.sh`): interactive chat 1,500 requests at 24 req/s (70% of the sustained 34), background rag 120 requests at 0.75 req/s (the CPU's sustained rag throughput), seed 42 / 7. Interactive traffic goes direct to vLLM in all runs (deviation from the runbook, which sends it through the router: at that rate router plus two benchmark clients would compete for the node's 4 vCPUs). Only the background route differs.

| Run | p50 TTFT | p99 TTFT | p50 ITL | p99 ITL | Out tok/s |
|---|---|---|---|---|---|
| 0: chat alone | 65 ms | 91 ms | 19.7 ms | 30.2 ms | 2971 |
| A: background rag on GPU | 9368 ms | 24702 ms | 111.5 ms | 203.7 ms | 2009 |
| B: background rag on CPU via router | 162 ms | 289 ms | 47.9 ms | 112.0 ms | 2942 |

Background rag itself: A on GPU p50 TTFT 478 ms (p99 23,971 ms), done in 164 s; B on CPU p50 TTFT 1,952 ms (p99 58,103 ms), done in 186 s.

**Finding: offloading background RAG to the CPU node protected the GPU's interactive users.** p99 TTFT fell 85x (24.7 s to 289 ms), p99 ITL fell 1.8x (204 ms to 112 ms), and throughput returned to the chat-alone level. Even 0.75 req/s of long-prompt work, added to 70% chat load, pushed the GPU past capacity.

Caveats:
- Run B is not as good as Run 0 (p99 TTFT 289 vs 91 ms; p99 ITL 112 vs 30 ms) and misses the 100 ms ITL target. Much of that gap is probably CPU contention on the 4-vCPU GPU node (router, second benchmark client), not GPU interference. Not isolated.
- Single run per condition; no repeats, so run-to-run variance is unknown.
- Files: `results/gpu/mix_*.json`, `results/gpu/bg_*.json`, `logs/offload.log`, `logs/bg_A.log`, `logs/bg_B.log`.

## Still to do
- GPU rag sweep results; then 5.3 tuning (configs A/B/C) before stopping the GPU.
- Stop GPU, then Phase 3 (CPU node), CPU sweeps, restart GPU for Gate 3b and 5.4 (router, offload A/B).
- Phase 4 monitoring (tunnels; check metric names against `/metrics`).
- Phase 6 analysis of all nodes; Phase 7 teardown (terminate both nodes, delete SG and key pair, confirm no volumes).
- Fill in the predictions table in `README.md`.
