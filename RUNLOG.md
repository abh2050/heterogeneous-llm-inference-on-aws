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

### Phase 5.2: GPU rag sweep (2048 in / 256 out): in progress
Rates 1 2 4 8 16 inf, tmux session `sweep2`, log `~/sweep_gpu_rag.log` on the node.

## Still to do
- GPU rag sweep results; then 5.3 tuning (configs A/B/C) before stopping the GPU.
- Stop GPU, then Phase 3 (CPU node), CPU sweeps, restart GPU for Gate 3b and 5.4 (router, offload A/B).
- Phase 4 monitoring (tunnels; check metric names against `/metrics`).
- Phase 6 analysis of all nodes; Phase 7 teardown (terminate both nodes, delete SG and key pair, confirm no volumes).
- Fill in the predictions table in `README.md`.
