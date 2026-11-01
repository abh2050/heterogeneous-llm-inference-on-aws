# Plan: execute the vLLM heterogeneous GPU/CPU runbook end to end

## Context
User pasted a 7-phase runbook (Mac control node + g6.xlarge L4 GPU node + c7i.4xlarge CPU node, Qwen2.5-1.5B, LiteLLM router) answering 5 questions: HF vs vLLM speedup, SLO capacity per node, $/1M output tokens, memory/concurrency tuning, and whether CPU offload of background RAG traffic lowers GPU tail latency. User chose to run it end to end. Starting state: `~/vllm-hetero` and `~/.ssh/config` do not exist; aws, jq, docker are installed. Budget ≈ $12–14.

## Step 0 (first, before any AWS work): write the learning README
Create `README.md` in the workspace (repository root), written as a study guide for the user. For each concept: what it is, why it matters here, which phase/question measures it, and a **"predict before you measure"** prompt with a back-of-envelope number to check against results.

Sections:
1. **How LLM inference works**: tokens, autoregressive decode, prefill (compute-bound) vs decode (memory-bandwidth-bound). Roofline estimate: L4 ~300 GB/s ÷ ~3.1 GB bf16 weights ≈ ~95 tok/s ceiling per sequence at batch 1 (why batching is the whole game).
2. **KV cache**: per-token size for Qwen2.5-1.5B (28 layers × 2 KV heads × 128 dim × K+V × 2 bytes ≈ 28 KB/token, thanks to GQA); ~117 MB per 4,096-token request; predicted pool size on L4 at 0.90 vs 0.50 utilization, compared with the Step 2.6 log line.
3. **vLLM's engine ideas**: PagedAttention, continuous batching vs HF static batching + padding, prefix caching (why 5.1 turns it off), chunked prefill, scheduler limits (`--max-num-seqs`), `--gpu-memory-utilization` vs `--max-model-len` (the lab correction).
4. **Latency and throughput metrics**: TTFT, ITL, TPOT, E2E latency, output throughput, p50 vs p99, SLOs, goodput / "capacity at SLO".
5. **Load testing and queueing**: open-loop Poisson arrivals (`--request-rate`), `inf` burst, saturation knee, Little's law, why `num_requests_waiting` is the tell, `--ignore-eos` and fixed seeds.
6. **Hardware**: L4 GPU (bandwidth, tensor cores, 24 GB); Sapphire Rapids CPU (AVX-512, AMX bf16 tiles, hyperthreads vs physical cores, OMP thread pinning, memory bandwidth as the CPU decode limit); what CPU inference is actually good for.
7. **Heterogeneous serving**: workload classes (interactive vs batch/RAG), prefill interference / head-of-line blocking, routing by model name with LiteLLM, router hop overhead, when a CPU tier beats "buy more GPU".
8. **Cost**: $/hr → $/1M output tokens, why only SLO-meeting rows count, GPU vs CPU cost comparison.
9. **Observability**: Prometheus scraping, counters vs gauges vs histograms, `rate()` and `histogram_quantile()`, the key vLLM metrics.
10. **Experiment hygiene**: controlled variables, colocated benchmark client, warm-up, one engine per process, reproducibility.
11. **AWS infrastructure basics used**: vCPU quotas, security groups (SG-to-SG rules), SSH tunnels/LocalForward, tagging, DeleteOnTermination, stop vs terminate billing.
12. **Learning checklist + results table** to fill in as each Gate passes (predictions vs measured).

Stop after the README for the user to read it before any AWS step.

## How I'll execute
- Follow the runbook phase by phase; run every Gate check and stop on a failed gate.
- **Pause for user confirmation** before: creating the budget (Phase 1.1), any quota increase request, launching the GPU node (2.2), launching the CPU node (3.1), and termination (7). Report running hours/cost at each pause.
- Long-running servers start detached over SSH, e.g. `ssh gpu "tmux new -d -s server '…'"`, then poll logs/health non-interactively. Sweeps run in tmux on the node; I poll for result files.
- AWS calls: AWS MCP `run_script` where practical (per its instructions); the AWS CLI with `--profile awsnew` for steps that write local files (key-pair .pem, env.sh).
- No paid model APIs are used; only EC2 time.

## Fixes I'll apply to the runbook as I go
1. **1.1 budget**: take the account ID from `aws sts get-caller-identity` instead of the hardcoded `054772656600`.
2. **1.5 version**: look up the real newest tag in the vLLM CPU ECR repo and confirm a matching `vllm==X` exists on PyPI before writing env.sh (0.24.0 is a placeholder).
3. **1.4 SSH rule**: if my public IP changes between sessions, re-authorize port 22 (note in env.sh how).
4. **3.4 CPU threads**: the benchmark client and API server share the box with the OMP threads; bind `0-6` (leave core 7 + hyperthreads for frontend/client) if Gate 3a confirms vCPUs 0–7 are distinct cores — and note it in results.
5. **5.3 result files**: write tuning results to `~/results/tuning/` (suffix per config) so `analyze.py`'s `*_chat_r*.json` glob doesn't mix tuned configs into the capacity table.
6. **5.4.4 offload load**: R_BG = CPU rag capacity may be tiny (~0.1–0.25 rps), making Run A barely loaded. Keep the runbook's rule (fair comparison), but also record GPU KV/queue metrics during Run A and flag in the write-up if background load was too light to matter. Ensure background run outlasts the interactive run (check prompt counts vs rates).
7. **6.1 field names**: run `jq keys` first and adapt analyze.py if vLLM's result keys differ.
8. **Gate 4 fallback**: if Prometheus can't reach 127.0.0.1-bound tunnels, switch LocalForward to `0.0.0.0:` as the runbook says.

## Session order (stop/start instances between sessions to save cost)
1. Phase 0 (free): env.sh, prometheus.yml, docker-compose.yml, `analyze.py` written early; Gate 0; stack down.
2. Phases 1–4: account prep → GPU node → CPU node → monitoring. Gates 1, 2a, 2b, 3a, 3b, 4.
3. 5.1 engine compare, 5.2 sweeps (stop early when p99 TTFT > 10 s), 5.3 tuning.
4. 5.4 routing + offload A/B, Phase 6 analysis, Phase 7 teardown.

## Files created
- `~/vllm-hetero/{env.sh, docker-compose.yml, prometheus/prometheus.yml, analyze.py, results/}`
- `~/.ssh/config` (Host gpu / Host cpu), `~/.ssh/vllm-hetero.pem`
- On nodes: `~/engine_compare.py`, `~/sweep.sh`, `~/router.yaml`, `~/results/*.json`

## Verification
- Every gate in the runbook, as written.
- Gate 6: for each node × shape — capacity (req/s) at SLO, output tok/s, $/1M output tokens; offload delta in p99 TTFT/ITL (A vs B); HF vs vLLM tok/s at batch 1/8/32.
- Gate 7: tagged volume query returns `[]`, no running `Project=vllm-hetero` instances, SG and key pair deleted, local compose stack down. Budget alert left in place.
- Final summary to user includes actual billed hours and estimated cost.
