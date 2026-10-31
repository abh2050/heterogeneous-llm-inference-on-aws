#!/usr/bin/env bash
# Re-check of 5.3 rate-56 results on WARM servers (the first run after a restart includes warm-up).
source ~/venv/bin/activate
MODEL=Qwen/Qwen2.5-1.5B-Instruct
# Config C server is still up and warm from tuning.sh.
mkdir -p ~/results/tuning/seqC_warm
RESDIR=~/results/tuning/seqC_warm ~/sweep.sh gpu chat 56 2>&1 | tee ~/results/tuning/seqC_warm/sweep.log
# Config A: restart, warm up with one throwaway run, then measure.
tmux kill-session -t server 2>/dev/null; sleep 10
mkdir -p ~/results/tuning/memA_warm /tmp/warmup
tmux new -d -s server "vllm serve $MODEL --dtype bfloat16 --max-model-len 4096 --port 8000 --gpu-memory-utilization 0.50 2>&1 | tee ~/results/tuning/memA_warm/server.log"
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
RESDIR=/tmp/warmup ~/sweep.sh gpu chat 56 >/dev/null 2>&1
RESDIR=~/results/tuning/memA_warm ~/sweep.sh gpu chat 56 2>&1 | tee ~/results/tuning/memA_warm/sweep.log
echo WARM_DONE
