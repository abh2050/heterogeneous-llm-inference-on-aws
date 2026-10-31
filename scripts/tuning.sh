#!/usr/bin/env bash
# 5.3: restart the GPU server under three configs, sweep chat at 56 req/s (baseline capacity) and inf.
# Results go to ~/results/tuning/<config>/ so analyze.py's glob does not mix them with the baseline.
source ~/venv/bin/activate
MODEL=Qwen/Qwen2.5-1.5B-Instruct

run() {
  name=$1; shift
  tmux kill-session -t server 2>/dev/null; sleep 10
  mkdir -p ~/results/tuning/$name
  tmux new -d -s server "vllm serve $MODEL --dtype bfloat16 --max-model-len 4096 --port 8000 $* 2>&1 | tee ~/results/tuning/$name/server.log"
  until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
  RESDIR=~/results/tuning/$name ~/sweep.sh gpu chat 56 inf 2>&1 | tee ~/results/tuning/$name/sweep.log
}

run memA --gpu-memory-utilization 0.50
run seqB --gpu-memory-utilization 0.90 --max-num-seqs 16
run seqC --gpu-memory-utilization 0.90 --max-num-seqs 256
echo TUNING_DONE
