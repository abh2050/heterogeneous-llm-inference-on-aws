#!/usr/bin/env bash
set -euo pipefail
NODE=$1; SHAPE=$2; shift 2; RATES="$@"
MODEL=Qwen/Qwen2.5-1.5B-Instruct
case $SHAPE in
  chat) IN=256;  OUT=128 ;;
  rag)  IN=2048; OUT=256 ;;
esac
if [ "$NODE" = gpu ]; then
  N=300; RES=${RESDIR:-$HOME/results}
  BENCH="vllm bench serve"
else
  N=60; RES=/results
  BENCH="docker run --rm --network host -v $HOME/results:/results \
         -v $HOME/.cache/huggingface:/root/.cache/huggingface --entrypoint vllm $CPU_IMAGE bench serve"
fi
for R in $RATES; do
  echo "=== $NODE $SHAPE rate=$R ==="
  $BENCH --backend vllm --base-url http://localhost:8000 --model $MODEL \
    --dataset-name random --random-input-len $IN --random-output-len $OUT \
    --num-prompts $N --request-rate $R --seed 42 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir $RES --result-filename ${NODE}_${SHAPE}_r${R}.json
done
