#!/usr/bin/env bash
# 5.4.3: router hop cost. Same load direct (port 8000) and through LiteLLM (port 4000). Run on the GPU node.
source ~/venv/bin/activate; source ~/router.env
COMMON="--backend openai --tokenizer Qwen/Qwen2.5-1.5B-Instruct --dataset-name random --random-input-len 256 --random-output-len 128 --seed 42 --ignore-eos --percentile-metrics ttft,itl --metric-percentiles 50,99"
bench() { # $1=base-url $2=model $3=num-prompts $4=result-name
  vllm bench serve $COMMON --base-url $1 --model $2 --num-prompts $3 --request-rate 4 \
    ${4:+--save-result --result-dir $HOME/results --result-filename $4}
}
echo "warm-up direct"; bench http://localhost:8000 Qwen/Qwen2.5-1.5B-Instruct 40 >/dev/null 2>&1
echo "warm-up router"; bench http://localhost:4000 qwen-fast 40 >/dev/null 2>&1
echo "direct";  bench http://localhost:8000 Qwen/Qwen2.5-1.5B-Instruct 200 router_direct.json
echo "router";  bench http://localhost:4000 qwen-fast 200 router_overhead.json
echo OVERHEAD_DONE
