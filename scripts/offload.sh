#!/usr/bin/env bash
# 5.4.4 offload experiment. Run on the GPU node. R_CHAT ~ 70% of sustained GPU chat capacity (34 req/s), R_BG = CPU sustained rag throughput (~0.76 req/s).
# Interactive traffic goes direct to vLLM (:8000) in every run, so the router and client do not compete for the node's 4 vCPUs.
# Run 0: chat alone | Run A: + background rag on the GPU | Run B: + background rag on the CPU via router (qwen-batch).
source ~/venv/bin/activate; source ~/router.env
M=Qwen/Qwen2.5-1.5B-Instruct
R_CHAT=${R_CHAT:-24}; R_BG=${R_BG:-0.75}
CHAT="--backend openai --tokenizer $M --dataset-name random --random-input-len 256 --random-output-len 128 --num-prompts 1500 --request-rate $R_CHAT --seed 42 --ignore-eos --percentile-metrics ttft,itl --metric-percentiles 50,99"
BG="--backend openai --tokenizer $M --dataset-name random --random-input-len 2048 --random-output-len 256 --num-prompts 120 --request-rate $R_BG --seed 7 --ignore-eos --percentile-metrics ttft,itl --metric-percentiles 50,99"
chat() { vllm bench serve $CHAT --base-url http://localhost:8000 --model $M --save-result --result-dir $HOME/results --result-filename $1; }

echo "== Run 0: interactive alone"; chat mix_0_alone.json
sleep 20
echo "== Run A: background on GPU"
vllm bench serve $BG --base-url http://localhost:8000 --model $M --save-result --result-dir $HOME/results --result-filename bg_A_gpu.json > ~/bg_A.log 2>&1 &
BGPID=$!; sleep 30; chat mix_A_allgpu.json; wait $BGPID
sleep 20
echo "== Run B: background on CPU via router"
vllm bench serve $BG --base-url http://localhost:4000 --model qwen-batch --save-result --result-dir $HOME/results --result-filename bg_B_cpu.json > ~/bg_B.log 2>&1 &
BGPID=$!; sleep 30; chat mix_B_offload.json; wait $BGPID
echo OFFLOAD_DONE
