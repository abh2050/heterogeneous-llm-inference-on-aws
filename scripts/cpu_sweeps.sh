#!/usr/bin/env bash
# CPU-node sweeps (5.2). The first request after server start takes ~85 s (cold kernels), so each shape
# gets a throwaway warm-up run before measuring. Run on the CPU node.
export CPU_IMAGE=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.29.0
MODEL=Qwen/Qwen2.5-1.5B-Instruct
warm() {  # $1=input len $2=output len
  docker run --rm --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface --entrypoint vllm $CPU_IMAGE bench serve \
    --backend vllm --base-url http://localhost:8000 --model $MODEL --dataset-name random \
    --random-input-len $1 --random-output-len $2 --num-prompts 6 --request-rate 1 --seed 1 --ignore-eos >/dev/null 2>&1
}
echo "warm-up chat"; warm 256 128
~/sweep.sh cpu chat 0.25 0.5 1 2 4
echo "warm-up rag"; warm 2048 256
~/sweep.sh cpu rag 0.1 0.25 0.5 1
echo CPU_SWEEPS_DONE
