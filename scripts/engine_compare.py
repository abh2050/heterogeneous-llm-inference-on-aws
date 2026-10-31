import sys, time, json

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BATCHES = [1, 8, 32]
NEW_TOKENS = 128
PROMPT = "Explain how a heat exchanger works to a new process engineer."

def run_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    results = []
    for b in BATCHES:
        inputs = tok([PROMPT] * b, return_tensors="pt", padding=True).to("cuda")
        model.generate(**inputs, max_new_tokens=8, do_sample=False)          # warm up
        torch.cuda.synchronize(); start = time.time()
        model.generate(**inputs, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize(); seconds = time.time() - start
        results.append({"engine": "hf", "batch": b, "seconds": seconds,
                        "tok_per_s": b * NEW_TOKENS / seconds})
        print(results[-1])
    return results

def run_vllm():
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_prefix_caching=False)
    params = SamplingParams(temperature=0, max_tokens=NEW_TOKENS, ignore_eos=True)
    llm.generate([PROMPT], SamplingParams(max_tokens=8), use_tqdm=False)    # warm up
    results = []
    for b in BATCHES:
        start = time.time()
        llm.generate([PROMPT] * b, params, use_tqdm=False)
        seconds = time.time() - start
        results.append({"engine": "vllm", "batch": b, "seconds": seconds,
                        "tok_per_s": b * NEW_TOKENS / seconds})
        print(results[-1])
    return results

if __name__ == "__main__":
    engine = sys.argv[1]
    results = run_hf() if engine == "hf" else run_vllm()
    with open(f"/home/ubuntu/results/engine_{engine}.json", "w") as f:
        json.dump(results, f, indent=2)
