import json, glob
import pandas as pd

PRICE_PER_HOUR = {"gpu": 0.8048, "cpu": 0.7140}
SLO_TTFT_MS, SLO_ITL_MS = 1000, 100

rows = []
# Tuning runs (5.3) live in results/*/tuning/ and are excluded by this glob on purpose.
for path in glob.glob("results/*/*_chat_r*.json") + glob.glob("results/*/*_rag_r*.json"):
    name = path.split("/")[-1].removesuffix(".json")
    node, shape, rate = name.split("_")[:3]
    d = json.load(open(path))
    tps = d["output_throughput"]
    rows.append({
        "node": node, "shape": shape, "rate": rate[1:],
        "out_tok_s": round(tps, 1),
        "p99_ttft_ms": round(d["p99_ttft_ms"]),
        "p99_itl_ms": round(d["p99_itl_ms"], 1),
        "meets_slo": d["p99_ttft_ms"] <= SLO_TTFT_MS and d["p99_itl_ms"] <= SLO_ITL_MS,
        "usd_per_1M_out": round(PRICE_PER_HOUR[node] / (tps * 3600) * 1e6, 3),
    })

df = pd.DataFrame(rows).sort_values(["node", "shape", "out_tok_s"])
print(df.to_string(index=False))
print("\nCapacity at SLO (highest throughput row that meets the SLO):")
print(df[df.meets_slo].groupby(["node", "shape"]).tail(1).to_string(index=False))
