import argparse
import math
from pathlib import Path
import polars as pl

p = argparse.ArgumentParser()
p.add_argument("--scores", default="data/ai_scores.tsv")
p.add_argument("--top_pct", type=float, default=20.0)
args = p.parse_args()

df = pl.read_csv(args.scores, separator="\t")

df = df.with_columns(
    pl.col("total_tokens_seen")
    .diff()
    .fill_null(pl.col("total_tokens_seen"))
    .cast(pl.Int64)
    .alias("tok_len")
)

k = math.ceil(df.height * args.top_pct / 100)
top_k = (
    df.sort("AI_Score", descending=True)
    .head(k)
    .select(["doc_id", "AI_Score", "tok_len"])
)

tokens_total = df["total_tokens_seen"][-1]
tokens_in_top = int(top_k["tok_len"].sum())
pct_tokens = tokens_in_top / tokens_total * 100

out_path = Path(f"data/top{int(args.top_pct)}.tsv")
out_path.parent.mkdir(parents=True, exist_ok=True)
top_k.write_csv(out_path, separator="\t")
top_k.write_parquet("data/top20.parquet")

print(
    f"✓ wrote {out_path}  "
    f"Top {args.top_pct}% Attention Influence Score "
    f"(k={k} docs, {tokens_in_top:,} tokens ≈ {pct_tokens:.2f}% "
    f"of {tokens_total:,} total tokens)"
)
