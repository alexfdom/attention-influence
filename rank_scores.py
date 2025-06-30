from config import logger
import argparse
import math
from pathlib import Path
import polars as pl

p = argparse.ArgumentParser()
p.add_argument("--scores", default="data/ai_scores.tsv")
p.add_argument("--top_pct", type=float, default=20.0)
p.add_argument("--bottom_pct", type=float, default=None)
args = p.parse_args()

logger.info(f"Reading scores from: {args.scores}")
df = pl.read_csv(args.scores, separator="\t")

df = df.with_columns(
    pl.col("total_tokens_seen")
    .diff()
    .fill_null(pl.col("total_tokens_seen"))
    .cast(pl.Int64)
    .alias("tok_len")
)

k = math.ceil(df.height * (args.bottom_pct if args.bottom_pct is not None else args.top_pct) / 100)

if args.bottom_pct is None:
    logger.info(f"Selecting top {args.top_pct:.1f}% documents by AI_Score (k={k})")
    selected_k = (
        df.sort("AI_Score", descending=True)
        .head(k)
        .select(["doc_id", "AI_Score", "tok_len"])
    )
    percent = args.top_pct
    selected = "Top"
    file_prefix = f"top{int(percent)}"
else:
    logger.info(f"Selecting bottom {args.bottom_pct:.1f}% documents by AI_Score (k={k})")
    selected_k = (
        df.sort("AI_Score", descending=False)
        .head(k)
        .select(["doc_id", "AI_Score", "tok_len"])
    )
    percent = args.bottom_pct
    selected = "Bottom"
    file_prefix = f"bottom{int(percent)}"

tokens_total = df["total_tokens_seen"][-1]
tokens_in_selected = int(selected_k["tok_len"].sum())
pct_tokens = tokens_in_selected / tokens_total * 100

out_path = Path(f"data/{file_prefix}.tsv")
out_path.parent.mkdir(parents=True, exist_ok=True)
selected_k.write_csv(out_path, separator="\t")
selected_k.write_parquet(f"data/{file_prefix}.parquet")

logger.info(
    f"✓ wrote {out_path}  "
    f"{selected} {percent}% Attention Influence Score "
    f"(k={k} docs, {tokens_in_selected:,} tokens ≈ {pct_tokens:.2f}% "
    f"of {tokens_total:,} total tokens)"
)

