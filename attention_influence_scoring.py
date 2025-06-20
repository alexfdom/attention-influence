import argparse
from pathlib import Path
import os
from typing import Union, Generator, Tuple, Optional
from numpy.typing import NDArray
import numpy as np
import glob
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
import polars as pl

MAGIC, VERSION = 20240520, 1
HEADER_BYTES = 256 * 4


def load_ckpt(path: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    return model.to(device).eval()


def read_shard(path: Union[str, Path]) -> torch.Tensor:
    """Read one .bin shard written by `write_datafile` and return uint16 tokens."""
    path = Path(path)

    with path.open("rb", buffering=0) as f:
        header_raw = f.read(HEADER_BYTES)

    header: NDArray[np.int32] = np.frombuffer(header_raw, dtype="<i4", count=256)
    magic, version, n_tokens = map(int, header[:3])
    assert (magic, version) == (MAGIC, VERSION), f"{path}: bad header"

    with path.open("rb", buffering=0) as f:
        f.seek(HEADER_BYTES)
        payload = f.read(n_tokens * 2)

    tok_arr: NDArray[np.uint16] = np.frombuffer(payload, dtype="<u2", count=n_tokens)
    assert tok_arr.size == n_tokens

    return torch.from_numpy(tok_arr).contiguous()


def gen_docs(
    pattern: str,
    eot_id: int,
    *,
    with_counter: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Yield (doc_id, tokens) where each sequence ends with `eot_id`.

    ─ ID format ───────────────────────────────────────────────────────────
      single-shard :  <stem>:<start>-<end>
      cross-shard  :  <stem1>:<start>-<end>—<stem2>:0-<end>
                      (the doc always starts at index 0 in the second shard)
    """
    tail: Optional[torch.Tensor] = None  # unfinished doc carried forward
    tail_stem: Optional[str] = None  # first shard’s stem
    tail_start: Optional[int] = None  # start idx in first shard

    num_toks_seen = 0

    for shard_path in sorted(glob.glob(pattern)):
        shard_tokens = read_shard(shard_path)
        if shard_tokens.numel() == 0:
            continue
        stem = os.path.splitext(os.path.basename(shard_path))[0]

        if tail is not None:
            offset = len(tail)
            if tail.device != shard_tokens.device:
                tail = tail.to(shard_tokens.device)
            shard_tokens = torch.cat((tail, shard_tokens))
        else:
            offset = 0

        eot_positions = torch.where(shard_tokens == eot_id)[0].tolist()
        pos = 0

        for nxt in eot_positions:
            if offset == 0:  # whole doc inside this shard
                doc_id = f"{stem}:{pos}-{nxt}"
            else:  # spans two shards
                end1 = tail_start + offset - 1
                end2 = nxt - offset
                doc_id = f"{tail_stem}:{tail_start}-{end1}—{stem}:0-{end2}"
                # Reset cross-shard bookkeeping
                tail = tail_stem = tail_start = None
                offset = 0

            tokens_slice = shard_tokens[pos : nxt + 1]
            num_toks_seen += len(tokens_slice)

            if with_counter:
                yield doc_id, tokens_slice, num_toks_seen
            else:
                yield doc_id, tokens_slice

            pos = nxt + 1

        # Anything after the last EOT becomes the next shard’s tail
        if pos < len(shard_tokens):
            tail = shard_tokens[pos:]
            if tail_stem is None:
                tail_stem = stem
                tail_start = pos
        else:
            tail = tail_stem = tail_start = None

    if tail is not None and len(tail):
        end1 = tail_start + len(tail) - 1
        doc_id = f"{tail_stem}:{tail_start}-{end1}—{tail_stem}:EOF"
        num_toks_seen += len(tail)
        if with_counter:
            yield doc_id, tail, num_toks_seen
        else:
            yield doc_id, tail


def ce_loss(model, ids: torch.Tensor) -> torch.Tensor:
    if bos_id is not None and ids[0] != bos_id:
        bos = torch.tensor([bos_id], device=ids.device, dtype=torch.long)
        ids = torch.cat([bos, ids])
    inp = ids[:-1].unsqueeze(0)
    tgt = ids[1:].unsqueeze(0)
    with torch.no_grad():
        return model(inp, labels=tgt).loss * tgt.numel()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--reference_model", required=True)
    p.add_argument("--data", required=True, help="glob for *.bin shards")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="data/ai_scores.tsv")
    p.add_argument(
        "--max_tokens",
        type=lambda s: int(s.replace("_", "")),
        default=None,
        help="stop after this many total tokens seen (underscores allowed, e.g. 10_000_000)",
    )
    args = p.parse_args()

    rank = int(os.getenv("RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

    device = torch.device(args.device)
    base_model = load_ckpt(args.base_model, device)
    reference_model = load_ckpt(args.reference_model, device)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-llama-2-1.3b", use_fast=True)
    bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or None

    EOT = tokenizer.eos_token_id

    # scoring loop
    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.out, "w", encoding="utf-8")
        out_f.write("doc_id\tL_base\tL_ref\tAI_Score\ttotal_tokens_seen\n")

        rows = []
    else:
        out_f = None

    for doc_id, tok_buf, num_toks_seen in tqdm(
        gen_docs(args.data, EOT, with_counter=True), disable=rank != 0
    ):
        Lb = Lr = 0.0
        for i in range(0, len(tok_buf) - 1, args.seq_len):
            chunk = tok_buf[i : i + args.seq_len + 1].to(device, torch.long)
            if len(chunk) < 2:
                break
            Lb += ce_loss(base_model, chunk).item()
            Lr += ce_loss(reference_model, chunk).item()

        if world > 1:
            tensor = torch.tensor([Lb, Lr], device=device)
            dist.all_reduce(tensor)
            Lb, Lr = tensor.tolist()

        attention_influence_score = (Lr - Lb) / max(Lb, 1e-12)
        if rank == 0:
            out_f.write(
                f"{doc_id}\t{Lb:.4f}\t{Lr:.4f}\t{attention_influence_score:.6f}\t{num_toks_seen}\n"
            )
            row = {
                "doc_id": doc_id,
                "L_base": round(Lb, 4),
                "L_ref": round(Lr, 4),
                "AI_Score": round(attention_influence_score, 6),
                "total_tokens_seen": num_toks_seen,
            }
            rows.append(row)

            if args.max_tokens is not None and num_toks_seen >= args.max_tokens:
                break

    if rank == 0:
        out_f.close()
        parquet_path = Path(args.out).with_suffix(".parquet")
        pl.DataFrame(rows).write_parquet(parquet_path)
        print(f"✓ wrote {parquet_path} (and TSV)")
    if world > 1:
        dist.barrier()
