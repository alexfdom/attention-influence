from config import logger
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attention_influence_scoring import read_shard


def _load_slice(shard_path: Path, start: int, end: int | None) -> torch.Tensor:
    """Read tokens [start : end] (inclusive); if end is None, read to EOF."""
    toks = read_shard(shard_path)
    return toks[start : None if end is None else end + 1]


def get_tokens(doc_id: str) -> torch.Tensor:
    """
    Resolve a FineWeb doc_id to its exact token tensor.
    Handles both single-shard and cross-shard forms.
    """
    if "—" not in doc_id:
        stem, span = doc_id.rsplit(":", 1)
        start, end = map(int, span.split("-"))
        shard_path = Path(stem).with_suffix(".bin")
        return _load_slice(shard_path, start, end)

    first, second = doc_id.split("—")
    stem1, span1 = first.rsplit(":", 1)
    start1, end1 = map(int, span1.split("-"))

    dirpath = Path(stem1).parent

    stem2, tail = second.split(":", 1)
    shard1 = _load_slice(Path(stem1).with_suffix(".bin"), start1, end1)

    if "/" not in stem2:
        stem2 = str(dirpath / stem2)

    shard2_full = read_shard(Path(stem2).with_suffix(".bin"))
    if tail == "EOF":
        shard2 = shard2_full
    else:
        s2, e2 = map(int, tail.split("-"))
        shard2 = shard2_full[s2 : e2 + 1]

    return torch.cat((shard1, shard2))


def load_ckpt(path: str, device: torch.device) -> torch.nn.Module:
    return (
        AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )


@torch.inference_mode()
def ce_loss(model, ids: torch.Tensor, bos_id: int | None, device) -> float:
    ids = ids.to(device, torch.long)
    if bos_id is not None and ids[0] != bos_id:
        ids = torch.cat([torch.tensor([bos_id], device=device), ids])
    inp = ids[:-1].unsqueeze(0)
    tgt = ids[1:].unsqueeze(0)
    return (model(inp, labels=tgt).loss * tgt.numel()).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_id", required=True, help="format: <stem>:<start>-<end>")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--reference_model", required=True)
    ap.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="window length used by the multi-file scorer",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    logger.info(f"Loading base model from: {args.base_model}")
    base = load_ckpt(args.base_model, device)
    logger.info(f"Loading base model from: {args.base_model}")
    ref = load_ckpt(args.reference_model, device)
    tokzr = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    bos_id = tokzr.bos_token_id or tokzr.cls_token_id or None

    logger.info(f"Retrieving tokens for document ID: {args.doc_id}")
    tokens = get_tokens(args.doc_id)
    Lb = Lr = 0.0

    for i in range(0, len(tokens) - 1, args.seq_len):
        chunk = tokens[i : i + args.seq_len + 1]
        if chunk.numel() < 2:
            break
        Lb += ce_loss(base, chunk, bos_id, device)
        Lr += ce_loss(ref, chunk, bos_id, device)

    ai_score = (Lr - Lb) / max(Lb, 1e-12)

    logger.info(f"Tokens       : {tokens.numel()}")
    logger.info(f"L_base     : {Lb:.4f}")
    logger.info(f"L_ref      : {Lr:.4f}")
    logger.info(f"AI score     : {ai_score:.6f}\n")
    logger.info("Document ↓")
    logger.info(tokzr.decode(tokens))


if __name__ == "__main__":
    main()
