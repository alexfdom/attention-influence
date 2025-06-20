from __future__ import annotations
from typing import Iterable, List, Tuple
from transformers import PreTrainedModel
import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_retrieval_heads(filename: str) -> List[Tuple[int, int]]:
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    retrieval_heads = [
        tuple(map(int, re.split(r"[ ,]+", line.strip())))
        for line in lines
        if line.strip()
    ]
    return retrieval_heads


def zero_out_heads(
    model: PreTrainedModel, retrieval_heads: Iterable[Tuple[int, int]]
) -> None:
    first_layer = model.model.layers[0]
    if not hasattr(first_layer, "self_attn"):
        raise ValueError("Unexpected model structure: no `.self_attn` found.")

    num_heads = model.config.num_attention_heads
    head_dim = first_layer.self_attn.head_dim

    for layer_idx, head_idx in retrieval_heads:
        if head_idx >= num_heads:
            raise IndexError(
                f"Head index {head_idx} exceeds number of heads ({num_heads}) in layer {layer_idx}"
            )

        attn = model.model.layers[layer_idx].self_attn

        row_slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)

        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            with torch.no_grad():
                proj.weight[row_slice].zero_()


def patch_generation_config(gen_cfg):
    """
    Ensure the GenerationConfig no longer contains sampling-only
    parameters when do_sample==False, so save_pretrained() passes validation.
    """
    if gen_cfg.do_sample is False:
        for fld in ("temperature", "top_p", "top_k", "top_n"):
            if hasattr(gen_cfg, fld) and getattr(gen_cfg, fld) is not None:
                setattr(gen_cfg, fld, None)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--base_model", required=True, help="HF model id or local path")
    cli.add_argument(
        "--retrieval_heads",
        default="data/retrieval_heads.txt",
        help="file with one 'layer,head' per line",
    )
    args = cli.parse_args()

    retrieval_heads = load_retrieval_heads(args.retrieval_heads)

    print(
        f"Masking {len(retrieval_heads)} retrieval heads → {retrieval_heads[:8]}{' …' if len(retrieval_heads) > 8 else ''}"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    patch_generation_config(base_model.generation_config)
    base_model.save_pretrained("base-model", safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained("base-model")
    print("✔ Base model written to: base-model")

    zero_out_heads(base_model, retrieval_heads) # base_model mutated in-place → becomes the masked reference model 

    patch_generation_config(base_model.generation_config)
    base_model.save_pretrained("reference-model", safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained("reference-model")
    print("✔ Reference model (masked heads) written to: reference-model")
