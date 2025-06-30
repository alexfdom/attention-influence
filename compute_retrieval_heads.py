from config import (
    logger,
    TOK,
    FILE_PATH_RETRIEVAL_HEADS
)
import math
import json
import argparse
import torch
import os
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict


def flat_index(layer_idx: int, head_idx: int, heads: int) -> int:
    """Convert (layer, head) into a single number 0 … L·H-1."""
    return layer_idx * heads + head_idx


@torch.no_grad()
def handle(prompt: str, gold: str, heads: int):
    full_ids = TOK(prompt + gold, return_tensors="pt").to(args.device).input_ids[0]
    prompt_ids = TOK(prompt, return_tensors="pt").to(args.device).input_ids[0]
    prompt_len, sequence_len = len(prompt_ids), len(full_ids)

    # Map token ID → positions in prompt (excluding answer span)
    token_positions_in_prompt = defaultdict(list)
    for pos, token_id in enumerate(full_ids[:prompt_len]):
        token_positions_in_prompt[token_id].append(pos)

    # How the model attends once the answer is already present in the context (we’re evaluating copy-and-paste ability while the model is generating the answer)
    output = model(full_ids[None], use_cache=False)
    all_attn = output.attentions
    for t in range(prompt_len, sequence_len):
        answer_token_id = int(full_ids[t])
        # protective wrapper: skips any answer-side token that the tokenizer has never produced anywhere in the prompt portion
        if answer_token_id not in token_positions_in_prompt:
            continue

        for layer_idx, layer_attn in enumerate(
            all_attn
        ):  # all_attn[layer_idx].shape == (1, H, T, T) — weight = all_attn[l][batch_idx, head_idx, query_pos=t, key_pos=k] == how much head h in layer l attends to token k while computing the representation for token t
            layer_attn = all_attn[layer_idx]  # (1, H, T, T)
            prompt_slice = layer_attn[0, :, t, :prompt_len]  # (HEADS , prompt_len)
            max_prompt_pos_per_head = prompt_slice.argmax(-1)  # (H ,)
            for head_idx in range(heads):
                copy_attempts[flat_index(layer_idx, head_idx, heads)] += 1

                attended_pos = int(max_prompt_pos_per_head[head_idx])
                attended_token_id = int(full_ids[attended_pos])

                if attended_token_id == answer_token_id:
                    copy_hits[flat_index(layer_idx, head_idx, heads)] += 1


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--model", required=True)
    argp.add_argument("--probes", default="data/probes.jsonl")
    argp.add_argument("--top_pct", type=float, default=5)
    argp.add_argument("--device", default="cuda")
    args = argp.parse_args()

    logger.info("Loading model from %s", args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            output_attentions=True,
        )
        .to(args.device)
        .eval()
    )

    LAYERS, HEADS = model.config.num_hidden_layers, model.config.num_attention_heads
    total = LAYERS * HEADS
    logger.info("Model loaded with %d layers × %d heads = %d total attention heads", LAYERS, HEADS, total)

    copy_hits = torch.zeros(total, dtype=torch.long)
    copy_attempts = torch.zeros(total, dtype=torch.long)

    logger.info("Reading probes from %s", args.probes)
    with open(args.probes, encoding="utf-8") as probe_file:
        for probe_line in tqdm(probe_file, total=800, desc="probes"):
            probe_json = json.loads(probe_line)
            demonstration_text = probe_json["prompt"]
            gold_answer_string = probe_json["answer"]
            handle(demonstration_text, gold_answer_string, HEADS)

    logger.info("Finished processing all probes. Computing recall scores per head.")
    
    recall_per_head = copy_hits.to(torch.float32) / (copy_attempts + 1e-12)
    top_k = math.ceil(total * args.top_pct / 100)
    top_ids = torch.topk(recall_per_head, k=top_k).indices.tolist()

    output_lines = [
        f"{layer},{head}"
        for flat_id in top_ids
        for layer, head in [divmod(flat_id, HEADS)]
    ]

    with open(FILE_PATH_RETRIEVAL_HEADS, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"✔ wrote {FILE_PATH_RETRIEVAL_HEADS} ({len(output_lines)} heads — top {args.top_pct}% of {total} model heads)")
