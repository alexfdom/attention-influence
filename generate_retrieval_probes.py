from __future__ import annotations
import os
import random
import uuid
from transformers import AutoTokenizer
from datasets import load_dataset
import pysbd
import multiprocessing as mp
from tqdm import tqdm
import json

random.seed(42)
tok = AutoTokenizer.from_pretrained(
    "hfl/chinese-llama-2-1.3b", use_fast=True, trust_remote_code=True
)
EOS = tok.eos_token_id
assert EOS < 2**16

MAX_TOKENS_PER_PROBE = 4_096
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PATH = os.path.join(OUTPUT_DIR, "probes.jsonl")

def n_tokens(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))

def process_doc(doc):
    try:
        seg = pysbd.Segmenter(language="en", clean=True)
        selected_sentences = []
        for sent in seg.segment(doc["text"]):
            s = sent.strip()
            n_tok = n_tokens(s)
            if 5 <= n_tok <= 30:
                selected_sentences.append(s)
        return selected_sentences
    except Exception:
        return []

def rand_key() -> str:
    """Return a 32-char mixed-alnum key."""
    return uuid.uuid4().hex + uuid.uuid4().hex[:16]


def build_context(k: int) -> dict[str, str]:
    """Return a dict with k unique random keys and sentences."""
    keys = {rand_key() for _ in range(k)}
    vals = random.sample(sentence_pool, k)
    return dict(zip(keys, vals))


def make_probe(probe_id: int) -> dict:
    """Assemble one probe; fall back to smaller `k` until <= 4 096 token budget."""

    for k in range(60, 29, -5):
        ctx = build_context(k)
        keys = list(ctx.keys())
        demo_keys = random.sample(keys, 4)  # 3 demos + 1 query
        q1, q2, q3, q_query = demo_keys
        prompt = (
            "Please extract the value corresponding to the specified key from the "
            "following JSON object. Output only the value of the corresponding key "
            "and nothing else. The JSON data is as follows:\n"
            f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
            # --- 3 in-context demonstrations (NO “Question/Answer:” prefixes) ---
            f"{q1}\n{ctx[q1]}\n"
            f"{q2}\n{ctx[q2]}\n"
            f"{q3}\n{ctx[q3]}\n"
            # --- Query: the actual test case — the key for which the model must retrieve a value ---
            f"{q_query}\n"
            "answer:"
        )
        # token budget check
        if n_tokens(prompt) + 1 <= MAX_TOKENS_PER_PROBE:
            return {"id": probe_id, "prompt": prompt, "answer": ctx[q_query]}

    raise RuntimeError("Could not fit a probe under the 4 096-token cap.")


if __name__ == "__main__":
    print("Loading WebText sentences …")
    web = load_dataset("stas/openwebtext-10k", split="train")
    docs = list(web)
    print(f"Loaded {len(docs)} documents.")

    seg = pysbd.Segmenter(language="en", clean=False)
    sentence_pool: list[str] = []

    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        results = list(tqdm(pool.imap(process_doc, docs, chunksize=100), total=len(docs)))

    sentence_pool = [s for sublist in results for s in sublist]
    print(f"Collected {len(sentence_pool):,} candidate sentences.")

    with open(f"{FILE_PATH}", "w", encoding="utf-8") as f:
        for i in tqdm(range(800), desc="Generating probes"):
            probe = make_probe(i)
            f.write(json.dumps(probe, ensure_ascii=False) + "\n")

    print("Done → probes.jsonl (800 lines)")
