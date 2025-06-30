from __future__ import annotations
from config import (
    logger,
    TOK,
    EOS,
    MAX_TOKENS_PER_PROBE,
    NUM_PROBES,
    HASH_KEY_LENGHT,
    FILE_PATH_PROBES,
)
import string
import random
from transformers import AutoTokenizer
from datasets import load_dataset
import pysbd
from dataclasses import dataclass
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import json

logger.info("Tokenizer initialized (EOS token ID: %d)", EOS)
logger.info("Max tokens per probe: %d", MAX_TOKENS_PER_PROBE)
logger.info("Output file will be saved to: %s", FILE_PATH_PROBES)

def n_tokens(text: str) -> int:
    return len(TOK.encode(text, add_special_tokens=False))


def process_doc(doc: dict, seg: pysbd.Segmenter) -> list[str]:
    try:
        selected_sentences = []
        for sent in seg.segment(doc["text"]):
            s = sent.strip()
            n_tok = n_tokens(s)
            if 5 <= n_tok <= 30:
                selected_sentences.append(s)
        return selected_sentences
    except Exception as e:
        logger.warning("Error processing document: %s", e)
        return []


def rand_key(length: int = 32) -> str:
    """Return a random mixed-alphanumeric key of the given length."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def _build_context(k: int, pool: list[str]) -> dict[str, str]:
    """Return a dict with k unique random keys and sentences."""
    keys = {rand_key(HASH_KEY_LENGHT) for _ in range(k)}
    vals = random.sample(pool, k)
    return dict(zip(keys, vals))


@dataclass
class Probe:
    id: int
    prompt: str
    answer: str


def make_probe(probe_id: int, pool: list[str]) -> Probe:
    """Assemble one probe; fall back to smaller `k` until <= 4 096 token budget."""

    for k in range(60, 29, -5):
        ctx = _build_context(k, pool)
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
            return Probe(id=probe_id, prompt=prompt, answer=ctx[q_query])
        
    logger.error("Probe %d could not fit within the token budget.", probe_id)
    raise RuntimeError("Could not fit a probe under the 4 096-token cap.")


if __name__ == "__main__":
    logger.info("Loading WebText sentences …")
    web = load_dataset("stas/openwebtext-10k", split="train")
    docs = list(web)
    logger.info("Loaded %d documents.", len(docs))

    SEG = pysbd.Segmenter(language="en", clean=False)
    sentence_pool: list[str] = []

    logger.info("Segmenting sentences using multiprocessing …")
    worker_task = partial(process_doc, seg=SEG)
    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        results = list(
            tqdm(pool.imap(worker_task, docs, chunksize=100), total=len(docs))
        )

    sentence_pool = [s for sublist in results for s in sublist]
    logger.info("Collected %d candidate sentences.", len(sentence_pool))

    logger.info("Generating %d probes …", NUM_PROBES)
    with open(FILE_PATH_PROBES, "w", encoding="utf-8") as f:
        for i in tqdm(range(NUM_PROBES), desc="Generating probes"):
            probe = make_probe(i, sentence_pool)
            f.write(json.dumps(probe.__dict__, ensure_ascii=False) + "\n")

    logger.info("Done → probes.jsonl (%d lines)", NUM_PROBES)
