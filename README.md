# Attention Influence

This work builds on [AttentionInfluence: Adopting Attention Head Influence for Weak-to-Strong Pretraining Data Selection](https://arxiv.org/abs/2505.07293). Following the paper's insights, we implement an efficient **data-selection** pipeline that upgrades a language‑model’s **context-retrieval reasoning**. Each candidate document receives an **AI Score** computed by comparing the loss from the **base model** (all heads active) with the loss from a **reference model** where the retrieval heads have been masked out.

| AI Score | What it means                                                |
| -------- | ------------------------------------------------------------ |
| **> 0**  | Document relies on *re‑using* earlier tokens → copy‑heads are *helpful*. |
| **< 0**  | Document introduces mostly fresh tokens → copy‑heads are *useless or harmful*. |

> Low scores aren’t "bad"—they just don’t strengthen copy‑oriented heads. For reasoning‑centric fine‑tunes we **up-sample** *high‑AI‑score subset on top of an already high quality corpus*.

## Why use Attention Influence?

- **Low‑cost filter** – Runs on a 1–2 B model; far cheaper than full‑LLM scoring.
- **Plug‑and‑play** – No re‑training; detection and masking are pure forward-passes.
- **Best for** – In‑context retrieval, chain‑of‑thought tasks, RAG, open‑book QA.
- **Storage** – after scoring, training just reads the marked slices of the original files—no extra copies, only a small index.

## Weak → Strong scaling

At a glance: **Baseline clean data** + **top-20 % AI-scored subset** → sample from both, but give the AI-scored docs more weight.

1. **Detect retrieval heads**

   ```bash
   python generate_retrieval_probes.py
   ```

   Output → data/probes.jsonl

   ```bash
   python compute_retrieval_heads.py \
       --model hfl/chinese-llama-2-1.3b \
       --probes data/probes.jsonl \
       --top_pct 5
   ```

   Output → data/retrieval_heads.txt

2. **Mask those heads** in a copy of the same checkpoint.

   ```bash
   python make_reference_model.py \
       --base_model hfl/chinese-llama-2-1.3b \
       --retrieval_heads data/retrieval_heads.txt
   ```

   Output → reference-model/

3. **Dataset**

   Preprocesses the FineWeb dataset into a tokenized format suitable for training and evaluation. The script shards the data into smaller binary files.

   ```bash
   python fineweb.py --version "10B"
   ```

   Output → data/fineweb10B/*.bin

4. **Score documents**

   ```bash
   python attention_influence_scoring.py \
       --base_model hfl/chinese-llama-2-1.3b \
       --reference_model reference-model \
       --data 'data/fineweb10B/fineweb_train_*.bin' \
       --seq_len 4096
   ```

   Outputs:

   ​	→ data/ai_scores.tsv (just for quick inspection during the early stages of development)

   ​	→ data/ai_scores.parquet

5. **Select top‑k** (e.g. top 20 % per domain):

   ```bash
   python rank_scores.py \
       --scores data/ai_scores.tsv \
       --top_pct 20
   ```

   Outputs:

   ​	→ data/top20.tsv (just for quick inspection during the early stages of development)

   ​	→ data/top20.parquet

6. **Validate documents (Inspection Only)**

   ```bash
   python validate_docs.py \
       --doc_id data/fineweb10B/fineweb_train_000003:95137733-95138028 \
       --base_model hfl/chinese-llama-2-1.3b \
       --reference_model reference-model
   ```

7. **Large‑model pre‑train**

   Train a 7 B (or larger) model on a **mixed diet** — e.g. 70 % high-AI subset and 30 % baseline clean data.

8. **Evaluate** on reasoning benchmarks (GSM8K, MMLU, HotpotQA…).

## Limitations & Tips

- **Repetition bias** – Duplicate phrases boost the score; apply de-dup filters.
- **Cross‑domain** – Don’t mix scores across unrelated domains. Re-detect heads if your data is code, biomedical, legal, etc., generate a new probe set, rerun the head-detection step, and you’ll get fresh retrieval heads—no retraining required.

## References

- [AttentionInfluence: Adopting Attention Head Influence for Weak-to-Strong Pretraining Data Selection](https://arxiv.org/abs/2505.07293)

   ```txt
   @misc{hua2025attentioninfluence,
   title        = {AttentionInfluence: Adopting Attention Head Influence for Weak-to-Strong Pretraining Data Selection},
   author       = {Kai Hua and Steven Wu and Ge Zhang and Ke Shen},
   year         = {2025},
   eprint       = {2505.07293},
   archivePrefix= {arXiv},
   primaryClass = {cs.CL},
   url          = {https://arxiv.org/abs/2505.07293},
   }
   ```

