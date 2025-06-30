#!/bin/bash
set -e

echo "--> Step 1: Detecting retrieval heads..."
echo "--> --> Step 1.1: Generating retrieval probes..."
python generate_retrieval_probes.py

echo "--> --> Step 1.2: Computing retrieval heads..."
python compute_retrieval_heads.py \
    --model hfl/chinese-llama-2-1.3b \
    --probes data/probes.jsonl \
    --top_pct 5

echo "--> Step 2: Masking those heads in a copy of the same checkpoint..."
python make_reference_model.py \
    --base_model hfl/chinese-llama-2-1.3b \
    --retrieval_heads data/retrieval_heads.txt

echo "--> Step 3: Preprocesses the FineWeb dataset into a tokenized format..."
if ls data/fineweb10B/*.bin 1> /dev/null 2>&1; then
  echo "✅ Skipping: tokenized files already present. ⚠️ Be aware of the tokenizer used."
else
  python fineweb.py --version "10B"
fi

echo "--> Step 4: Scoring documents..."
MAX_TOKENS=100_000
python attention_influence_scoring.py \
    --base_model hfl/chinese-llama-2-1.3b \
    --reference_model reference-model \
    --data 'data/fineweb10B/fineweb_train_*.bin' \
    --seq_len 4096 \
    --max_tokens "$MAX_TOKENS"
echo "⚠️ Processed $MAX_TOKENS tokens. You can increase this value or run a full scoring over the complete *.bin files."

echo "--> Step 5: Selecting top-k..."
python rank_scores.py \
       --scores data/ai_scores.tsv \
       --top_pct 20

echo "✅ Pipeline completed successfully."