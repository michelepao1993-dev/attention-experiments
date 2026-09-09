#!/usr/bin/env bash
set -euo pipefail

MEGATRON_PATH="${MEGATRON_PATH:?Set MEGATRON_PATH to the Megatron-LM repository}"

python "$MEGATRON_PATH/tools/preprocess_data.py" \
    --input wikipedia_20230601en.json \
    --output-prefix training_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model google/gemma-2-2b \
    --json-keys text \
    --append-eod \
    --workers 16