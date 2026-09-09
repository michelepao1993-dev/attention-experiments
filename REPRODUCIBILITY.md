# Reproducibility guide

## Levels of reproduction

This repository is designed for three levels of reproducibility.

1. **Paper artifact reproduction.** Regenerate paper tables from sanitized CSV summaries included under `results/evals/`.
2. **Implementation checks.** Run compact unit/reference/integration tests for the MoSAR controlled-decay implementation. Development-only pre-launch gates are intentionally omitted.
3. **Full HPC rerun.** Rerun training/evaluation if external dataset, tokenizer, Megatron-Bridge runtime, container, and checkpoints are provided.

## Environment

The training stack assumes Python 3.12 inside a Megatron-Bridge container/runtime, PyTorch with CUDA and Transformer Engine, and NVIDIA Megatron-Bridge/Megatron-Core on `PYTHONPATH`. The repository keeps only reusable training/evaluation templates; internal pre-launch scripts are not part of the anonymous release.

Copy and edit:

```bash
cp configs/main_constants.env.example configs/main_constants.env
```

The Slurm templates source this file automatically.

## Dataset and tokenizer provenance

Training uses an English Wikipedia corpus converted to the Megatron indexed-dataset format.

### Source corpus

* Hugging Face dataset: `graelo/wikipedia`
* Configuration: `20230601.en`
* Wikipedia snapshot: June 1, 2023
* Dataset split used by the Hugging Face source: `train`

The source dataset was downloaded and exported to line-delimited JSON with:

```python
from datasets import load_dataset

wikipedia_en = load_dataset(
    "graelo/wikipedia",
    "20230601.en",
    cache_dir="./cache",
    trust_remote_code=True,
)

wikipedia_en["train"].to_json(
    "wikipedia_20230601en.json",
    orient="records",
    lines=True,
    num_proc=32,
)
```

### Megatron preprocessing

The JSON corpus was converted to the Megatron indexed-dataset format using the pinned Megatron-LM `tools/preprocess_data.py` implementation:

```bash
python tools/preprocess_data.py \
  --input wikipedia_20230601en.json \
  --output-prefix training_data \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model google/gemma-2-2b \
  --json-keys text \
  --append-eod \
  --workers 16
```

This produces the indexed dataset prefix:

```text
training_data_text_document
```

with the corresponding `.bin` and `.idx` files.

Training uses a Megatron dataset split of:

```text
990,10,0
```

and sequence length 2048.

The resulting indexed corpus contains approximately `4,975,435,463` tokens, estimated from the indexed `.bin` size assuming int32 token storage.

### Tokenizer

The training-time tokenizer directory was created from the Hugging Face `google/gemma-2b` tokenizer using `AutoTokenizer.from_pretrained(...)` and `save_pretrained(...)`.

Dataset preprocessing used `google/gemma-2-2b`. Gemma 2 uses the same 256k SentencePiece tokenizer as Gemma 1, so these tokenizer sources are compatible.

* Vocabulary size: `256000`
* Training-time tokenizer path: user-configurable through `TOKENIZER_PATH`
* Original local directory name: `gemma_tokenizer`
* Tokenizer class: `GemmaTokenizer`

The tokenizer itself is not included in this anonymous repository and must be obtained from the corresponding Hugging Face Gemma release.



### Reference runtime

- Megatron-Bridge commit: `0fbfe7d3e970fbd75c1281d71cee586ce1f3df5e`
- Megatron-LM commit: `9539a12e1b04a68423f57b3eb41d6125161dca24`
- Container SHA-256: `cc395254186979684809f8e103bfefa8f68c00644f123c4b9859ce6ff7aa1d1b`

Package versions:
- Python: `3.12.3`
- PyTorch: `2.11.0a0+eb65b36914.nv26.02`
- CUDA used to build PyTorch: `13.1`
- cuDNN: `92000`
- Transformer Engine: `2.14.0+71bbefbf`

### Anonymization note

Machine-specific absolute paths, usernames, allocation names, job IDs, and node names are intentionally omitted from this anonymous release. Reproduction scripts assume that users set `PROJECT_ROOT`, `SCRATCH_ROOT`, and container paths in their own environment.


## Training configuration

Canonical 610k-step setting: sequence length 2048, global batch size 8, micro batch size 1, model seed 1239, data seed 2031, peak LR 3e-4, min LR 3e-5.

Segmented pre-training template:

```bash
source configs/main_constants.env
START=0 END=50000 bash slurm/pretraining/submit_main_wave_parallel_v1.sh
```

## Evaluation configuration

Use sequence lengths 2048, 4096, and 8192. The post-training launchers support `ROUTING_INFERENCE=soft` and `ROUTING_INFERENCE=hard_top1`; hard top-1 is meaningful only for learned MoSAR runs.

Example final soft evaluation at 8192:

```bash
source configs/main_constants.env
STEP=610000 SEQUENCE_LENGTH=8192 EVAL_ITERS=16 ROUTING_INFERENCE=soft \
  bash slurm/posttraining/submit_posttraining_eval_parallel_v1.sh
```

Example final hard evaluation for MoSAR checkpoints:

```bash
source configs/main_constants.env
STEP=610000 SEQUENCE_LENGTH=8192 EVAL_ITERS=16 ROUTING_INFERENCE=hard_top1 RUNS="mosar0 cost001" \
  bash slurm/posttraining/submit_posttraining_eval_parallel_v1.sh
```

## Checkpoints

Checkpoints are external artifacts and should not be committed to the anonymous repository. For final paper evaluation, only the final 610k checkpoints are needed. See `CHECKPOINTS.md` for layout.

## Included results

The CSV summaries in `results/evals/` are sanitized paper-table inputs and omit original cluster log paths.

## Training-curve generation

From the repository root:

```bash
python3 analysis/make_training_curve_pgfplots.py
```