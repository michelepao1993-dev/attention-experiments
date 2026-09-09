# MoSAR: Mixture of Semantic Attention Regimes

Anonymous reproducibility release candidate for the MoSAR controlled-decay attention experiments.

MoSAR replaces a fixed attention sparsity prior with an input-conditioned controlled-decay geometry. Query-side and key-side routers select mixtures over short, medium, and global regimes; the selected regime pair induces a distance-dependent decay field over query-key interactions. The same code also supports fixed-regime and positional baselines used in the paper.


## Repository layout

```text
data/                   Dataset, tokenizer, and Megatron preprocessing scripts
runtime_source/        MoSAR implementation and Megatron-Bridge integration
slurm/pretraining/     Portable Slurm templates for matched pre-training/eval
slurm/posttraining/    Portable Slurm templates for length extrapolation / hard routing
configs/               Environment templates; copy and edit before running
analysis/              Paper table/figure builders and optional log collectors
results/evals/         Sanitized CSV summaries used by the paper tables
results/training_curve/ Sanitized points used by the training-curve figure
tables/                Generated LaTeX tables from included CSV summaries
CHECKPOINTS.md         Policy and expected layout for external checkpoints
```

## Main variants

The primary 610k-step run compares `vanilla`, `alibi`, `prope075`, `s_fixed`, `m_fixed`, `mosar0`, `cost001`, and `rope_m_mask`.


## Important reproducibility notes

This package does **not** include checkpoints, tokenizer files, the indexed pre-training corpus, or the HPC container. These are external artifacts. To rerun training/evaluation, copy `configs/main_constants.env.example` to `configs/main_constants.env` and set:

```bash
SCRATCH_ROOT=/path/to/scratch
MEGATRON_BRIDGE_PATH=/path/to/Megatron-Bridge
CONTAINER_PATH=/path/to/container.sif
DATA_PREFIX=/path/to/dataset_prefix_without_bin_idx
TOKENIZER_PATH=/path/to/tokenizer_directory
```

Final checkpoint evaluation expects external checkpoints under `${CHECKPOINT_ROOT}/${GROUP}/${RUN_ID}`. See `CHECKPOINTS.md`.


## Quick local checks

```bash
cd runtime_source
python -m compileall mosar integrations experiments tests
pytest tests/unit tests/reference tests/integration
```

Full training and evaluation require the Megatron-Bridge runtime and GPU cluster environment. Development-only pre-launch checks, job-health scripts, and internal recovery utilities are intentionally omitted from this anonymous reproduction package.


## Paper-table generation

From the repository root:

```bash
python3 analysis/make_results_tables.py \
  --group anonymous_main_v1 \
  --step 610000 \
  --analysis-root results/evals
```

This writes LaTeX tables under `tables/`.
