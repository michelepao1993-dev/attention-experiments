# External checkpoints

Model checkpoints are not included in this anonymous repository.

The training and evaluation scripts expect checkpoints under:

```text
${CHECKPOINT_ROOT}/${GROUP}/${RUN_ID}/
```

where `RUN_ID` is one of:

```text
vanilla
alibi
prope075
s_fixed
m_fixed
mosar0
cost001
rope_m_mask
```

For the results reported in the paper, the final checkpoint corresponds to step `610000`.

A checkpoint directory used for evaluation must contain the Megatron checkpoint produced by the pinned training runtime together with:

```text
latest_checkpointed_iteration.txt
```

whose contents must match the requested evaluation step, e.g.:

```text
610000
```

Example layout:

```text
${CHECKPOINT_ROOT}/
└── anonymous_main_v1/
    ├── vanilla/
    │   ├── latest_checkpointed_iteration.txt
    │   └── ...
    ├── mosar0/
    │   ├── latest_checkpointed_iteration.txt
    │   └── ...
    └── cost001/
        ├── latest_checkpointed_iteration.txt
        └── ...
```

The checkpoint serialization format is the native format produced by the pinned Megatron-Bridge/Megatron-Core runtime described in `REPRODUCIBILITY.md`.

Checkpoints are required only for full evaluation or continued training. They are not required for the local implementation tests or for regenerating the paper tables and figures from the sanitized results included in this repository.
