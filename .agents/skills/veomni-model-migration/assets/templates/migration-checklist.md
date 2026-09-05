# {{MODEL_NAME}} migration checklist

Generated: {{GENERATED_AT}}
Upstream revision: `{{UPSTREAM_REVISION}}`
Target backends: `{{BACKENDS}}`

## Inputs and scope

- [ ] Upstream source URL, immutable commit, and license recorded
- [ ] Weight source, immutable revision/SHA-256, format, and license recorded
- [ ] Dependency/runtime versions recorded
- [ ] Trainable and frozen modules identified from the real call graph
- [ ] Forward tensor contract and objective documented
- [ ] Nearest VeOmni references selected by structural comparison

## Checkpoint

- [ ] All source keys and shapes inventoried safely
- [ ] Every key classified as identity/rename/layout/split/concat/drop/init
- [ ] Converter rejects collisions, unexplained keys, and shape mismatches
- [ ] Conversion report records coverage and non-identity mappings
- [ ] Converted weights load strictly into the target model
- [ ] Fixed-input upstream parity passes, or alternative invariant is justified

## Data and training

- [ ] Raw example schema and one real-format fixture documented
- [ ] Every transform has input/output dtype and shape evidence
- [ ] Offline caches include provenance and schema version
- [ ] One-process forward/backward has finite loss and gradients
- [ ] Deterministic tiny-overfit loss rule passes
- [ ] FSDP, SP, EP, and backend kernels added one dimension at a time
- [ ] Example configs contain no machine-specific absolute paths

## Checkpoint lifecycle

- [ ] Common VeOmni checkpoint path saves model and training state
- [ ] Fresh process restores global step, optimizer, scheduler, RNG, and data position
- [ ] Resumed next step matches uninterrupted control within declared tolerance
- [ ] Exported weights load in the intended inference consumer

## Handoff

- [ ] Unit/toy tests and exact E2E command committed
- [ ] E2E report contains environment, commands, logs, loss series, and resume evidence
- [ ] Ascend experience report completed when NPU is in scope
- [ ] Clean-worktree reproduction performed or remaining items marked BLOCKED
- [ ] No `TBD`, unchecked acceptance item, unpinned revision, or secret remains
