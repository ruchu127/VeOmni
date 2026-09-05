# Planning and source analysis

The planning packet is the contract between analysis, implementation, and review. It prevents a plausible-looking port from silently changing the model or training objective.

## Pin the inputs

Record all of the following in `migration-manifest.yaml`:

- upstream repository URL and immutable commit;
- model checkpoint repository plus revision or local SHA-256 checksums;
- upstream and checkpoint licenses;
- Python, PyTorch, Transformers, Diffusers, accelerator runtime, and kernel versions;
- the exact target VeOmni commit used as the base.

Keep upstream source in `.agents_workspace/`; do not commit the checkout or model weights.

## Build four inventories

### Architecture inventory

Trace the actual training call graph and record:

- top-level model class and trainable submodule;
- configuration classes and construction path;
- forward inputs, outputs, dtype, shape notation, optional values, and masks;
- normalization, attention, positional encoding, MLP/MoE, patchification, and output head;
- condition encoders, schedulers/noise process, and loss construction;
- gradient-checkpointing boundaries and modules that FSDP must wrap together.

For multimodal and diffusion models, distinguish raw examples, encoded conditions, noisy model inputs, and targets. Do not collapse these into a generic `inputs` field.

### Checkpoint inventory

Inspect keys and shapes without executing untrusted pickle code. Prefer safetensors metadata. For PyTorch checkpoints use `torch.load(..., weights_only=True)` only in a trusted environment.

Classify every source tensor as one of:

- exact identity;
- renamed or prefix-stripped;
- transposed/permuted/reshaped;
- split into multiple targets;
- concatenated or stacked from multiple sources;
- duplicated/tied;
- intentionally dropped with a reason;
- missing and intentionally initialized with a reason.

The converter must fail on collisions, unexplained missing keys, unexplained extra keys, and shape mismatches. A successful `strict=False` load is not conversion evidence.

### Data inventory

Record one raw example and the tensor contract after every transformation. Include decoding, sampling, resize/crop, normalization, tokenizer/processor calls, latent caching, random augmentation, collator behavior, and deterministic seeds. Mark which transformations run offline, on CPU workers, or on the accelerator.

### VeOmni interface inventory

Map each upstream responsibility to a current VeOmni interface. Inspect the current code rather than relying on remembered paths:

| Responsibility | Typical VeOmni location |
|---|---|
| model/config registration | `veomni/models/loader.py`, model package `__init__.py` |
| Transformers patching | `veomni/patchgen/`, generated model package |
| DiT condition/trainable split | `veomni/models/diffusers/`, `DiTTrainer` |
| sample transform/collation | `veomni/data/`, model-specific dataset builder |
| distributed plan | model `parallel_plan.py`, distributed modules |
| optimizer/scheduler/training loop | trainer and task entry point selected by category |
| DCP save/resume and export | `veomni/checkpoint/`, training checkpoint config |

## Choose a route

Prefer, in order:

1. native supported upstream format with no weight rewrite;
2. a thin VeOmni wrapper that preserves upstream keys;
3. patchgen over a pinned Transformers implementation;
4. a source-derived implementation plus explicit converter.

The shortest code path is not automatically safest. Prefer the route with the smallest semantic distance and the strongest automated parity check.

## Define pass criteria before coding

Use numeric tolerances appropriate to dtype and backend. Include:

- expected converter coverage, normally 100% of required model parameters;
- maximum forward absolute/relative error for a deterministic fixture;
- finite loss and gradient requirements;
- overfit-fixture steps and required loss trend;
- checkpoint-resume parameter and next-loss tolerances;
- supported world sizes and parallel dimension divisibility constraints.

If a criterion cannot be evaluated, mark it `BLOCKED` with the missing resource. Do not replace it with a weaker unrelated test.
