# Implementation protocol

## Preserve the upstream contract

Use the pinned upstream source as the semantic reference. Keep a provenance header in source-derived files with repository URL, commit, upstream path, and license. Avoid broad copies: include only code required for construction, forward, and training.

Keep configuration defaults centralized. Machine paths, device counts, and dataset locations belong in command-line overrides or YAML, not Python modules.

## Model integration

Registration must work in a clean process without first importing a test helper. Verify both configuration and modeling registry lookups.

For a Transformers model, patch only behavior that VeOmni requires. Generate model files through patchgen. Preserve upstream function signatures and output fields, and add backend-specific patches without importing unavailable accelerator libraries on other backends.

For a Diffusers or custom DiT, separate the frozen condition path from the trainable transformer path when their lifecycle differs. The condition model owns encoding, noise/timestep sampling, and target construction. The trainable model accepts tensors and returns a `ModelOutput` containing scalar loss. Keep the trainable parameter namespace stable so DCP and inference export can use it.

## Weight conversion

Start from `weight-converter.py`. Make mapping rules declarative where possible and use small, named transform functions for split/concat/layout changes. Produce a conversion report containing:

- source and destination fingerprints;
- source, converted, dropped, and initialized key counts;
- every non-identity mapping and its source/target shapes;
- collisions, missing required keys, unexpected source keys, and dtype changes;
- strict target-model load result.

Make conversion deterministic and atomic: write to a new output path, validate it, then finalize. Never overwrite the only copy of a source checkpoint.

## Data path

Make a tiny deterministic fixture first. It should exercise the real tensor contract while being small enough to overfit quickly. A random-tensor fixture can prove model mechanics, but it cannot replace one decoded real-format sample for the full E2E report.

For expensive encoders, support offline cached conditions only if the cache records source-example identity, encoder/checkpoint revision, preprocessing settings, tensor dtype/shape, and a schema version. The training path must reject an incompatible cache.

## Training configuration

Start with correctness settings: one process, eager or native attention, no sequence/expert parallelism, deterministic seed, short sequence/resolution, and frequent logging/checkpointing. Add FSDP, SP, EP, fused kernels, and multi-node execution one dimension at a time.

Every example config must expose:

- model/config/checkpoint inputs;
- dataset and transform settings;
- precision and attention implementation;
- global and micro batch sizes plus gradient accumulation implications;
- optimizer, scheduler, maximum steps, and seed;
- FSDP/SP/EP dimensions and constraints;
- checkpoint output, interval, retention, and optional load path.

## Checkpoint save and resume

Use VeOmni's existing checkpoint manager. Do not add a model-private resume path unless the common interface cannot represent necessary state and the reason is documented.

Test resume in a fresh process. Compare an uninterrupted `N+1` step run with an `N` step save followed by resume and one step. Use the same next batch and RNG state. Verify model, optimizer, scheduler, scaler where applicable, global step, and data position.
