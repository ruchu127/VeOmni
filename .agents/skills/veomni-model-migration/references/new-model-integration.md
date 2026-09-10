# New-model integration protocol

Use this reference after the unified skill classifies a task as new-model mode
or as the implementation route inside migration mode. Migration-specific
provenance, conversion, and save/resume gates remain in the parent skill.

## Before implementation

Track these phases:

1. analyze the upstream model;
2. create the model implementation or patchgen configuration;
3. define the parallel plan;
4. write model and training configurations;
5. integrate with the appropriate trainer;
6. test and document the model.

Read the upstream `config.json`, modeling files, processor configuration, and
the closest existing model in `veomni/models/`. For Transformers, use the
repository's pinned version and inspect the patchgen path before creating new
files. Read `.agents/knowledge/testing.md` before adding or wiring tests so the
new coverage lands in a test target that CI actually runs.

## Model package and patchgen

Create `veomni/models/transformers/<model_name>/` with, as applicable:

- `__init__.py` for `MODELING_REGISTRY`, `MODEL_CONFIG_REGISTRY`, and
  `MODEL_PROCESSOR_REGISTRY` registration;
- `<model_name>_gpu_patch_gen_config.py`;
- `<model_name>_npu_patch_gen_config.py`;
- `parallel_plan.py`;

- generated GPU/NPU modeling files produced by patchgen.
For Transformers modeling, switch to `/veomni-patchgen-model`. It owns the
GPU/NPU patchgen configs, ExtraParallel plan, MoE checkpoint converter,
registration, code generation, and model-level test cases. Do not re-derive
that workflow here. Diffusion models under `veomni/models/diffusers/` and other
non-Transformers architectures have no patchgen output; follow the closest
existing direct-modeling integration instead.


Use declarative patchgen operations (`replace_class`, `override_method`,
`replace_function`, `modify_init`, `add_post_import_block`,
`drop_import_names`) for model behavior. Declare `OpSlot`s for sequence
parallel attention/loss and MoE expert dispatch. Stack MoE weights in the
layout expected by the runtime converter. Register model classes in the model
package; Transformers models do not need an entry in `veomni/models/auto.py`.

Run `make patchgen` after changing a patchgen config. Never edit files under
`generated/` manually.

## Parallel plan and configuration

`parallel_plan.py` is not an FSDP wrapping policy: FSDP2 wraps models
generically in `build_parallelize_model()`. Define a `ParallelPlan` only for
ExtraParallel sharding, such as expert or embedding sharding, and follow a
structurally similar existing plan. A model without ExtraParallel does not need
one.

Create the model configuration in `configs/model_configs/<model_family>/` and
the training YAML under the matching text, multimodal, or DiT directory. Keep
model path, data, optimizer, precision, parallel dimensions, checkpoint, seed,
and batch-size settings explicit.

## Trainer and data integration

Use `TextTrainer`, `VLMTrainer`, or `DitTrainer` according to the model. Add
model-specific transforms or collators only when the existing data contract
cannot represent the model. For VLMs, keep multimodal metadata preparation in
the collator and follow `.agents/knowledge/multimodal_metadata.md`.

## Tests and documentation

Add a minimal toy configuration under `tests/toy_config/<model_name>_toy/`.
Add cases to the existing enumerated tables in
`tests/models/test_model_registry.py` and `tests/models/test_models_patch.py`
instead of creating an unwired model test file. Test clean-process model loading
through `veomni.models.auto`, forward output shapes, patch application, and the
selected trainer path. Add an E2E `pytest.param` to
`tests/e2e/test_e2e_parallel.py` when feasible. Run `make quality` and focused
model tests. Add a short training command and update the supported models
documentation when appropriate.

Migration mode must extend this checklist with the parent skill's converter,
parity, training-loss, and checkpoint-resume evidence; a model that only loads
or completes one forward pass is not a completed migration.
