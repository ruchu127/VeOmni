---
name: veomni-model-migration
description: "Single entry point for adding or migrating a model into VeOmni. Covers new-model integration and end-to-end migration from pinned upstream source or checkpoints, including architecture analysis, model or patchgen implementation, parallel plans, data and training configuration, weight conversion, checkpoint save/resume, and evidence-backed validation. Do not use for a bug fix to an already supported model or for a Transformers v5 patchgen-only refresh."
---

# VeOmni Model Onboarding and Migration

Use this skill as the single model-onboarding entry point. First classify the
request, then run only the gates required by that mode. Keep the implementation
protocol shared; do not duplicate a separate legacy workflow.

## 1. Classify the request

| Mode | Select when | Required evidence |
| --- | --- | --- |
| **New-model mode** | VeOmni does not support the model yet and the task is ordinary model integration | Clean imports, registry/config lookup, model tests, and a short E2E run when feasible |
| **Migration mode** | The task ports an external repository/checkpoint, changes model families or frameworks, requires weight conversion, or explicitly asks for reproducible training and resume validation | Everything in new-model mode, plus pinned provenance, converter coverage, numerical parity where possible, training loss evidence, checkpoint save/resume, and a report |
| **Transformers v5 patchgen mode** | The model is already integrated and only its patchgen-generated v5 path needs to be added or refreshed | Follow `veomni-migrate-transformers-v5`; do not run the full migration gates unless the task also changes model onboarding or checkpoint semantics |

For a request that only says “add support for model X”, start in new-model
mode. If source pinning, checkpoint conversion, data migration, or
save/resume is part of the request, upgrade to migration mode. A supported
model bug or failed test belongs to `veomni-debug`.

## 2. Establish the common model contract

Before editing code, identify the model category (Transformers, Diffusers, or
custom PyTorch), trainable and frozen modules, forward inputs/outputs, data
contract, loss, parallel dimensions, and the nearest structural reference.
Read [new-model-integration.md](references/new-model-integration.md) for the
implementation checklist shared by both modes.

For migration mode, create an evidence packet before writing model code:

```bash
python .agents/skills/veomni-model-migration/scripts/analyze_upstream.py \
  --model-name <display-name> \
  --upstream <local-upstream-checkout> \
  --veomni-root . \
  --backend <gpu|npu|both> \
  --output .agents_workspace/migrations/<model-slug>
```

Pin the upstream URL and commit, checkpoint revision or checksum, licenses,
and dependency versions. Read [planning.md](references/planning.md), then
complete `migration-manifest.yaml` and `migration-plan.md` before
implementation. Replace every analyzer `TBD` with source-backed evidence.

## 3. Implement the smallest complete path

Implement one real batch through data preparation, condition processing,
trainable-model forward, scalar loss, backward, optimizer step, and (in
migration mode) checkpoint save/resume before adding performance features.
Read [implementation.md](references/implementation.md) before coding so the
upstream contract, converter, data path, and common checkpoint interface are
preserved.

- **Transformers:** follow [new-model-integration.md](references/new-model-integration.md).
  When generated modeling is required, also follow
  `veomni-migrate-transformers-v5`; edit patchgen configs and regenerate
  outputs, never edit `generated/` directly.
- **Diffusers:** follow `docs/usage/support_new_models/dit_model_guide.md`
  and preserve Diffusers-compatible load/save keys.
- **Custom PyTorch:** vendor only required Apache-compatible source with
  provenance, or implement an isomorphic VeOmni wrapper. Add an explicit
  converter and prove parameter coverage and numerical parity.

Treat GPU and NPU as separate observable contracts. Shared model logic is
preferred, but backend-specific imports must not make the other backend
unimportable.

## 4. Validate by mode

Run validations in increasing cost order:

1. static imports, config parsing, registry lookup, and generated-file drift;
2. toy-model forward and model-specific unit tests;
3. one-batch forward/backward with finite loss and gradients;
4. a short E2E run with the selected trainer and data path.

Migration mode additionally requires [validation.md](references/validation.md):

5. strict converter key/shape/coverage checks and a state-dict round trip;
6. fixed-input upstream-versus-VeOmni parity, or a documented invariant when
   equivalent inference is unavailable;
7. a deterministic overfit fixture with recorded loss evidence;
8. checkpoint save, fresh-process resume, and comparison with an uninterrupted
   control at the same step;
9. the exact target-backend production command.

The migration gates override the looser “E2E if feasible” wording in the
new-model implementation reference. Do not claim completion when a required
criterion is unavailable; mark it `BLOCKED` and record the missing resource.

## 5. Required deliverables

### New-model mode

- model registration/implementation or patchgen configs;
- parallel plan, data/training configuration, and toy configuration;
- model unit tests, a runnable E2E command, and relevant documentation.

### Migration mode

Everything in new-model mode, plus:

- `migration-manifest.yaml` and `migration-plan.md`;
- strict weight converter or proof of identity-compatible checkpoint keys;
- data/checkpoint provenance and reproducible configuration;
- completed E2E report with commands, revisions, loss evidence, and resume
  evidence;
- completed Ascend experience report when NPU is in scope.

Before handoff, reconcile the final diff against the plan and rerun the
analyzer or perform the equivalent manual audit. Never fabricate success when
the license, weights, hardware, numerical checks, loss, or resume evidence is
missing.
