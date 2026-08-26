---
name: veomni-model-migration
description: "Plan, implement, and verify an unsupported model migration into VeOmni from an upstream repository or checkpoint. Use for end-to-end model onboarding that includes source analysis, interface mapping, model or patchgen integration, weight conversion, data and training configuration, checkpoint save/resume, and evidence-backed validation. Do not use for a small fix to an already supported model."
---

# VeOmni Model Migration

Migrate a model from pinned upstream source and weights to a reproducible VeOmni training path. Treat a loading model or a completed forward pass as intermediate evidence, not completion. Completion requires a short training run, checkpoint save and resume, and a report containing the exact commands and observed results.

This protocol is model-family neutral. Do not copy names, paths, tensor rules, or configuration fields from an example until source inspection proves they apply to the target.

## Start with an evidence packet

Run the deterministic analyzer before editing VeOmni:

```bash
python .agents/skills/veomni-model-migration/scripts/analyze_upstream.py \
  --model-name <display-name> \
  --upstream <local-upstream-checkout> \
  --veomni-root . \
  --backend <gpu|npu|both> \
  --output .agents_workspace/migrations/<model-slug>
```

The analyzer uses only local files and the Python standard library. It records the upstream revision, architecture clues, an initial integration route, a file checklist, a validation matrix, and editable copies of all required templates. Review its inferences against the source. Replace every `TBD` backed by direct evidence before implementation.

If the upstream code is not locally available, clone or export an exact revision first. Record the repository URL, commit, model-weight revision or checksum, license, and dependency versions. Never analyze a moving default branch and call the result reproducible.

## Gate 1: approve the migration contract

Read [planning.md](references/planning.md) and complete `migration-manifest.yaml` plus `migration-plan.md`. Do not write model code until the packet identifies:

- the trainable module, frozen condition modules, objective, and data contract;
- the upstream-to-VeOmni interface map and the nearest structural references;
- every checkpoint namespace/shape transform, including an explicit identity mapping when no conversion is needed;
- target backends and parallel dimensions;
- unit, numerical-parity, training, and checkpoint-resume evidence with pass criteria.

Use structural similarity, not a shared product name, to choose reference integrations. Compare forward signatures, tensor layouts, normalization, attention, positional encoding, expert layout, condition encoders, output/loss contract, and checkpoint keys.

## Gate 2: implement the smallest complete vertical slice

Read [implementation.md](references/implementation.md). Implement one batch through data preparation, condition processing, trainable-model forward, scalar loss, backward, optimizer step, checkpoint save, and resume before adding performance features.

Select the integration route from evidence:

- Transformers-native model: use the repository's `veomni-new-model` protocol. If generated modeling is required, also follow `veomni-migrate-transformers-v5`; edit patchgen configs and regenerate outputs, never edit `generated/` directly.
- Diffusers-native model: follow `docs/usage/support_new_models/dit_model_guide.md` and preserve diffusers-compatible load/save keys.
- Custom PyTorch model: vendor only the required Apache-compatible source with provenance, or implement an isomorphic VeOmni wrapper. Add an explicit checkpoint converter and prove parameter coverage and numerical parity. Do not rely on an unpinned editable checkout at runtime.

Treat GPU and NPU support as separate observable contracts. Shared model logic is preferred, but backend-specific kernels must have a correct PyTorch fallback. A backend import must not make the other backend unimportable.

## Gate 3: prove correctness and resumability

Read [validation.md](references/validation.md) and fill `e2e-report.md`. Run validations in increasing cost order:

1. static imports, configuration parsing, registry lookup, and generated-file drift checks;
2. converter key/shape/coverage checks and a tiny-model state-dict round trip;
3. upstream-versus-VeOmni forward parity on fixed inputs where equivalent inference exists;
4. one-batch forward/backward with finite loss and gradients on trainable parameters;
5. a multi-step run showing the expected loss behavior for a deterministic overfit fixture;
6. checkpoint save, fresh-process resume, and the next optimizer step;
7. target-backend E2E with the exact production command.

For save/resume, compare the resumed run with an uninterrupted control at the same step. At minimum record restored global step, optimizer/scheduler state, parameter checksum, next-batch identity, and next-step loss tolerance.

## Required deliverables

Keep these artifacts discoverable from the PR:

- the general Skill, its references, analyzer, and reusable templates;
- target model registration/implementation or patchgen files;
- a strict, auditable weight converter or proof that the native checkpoint is identity-compatible;
- data and training configurations with no machine-specific absolute paths;
- toy/unit tests and an executable E2E command;
- a completed E2E report with environment, revisions, commands, loss evidence, and checkpoint-resume evidence;
- a completed Ascend experience report when NPU is in scope.

Before handoff, rerun the analyzer or manually reconcile the original plan against the final diff. Every changed migration decision must be reflected in the manifest and report.

## Stop conditions

Stop and report missing evidence instead of fabricating success when upstream training code or weights are unavailable, the license is incompatible, a checkpoint transform has unexplained keys, required hardware is unavailable, loss is non-finite, or resume does not reproduce the uninterrupted control. A documented limitation is acceptable evidence; an unchecked box presented as complete is not.
