# Validation and reporting

The E2E report is an evidence index. Link logs or committed small fixtures; do not paste a success claim without the command and observation that support it.

## Required validation matrix

| Layer | Required evidence | Typical failure caught |
|---|---|---|
| source pin | URL, commit, license, dependency versions | irreproducible moving target |
| import/config | clean-process imports and registry lookup | hidden import order or backend dependency |
| checkpoint conversion | 100% accounted keys, shapes, strict load | silent random initialization or wrong layout |
| forward | fixed-input parity or documented alternative invariant | semantic port error |
| backward | finite scalar loss and gradients | detached path or unsupported op |
| tiny overfit | recorded step/loss series and acceptance rule | incorrect objective/data pairing |
| distributed | target FSDP/SP/EP topology and world size | sharding path mismatch or collective hang |
| save/resume | uninterrupted-versus-resumed comparison | missing optimizer/RNG/data state |
| export/inference | target consumer loads exported weights | incompatible checkpoint namespace |

## Loss evidence

Use a deterministic repeated fixture when the acceptance criterion is loss decrease. Record every step used for the claim. Define the rule before running, for example: the median of the final three losses is lower than the median of the first three and all values are finite. A stochastic production batch can supplement this but is a poor sole acceptance test.

## Numerical parity

Run upstream and VeOmni in evaluation mode with identical weights, inputs, masks, dtype, and seed. Compare stable intermediate boundaries as well as final output when the full objective includes randomness. Report max absolute error, max relative error, and the compared tensor shape. If backend kernels are intentionally numerically different, establish a PyTorch reference path first and then choose a justified backend tolerance.

## Resume evidence

Record:

- checkpoint path and saved global step;
- files or DCP metadata present;
- restored step and state components;
- pre-save and post-load parameter fingerprints;
- identity/fingerprint of the next batch;
- uninterrupted and resumed next-step loss;
- allowed and observed difference.

Loading weights alone is not training resume. Optimizer and scheduler state must be exercised by the next step.

## Review the report

Search for `TBD`, unchecked boxes, machine-local absolute paths, unpinned `main` revisions, and commands that depend on shell history. Re-run commands from a clean worktree or container using only documented inputs. Classify unavailable expensive or hardware tests as `BLOCKED`, never `PASS`.
