# {{MODEL_NAME}} VeOmni migration E2E report

Status: `TBD` (`PASS`, `FAIL`, or `BLOCKED`)
Generated: {{GENERATED_AT}}

## Revisions and environment

| Item | Value |
|---|---|
| upstream source | `{{UPSTREAM_PATH}}` |
| upstream revision | `{{UPSTREAM_REVISION}}` |
| weights + revision/SHA-256 | TBD |
| VeOmni revision | TBD |
| OS/Python/PyTorch | TBD |
| accelerator/runtime | TBD |
| Transformers/Diffusers | TBD |
| world size / DP / FSDP / SP / EP | TBD |

## Reproduction inputs

- Dataset or deterministic fixture: TBD
- Raw example schema and provenance: TBD
- Model/data/training configs: TBD
- Converted checkpoint and conversion report: TBD

## Commands and results

### Environment setup

```bash
TBD
```

### Analyze and convert

```bash
TBD
```

Observed converter coverage and strict-load result: TBD

### Forward parity

```bash
TBD
```

Compared tensors, shapes, dtype, max absolute/relative error, tolerance: TBD

### Training

```bash
TBD
```

Predeclared loss acceptance rule: TBD

| global step | loss | gradient norm | notes |
|---:|---:|---:|---|
| 0 | TBD | TBD | |

Result against acceptance rule: TBD

### Checkpoint save and fresh-process resume

```bash
TBD
```

| Evidence | Uninterrupted control | Resumed run | Difference/tolerance |
|---|---|---|---|
| restored global step | TBD | TBD | exact |
| parameter fingerprint | TBD | TBD | exact or justified |
| next-batch fingerprint | TBD | TBD | exact |
| next-step loss | TBD | TBD | TBD |
| optimizer/scheduler state | TBD | TBD | exact fields |

## Quality and tests

```bash
TBD
```

Results: TBD

## Known limitations and blocked checks

- TBD

## Conclusion

TBD: state only claims directly supported by the evidence above.
