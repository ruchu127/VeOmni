# SeedVR2-3B Ascend migration experience report

## Tested environment

| Component | Observed value |
|---|---|
| Hardware | 16 × Ascend 910, 65,536 MB HBM each; tests used two idle devices |
| OS architecture | Linux aarch64 |
| CANN toolkit | 9.1.0, `V100R001C11SPC001B243` |
| Python | 3.11.6 for the validation overlay |
| Torch / torch-npu | 2.7.1+cpu / 2.7.1.post8 on the host |
| Transformers | 5.9.0 in an isolated validation overlay |
| VeOmni locked target | Python 3.12, Torch/torch-npu 2.10.0 for `npu_aarch64` |

The host's preinstalled Torch stack is older than current VeOmni's lock. The
code and full 3B run succeeded with that stack plus Transformers 5.9.0, but the
version warning is not treated as harmless proof of the locked environment.
Clean reproduction should use `uv sync --frozen --extra npu_aarch64`.

## Setup lessons

Source the toolkit environment before importing torch-npu:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Without it, import failed because `libhccl.so` was not discoverable. Use
`ASCEND_RT_VISIBLE_DEVICES` to select genuinely idle devices; `npu-smi info`
showed several other cards occupied by unrelated jobs, which were left alone.

The temporary validation overlay inherited the system Torch package and did
not contain its own `torchrun` entry point. In that one environment,
`train.sh` resolved `/usr/local/bin/torchrun`, whose shebang used the system
Transformers 4.57 and failed. Calling the overlay interpreter with
`python -m torch.distributed.run` fixed the interpreter mismatch. A normal
locked `uv sync` installs its own Torch and torchrun, so the documented
`train.sh` command is the clean path.

## Observed runtime behavior

- Full 3B FSDP2 training reached 41.94 GB peak HBM per rank with tiny latent
  inputs and gradient checkpointing.
- The reference window-index path runs int64 `argsort` on AiCPU on this stack.
  It is correct but a visible performance warning.
- HCCL reports that gather is implemented with all-gather. DCP save still
  completed successfully.
- CANN emitted a 32-byte padding allocator warning and a base-format tensor
  warning in the tiny test. Neither caused numerical failure.
- DCP for the full optimizer state was about 38 GB per checkpoint. Plan disk
  capacity and retention before using frequent save intervals.

## Recommended next optimizations

1. Validate again on the repository-locked Torch/torch-npu 2.10 environment.
2. Replace the Python per-window SDPA reference with a registered Ascend
   attention backend while preserving the strict parity test.
3. Avoid int64 window-index sorting on AiCPU or cache host-built indices when
   shapes repeat.
4. Add Ulysses-aware slice/gather support before enabling sequence parallelism.
5. Use real upstream-compatible VAE latents for quality and throughput tests;
   keep the deterministic toy fixture for fast checkpoint regressions.
