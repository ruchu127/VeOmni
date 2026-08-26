# SeedVR2-3B migration and reproduction guide

This is the worked example for the generic `veomni-model-migration` Skill. It
integrates the public SeedVR2-3B NaDiT checkpoint into VeOmni, converts its
weights with a fail-closed audit, prepares cached latent data, runs FSDP2
training, and verifies DCP resume.

The semantic source is the Apache-2.0
[ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) repository at
commit `e4de8c24441a67e1b7df56abea10645059bb1185`. The checkpoint source is
[ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B).
Do not substitute a moving upstream branch when checking parity.

## What was migrated

The wrapper preserves all 635 upstream tensor names below a single `dit.`
module. The CUDA-only Apex norms and FlashAttention call are replaced by
state-dict-compatible PyTorch LayerNorm/RMSNorm and SDPA reference paths. The
condition model reconstructs the upstream `v_lerp` flow contract:

- clean and degraded VAE latents are `[16,T,H,W]`;
- the model input is noisy latent + degraded latent + mask, or 33 channels;
- text conditions are `[L,5120]`;
- the target is `noise - clean_latent`;
- the model predicts 16 latent channels.

The public SeedVR repository is inference-only and does not publish its
training dataset pipeline. Consequently, raw video/VAE encoding remains a
pinned upstream preprocessing boundary. VeOmni accepts validated cached
latents and text embeddings; it does not claim that arbitrary tensors are a
real restoration dataset.

## Clean-tree reproduction

Install the repository-locked environment. Use `npu_aarch64` on an aarch64
Ascend host and `npu` on an x86_64 Ascend host.

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
uv sync --frozen --extra npu_aarch64 --group test --group lint
source .venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Optionally reproduce the Skill's upstream-analysis packet before changing
code:

```bash
git clone https://github.com/ByteDance-Seed/SeedVR.git /tmp/SeedVR
git -C /tmp/SeedVR checkout e4de8c24441a67e1b7df56abea10645059bb1185
python .agents/skills/veomni-model-migration/scripts/analyze_upstream.py \
  --upstream /tmp/SeedVR \
  --output /tmp/seedvr2-migration
```

Download only the required official artifacts, then convert the EMA
checkpoint. The converter refuses a non-empty output directory, uses
`weights_only=True` and mmap loading, builds the full target model on `meta`,
and requires zero missing, unexpected, or shape-mismatched tensors before it
writes sharded safetensors.

```bash
huggingface-cli download ByteDance-Seed/SeedVR2-3B \
  seedvr2_ema_3b.pth pos_emb.pt neg_emb.pt \
  --local-dir checkpoints/SeedVR2-3B

python scripts/seedvr2/convert_seedvr2_weights.py \
  --source checkpoints/SeedVR2-3B/seedvr2_ema_3b.pth \
  --output-dir pretrained_models/SeedVR2-3B-veomni \
  --max-shard-size 5GB
```

## Data preparation

For real data, create a JSONL manifest whose paths are relative to the
manifest or absolute. Every line has this contract:

```json
{"sample_id":"clip-0001","clean_latents":"clean/0001.pt","degraded_latents":"degraded/0001.pt","prompt_embeds":"text/0001.pt"}
```

The tensors must come from parity-checked SeedVR VAE/text preprocessing. Pack
and validate them as follows:

```bash
python scripts/seedvr2/prepare_seedvr2_data.py \
  --manifest data/seedvr2/manifest.jsonl \
  --output-dir output/seedvr2_3b_data \
  --shard-size 1000 \
  --pad-to-multiple 8
```

For a mechanics-only smoke test, the same command can create deterministic,
contract-valid toy tensors. This fixture is suitable for checking loading,
forward/backward, loss, and checkpointing, not model quality:

```bash
python scripts/seedvr2/prepare_seedvr2_data.py \
  --make-toy output/seedvr2_toy_inputs \
  --output-dir output/seedvr2_3b_data \
  --toy-samples 8 --pad-to-multiple 8 --shard-size 4
```

The parquet directory contains only parquet shards so VeOmni's directory
loader cannot mix formats. A SHA-bearing dataset manifest is written next to
the directory.

## Train, save, and resume

Edit the machine paths in `configs/dit/seedvr2_3b.yaml` or override them on the
CLI. The checked-in config uses correctness-first eager kernels, FSDP2, BF16,
gradient checkpointing, micro batch size 1, and DCP saves.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash train.sh \
  tasks/train_dit.py configs/dit/seedvr2_3b.yaml \
  --train.global_batch_size 2 \
  --train.max_steps 4 \
  --train.checkpoint.save_steps 2
```

Resume in a new process. For a finite iterable smoke dataset, ensure another
epoch or enough remaining samples are available; otherwise the restored
dataloader is correctly already exhausted.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash train.sh \
  tasks/train_dit.py configs/dit/seedvr2_3b.yaml \
  --train.global_batch_size 2 \
  --train.num_train_epochs 2 \
  --train.max_steps 5 \
  --train.checkpoint.load_path output/seedvr2_3b_train/checkpoints/global_step_4
```

Review the [E2E report](./seedvr2_3b_e2e_report.md) for the measured loss,
checkpoint, and load evidence, and the
[Ascend report](./seedvr2_3b_ascend_report.md) for environment-specific notes.

## Known boundaries

- The reference attention path is correctness-oriented. It does not yet use
  SeedVR-specific fused attention or VeOmni Ulysses sequence parallelism.
- Real-data quality depends on upstream-compatible VAE degradation, scaling,
  and text preprocessing. The packer validates tensor contracts, not semantic
  quality.
- Public upstream training code is unavailable; the flow objective is derived
  from the released scheduler and inference construction and is identified as
  such in the E2E report.
