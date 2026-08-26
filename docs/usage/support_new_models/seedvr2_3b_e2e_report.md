# SeedVR2-3B end-to-end migration report

## Result

The migration passed the full mechanics path on 2026-08-26: official weights
were strictly converted and loaded, a two-Ascend FSDP2 job completed four
forward/backward/optimizer steps with a lower ending loss, DCP checkpoints were
saved, and a fresh process loaded `global_step_4` and continued training.

This validates the VeOmni training contract. The data was a deterministic
cached-latent fixture, so this report does not claim restoration quality.

## Revisions and artifacts

| Item | Evidence |
|---|---|
| VeOmni base | `1d5651fcb9d569441188395a1a23bc79fb7edb62` |
| SeedVR source | `e4de8c24441a67e1b7df56abea10645059bb1185` |
| Official checkpoint | `seedvr2_ema_3b.pth`, 13,566,090,228 bytes |
| Source SHA-256 | `6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b` |
| Source inventory | 635 FP32 tensors; 3,391,476,448 parameters |
| Target audit | missing 0; unexpected 0; shape mismatches 0; same parameter count |
| Converted output | 3 safetensors shards + index + config + conversion report |

Converted shard fingerprints:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 4,964,880,788 | `6abd2008a180a52a579672124f502b1d8c1c55b5edb52260ed8935b9b397009c2` |
| `model-00002-of-00003.safetensors` | 4,934,738,900 | `ec3ccf0bc4b35cacdf40ce7e0cb7f5fc452b4b9bbe80d541657c23e45b7cdebc` |
| `model-00003-of-00003.safetensors` | 3,666,351,920 | `c09eebd1d9c7f2e4fa76d97c0d218a209cbfb4769fa7d86f2a206a0bcf3933ac9` |

VeOmni's streaming loader loaded all 3,391,476,448 parameters to `npu:0`
with zero meta tensors. Exact equality checks passed for
`vid_in.proj.weight`, `blocks.0.attn.proj_qkv.vid.weight`, and
`vid_out.proj.bias` against the official source.

## Automated and device tests

- Generic Skill/analyzer, SeedVR model, and DiT registration tests: `5 passed`.
- Tiny CPU path: condition construction, forward, gradient-checkpointed
  backward, finite gradients, safe serialization round trip.
- Tiny Ascend BF16 path: output `(4,4)` on `npu:0`, loss
  `2.0808558464050293`, finite input-projection gradients.
- Full checkpoint state dict: 635/635 names and shapes match the target meta
  model; the RoPE frequency buffer also matches bit-for-bit.

## Full 3B training evidence

The main run used two Ascend 910 devices, FSDP2 shard size 2, BF16 mixed
precision, gradient checkpointing, eager SDPA, global batch 2, micro batch 1,
and deterministic toy parquet with eight samples.

| Optimizer step | MSE loss |
|---:|---:|
| 1 | 3.49 |
| 2 | 2.48 |
| 3 | 2.32 |
| 4 | 2.91 |

Loss is stochastic because each condition call samples flow noise, so monotonic
decrease is not expected. The ending loss is below the starting loss and the
minimum is 33.5% below the start. DCP checkpoints were successfully written at
`global_step_2` and `global_step_4`; the latter occupied about 38 GB. Peak
reported device memory was 41.94 GB per rank.

In a new two-rank process, VeOmni reported successful loading from
`global_step_4`. Continuing into the next finite-data epoch produced losses
`1.79`, `1.68`, `2.47`, and `2.19`, and wrote new DCP checkpoints through
`global_step_9`. This proves model, optimizer, scheduler, trainer state, and
dataloader state are loadable. The finite iterable was exhausted when a resume
was attempted with only the original epoch, which is expected and is called
out in the reproduction guide.

## Interpretation and limitations

The public SeedVR release contains inference but not training code. The
`v_lerp` target and 33-channel restoration input are therefore reconstructed
from the released scheduler and inference graph. This is a source-backed
inference, not a claim about private upstream training details.

The test fixture proves software mechanics only. A quality validation must use
real clean/degraded pairs encoded with the pinned SeedVR VAE pipeline and then
compare model outputs or loss against the upstream implementation. Sequence
parallelism and fused Ascend attention remain follow-up performance work.
