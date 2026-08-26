#!/usr/bin/env python3
"""Strictly convert the official SeedVR2-3B EMA checkpoint for VeOmni."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file

from veomni.models.diffusers.seedvr2.seedvr2_transformer.configuration_seedvr2_transformer import (
    SeedVR2TransformerConfig,
)
from veomni.models.diffusers.seedvr2.seedvr2_transformer.modeling_seedvr2_transformer import (
    SeedVR2TransformerModel,
)


UPSTREAM_REPOSITORY = "https://github.com/ByteDance-Seed/SeedVR"
UPSTREAM_REVISION = "e4de8c24441a67e1b7df56abea10645059bb1185"
UPSTREAM_FILENAME = "seedvr2_ema_3b.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(payload, dict):
        for wrapper in ("state_dict", "model", "module"):
            if isinstance(payload.get(wrapper), dict):
                payload = payload[wrapper]
                break
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("source does not contain a string-keyed state dict")
    non_tensors = [key for key, value in payload.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise TypeError(f"source contains non-tensor values: {non_tensors[:20]}")
    return payload


def expected_state(config: SeedVR2TransformerConfig) -> dict[str, torch.Tensor]:
    with torch.device("meta"):
        model = SeedVR2TransformerModel(config)
    return model.state_dict()


def validate_mapping(
    source: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict]:
    target_view = {f"dit.{key}": value for key, value in source.items()}
    missing = sorted(set(expected) - set(target_view))
    unexpected = sorted(set(target_view) - set(expected))
    shape_mismatches = [
        {
            "key": key,
            "source": list(target_view[key].shape),
            "expected": list(expected[key].shape),
        }
        for key in sorted(set(target_view) & set(expected))
        if target_view[key].shape != expected[key].shape
    ]
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "checkpoint contract mismatch: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, shape_mismatches={shape_mismatches[:20]}"
        )
    source_numel = sum(tensor.numel() for tensor in source.values())
    target_numel = sum(tensor.numel() for tensor in target_view.values())
    if source_numel != target_numel:
        raise ValueError(f"parameter count changed: source={source_numel}, target={target_numel}")
    audit = {
        "load_rule": "VeOmni model._checkpoint_conversion_mapping maps target_key = 'dit.' + stored_key",
        "storage_rule": "preserve official unprefixed keys so the VeOmni streaming loader applies the mapping exactly once",
        "source_tensor_count": len(source),
        "target_tensor_count": len(target_view),
        "source_numel": source_numel,
        "target_numel": target_numel,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }
    return source, audit


def save_sharded(state: dict[str, torch.Tensor], output_dir: Path, max_shard_size: str) -> dict:
    split = split_torch_state_dict_into_shards(
        state, filename_pattern="model{suffix}.safetensors", max_shard_size=max_shard_size
    )
    fingerprints = {}
    for filename, keys in split.filename_to_tensors.items():
        path = output_dir / filename
        shard = {key: state[key].contiguous() for key in keys}
        save_file(shard, str(path), metadata={"format": "pt", "model": "SeedVR2-3B"})
        fingerprints[filename] = {"sha256": sha256(path), "bytes": path.stat().st_size, "tensors": len(keys)}
        del shard
    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return {"is_sharded": split.is_sharded, "shards": fingerprints, "metadata": split.metadata}


def convert(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SeedVR2TransformerConfig.from_pretrained(args.config)
    source = load_state_dict(args.source)
    converted, audit = validate_mapping(source, expected_state(config))
    source_dtype_counts: dict[str, int] = {}
    for tensor in source.values():
        source_dtype_counts[str(tensor.dtype)] = source_dtype_counts.get(str(tensor.dtype), 0) + 1

    source_fingerprint = None if args.skip_source_hash else sha256(args.source)
    storage = None if args.dry_run else save_sharded(converted, args.output_dir, args.max_shard_size)
    if not args.dry_run:
        config.save_pretrained(args.output_dir)
    report = {
        "model": "SeedVR2-3B",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_filename": UPSTREAM_FILENAME,
        "source": str(args.source.resolve()),
        "source_bytes": args.source.stat().st_size,
        "source_sha256": source_fingerprint,
        "source_dtypes": source_dtype_counts,
        "audit": audit,
        "storage": storage,
        "dry_run": args.dry_run,
    }
    report_path = args.output_dir / "conversion-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="official seedvr2_ema_3b.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model_configs/seedvr2_3b.json"))
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-source-hash", action="store_true", help="only for fast local iteration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    report = convert(args)
    print(json.dumps({"audit": report["audit"], "dry_run": report["dry_run"]}, indent=2))


if __name__ == "__main__":
    main()
