#!/usr/bin/env python3
"""Pack cached SeedVR2 latent pairs into VeOmni offline-training parquet.

The public SeedVR2 repository ships inference code, not its training data
pipeline.  This boundary tool accepts latents produced by a pinned copy of the
upstream VAE (or equivalent, parity-checked preprocessing), validates the
training contract, and writes the byte-pickled parquet format consumed by
``dit_offline``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset


FIELDS = ("clean_latents", "degraded_latents", "prompt_embeds")


def _load_tensor(path: Path, field: str) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        candidates = (field, "latents", "embeddings", "prompt_embeds", "tensor")
        matches = [value[key] for key in candidates if key in value and isinstance(value[key], torch.Tensor)]
        if len(matches) != 1:
            raise ValueError(f"{path}: cannot select one tensor for {field}; keys={sorted(value)}")
        value = matches[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path}: expected a tensor for {field}, got {type(value).__name__}")
    return value.detach().cpu()


def _validate(sample: dict[str, torch.Tensor], sample_id: str) -> None:
    clean, degraded, text = (sample[field] for field in FIELDS)
    if clean.ndim == 5 and clean.shape[0] == 1:
        clean = sample["clean_latents"] = clean[0]
    if degraded.ndim == 5 and degraded.shape[0] == 1:
        degraded = sample["degraded_latents"] = degraded[0]
    if clean.ndim != 4 or clean.shape[0] != 16:
        raise ValueError(f"{sample_id}: clean_latents must be [16,T,H,W], got {tuple(clean.shape)}")
    if degraded.shape != clean.shape:
        raise ValueError(f"{sample_id}: degraded_latents {tuple(degraded.shape)} != clean {tuple(clean.shape)}")
    if text.ndim == 3 and text.shape[0] == 1:
        text = sample["prompt_embeds"] = text[0]
    if text.ndim != 2 or text.shape[-1] != 5120:
        raise ValueError(f"{sample_id}: prompt_embeds must be [L,5120], got {tuple(text.shape)}")
    for field, tensor in sample.items():
        if not tensor.is_floating_point():
            raise TypeError(f"{sample_id}: {field} must be floating point, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{sample_id}: {field} contains non-finite values")


def _pickle_row(sample: dict[str, torch.Tensor], sample_id: str) -> dict[str, bytes]:
    row: dict[str, Any] = {**sample, "sample_id": sample_id}
    return {key: pickle.dumps(value) for key, value in row.items()}


def _write_shard(rows: list[dict[str, bytes]], output_dir: Path, index: int) -> None:
    Dataset.from_list(rows).to_parquet(str(output_dir / f"shard_{index:04d}.parquet"))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def pack_manifest(manifest: Path, output_dir: Path, shard_size: int, pad_to_multiple: int) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"manifest is empty: {manifest}")
    if pad_to_multiple > 1:
        entries += [entries[index % len(entries)] for index in range((-len(entries)) % pad_to_multiple)]

    rows: list[dict[str, bytes]] = []
    shapes: dict[str, list[list[int]]] = {field: [] for field in FIELDS}
    shard_index = 0
    for index, entry in enumerate(entries):
        missing = sorted(set(FIELDS) - entry.keys())
        if missing:
            raise ValueError(f"manifest row {index} is missing {missing}")
        sample_id = str(entry.get("sample_id", index))
        sample = {field: _load_tensor(_resolve(manifest.parent, entry[field]), field) for field in FIELDS}
        _validate(sample, sample_id)
        for field, tensor in sample.items():
            shape = list(tensor.shape)
            if shape not in shapes[field]:
                shapes[field].append(shape)
        rows.append(_pickle_row(sample, sample_id))
        if len(rows) == shard_size:
            _write_shard(rows, output_dir, shard_index)
            rows, shard_index = [], shard_index + 1
    if rows:
        _write_shard(rows, output_dir, shard_index)
        shard_index += 1

    report = {
        "format": "veomni-dit-offline-v1",
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "samples": len(entries),
        "shards": shard_index,
        "shapes": shapes,
        "fields": list(FIELDS),
    }
    report_path = output_dir.parent / f"{output_dir.name}.manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def make_toy_inputs(output_dir: Path, count: int, seed: int) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty toy input directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lines = []
    for index in range(count):
        clean = torch.randn((16, 1, 2, 2), generator=generator, dtype=torch.bfloat16)
        degraded = clean + 0.05 * torch.randn(clean.shape, generator=generator, dtype=torch.bfloat16)
        prompt = torch.randn((2, 5120), generator=generator, dtype=torch.bfloat16)
        paths = {}
        for field, tensor in zip(FIELDS, (clean, degraded, prompt)):
            path = output_dir / f"{index:04d}_{field}.pt"
            torch.save(tensor, path)
            paths[field] = path.name
        lines.append(json.dumps({"sample_id": f"toy-{index:04d}", **paths}))
    manifest = output_dir / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSONL with paths for the three required tensor fields")
    source.add_argument("--make-toy", type=Path, metavar="DIR", help="create deterministic contract-valid toy inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--toy-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--pad-to-multiple", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_size < 1 or args.pad_to_multiple < 1 or args.toy_samples < 1:
        raise ValueError("sample, shard, and padding counts must be positive")
    manifest = args.manifest
    if args.make_toy is not None:
        manifest = make_toy_inputs(args.make_toy, args.toy_samples, args.seed)
    report = pack_manifest(manifest.resolve(), args.output_dir.resolve(), args.shard_size, args.pad_to_multiple)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
