#!/usr/bin/env python3
"""Create an evidence-backed VeOmni migration packet from local upstream source.

The analyzer is deliberately conservative. It reports source evidence and an
initial route, but leaves semantic decisions as TBD when files cannot prove them.
It has no network or third-party Python dependency.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
MAX_SOURCE_BYTES = 2_000_000
TOKEN_GROUPS = {
    "text": ("text", "token", "caption", "prompt", "language", "encoder_hidden_states"),
    "vision": ("vision", "image", "video", "pixel", "frame", "vae", "latent"),
    "audio": ("audio", "speech", "waveform", "mel", "vocoder"),
    "diffusion": ("diffusion", "dit", "noise", "timestep", "scheduler", "sigma", "denois"),
    "moe": ("moe", "expert", "router", "num_experts"),
}


@dataclass
class PythonInventory:
    files_scanned: int = 0
    parse_errors: list[str] = field(default_factory=list)
    classes: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    model_name: str
    model_slug: str
    upstream_path: str
    upstream_revision: str
    upstream_remote: str
    license_files: list[str]
    config_files: list[str]
    config_signals: dict[str, Any]
    python: PythonInventory
    token_counts: dict[str, int]
    detected_category: str
    detected_ecosystem: str
    is_moe: bool
    recommended_route: str
    confidence_notes: list[str]
    backend: str
    veomni_revision: str
    reference_paths: list[str]


def _run_git(path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
    return result.stdout.strip() or "UNKNOWN"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError("model name does not contain a usable ASCII letter or digit")
    return slug


def iter_files(root: Path, suffixes: tuple[str, ...] | None = None) -> Iterable[Path]:
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
        for name in sorted(filenames):
            path = Path(current, name)
            if suffixes is None or path.suffix.lower() in suffixes:
                yield path


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def scan_python(root: Path) -> PythonInventory:
    inventory = PythonInventory()
    for path in iter_files(root, (".py",)):
        if path.stat().st_size > MAX_SOURCE_BYTES:
            inventory.parse_errors.append(f"{relative(path, root)}: skipped (> {MAX_SOURCE_BYTES} bytes)")
            continue
        inventory.files_scanned += 1
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            inventory.parse_errors.append(f"{relative(path, root)}: {type(exc).__name__}: {exc}")
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inventory.functions.append({"name": node.name, "file": relative(path, root)})
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except Exception:
                        bases.append(type(base).__name__)
                methods = sorted(
                    child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                inventory.classes.append(
                    {"name": node.name, "bases": bases, "methods": methods, "file": relative(path, root)}
                )
            elif isinstance(node, ast.Import):
                inventory.imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                inventory.imports.append(node.module)
    inventory.classes.sort(key=lambda item: (item["file"], item["name"]))
    inventory.functions.sort(key=lambda item: (item["file"], item["name"]))
    inventory.imports = sorted(set(inventory.imports))
    return inventory


def _walk_json(value: Any, result: dict[str, list[Any]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {
                "model_type",
                "architectures",
                "_class_name",
                "processor_class",
                "num_experts",
                "num_local_experts",
            }:
                result.setdefault(key, []).append(child)
            _walk_json(child, result)
    elif isinstance(value, list):
        for child in value:
            _walk_json(child, result)


def scan_configs(root: Path) -> tuple[list[str], dict[str, Any]]:
    config_files: list[str] = []
    signals: dict[str, list[Any]] = {}
    for path in iter_files(root, (".json",)):
        name = path.name.lower()
        if "config" not in name and name not in {"model_index.json", "params.json"}:
            continue
        config_files.append(relative(path, root))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        _walk_json(payload, signals)
    return sorted(config_files), signals


def count_tokens(root: Path, inventory: PythonInventory, config_signals: dict[str, Any]) -> dict[str, int]:
    corpus_parts = [
        " ".join(path.as_posix().lower().split("/")) for path in iter_files(root) if path.stat().st_size < 500_000
    ]
    corpus_parts.extend(item["name"].lower() for item in inventory.classes)
    corpus_parts.extend(inventory.imports)
    corpus_parts.append(json.dumps(config_signals, sort_keys=True).lower())
    corpus = "\n".join(corpus_parts)
    return {group: sum(corpus.count(token) for token in tokens) for group, tokens in TOKEN_GROUPS.items()}


def infer(
    config_signals: dict[str, Any],
    inventory: PythonInventory,
    token_counts: dict[str, int],
) -> tuple[str, str, bool, list[str]]:
    notes: list[str] = []
    has_hf_config = bool(config_signals.get("model_type") or config_signals.get("architectures"))
    has_diffusers_config = bool(config_signals.get("_class_name"))
    bases = [base for item in inventory.classes for base in item["bases"]]
    imports_transformers = any(
        name == "transformers" or name.startswith("transformers.") for name in inventory.imports
    )
    imports_diffusers = any(name == "diffusers" or name.startswith("diffusers.") for name in inventory.imports)
    native_transformers_class = any("PreTrainedModel" in base for base in bases)
    native_diffusers_class = any("ModelMixin" in base or "DiffusionPipeline" in base for base in bases)

    if has_diffusers_config or native_diffusers_class:
        ecosystem = "diffusers"
    elif has_hf_config or native_transformers_class:
        ecosystem = "transformers"
    else:
        ecosystem = "custom-pytorch"
        notes.append("No standard Transformers/Diffusers config contract was proven; inspect construction manually.")
        if imports_diffusers:
            notes.append(
                "Diffusers is imported by an auxiliary component, but no native trainable Diffusers model contract was proven."
            )
        if imports_transformers:
            notes.append(
                "Transformers is imported by an auxiliary component, but no native trainable PreTrainedModel contract was proven."
            )

    is_moe = (
        bool(config_signals.get("num_experts") or config_signals.get("num_local_experts")) or token_counts["moe"] >= 4
    )
    if token_counts["diffusion"] >= 4:
        category = "dit"
    elif token_counts["audio"] >= 3 and token_counts["vision"] >= 3:
        category = "omni"
    elif token_counts["vision"] >= 3 and token_counts["text"] >= 2:
        category = "vlm"
    else:
        category = "text"

    if category == "text" and ecosystem == "custom-pytorch":
        notes.append(
            "Text was selected by fallback; confirm modality and objective from the forward/training call graph."
        )
    if is_moe:
        notes.append(
            "MoE signals were detected; verify expert tensor layout and the exact parallel-plan parameter paths."
        )
    return category, ecosystem, is_moe, notes


def route_for(category: str, ecosystem: str) -> str:
    if ecosystem == "transformers":
        return "transformers-patchgen"
    if ecosystem == "diffusers":
        return "diffusers-native-wrapper"
    if category == "dit":
        return "custom-dit-wrapper-and-explicit-converter"
    return "custom-pytorch-wrapper-and-explicit-converter"


def references_for(root: Path, category: str, is_moe: bool, ecosystem: str) -> list[str]:
    candidates: list[str]
    if ecosystem == "transformers" and category == "text":
        candidates = ["veomni/models/transformers/qwen3_moe" if is_moe else "veomni/models/transformers/qwen3"]
    elif ecosystem == "transformers" and category == "vlm":
        candidates = ["veomni/models/transformers/qwen3_vl_moe" if is_moe else "veomni/models/transformers/qwen3_vl"]
    elif ecosystem == "transformers" and category == "omni":
        candidates = [
            "veomni/models/transformers/qwen3_omni_moe" if is_moe else "veomni/models/transformers/qwen2_5_omni"
        ]
    elif category == "dit":
        candidates = [
            "veomni/models/diffusers/wan_t2v",
            "veomni/models/diffusers/minimax_h3",
            "docs/usage/support_new_models/dit_model_guide.md",
        ]
    else:
        candidates = ["veomni/models", "veomni/trainer"]
    return [candidate for candidate in candidates if (root / candidate).exists()]


def proposed_files(slug: str, category: str, ecosystem: str, backend: str, is_moe: bool) -> list[str]:
    files: list[str] = []
    if ecosystem == "transformers":
        base = f"veomni/models/transformers/{slug}"
        files.extend([f"{base}/__init__.py", f"{base}/{slug}_gpu_patch_gen_config.py"])
        if backend in {"npu", "both"}:
            files.append(f"{base}/{slug}_npu_patch_gen_config.py")
        if is_moe:
            files.extend([f"{base}/parallel_plan.py", f"{base}/checkpoint_tensor_converter.py"])
        config = f"configs/{'text' if category == 'text' else 'multimodal'}/{slug}.yaml"
    else:
        base = f"veomni/models/diffusers/{slug}"
        files.extend(
            [
                f"{base}/__init__.py",
                f"{base}/{slug}_condition/__init__.py",
                f"{base}/{slug}_condition/configuration_{slug}_condition.py",
                f"{base}/{slug}_condition/modeling_{slug}_condition.py",
                f"{base}/{slug}_transformer/__init__.py",
                f"{base}/{slug}_transformer/configuration_{slug}_transformer.py",
                f"{base}/{slug}_transformer/modeling_{slug}_transformer.py",
            ]
        )
        config = f"configs/dit/{slug}.yaml"
    files.extend(
        [
            config,
            f"scripts/model_conversion/convert_{slug}.py",
            f"tests/models/test_{slug}.py",
            f"tests/toy_config/{slug}_toy/config.json",
            f"docs/examples/{slug}_migration_report.md",
        ]
    )
    return files


def build_evidence(args: argparse.Namespace) -> Evidence:
    upstream = args.upstream.resolve()
    veomni = args.veomni_root.resolve()
    python_inventory = scan_python(upstream)
    config_files, config_signals = scan_configs(upstream)
    token_counts = count_tokens(upstream, python_inventory, config_signals)
    category, ecosystem, is_moe, notes = infer(config_signals, python_inventory, token_counts)
    if args.category != "auto":
        notes.append(f"Category was explicitly overridden from inferred {category!r} to {args.category!r}.")
        category = args.category
    slug = slugify(args.model_name)
    licenses = [
        relative(path, upstream)
        for path in iter_files(upstream)
        if path.name.lower().startswith(("license", "copying", "notice"))
    ]
    if not licenses:
        notes.append("No license file was detected. Resolve licensing before copying source.")
    return Evidence(
        model_name=args.model_name,
        model_slug=slug,
        upstream_path=str(upstream),
        upstream_revision=_run_git(upstream, "rev-parse", "HEAD"),
        upstream_remote=_run_git(upstream, "remote", "get-url", "origin"),
        license_files=sorted(licenses),
        config_files=config_files,
        config_signals=config_signals,
        python=python_inventory,
        token_counts=token_counts,
        detected_category=category,
        detected_ecosystem=ecosystem,
        is_moe=is_moe,
        recommended_route=route_for(category, ecosystem),
        confidence_notes=notes,
        backend=args.backend,
        veomni_revision=_run_git(veomni, "rev-parse", "HEAD"),
        reference_paths=references_for(veomni, category, is_moe, ecosystem),
    )


def render_plan(evidence: Evidence) -> str:
    files = proposed_files(
        evidence.model_slug,
        evidence.detected_category,
        evidence.detected_ecosystem,
        evidence.backend,
        evidence.is_moe,
    )

    def class_score(item: dict[str, Any]) -> tuple[int, str, str]:
        name = item["name"].lower()
        path = item["file"].lower()
        score = 0
        if path.startswith("models/"):
            score += 4
        if any(token in name for token in ("transformer", "model", "dit", "backbone")):
            score += 5
        if any(token in name for token in ("output", "attention", "embedding", "norm", "mlp", "vae")):
            score -= 2
        if path.startswith(("common/", "data/", "projects/")):
            score -= 2
        return (-score, path, name)

    classes = sorted((item for item in evidence.python.classes if "forward" in item["methods"]), key=class_score)
    class_lines = [
        f"- `{item['name']}` in `{item['file']}` (bases: {', '.join(item['bases']) or 'none recorded'})"
        for item in classes[:40]
    ] or ["- TBD: no class with a directly defined `forward` method was detected."]
    notes = [f"- {note}" for note in evidence.confidence_notes] or ["- No analyzer warnings."]
    refs = [f"- `{path}`" for path in evidence.reference_paths] or ["- TBD: select a structural VeOmni reference."]
    file_lines = [f"- [ ] `{path}`" for path in files]
    return f"""# {evidence.model_name} → VeOmni migration plan

Generated from local evidence. Analyzer inferences must be reviewed before implementation.

## Source pin

- Upstream: `{evidence.upstream_remote}`
- Revision: `{evidence.upstream_revision}`
- Local source: `{evidence.upstream_path}`
- License files: {", ".join(f"`{item}`" for item in evidence.license_files) or "TBD"}
- VeOmni base: `{evidence.veomni_revision}`
- Target backend: `{evidence.backend}`

## Analyzer inference

- Category: `{evidence.detected_category}`
- Ecosystem: `{evidence.detected_ecosystem}`
- MoE signals: `{evidence.is_moe}`
- Initial route: `{evidence.recommended_route}`

### Review notes

{chr(10).join(notes)}

### Forward-bearing upstream classes

{chr(10).join(class_lines)}

### Candidate structural references

{chr(10).join(refs)}

## Contract decisions

- Trainable module and objective: TBD
- Frozen/condition modules: TBD
- Raw example schema: TBD
- Model input/output tensor contract: TBD
- Checkpoint format and immutable fingerprint: TBD
- Parallel dimensions and divisibility constraints: TBD
- Forward-parity tensors and tolerances: TBD
- Tiny-overfit loss acceptance rule: TBD
- Save/resume comparison tolerance: TBD

## Migration scope and file checklist

This is an initial list. Add or remove entries only with a reason in the decision log.

{chr(10).join(file_lines)}

## Implementation sequence

1. Pin source, weights, licenses, dependencies, and fill the contract decisions.
2. Inventory checkpoint keys/shapes and complete mapping rules before loading with relaxed strictness.
3. Implement registry/config construction and a CPU/meta-device tiny model.
4. Implement the condition/data path and one scalar-loss forward/backward batch.
5. Convert and strictly load real weights; run fixed-input upstream parity.
6. Run a deterministic tiny overfit fixture without distributed optimizations.
7. Add checkpoint save plus fresh-process resume and compare with uninterrupted control.
8. Add FSDP, then SP/EP, then backend kernels one dimension at a time.
9. Run target-backend E2E and complete both reports.

## Validation matrix

| Check | Command | Pass criterion | Status/evidence |
|---|---|---|---|
| clean import + registry | TBD | fresh process resolves config and model | TBD |
| converter coverage | TBD | all required keys accounted; strict load | TBD |
| forward parity | TBD | agreed abs/rel tolerance | TBD |
| forward/backward | TBD | finite loss and trainable gradients | TBD |
| tiny overfit | TBD | predeclared loss rule passes | TBD |
| distributed topology | TBD | target FSDP/SP/EP run completes | TBD |
| checkpoint save/resume | TBD | next step matches control tolerance | TBD |
| exported inference load | TBD | target consumer loads and runs | TBD |
| quality/tests | TBD | selected lint and tests pass | TBD |

## Decision log

| Date | Decision | Direct evidence | Consequence |
|---|---|---|---|
| TBD | TBD | TBD | TBD |
"""


def render_manifest(evidence: Evidence) -> str:
    return f"""schema_version: 1
model:
  name: {json.dumps(evidence.model_name)}
  slug: {evidence.model_slug}
  category: {evidence.detected_category}
  ecosystem: {evidence.detected_ecosystem}
  is_moe: {str(evidence.is_moe).lower()}
source:
  repository: {json.dumps(evidence.upstream_remote)}
  revision: {json.dumps(evidence.upstream_revision)}
  local_checkout: {json.dumps(evidence.upstream_path)}
  license_files: {json.dumps(evidence.license_files)}
weights:
  repository_or_path: TBD
  revision_or_sha256: TBD
  format: TBD
target:
  veomni_revision: {evidence.veomni_revision}
  backends: {evidence.backend}
  integration_route: {evidence.recommended_route}
contract:
  trainable_module: TBD
  objective: TBD
  raw_example_schema: TBD
  model_inputs: TBD
  model_outputs: TBD
  checkpoint_mapping: TBD
validation:
  forward_parity: TBD
  tiny_overfit: TBD
  checkpoint_resume: TBD
"""


def copy_templates(script_path: Path, output: Path, replacements: dict[str, str]) -> None:
    template_root = script_path.parent.parent / "assets" / "templates"
    for source in sorted(template_root.iterdir()):
        if not source.is_file():
            continue
        content = source.read_text(encoding="utf-8")
        for key, value in replacements.items():
            content = content.replace("{{" + key + "}}", value)
        (output / source.name).write_text(content, encoding="utf-8", newline="\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Human-readable target model name")
    parser.add_argument("--upstream", required=True, type=Path, help="Pinned local upstream source checkout")
    parser.add_argument("--veomni-root", type=Path, default=Path.cwd(), help="VeOmni repository root")
    parser.add_argument("--backend", choices=("gpu", "npu", "both"), default="both")
    parser.add_argument("--category", choices=("auto", "text", "vlm", "omni", "dit"), default="auto")
    parser.add_argument("--output", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--force", action="store_true", help="Replace analyzer-owned files in an existing directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.upstream.is_dir():
        print(f"error: upstream directory does not exist: {args.upstream}", file=sys.stderr)
        return 2
    if not (args.veomni_root / "veomni").is_dir() or not (args.veomni_root / ".git").exists():
        print(f"error: not a VeOmni checkout: {args.veomni_root}", file=sys.stderr)
        return 2
    owned = {"analysis.json", "migration-plan.md", "migration-manifest.yaml"}
    template_root = Path(__file__).resolve().parent.parent / "assets" / "templates"
    owned.update(path.name for path in template_root.iterdir() if path.is_file())
    if args.output.exists() and not args.force:
        conflicts = sorted(name for name in owned if (args.output / name).exists())
        if conflicts:
            print(
                f"error: output already contains analyzer files: {', '.join(conflicts)}; use --force", file=sys.stderr
            )
            return 2
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence(args)
    payload = asdict(evidence)
    (args.output / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (args.output / "migration-plan.md").write_text(render_plan(evidence), encoding="utf-8", newline="\n")
    (args.output / "migration-manifest.yaml").write_text(render_manifest(evidence), encoding="utf-8", newline="\n")
    replacements = {
        "MODEL_NAME": evidence.model_name,
        "MODEL_SLUG": evidence.model_slug,
        "UPSTREAM_PATH": evidence.upstream_path,
        "UPSTREAM_REVISION": evidence.upstream_revision,
        "BACKENDS": evidence.backend,
        "GENERATED_AT": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    copy_templates(Path(__file__).resolve(), args.output, replacements)
    print(f"wrote migration packet: {args.output.resolve()}")
    print(
        f"inference: category={evidence.detected_category}, ecosystem={evidence.detected_ecosystem}, "
        f"route={evidence.recommended_route}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
