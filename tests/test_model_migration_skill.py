import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = REPO_ROOT / ".agents/skills/veomni-model-migration/scripts/analyze_upstream.py"


def run_analyzer(upstream: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--model-name",
            "Example Model",
            "--upstream",
            str(upstream),
            "--veomni-root",
            str(REPO_ROOT),
            "--output",
            str(output),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_analyzer_creates_transformers_migration_packet(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    (upstream / "config.json").write_text(
        json.dumps({"model_type": "example", "architectures": ["ExampleForCausalLM"]}), encoding="utf-8"
    )
    (upstream / "modeling_example.py").write_text(
        "from transformers import PreTrainedModel\n"
        "class ExampleForCausalLM(PreTrainedModel):\n"
        "    def forward(self, input_ids):\n"
        "        return input_ids\n",
        encoding="utf-8",
    )
    output = tmp_path / "packet"

    result = run_analyzer(upstream, output, "--backend", "npu", "--category", "text")

    assert result.returncode == 0, result.stderr
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["detected_ecosystem"] == "transformers"
    assert analysis["recommended_route"] == "transformers-patchgen"
    assert analysis["license_files"] == ["LICENSE"]
    plan = (output / "migration-plan.md").read_text(encoding="utf-8")
    assert "veomni/models/transformers/example_model/example_model_npu_patch_gen_config.py" in plan
    assert (output / "weight-converter.py").is_file()
    assert (output / "e2e-report.md").is_file()


def test_analyzer_detects_custom_dit_and_refuses_implicit_overwrite(tmp_path):
    upstream = tmp_path / "upstream"
    source_dir = upstream / "models" / "dit"
    source_dir.mkdir(parents=True)
    (upstream / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    (source_dir / "model.py").write_text(
        "import diffusers\n"
        "import torch\n"
        "class AuxiliaryVAE(diffusers.AutoencoderKL):\n"
        "    pass\n"
        "class VideoDiffusionTransformer(torch.nn.Module):\n"
        "    def forward(self, video_latent, timestep, noise):\n"
        "        return video_latent + noise\n",
        encoding="utf-8",
    )
    output = tmp_path / "packet"

    first = run_analyzer(upstream, output, "--category", "dit")
    second = run_analyzer(upstream, output, "--category", "dit")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 2
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["detected_category"] == "dit"
    assert analysis["detected_ecosystem"] == "custom-pytorch"
    assert analysis["recommended_route"] == "custom-dit-wrapper-and-explicit-converter"
    assert "use --force" in second.stderr
