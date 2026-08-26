import torch

from veomni.models.diffusers.seedvr2.seedvr2_condition.configuration_seedvr2_condition import (
    SeedVR2ConditionConfig,
)
from veomni.models.diffusers.seedvr2.seedvr2_condition.modeling_seedvr2_condition import SeedVR2ConditionModel
from veomni.models.diffusers.seedvr2.seedvr2_transformer.configuration_seedvr2_transformer import (
    SeedVR2TransformerConfig,
)
from veomni.models.diffusers.seedvr2.seedvr2_transformer.modeling_seedvr2_transformer import (
    SeedVR2TransformerModel,
)


def tiny_config() -> SeedVR2TransformerConfig:
    return SeedVR2TransformerConfig(
        vid_in_channels=9,
        vid_out_channels=4,
        vid_dim=32,
        vid_out_norm="rms",
        txt_in_dim=48,
        txt_in_norm=None,
        txt_dim=32,
        emb_dim=192,
        heads=4,
        head_dim=8,
        expand_ratio=2,
        norm="rms",
        qk_norm="rms",
        patch_size=(1, 2, 2),
        num_layers=2,
        mm_layers=1,
        window=(1, 1, 1),
        window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"],
        rope_dim=6,
    )


def test_seedvr2_tiny_condition_forward_backward():
    torch.manual_seed(7)
    condition = SeedVR2ConditionModel(
        SeedVR2ConditionConfig(fixed_timestep=1000, latent_channels=4, text_dim=48, seed=11)
    )
    batch = condition.process_condition(
        clean_latents=[torch.randn(4, 1, 2, 2)],
        degraded_latents=[torch.randn(4, 1, 2, 2)],
        prompt_embeds=[torch.randn(2, 48)],
    )
    model = SeedVR2TransformerModel(tiny_config())
    model.gradient_checkpointing_enable()

    output = model(**batch)
    loss = output.loss["mse"]
    loss.backward()

    assert output.predictions.shape == (4, 4)
    assert torch.isfinite(loss)
    assert model.dit.vid_in.proj.weight.grad is not None
    assert torch.isfinite(model.dit.vid_in.proj.weight.grad).all()


def test_seedvr2_checkpoint_namespace_and_round_trip(tmp_path):
    model = SeedVR2TransformerModel(tiny_config())
    state_dict = model.state_dict()

    assert "dit.vid_in.proj.weight" in state_dict
    assert "dit.blocks.0.attn.proj_qkv.vid.weight" in state_dict
    assert "dit.blocks.1.attn.proj_qkv.all.weight" in state_dict

    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = SeedVR2TransformerModel.from_pretrained(tmp_path)

    assert set(restored.state_dict()) == set(state_dict)
    for key in state_dict:
        assert torch.equal(restored.state_dict()[key], state_dict[key]), key


def test_dit_trainer_registers_offline_transform():
    from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY
    from veomni.trainer import dit_trainer  # noqa: F401

    assert "dit_offline" in DATA_TRANSFORM_REGISTRY.valid_keys()
