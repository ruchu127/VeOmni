from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ..seedvr2_core import NaDiT
from .configuration_seedvr2_transformer import SeedVR2TransformerConfig


@dataclass
class SeedVR2TransformerOutput(ModelOutput):
    loss: dict[str, torch.Tensor] | None = None
    predictions: torch.Tensor | None = None


class SeedVR2TransformerModel(PreTrainedModel):
    config_class = SeedVR2TransformerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["NaMMSRTransformerBlock"]
    _checkpoint_conversion_mapping = {"^": "dit."}

    def __init__(self, config: SeedVR2TransformerConfig, **kwargs):
        super().__init__(config)
        self.dit = NaDiT(
            vid_in_channels=config.vid_in_channels,
            vid_out_channels=config.vid_out_channels,
            vid_dim=config.vid_dim,
            vid_out_norm=config.vid_out_norm,
            txt_in_dim=config.txt_in_dim,
            txt_in_norm=config.txt_in_norm,
            txt_dim=config.txt_dim,
            emb_dim=config.emb_dim,
            heads=config.heads,
            head_dim=config.head_dim,
            expand_ratio=config.expand_ratio,
            norm=config.norm,
            norm_eps=config.norm_eps,
            ada=config.ada,
            qk_bias=config.qk_bias,
            qk_norm=config.qk_norm,
            patch_size=config.patch_size,
            num_layers=config.num_layers,
            mm_layers=config.mm_layers,
            mlp_type=config.mlp_type,
            block_type=config.block_type,
            window=config.window,
            window_method=config.window_method,
            rope_type=config.rope_type,
            rope_dim=config.rope_dim,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.dit.set_gradient_checkpointing(True)

    def gradient_checkpointing_disable(self):
        self.dit.set_gradient_checkpointing(False)

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        vid_shape: torch.Tensor,
        txt_shape: torch.Tensor,
        timestep: torch.Tensor,
        training_target: torch.Tensor | None = None,
        disable_cache: bool = False,
        **kwargs,
    ) -> SeedVR2TransformerOutput:
        prediction = self.dit(
            vid=vid,
            txt=txt,
            vid_shape=vid_shape,
            txt_shape=txt_shape,
            timestep=timestep,
            disable_cache=disable_cache,
        ).vid_sample
        loss = None
        if training_target is not None:
            loss = {"mse": F.mse_loss(prediction.float(), training_target.float())}
        return SeedVR2TransformerOutput(loss=loss, predictions=prediction)
