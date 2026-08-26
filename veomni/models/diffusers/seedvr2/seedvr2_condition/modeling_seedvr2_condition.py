from __future__ import annotations

from typing import Any

import torch
from einops import rearrange
from transformers import PreTrainedModel

from .configuration_seedvr2_condition import SeedVR2ConditionConfig


class SeedVR2ConditionModel(PreTrainedModel):
    """Build SeedVR2 flow-matching inputs from cached latent training pairs.

    Each example contains a clean target latent, its degraded/restoration input
    latent, and the pinned upstream 5120-wide prompt embedding. The expensive
    VAE/text encoding stage is deliberately an offline data-preparation step.
    """

    config_class = SeedVR2ConditionConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: SeedVR2ConditionConfig, **kwargs):
        super().__init__(config)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.seed)
        self.post_init()

    @torch.no_grad()
    def get_condition(self, **inputs) -> dict[str, Any]:
        required = {"clean_latents", "degraded_latents", "prompt_embeds"}
        missing = sorted(required - inputs.keys())
        if missing:
            raise ValueError(
                "SeedVR2 online encoding is not bundled because the upstream release is inference-only. "
                f"Prepare cached latents first; missing fields: {missing}"
            )
        return inputs

    @staticmethod
    def _one(value, name: str) -> torch.Tensor:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(
                    f"SeedVR2 currently requires micro_batch_size=1 for variable-size latents; {name} has {len(value)}"
                )
            value = value[0]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value

    @torch.no_grad()
    def process_condition(self, clean_latents, degraded_latents, prompt_embeds, **kwargs) -> dict[str, Any]:
        clean = self._one(clean_latents, "clean_latents")
        degraded = self._one(degraded_latents, "degraded_latents")
        text = self._one(prompt_embeds, "prompt_embeds")
        if clean.ndim == 5 and clean.shape[0] == 1:
            clean = clean[0]
        if degraded.ndim == 5 and degraded.shape[0] == 1:
            degraded = degraded[0]
        if clean.ndim != 4 or degraded.ndim != 4:
            raise ValueError("clean_latents and degraded_latents must have shape [C, T, H, W]")
        if clean.shape != degraded.shape or clean.shape[0] != self.config.latent_channels:
            raise ValueError(
                f"expected matching {self.config.latent_channels}-channel latents, "
                f"got {tuple(clean.shape)} and {tuple(degraded.shape)}"
            )
        if text.ndim == 3 and text.shape[0] == 1:
            text = text[0]
        if text.ndim != 2 or text.shape[-1] != self.config.text_dim:
            raise ValueError(f"prompt_embeds must have shape [L, {self.config.text_dim}], got {tuple(text.shape)}")

        noise = torch.randn(clean.shape, dtype=clean.dtype, device="cpu", generator=self.generator).to(clean.device)
        if self.config.fixed_timestep is None:
            timestep = torch.rand(1, device="cpu", generator=self.generator).to(clean.device)
            timestep = timestep * self.config.num_train_timesteps
        else:
            timestep = torch.tensor([self.config.fixed_timestep], device=clean.device, dtype=torch.float32)
        ratio = timestep.to(clean.dtype) / self.config.num_train_timesteps
        while ratio.ndim < clean.ndim:
            ratio = ratio.unsqueeze(-1)
        noisy = (1 - ratio) * clean + ratio * noise
        if self.config.condition_noise_scale:
            condition_noise = torch.randn(
                degraded.shape, dtype=degraded.dtype, device="cpu", generator=self.generator
            ).to(degraded.device)
            degraded = degraded + self.config.condition_noise_scale * condition_noise

        clean_flat = rearrange(clean, "c t h w -> (t h w) c")
        noisy_flat = rearrange(noisy, "c t h w -> (t h w) c")
        degraded_flat = rearrange(degraded, "c t h w -> (t h w) c")
        condition_mask = torch.ones((*degraded_flat.shape[:-1], 1), device=degraded.device, dtype=degraded.dtype)
        frames, height, width = clean.shape[1:]
        return {
            "vid": torch.cat([noisy_flat, degraded_flat, condition_mask], dim=-1),
            "txt": text,
            "vid_shape": torch.tensor([[frames, height, width]], device=clean.device),
            "txt_shape": torch.tensor([[text.shape[0]]], device=text.device),
            "timestep": timestep.to(clean.dtype),
            "training_target": rearrange(noise, "c t h w -> (t h w) c") - clean_flat,
        }
