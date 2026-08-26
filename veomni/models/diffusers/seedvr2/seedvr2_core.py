# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""SeedVR2 NaDiT core adapted for VeOmni.

Source: https://github.com/ByteDance-Seed/SeedVR
Revision: e4de8c24441a67e1b7df56abea10645059bb1185
Upstream paths: ``models/dit_v2`` and ``common/cache.py``.

The module preserves the upstream parameter hierarchy so the public checkpoint
can be converted by adding only the VeOmni wrapper prefix. CUDA-only Apex and
FlashAttention calls are replaced with state-dict-compatible PyTorch reference
implementations. Sequence parallelism is intentionally not implemented here;
FSDP data parallelism remains available through the normal VeOmni trainer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple


class Cache:
    def __init__(self, disable: bool = False, prefix: str = "", cache: Optional[dict] = None):
        self.cache = cache if cache is not None else {}
        self.disable = disable
        self.prefix = prefix

    def __call__(self, key: str, fn: Callable):
        if self.disable:
            return fn()
        key = self.prefix + key
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def namespace(self, namespace: str) -> "Cache":
        return Cache(self.disable, self.prefix + namespace + ".", self.cache)

    def get(self, key: str):
        return self.cache[self.prefix + key]


def flatten(hidden_states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not hidden_states:
        raise ValueError("at least one tensor is required")
    shape = torch.stack([torch.tensor(item.shape[:-1], device=hidden_states[0].device) for item in hidden_states])
    return torch.cat([item.flatten(0, -2) for item in hidden_states]), shape


def unflatten(hidden_states: torch.Tensor, hidden_shape: torch.Tensor) -> list[torch.Tensor]:
    lengths = hidden_shape.prod(-1)
    return [item.unflatten(0, shape.tolist()) for item, shape in zip(hidden_states.split(lengths.tolist()), hidden_shape)]


def _concat(vid: torch.Tensor, txt: torch.Tensor, vid_len: torch.Tensor, txt_len: torch.Tensor) -> torch.Tensor:
    vid_parts = torch.split(vid, vid_len.tolist())
    txt_parts = torch.split(txt, txt_len.tolist())
    return torch.cat(list(chain(*zip(vid_parts, txt_parts))))


def concat_idx(vid_len: torch.Tensor, txt_len: torch.Tensor) -> tuple[Callable, Callable]:
    vid_idx = torch.arange(vid_len.sum(), device=vid_len.device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + txt_len.sum(), device=vid_len.device)
    target_idx = _concat(vid_idx, txt_idx, vid_len, txt_len)
    source_idx = torch.argsort(target_idx)
    return (
        lambda vid, txt: torch.index_select(torch.cat([vid, txt]), 0, target_idx),
        lambda all_states: torch.index_select(all_states, 0, source_idx).split([len(vid_idx), len(txt_idx)]),
    )


def _repeat_concat(
    vid: torch.Tensor,
    txt: torch.Tensor,
    vid_len: torch.Tensor,
    txt_len: torch.Tensor,
    txt_repeat: list[int],
) -> torch.Tensor:
    vid_parts = torch.split(vid, vid_len.tolist())
    txt_parts = torch.split(txt, txt_len.tolist())
    repeated_txt = list(chain(*([[item] * count for item, count in zip(txt_parts, txt_repeat)])))
    return torch.cat(list(chain(*zip(vid_parts, repeated_txt))))


def repeat_concat_idx(vid_len: torch.Tensor, txt_len: torch.Tensor, txt_repeat: torch.Tensor):
    vid_idx = torch.arange(vid_len.sum(), device=vid_len.device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + txt_len.sum(), device=vid_len.device)
    repeat_list = txt_repeat.tolist()
    target_idx = _repeat_concat(vid_idx, txt_idx, vid_len, txt_len, repeat_list)
    source_idx = torch.argsort(target_idx)
    repeated_txt_len = (txt_len * txt_repeat).tolist()

    def unconcat_coalesce(all_states: torch.Tensor):
        vid_out, txt_out = all_states[source_idx].split([len(vid_idx), len(target_idx) - len(vid_idx)])
        pooled = []
        for sample, repeat_count in zip(txt_out.split(repeated_txt_len), repeat_list):
            pooled.append(sample.reshape(-1, repeat_count, *sample.shape[1:]).mean(1))
        return vid_out, torch.cat(pooled)

    return lambda vid, txt: torch.cat([vid, txt])[target_idx], unconcat_coalesce


def window_idx(hidden_shape: torch.Tensor, window_fn: Callable):
    hidden_idx = torch.arange(hidden_shape.prod(-1).sum(), device=hidden_shape.device).unsqueeze(-1)
    samples = unflatten(hidden_idx, hidden_shape)
    windows = [window_fn(item) for item in samples]
    window_count = torch.tensor([len(item) for item in windows], device=hidden_shape.device)
    flattened, target_shape = flatten(list(chain(*windows)))
    target_idx = flattened.squeeze(-1)
    source_idx = torch.argsort(target_idx)
    return (
        lambda hidden_states: torch.index_select(hidden_states, 0, target_idx),
        lambda hidden_states: torch.index_select(hidden_states, 0, source_idx),
        target_shape,
        window_count,
    )


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    value = rearrange(value, "... (d r) -> ... d r", r=2)
    first, second = value.unbind(dim=-1)
    return rearrange(torch.stack((-second, first), dim=-1), "... d r -> ... (d r)")


def apply_rotary_emb(freqs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    dtype = value.dtype
    if freqs.ndim == 2 or value.ndim == 3:
        freqs = freqs[-value.shape[-2] :]
    rotary_dim = freqs.shape[-1]
    middle = value[..., :rotary_dim]
    transformed = middle * freqs.cos() + rotate_half(middle) * freqs.sin()
    return torch.cat((transformed, value[..., rotary_dim:]), dim=-1).to(dtype)


class RotaryFrequencies(nn.Module):
    def __init__(self, dim: int, freqs_for: str, max_freq: int = 10, theta: int = 10000):
        super().__init__()
        if freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        else:
            raise ValueError(f"unsupported frequency type: {freqs_for}")
        self.freqs_for = freqs_for
        self.register_buffer("freqs", freqs)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", positions.to(self.freqs.dtype), self.freqs)
        return torch.repeat_interleave(freqs, 2, dim=-1)

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        axes = []
        for index, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                positions = torch.linspace(-1, 1, steps=dim, device=self.freqs.device)
            else:
                positions = torch.arange(dim, device=self.freqs.device)
            freqs = self.forward(positions)
            view = [1] * len(dims) + [freqs.shape[-1]]
            view[index] = dim
            axes.append(freqs.reshape(view).expand(*dims, freqs.shape[-1]))
        return torch.cat(axes, dim=-1)


class MMRotaryEmbedding3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.rope = RotaryFrequencies(dim // 3, freqs_for="lang")
        self.mm = True

    @lru_cache(maxsize=128)
    def _get_axial_freqs(self, *dims: int) -> torch.Tensor:
        return self.rope.get_axial_freqs(*dims)

    def get_freqs(self, vid_shape: torch.Tensor, txt_shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vid_freqs = self._get_axial_freqs(1024, 128, 128)
        txt_freqs = self._get_axial_freqs(1024)
        vid_items, txt_items = [], []
        for (frames, height, width), length in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_items.append(vid_freqs[length : length + frames, :height, :width].reshape(-1, vid_freqs.size(-1)))
            txt_items.append(txt_freqs[:length].repeat(1, 3).reshape(-1, vid_freqs.size(-1)))
        return torch.cat(vid_items), torch.cat(txt_items)

    def forward(
        self,
        vid_q: torch.Tensor,
        vid_k: torch.Tensor,
        vid_shape: torch.Tensor,
        txt_q: torch.Tensor,
        txt_k: torch.Tensor,
        txt_shape: torch.Tensor,
        cache: Cache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vid_freqs, txt_freqs = cache("mmrope_freqs_3d", lambda: self.get_freqs(vid_shape, txt_shape))
        vid_q = rearrange(apply_rotary_emb(vid_freqs, rearrange(vid_q, "l h d -> h l d").float()), "h l d -> l h d")
        vid_k = rearrange(apply_rotary_emb(vid_freqs, rearrange(vid_k, "l h d -> h l d").float()), "h l d -> l h d")
        txt_q = rearrange(apply_rotary_emb(txt_freqs, rearrange(txt_q, "l h d -> h l d").float()), "h l d -> l h d")
        txt_k = rearrange(apply_rotary_emb(txt_freqs, rearrange(txt_k, "l h d -> h l d").float()), "h l d -> l h d")
        return vid_q, vid_k, txt_q, txt_k


def get_norm_layer(norm_type: Optional[str]):
    def build(dim: int, eps: float, elementwise_affine: bool):
        if norm_type is None:
            return nn.Identity()
        if norm_type in {"layer", "fusedln"}:
            return nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        if norm_type in {"rms", "fusedrms"}:
            return nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        raise ValueError(f"unsupported norm type: {norm_type}")

    return build


@dataclass
class MMArg:
    vid: Any
    txt: Any


def _mm_args(key: str, values: tuple[Any, ...]) -> list[Any]:
    return [getattr(value, key) if isinstance(value, MMArg) else value for value in values]


def _mm_kwargs(key: str, values: dict[str, Any]) -> dict[str, Any]:
    return {name: getattr(value, key) if isinstance(value, MMArg) else value for name, value in values.items()}


class MMModule(nn.Module):
    def __init__(self, module: Callable[..., nn.Module], *args, shared_weights=False, vid_only=False, **kwargs):
        super().__init__()
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        if shared_weights:
            self.all = module(*_mm_args("vid", args), **_mm_kwargs("vid", kwargs))
        else:
            self.vid = module(*_mm_args("vid", args), **_mm_kwargs("vid", kwargs))
            self.txt = None if vid_only else module(*_mm_args("txt", args), **_mm_kwargs("txt", kwargs))

    def forward(self, vid: torch.Tensor, txt: torch.Tensor, *args, **kwargs):
        vid_module = self.all if self.shared_weights else self.vid
        vid = vid_module(vid, *_mm_args("vid", args), **_mm_kwargs("vid", kwargs))
        if not self.vid_only:
            txt_module = self.all if self.shared_weights else self.txt
            txt = txt_module(txt, *_mm_args("txt", args), **_mm_kwargs("txt", kwargs))
        return vid, txt


def _expand_dims(value: torch.Tensor, dim: int, ndim: int) -> torch.Tensor:
    shape = value.shape
    return value.reshape(shape[:dim] + (1,) * (ndim - len(shape)) + shape[dim:])


class AdaSingle(nn.Module):
    def __init__(self, dim: int, emb_dim: int, layers: list[str], modes: list[str] = ["in", "out"]):
        super().__init__()
        if emb_dim != 6 * dim:
            raise ValueError("AdaSingle requires emb_dim == 6 * dim")
        self.dim = dim
        self.emb_dim = emb_dim
        self.layers = layers
        for layer in layers:
            if "in" in modes:
                self.register_parameter(f"{layer}_shift", nn.Parameter(torch.randn(dim) / dim**0.5))
                self.register_parameter(f"{layer}_scale", nn.Parameter(torch.randn(dim) / dim**0.5 + 1))
            if "out" in modes:
                self.register_parameter(f"{layer}_gate", nn.Parameter(torch.randn(dim) / dim**0.5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        layer: str,
        mode: str,
        cache: Cache = Cache(disable=True),
        branch_tag: str = "",
        hid_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        index = self.layers.index(layer)
        emb = rearrange(emb, "b (d l g) -> b d l g", l=len(self.layers), g=3)[..., index, :]
        emb = _expand_dims(emb, 1, hidden_states.ndim + 1)
        if hid_len is not None:
            emb = cache(
                f"emb_repeat_{index}_{branch_tag}",
                lambda: torch.cat([item.repeat(length, *([1] * item.ndim)) for item, length in zip(emb, hid_len)]),
            )
        shift_a, scale_a, gate_a = emb.unbind(-1)
        shift_b = getattr(self, f"{layer}_shift", None)
        scale_b = getattr(self, f"{layer}_scale", None)
        gate_b = getattr(self, f"{layer}_gate", None)
        if mode == "in":
            return hidden_states * (scale_a + scale_b) + (shift_a + shift_b)
        if mode == "out":
            return hidden_states * (gate_a + gate_b)
        raise ValueError(f"unsupported AdaSingle mode: {mode}")


def timestep_embedding(timestep: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(half_dim, device=timestep.device, dtype=torch.float32) / half_dim
    frequencies = torch.exp(exponent)
    args = timestep.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, sinusoidal_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(self, timestep, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]
        hidden_states = timestep_embedding(timestep, self.sinusoidal_dim).to(dtype)
        hidden_states = self.act(self.proj_in(hidden_states))
        hidden_states = self.act(self.proj_hid(hidden_states))
        return self.proj_out(hidden_states)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, expand_ratio: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=False)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj_out(F.silu(self.proj_in_gate(hidden_states)) * self.proj_in(hidden_states))


class PatchIn(nn.Module):
    def __init__(self, in_channels: int, patch_size, dim: int):
        super().__init__()
        self.patch_size = _triple(patch_size)
        self.proj = nn.Linear(in_channels * math.prod(self.patch_size), dim)

    def forward(self, vid: torch.Tensor, vid_shape: torch.Tensor, cache: Cache):
        cache("vid_shape_before_patchify", lambda: vid_shape)
        temporal, height, width = self.patch_size
        if self.patch_size != (1, 1, 1):
            videos = unflatten(vid, vid_shape)
            for index, video in enumerate(videos):
                if temporal > 1 and vid_shape[index, 0] % temporal:
                    videos[index] = torch.cat([video[:1]] * (temporal - video.size(0) % temporal) + [video])
                videos[index] = rearrange(
                    videos[index],
                    "(t pt) (h ph) (w pw) c -> t h w (pt ph pw c)",
                    pt=temporal,
                    ph=height,
                    pw=width,
                )
            vid, vid_shape = flatten(videos)
        return self.proj(vid), vid_shape


class PatchOut(nn.Module):
    def __init__(self, out_channels: int, patch_size, dim: int):
        super().__init__()
        self.patch_size = _triple(patch_size)
        self.proj = nn.Linear(dim, out_channels * math.prod(self.patch_size))

    def forward(self, vid: torch.Tensor, vid_shape: torch.Tensor, cache: Cache):
        before_shape = cache.get("vid_shape_before_patchify")
        temporal, height, width = self.patch_size
        vid = self.proj(vid)
        if self.patch_size != (1, 1, 1):
            videos = unflatten(vid, vid_shape)
            for index, video in enumerate(videos):
                videos[index] = rearrange(
                    video,
                    "t h w (pt ph pw c) -> (t pt) (h ph) (w pw) c",
                    pt=temporal,
                    ph=height,
                    pw=width,
                )
                if temporal > 1 and before_shape[index, 0] % temporal:
                    videos[index] = videos[index][temporal - before_shape[index, 0] % temporal :]
            vid, vid_shape = flatten(videos)
        return vid, vid_shape


def make_windows(size: tuple[int, int, int], counts: tuple[int, int, int], shifted: bool):
    temporal, height, width = size
    count_t, count_h, count_w = counts
    scale = math.sqrt((45 * 80) / (height * width))
    resized_h, resized_w = round(height * scale), round(width * scale)
    win_h, win_w = math.ceil(resized_h / count_h), math.ceil(resized_w / count_w)
    win_t = math.ceil(min(temporal, 30) / count_t)
    shifts = (
        (0.5 if win_t < temporal else 0, 0.5 if win_h < height else 0, 0.5 if win_w < width else 0)
        if shifted
        else (0, 0, 0)
    )
    shift_t, shift_h, shift_w = shifts
    num_t = math.ceil((temporal - shift_t) / win_t) + (1 if shift_t else 0)
    num_h = math.ceil((height - shift_h) / win_h) + (1 if shift_h else 0)
    num_w = math.ceil((width - shift_w) / win_w) + (1 if shift_w else 0)
    result = []
    for index_w in range(num_w):
        for index_h in range(num_h):
            for index_t in range(num_t):
                starts = (
                    max(int((index_t - shift_t) * win_t), 0),
                    max(int((index_h - shift_h) * win_h), 0),
                    max(int((index_w - shift_w) * win_w), 0),
                )
                ends = (
                    min(int((index_t - shift_t + 1) * win_t), temporal),
                    min(int((index_h - shift_h + 1) * win_h), height),
                    min(int((index_w - shift_w + 1) * win_w), width),
                )
                if all(end > start for start, end in zip(starts, ends)):
                    result.append(tuple(slice(start, end) for start, end in zip(starts, ends)))
    return result


class VarlenAttention(nn.Module):
    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
        outputs = []
        q_ranges = cu_seqlens_q.tolist()
        k_ranges = cu_seqlens_k.tolist()
        for q_start, q_end, k_start, k_end in zip(q_ranges[:-1], q_ranges[1:], k_ranges[:-1], k_ranges[1:]):
            q_item = rearrange(q[q_start:q_end], "l h d -> 1 h l d")
            k_item = rearrange(k[k_start:k_end], "l h d -> 1 h l d")
            v_item = rearrange(v[k_start:k_end], "l h d -> 1 h l d")
            output = F.scaled_dot_product_attention(q_item, k_item, v_item)
            outputs.append(rearrange(output, "1 h l d -> l h d"))
        return torch.cat(outputs)


class NaSwinAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: Callable,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        window,
        window_method: str,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        self.head_dim = head_dim
        self.proj_qkv = MMModule(nn.Linear, dim, inner_dim * 3, bias=qk_bias, shared_weights=shared_weights)
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(qk_norm, dim=head_dim, eps=qk_norm_eps, elementwise_affine=True, shared_weights=shared_weights)
        self.norm_k = MMModule(qk_norm, dim=head_dim, eps=qk_norm_eps, elementwise_affine=True, shared_weights=shared_weights)
        self.rope = MMRotaryEmbedding3D(rope_dim) if rope_type == "mmrope3d" else None
        self.attn = VarlenAttention()
        self.window = _triple(window)
        self.window_method = window_method

    def forward(self, vid, txt, vid_shape, txt_shape, cache: Cache):
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def partition(value: torch.Tensor):
            slices = make_windows(value.shape[:-1], self.window, shifted="swin" in self.window_method)
            return [value[item] for item in slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform", lambda: window_idx(vid_shape, partition)
        )
        vid_qkv = window_partition(vid_qkv)
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)
        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        vid_len = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len = cache_win("all_len", lambda: vid_len + txt_len_win)
        concat, unconcat = cache_win("mm_pnp", lambda: repeat_concat_idx(vid_len, txt_len, window_count))
        if self.rope is not None:
            heads = txt_q.shape[1]
            txt_q_items = list(chain(*([[item] * count for item, count in zip(unflatten(rearrange(txt_q, "l h d -> l (h d)"), txt_shape), window_count)])))
            txt_k_items = list(chain(*([[item] * count for item, count in zip(unflatten(rearrange(txt_k, "l h d -> l (h d)"), txt_shape), window_count)])))
            txt_q_repeat, txt_shape_repeat = flatten(txt_q_items)
            txt_k_repeat, _ = flatten(txt_k_items)
            txt_q_repeat = rearrange(txt_q_repeat, "l (h d) -> l h d", h=heads)
            txt_k_repeat = rearrange(txt_k_repeat, "l (h d) -> l h d", h=heads)
            vid_q, vid_k, txt_q, txt_k = self.rope(
                vid_q, vid_k, window_shape, txt_q_repeat, txt_k_repeat, txt_shape_repeat, cache_win
            )
        q = concat(vid_q, txt_q)
        k = concat(vid_k, txt_k)
        v = concat(vid_v, txt_v)
        attention_dtype = q.dtype if q.device.type == "cpu" else torch.bfloat16
        cu_seqlens = F.pad(all_len.cumsum(0), (1, 0)).int()
        output = self.attn(
            q=q.to(attention_dtype),
            k=k.to(attention_dtype),
            v=v.to(attention_dtype),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=all_len.max().item(),
            max_seqlen_k=all_len.max().item(),
        ).to(vid_q.dtype)
        vid_out, txt_out = unconcat(output)
        vid_out = window_reverse(rearrange(vid_out, "l h d -> l (h d)"))
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        return self.proj_out(vid_out, txt_out)


class NaMMSRTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: Callable,
        norm_eps: float,
        ada: Callable,
        qk_bias: bool,
        qk_norm: Callable,
        mlp_type: str,
        shared_weights: bool,
        rope_type: str,
        rope_dim: int,
        is_last_layer: bool,
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights)
        self.attn = NaSwinAttention(
            vid_dim,
            txt_dim,
            heads,
            head_dim,
            qk_bias,
            qk_norm,
            norm_eps,
            rope_type,
            rope_dim,
            shared_weights,
            kwargs.pop("window"),
            kwargs.pop("window_method"),
        )
        self.mlp_norm = MMModule(
            norm, dim=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights, vid_only=is_last_layer
        )
        if mlp_type != "swiglu":
            raise ValueError("SeedVR2 integration currently supports the upstream SwiGLU MLP")
        self.mlp = MMModule(SwiGLUMLP, dim=dim, expand_ratio=expand_ratio, shared_weights=shared_weights, vid_only=is_last_layer)
        self.ada = MMModule(
            ada,
            dim=dim,
            emb_dim=emb_dim,
            layers=["attn", "mlp"],
            shared_weights=shared_weights,
            vid_only=is_last_layer,
        )
        self.is_last_layer = is_last_layer

    def forward(self, vid, txt, vid_shape, txt_shape, emb, cache: Cache):
        hidden_lengths = MMArg(cache("vid_len", lambda: vid_shape.prod(-1)), cache("txt_len", lambda: txt_shape.prod(-1)))
        ada_kwargs = {"emb": emb, "hid_len": hidden_lengths, "cache": cache, "branch_tag": MMArg("vid", "txt")}
        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = vid_attn + vid, txt_attn + txt
        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        return vid_mlp + vid_attn, txt_mlp + txt_attn, vid_shape, txt_shape


@dataclass
class NaDiTOutput:
    vid_sample: torch.Tensor


class NaDiT(nn.Module):
    def __init__(
        self,
        vid_in_channels: int,
        vid_out_channels: int,
        vid_dim: int,
        txt_in_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: Optional[str],
        norm_eps: float,
        ada: str,
        qk_bias: bool,
        qk_norm: Optional[str],
        patch_size,
        num_layers: int,
        block_type,
        mm_layers: int,
        mlp_type: str = "swiglu",
        rope_type: str = "mmrope3d",
        rope_dim: Optional[int] = None,
        window=None,
        window_method=None,
        txt_in_norm: Optional[str] = None,
        txt_in_norm_scale_factor: float = 0.01,
        vid_out_norm: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if ada != "single":
            raise ValueError("SeedVR2 integration supports AdaSingle only")
        norm_layer = get_norm_layer(norm)
        qk_norm_layer = get_norm_layer(qk_norm)
        rope_dim = rope_dim or head_dim // 2
        block_type = [block_type] * num_layers if isinstance(block_type, str) else list(block_type)
        window = [window] * num_layers if window is None or isinstance(window[0], int) else list(window)
        window_method = [window_method] * num_layers if isinstance(window_method, str) else list(window_method)
        self.vid_in = PatchIn(vid_in_channels, patch_size, vid_dim)
        self.txt_in = nn.Linear(txt_in_dim, txt_dim) if txt_in_dim != txt_dim else nn.Identity()
        self.emb_in = TimeEmbedding(256, max(vid_dim, txt_dim), emb_dim)
        self.blocks = nn.ModuleList(
            [
                NaMMSRTransformerBlock(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm_layer,
                    norm_eps=norm_eps,
                    ada=AdaSingle,
                    qk_bias=qk_bias,
                    qk_norm=qk_norm_layer,
                    shared_weights=not (index < mm_layers),
                    mlp_type=mlp_type,
                    window=window[index],
                    window_method=window_method[index],
                    rope_type=rope_type,
                    rope_dim=rope_dim,
                    is_last_layer=index == num_layers - 1,
                )
                for index in range(num_layers)
            ]
        )
        self.vid_out_norm = get_norm_layer(vid_out_norm)(vid_dim, norm_eps, True) if vid_out_norm else None
        if self.vid_out_norm is not None:
            self.vid_out_ada = AdaSingle(vid_dim, emb_dim, layers=["out"], modes=["in"])
        self.vid_out = PatchOut(vid_out_channels, patch_size, vid_dim)
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    def forward(self, vid, txt, vid_shape, txt_shape, timestep, disable_cache=False):
        cache = Cache(disable=disable_cache)
        txt = self.txt_in(txt)
        vid, vid_shape = self.vid_in(vid, vid_shape, cache)
        emb = self.emb_in(timestep, vid.device, vid.dtype)
        for block in self.blocks:
            kwargs = {
                "vid": vid,
                "txt": txt,
                "vid_shape": vid_shape,
                "txt_shape": txt_shape,
                "emb": emb,
                "cache": cache,
            }
            if self.gradient_checkpointing and self.training:
                vid, txt, vid_shape, txt_shape = torch.utils.checkpoint.checkpoint(
                    block, use_reentrant=False, **kwargs
                )
            else:
                vid, txt, vid_shape, txt_shape = block(**kwargs)
        if self.vid_out_norm is not None:
            vid = self.vid_out_norm(vid)
            vid = self.vid_out_ada(
                vid,
                emb=emb,
                layer="out",
                mode="in",
                hid_len=cache("vid_len", lambda: vid_shape.prod(-1)),
                cache=cache,
                branch_tag="vid",
            )
        vid, _ = self.vid_out(vid, vid_shape, cache)
        return NaDiTOutput(vid_sample=vid)
