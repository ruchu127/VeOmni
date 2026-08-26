from transformers import PretrainedConfig


class SeedVR2TransformerConfig(PretrainedConfig):
    model_type = "SeedVR2TransformerModel"
    condition_model_type = "SeedVR2ConditionModel"

    def __init__(
        self,
        vid_in_channels: int = 33,
        vid_out_channels: int = 16,
        vid_dim: int = 2560,
        vid_out_norm: str = "fusedrms",
        txt_in_dim: int = 5120,
        txt_in_norm: str = "fusedln",
        txt_dim: int = 2560,
        emb_dim: int = 15360,
        heads: int = 20,
        head_dim: int = 128,
        expand_ratio: int = 4,
        norm: str = "fusedrms",
        norm_eps: float = 1e-5,
        ada: str = "single",
        qk_bias: bool = False,
        qk_norm: str = "fusedrms",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_layers: int = 32,
        mm_layers: int = 10,
        mlp_type: str = "swiglu",
        block_type: str | list[str] = "mmdit_sr",
        window: tuple[int, int, int] | list[tuple[int, int, int]] = (4, 3, 3),
        window_method: str | list[str] | None = None,
        rope_type: str = "mmrope3d",
        rope_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if window_method is None:
            window_method = [
                method
                for _ in range(num_layers // 2)
                for method in ("720pwin_by_size_bysize", "720pswin_by_size_bysize")
            ]
        self.vid_in_channels = vid_in_channels
        self.vid_out_channels = vid_out_channels
        self.vid_dim = vid_dim
        self.vid_out_norm = vid_out_norm
        self.txt_in_dim = txt_in_dim
        self.txt_in_norm = txt_in_norm
        self.txt_dim = txt_dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = head_dim
        self.expand_ratio = expand_ratio
        self.norm = norm
        self.norm_eps = norm_eps
        self.ada = ada
        self.qk_bias = qk_bias
        self.qk_norm = qk_norm
        self.patch_size = tuple(patch_size)
        self.num_layers = num_layers
        self.mm_layers = mm_layers
        self.mlp_type = mlp_type
        self.block_type = block_type
        self.window = window
        self.window_method = window_method
        self.rope_type = rope_type
        self.rope_dim = rope_dim
