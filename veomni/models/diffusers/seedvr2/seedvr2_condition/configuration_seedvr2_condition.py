from transformers import PretrainedConfig


class SeedVR2ConditionConfig(PretrainedConfig):
    model_type = "SeedVR2ConditionModel"

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        fixed_timestep: float | None = None,
        condition_noise_scale: float = 0.0,
        latent_channels: int = 16,
        text_dim: int = 5120,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_train_timesteps = num_train_timesteps
        self.fixed_timestep = fixed_timestep
        self.condition_noise_scale = condition_noise_scale
        self.latent_channels = latent_channels
        self.text_dim = text_dim
        self.seed = seed
