from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("SeedVR2TransformerModel")
def register_seedvr2_transformer_config():
    from .configuration_seedvr2_transformer import SeedVR2TransformerConfig

    return SeedVR2TransformerConfig


@MODELING_REGISTRY.register("SeedVR2TransformerModel")
def register_seedvr2_transformer_modeling(architecture: str = None):
    from .modeling_seedvr2_transformer import SeedVR2TransformerModel

    return SeedVR2TransformerModel
