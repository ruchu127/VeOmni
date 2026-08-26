from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("SeedVR2ConditionModel")
def register_seedvr2_condition_config():
    from .configuration_seedvr2_condition import SeedVR2ConditionConfig

    return SeedVR2ConditionConfig


@MODELING_REGISTRY.register("SeedVR2ConditionModel")
def register_seedvr2_condition_modeling(architecture: str = None):
    from .modeling_seedvr2_condition import SeedVR2ConditionModel

    return SeedVR2ConditionModel
