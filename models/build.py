
from sentry_sdk.integrations.cloud_resource_context import context_getters


from .Swin.Swin_Decoder import build_life


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "SWIN":
        model = build_life(
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            juery_nums = config.MODEL.SWIN.JUERY_NUMS,
            num_experts = config.MODEL.SWIN.NUM_EXPERTS,
            top_k = config.MODEL.SWIN.TOP_K,
            depth = config.MODEL.SWIN.DEPTH
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
