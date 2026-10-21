"""The Hugging Face cache location."""


def hf_cache_dir() -> str:
    """The Hub cache that transformers and datasets use by default (HF_HUB_CACHE, else
    $HF_HOME/hub), so models loaded with an explicit cache_dir share it."""
    from huggingface_hub import constants
    return constants.HF_HUB_CACHE
