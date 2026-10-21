"""Model and processor loading for the text pipeline (keys of config.MODEL_REGISTRY)."""

import torch

from config import MODEL_REGISTRY


def _patch_transformers():
    """Compatibility shims for trust_remote_code models on recent transformers."""
    try:
        import transformers.integrations.tensor_parallel as tp
        orig_itp = tp.initialize_tensor_parallelism

        def safe_itp(*a, **kw):
            try:
                return orig_itp(*a, **kw)
            except OSError:
                return None, None, None

        tp.initialize_tensor_parallelism = safe_itp
        import transformers.modeling_utils as mu
        if hasattr(mu, "initialize_tensor_parallelism"):
            mu.initialize_tensor_parallelism = safe_itp
    except (ImportError, AttributeError):
        pass
    try:
        import transformers.integrations.accelerate as acc
        orig_cms = acc.compute_module_sizes

        def patched_cms(model, *a, **kw):
            if not hasattr(model, "all_tied_weights_keys"):
                tied = getattr(model, "_tied_weights_keys", None) or []
                model.all_tied_weights_keys = {k: k for k in tied}
            elif isinstance(model.all_tied_weights_keys, set):
                model.all_tied_weights_keys = {k: k for k in model.all_tied_weights_keys}
            return orig_cms(model, *a, **kw)

        acc.compute_module_sizes = patched_cms
    except (ImportError, AttributeError):
        pass


def load_processor(model_key: str):
    from transformers import AutoProcessor
    name = MODEL_REGISTRY[model_key]
    if model_key in ("internvl-8b", "internvl-14b"):
        # The remote InternVL tokenizer lacks the image-token attributes the processor expects.
        from transformers import AutoTokenizer
        tok_cls = type(AutoTokenizer.from_pretrained(name, trust_remote_code=True))
        orig_getattr = tok_cls.__getattr__
        tokens = {"start_image_token": "<img>", "end_image_token": "</img>",
                  "context_image_token": "<IMG_CONTEXT>", "video_token": "<|video_pad|>"}
        ids = {"start_image_token_id": "<img>", "end_image_token_id": "</img>",
               "context_image_token_id": "<IMG_CONTEXT>"}

        def tok_getattr(self, key):
            if key in tokens:
                return tokens[key]
            if key in ids:
                return self.convert_tokens_to_ids(ids[key])
            return orig_getattr(self, key)

        tok_cls.__getattr__ = tok_getattr
    return AutoProcessor.from_pretrained(name, trust_remote_code=True)


def load_model(model_key: str, processor, device_map="auto"):
    _patch_transformers()
    name = MODEL_REGISTRY[model_key]
    kw = dict(torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
    if model_key in ("qwen35-9b", "qwen35-9b-thinking"):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    elif model_key in ("internvl-8b", "internvl-14b"):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(name, **kw)
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    else:
        from transformers import AutoConfig, AutoModelForImageTextToText
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        if hasattr(cfg, "tp_plan"):
            cfg.tp_plan = None
        model = AutoModelForImageTextToText.from_pretrained(name, config=cfg, **kw)
    model.eval()
    return model
