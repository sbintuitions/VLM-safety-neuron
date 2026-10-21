"""Stage 1 of the visual pipeline: prefill, generation and behavioral probing.

The functions here are called by run_iterative_visual.py, which adds GPT-4o re-judging,
Stage 2 and the iterative loop. Run on its own, this script does one keyword-labeled
Stage-1 pass.

1. Load the ViSafe-Detect harmful and safe images (each paired with the procedural query)
2. Prefill with hooks on all (layer, component) pairs: a_n(x) at the last input token
3. Generation (max_new_tokens 512) -> keyword labels on the response (GPT-4o re-judging is
   applied by run_iterative_visual.py)
4. l2 logistic probe per (layer, component) on a stratified 80/20 split
5. sign(w) * |Cohen's d| on the validation split, z-standardized per layer (pooled over
   components); candidates have |z| > 3.0

Relies on VLMActivationExtractor from src/neuron_detection_vlm.py
for hook registration and input preparation.

Usage:
  python src/run_stage1_screening.py --model qwen3vl-8b [--dry-run]
  python src/run_stage1_screening.py --model qwen25vl-7b --n-harmful 200 --n-safe 200
"""

import argparse
import gc
import inspect
import json
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup: make src/ and the repo root importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT))
from paths import hf_cache_dir

from config import MODELS, DetectionConfig, load_pipeline_config
from neuron_detection_vlm import (
    VLMActivationExtractor, probe_scores, probe_split, repo_path, save_neurons,
)
from neuron_types import SafetyNeuron
from data_utils import SafetyQuery
from data.image_utils import resize_max_side
from refusal import final_answer, kw_label, thinking_setting

# a run stops if more than this fraction of samples fail in prefill or generation
MAX_FAILED_FRACTION = 0.01


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_registry() -> dict:
    return {"models": MODELS}


# ---------------------------------------------------------------------------
# Label JSONL loading
# ---------------------------------------------------------------------------

def load_labels(label_path: Path) -> list:
    """Load all labels from the JSONL file."""
    labels = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(json.loads(line))
    return labels


def filter_harmful_labels(labels: list) -> list:
    """The ViSafe-Detect harmful set."""
    return [l for l in labels if l["label"] == "harmful"]


def filter_safe_labels(labels: list) -> list:
    """The ViSafe-Detect safe set."""
    return [l for l in labels if l["label"] == "safe"]


# ---------------------------------------------------------------------------
# Image loading from data/visafe_detect/images/ (written by data/build/fetch_visafe_detect.py)
# ---------------------------------------------------------------------------

LOCAL_IMAGE_DIR = REPO_ROOT / "data" / "visafe_detect" / "images"
DEFAULT_LABEL_FILE = REPO_ROOT / "data" / "visafe_detect" / "labels.jsonl"


def _load_from_cache(label: dict, query_template: str, max_side: int) -> Optional[SafetyQuery]:
    """Try loading a single image from local cache."""
    source = label["source"]
    idx = label["source_index"]
    cache_path = LOCAL_IMAGE_DIR / source / f"{idx:05d}.jpg"
    if not cache_path.exists():
        return None
    try:
        img = Image.open(cache_path).convert("RGB")
    except Exception:
        return None
    return SafetyQuery(
        text=query_template,
        image=resize_max_side(img, max_side),
        label=label["label"],
        harm_category=label["harm_category"],
        metadata={
            "source": source,
            "source_index": idx,
        },
    )


def load_images_for_labels(
    labels: list,
    query_template: str,
    max_side: int = 1024,
    max_samples: int = None,
) -> List[SafetyQuery]:
    """Load the ViSafe-Detect images for these label rows from data/visafe_detect/images/.

    The images are the files written by data/build/fetch_visafe_detect.py.
    `source_index` is a local file key, not a row of the source dataset, so there is
    no fallback to the Hugging Face datasets: a missing file is an error.
    """
    if max_samples is not None:
        labels = labels[:max_samples]

    queries, missing = [], []
    for label in labels:
        q = _load_from_cache(label, query_template, max_side)
        if q is not None:
            queries.append(q)
        else:
            missing.append(f"{label['source']}/{label['source_index']:05d}.jpg")
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} ViSafe-Detect images are missing under data/visafe_detect/images/ "
            f"(e.g. {', '.join(missing[:3])}). Run: python data/build/fetch_visafe_detect.py"
        )
    print(f"  Loaded {len(queries)} images from data/visafe_detect/images/")
    return queries


def load_safe_images_for_screening(
    safe_labels: list,
    query_template: str,
    max_side: int = 1024,
    max_samples: int = None,
) -> List[SafetyQuery]:
    """Load the safe ViSafe-Detect images from the label JSONL."""
    # Reuse the same loader - safe labels also have source + source_index
    safe_queries = load_images_for_labels(
        safe_labels, query_template, max_side, max_samples
    )
    # Override label to "safe"
    for q in safe_queries:
        q.label = "safe"
    return safe_queries


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(hf_repo: str, model_key: str):
    """Load model and processor, handling per-family differences."""
    from transformers import (
        AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM,
        AutoTokenizer, AutoImageProcessor, AutoModel,
    )

    cache_dir = hf_cache_dir()

    print(f"Loading model: {hf_repo}")
    print(f"Cache dir: {cache_dir}")

    common_kw = dict(trust_remote_code=True, cache_dir=cache_dir)

    # InternVL: attn_implementation is not passed; its remote code uses flash attention when
    # flash_attn is installed and eager attention otherwise
    if model_key.startswith("internvl"):
        _attn_impl = "eager"
    else:
        _attn_impl = "flash_attention_2"
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            _attn_impl = "sdpa"
    if _attn_impl == "eager":
        print("  attn_implementation: set by the model's remote code")
    else:
        print(f"  attn_implementation={_attn_impl}")

    model_kw = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **common_kw,
    )
    if _attn_impl != "eager":
        model_kw["attn_implementation"] = _attn_impl

    # ---- InternVL: the transformers InternVLProcessor expects image-token attributes
    #      that the remote-code tokenizer lacks. Try AutoProcessor first; if it fails,
    #      build a processor from AutoTokenizer + AutoImageProcessor.
    if model_key.startswith("internvl"):
        print("  [InternVL] Loading processor (with fallback)")
        try:
            processor = AutoProcessor.from_pretrained(hf_repo, **common_kw)
            _ = processor.tokenizer  # fails here if the processor is incomplete
        except Exception as e:
            print(f"  [InternVL] AutoProcessor failed ({e}), using tokenizer+image_processor fallback")
            tokenizer = AutoTokenizer.from_pretrained(hf_repo, **common_kw)
            image_processor = AutoImageProcessor.from_pretrained(hf_repo, **common_kw)
            from types import SimpleNamespace
            processor = SimpleNamespace(
                tokenizer=tokenizer,
                image_processor=image_processor,
            )
        model = AutoModel.from_pretrained(hf_repo, **model_kw)

    # ---- Generic path: try VLM first, fall back to CausalLM (remote-code VLMs register there)
    else:
        processor = AutoProcessor.from_pretrained(hf_repo, **common_kw)
        errors = []
        for auto_cls in (AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel):
            try:
                model = auto_cls.from_pretrained(hf_repo, **model_kw)
                break
            except Exception as e:
                errors.append(f"{auto_cls.__name__}: {type(e).__name__}: {e}")
                print(f"  {errors[-1]}")
        else:
            raise RuntimeError(f"could not load {hf_repo}:\n  " + "\n  ".join(errors))
        if not any(k in name for name, _ in model.named_modules() for k in ("visual", "vision")):
            raise RuntimeError(f"{hf_repo} loaded as {type(model).__name__}, which has no vision "
                               "tower; the visual pipeline needs the vision-language class")

    model.eval()

    # InternVL: set img_context_token_id (required by custom generate())
    if hasattr(model, "img_context_token_id") and model.img_context_token_id is None:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        model.img_context_token_id = ctx_id
        print(f"  Set InternVL img_context_token_id = {ctx_id}")

    # InternVL on several GPUs: with device_map="auto", image_flags and vit_embeds can sit
    # on different devices inside the remote-code forward(). The forward below is the
    # remote-code one with two .to(device) calls added, plus a check that the number of
    # image-context tokens matches the number of image embeddings. Accelerate keeps the
    # original in _old_forward, so that is replaced when present.
    if model_key.startswith("internvl") and hasattr(model, "extract_feature"):
        import types
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from torch.nn import CrossEntropyLoss

        def _internvl_forward(self, pixel_values, input_ids=None,
                                   attention_mask=None, position_ids=None,
                                   image_flags=None, past_key_values=None,
                                   labels=None, use_cache=None,
                                   output_attentions=None,
                                   output_hidden_states=None,
                                   return_dict=None, **extra_kwargs):
            """InternVL forward() with image_flags and vit_embeds moved to matching devices."""
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            image_flags = image_flags.squeeze(-1)
            input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
            vit_embeds = self.extract_feature(pixel_values)
            image_flags = image_flags.to(vit_embeds.device)
            vit_embeds = vit_embeds[image_flags == 1]

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)
            selected = (input_ids_flat == self.img_context_token_id)
            vit_embeds = vit_embeds.to(input_embeds.device)
            vit_embeds = vit_embeds.reshape(-1, C)
            if int(selected.sum()) != vit_embeds.size(0):
                raise RuntimeError(f"InternVL: {int(selected.sum())} image-context tokens but "
                                   f"{vit_embeds.size(0)} image embeddings")
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
            input_embeds = input_embeds.reshape(B, N, C)

            outputs = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
                shift_labels = shift_labels.view(-1).to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output
            return CausalLMOutputWithPast(
                loss=loss, logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # Accelerate wraps forward via _old_forward; patch that if present
        if hasattr(model, "_old_forward"):
            model._old_forward = types.MethodType(_internvl_forward, model)
        else:
            model.forward = types.MethodType(_internvl_forward, model)
        print("  [InternVL] forward() replaced (image_flags / vit_embeds on matching devices)")

    print(f"Model loaded: {type(model).__name__}, device={next(model.parameters()).device}")
    return model, processor


def build_valid_keys(model):
    """Build valid forward() keys by inspecting model signature.

    """
    _all_valid = {
        "input_ids", "attention_mask", "position_ids",
        "pixel_values", "image_grid_thw", "image_sizes",
        "mm_token_type_ids", "token_type_ids",
        "image_bounds", "image_bound", "tgt_sizes",
        "temporal_ids",
        "image_embeds", "image_flags",
        "pixel_values_videos", "video_grid_thw",
        "pixel_position_ids", "image_position_ids",
    }
    try:
        fwd_sig = inspect.signature(model.forward)
        fwd_params = set(fwd_sig.parameters.keys())
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in fwd_sig.parameters.values()
        )
        if has_var_kw:
            valid_keys = set(_all_valid)
        else:
            valid_keys = _all_valid & fwd_params
            excluded = _all_valid - valid_keys
            if excluded:
                print(f"  Model forward() excludes: {excluded}")
    except Exception:
        valid_keys = set(_all_valid)

    # Gemma4 mm_token_type_ids requires torch>=2.6; skip if torch is older
    model_class = type(model).__name__.lower()
    if "gemma4" in model_class:
        torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if torch_version < (2, 6):
            valid_keys.discard("mm_token_type_ids")
            print(f"  [compat] Excluded mm_token_type_ids for Gemma (torch {torch.__version__} < 2.6)")

    print(f"  valid_keys: {sorted(valid_keys)}")
    return valid_keys


# ---------------------------------------------------------------------------
# Generation + activation extraction
# ---------------------------------------------------------------------------

def generate_and_extract(
    model, processor, extractor,
    harmful_queries: List[SafetyQuery],
    safe_queries: List[SafetyQuery],
    components: List[str],
    save_dir: Path,
    max_new_tokens: int = 512,
    valid_keys: set = None,
    gen_batch_size: int = 4,
) -> Tuple[dict, dict]:
    """Prefill with hooks (activations) and greedy generation (keyword labels) for every sample.

    For each sample:
      1. Prefill forward pass with hooks -> a_n(x) at the last input token
      2. generate() the full response -> strip reasoning -> keyword label

    Activations are saved for ALL samples (not just the probe set), so the labels can be
    refined later (keyword -> keyword + GPT-4o) without another GPU pass.

    Returns:
        (gen_stats, extract_meta)
    """
    all_queries = harmful_queries + safe_queries
    all_labels = ["harmful"] * len(harmful_queries) + ["safe"] * len(safe_queries)
    n_harmful = len(harmful_queries)
    n_samples = len(all_queries)
    n_layers = extractor.num_layers

    tokenizer = extractor.tokenizer
    thinking = bool(extractor.thinking_mode)

    if valid_keys is None:
        valid_keys = build_valid_keys(model)

    # ---- Setup activation hooks ----
    hidden_dims = {comp: extractor.component_dim(comp) for comp in components}

    print(f"Components: {list(hidden_dims.keys())}, dims: {hidden_dims}")

    layer_buffers = {li: {comp: [] for comp in hidden_dims} for li in range(n_layers)}

    # ---- Determine if we need split-layer prefill ----
    # Models with 48+ layers on several GPUs: hooks are registered for 32 layers at a time
    # and the prefill runs once per chunk, which keeps the number of active hooks (each
    # copies to CPU) per forward pass small.
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    use_split_prefill = (n_layers >= 48 and n_gpus >= 2)
    if use_split_prefill:
        CHUNK_SIZE = 32
        layer_chunks = []
        for start in range(0, n_layers, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_layers)
            layer_chunks.append(list(range(start, end)))
        print(f"  [split-prefill] {n_layers} layers on {n_gpus} GPUs -> "
              f"{len(layer_chunks)} chunks of ~{CHUNK_SIZE} layers each")
        print(f"  [split-prefill] Each sample needs {len(layer_chunks)} forward passes")
    else:
        layer_chunks = [list(range(n_layers))]

    all_caches = {}
    hooks_enabled = [False]
    # For batched prefill: store per-sample activations indexed by batch position
    # When prefill_bs > 1, hooks collect act[:, -1, :] for all samples in batch
    batch_caches = {}  # key -> list of np arrays (one per sample in batch)
    prefill_bs_active = [1]  # current prefill batch size (mutable)
    batched_prefill = [False]  # True while a padded batch (any size, also 1) is running

    def make_hook(layer_idx, comp):
        key = (layer_idx, comp)
        def hook_fn(act):
            if not hooks_enabled[0]:
                return
            bs = prefill_bs_active[0]
            if not batched_prefill[0]:
                if act.dim() == 2:  # [seq, dim]
                    all_caches[key] = act[-1, :].float().detach().cpu().numpy()
                else:
                    all_caches[key] = act[0, -1, :].float().detach().cpu().numpy()
            else:
                # Batched: collect all samples
                if act.dim() == 2:  # no batch axis: fall back to bs=1
                    raise RuntimeError("unbatched activation in a batched prefill")
                else:
                    batch_caches[key] = act[:bs, -1, :].float().detach().cpu().numpy()
        return hook_fn

    def _register_hooks_for_layers(layer_indices):
        """Register hooks only on specified layers. Returns list of hook handles."""
        chunk_hooks = []
        for layer_idx in layer_indices:
            for comp in hidden_dims:
                chunk_hooks.append(extractor.capture_hook(layer_idx, comp, make_hook(layer_idx, comp)))
        return chunk_hooks

    def _remove_hooks(hook_list):
        """Remove all hooks in the list."""
        for h in hook_list:
            h.remove()

    # For non-split mode, register all hooks upfront
    if not use_split_prefill:
        hooks = _register_hooks_for_layers(list(range(n_layers)))
        print(f"Registered {len(hooks)} hooks across {n_layers} layers x {len(hidden_dims)} components")
    else:
        hooks = []  # managed per-chunk inside the loop
        n_hooks_per_chunk = len(hidden_dims) * CHUNK_SIZE
        print(f"  [split-prefill] ~{n_hooks_per_chunk} hooks per chunk (vs {n_layers * len(hidden_dims)} total)")

    import signal

    # ---- Phase 1: Prefill (hooks ON) -> extract activations ----
    # Supports batched prefill (prefill_bs > 1) for non-split mode when
    # samples can be padded to uniform length. Falls back to bs=1 for
    # split-prefill (multi-GPU large models) or on OOM.
    PREFILL_TIMEOUT_S = 180 if use_split_prefill else 120
    PREFILL_BS = 1 if use_split_prefill else min(8, gen_batch_size)

    class PrefillTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise PrefillTimeout("Prefill forward pass timed out")

    all_prepared = []  # True if the sample's prefill succeeded, else None
    skipped_indices = []

    t_phase1 = time.time()
    if use_split_prefill:
        print(f"  Phase 1: Prefill (bs=1, split-layer, {len(layer_chunks)} passes/sample) for {n_samples} samples...")
    else:
        print(f"  Phase 1: Prefill (bs={PREFILL_BS}, extract activations) for {n_samples} samples...")
    print(f"  Per-sample timeout: {PREFILL_TIMEOUT_S}s")

    # Memory diagnostics
    if torch.cuda.is_available():
        for gpu_i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(gpu_i) / 1e9
            total = torch.cuda.get_device_properties(gpu_i).total_memory / 1e9
            print(f"  GPU {gpu_i}: {alloc:.1f}/{total:.1f} GB allocated")

    def _zero_activations(sample_idx):
        """Fill zero activations for a failed/skipped sample."""
        for layer_idx in range(n_layers):
            for comp in hidden_dims:
                if len(layer_buffers[layer_idx][comp]) <= sample_idx:
                    layer_buffers[layer_idx][comp].append(
                        np.zeros(hidden_dims[comp], dtype=np.float32))

    if use_split_prefill or PREFILL_BS <= 1:
        # ---- bs=1 path (split-prefill or fallback) ----
        prefill_bs_active[0] = 1
        for i, query in enumerate(tqdm(all_queries, desc="Prefill")):
            try:
                inputs = extractor._prepare_inputs(query)
                fwd_inputs = {k: v for k, v in inputs.items() if k in valid_keys}

                if use_split_prefill:
                    for chunk_layers in layer_chunks:
                        chunk_hooks = _register_hooks_for_layers(chunk_layers)
                        all_caches.clear()
                        hooks_enabled[0] = True
                        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                        signal.alarm(PREFILL_TIMEOUT_S)
                        try:
                            with torch.no_grad():
                                extractor.model(**fwd_inputs)
                            for gpu_i in range(n_gpus):
                                torch.cuda.synchronize(gpu_i)
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)
                            hooks_enabled[0] = False
                            _remove_hooks(chunk_hooks)
                        for (layer_idx, comp), act in all_caches.items():
                            layer_buffers[layer_idx][comp].append(act)
                else:
                    all_caches.clear()
                    hooks_enabled[0] = True
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(PREFILL_TIMEOUT_S)
                    try:
                        with torch.no_grad():
                            extractor.model(**fwd_inputs)
                        if n_gpus > 1:
                            for gpu_i in range(n_gpus):
                                torch.cuda.synchronize(gpu_i)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                    hooks_enabled[0] = False
                    for (layer_idx, comp), act in all_caches.items():
                        layer_buffers[layer_idx][comp].append(act)

                all_prepared.append(True)

            except Exception as e:
                hooks_enabled[0] = False
                skipped_indices.append(i)
                all_prepared.append(None)
                _zero_activations(i)
                etype = "timed out" if isinstance(e, PrefillTimeout) else f"failed ({type(e).__name__}: {e})"
                print(f"\n  WARNING: Sample {i} {etype}, skipped")
                torch.cuda.empty_cache()

            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

    else:
        # ---- Batched prefill path (bs > 1, non-split) ----
        prefill_bs_active[0] = PREFILL_BS
        n_batched_ok = 0
        n_fallback = 0

        for batch_start in tqdm(range(0, n_samples, PREFILL_BS),
                                desc=f"Prefill(bs={PREFILL_BS})",
                                total=(n_samples + PREFILL_BS - 1) // PREFILL_BS):
            batch_end = min(batch_start + PREFILL_BS, n_samples)
            batch_queries = all_queries[batch_start:batch_end]
            actual_bs = len(batch_queries)

            try:
                # Prepare all inputs in batch
                batch_inputs_list = []
                for q in batch_queries:
                    inp = extractor._prepare_inputs(q)
                    batch_inputs_list.append(inp)

                # Attempt to pad and batch
                # Find max sequence length in this batch
                max_len = max(inp["input_ids"].shape[1] for inp in batch_inputs_list)
                device = next(model.parameters()).device
                tokenizer_local = processor.tokenizer if hasattr(processor, "tokenizer") else processor
                pad_id = getattr(tokenizer_local, "pad_token_id", 0) or 0

                # Pad input_ids and attention_mask
                padded_ids = torch.full((actual_bs, max_len), pad_id, dtype=torch.long, device=device)
                padded_mask = torch.zeros((actual_bs, max_len), dtype=torch.long, device=device)

                has_pixel_values = any("pixel_values" in inp for inp in batch_inputs_list)
                can_batch_pixels = False

                if has_pixel_values:
                    # Check if all pixel_values have compatible shapes for concat
                    pv_shapes = [inp["pixel_values"].shape for inp in batch_inputs_list
                                 if "pixel_values" in inp]
                    if pv_shapes and all(len(s) == len(pv_shapes[0]) for s in pv_shapes):
                        if all(s[1:] == pv_shapes[0][1:] for s in pv_shapes):
                            can_batch_pixels = True

                for b, inp in enumerate(batch_inputs_list):
                    seq_len = inp["input_ids"].shape[1]
                    # Left-pad so last token is at position max_len-1
                    padded_ids[b, max_len - seq_len:] = inp["input_ids"][0]
                    padded_mask[b, max_len - seq_len:] = inp.get(
                        "attention_mask",
                        torch.ones(1, seq_len, dtype=torch.long)
                    )[0]

                batched_fwd = {"input_ids": padded_ids, "attention_mask": padded_mask}

                # Detect model type for vision key handling
                model_class_lower = type(model).__name__.lower()
                is_internvl_model = "internvl" in model_class_lower

                if has_pixel_values:
                    if is_internvl_model:
                        # InternVL: pixel_values are per-patch (variable count per image)
                        # Concat along dim=0, concat image_flags accordingly
                        all_pv = [inp["pixel_values"] for inp in batch_inputs_list
                                  if "pixel_values" in inp]
                        batched_fwd["pixel_values"] = torch.cat(all_pv, dim=0).to(device)
                        # image_flags: one per patch, all set to 1
                        all_flags = [inp.get("image_flags",
                                     torch.ones(inp["pixel_values"].shape[0], 1,
                                                dtype=torch.long, device=device))
                                     for inp in batch_inputs_list if "pixel_values" in inp]
                        batched_fwd["image_flags"] = torch.cat(all_flags, dim=0).to(device)
                    elif can_batch_pixels:
                        # Standard VLMs: pixel_values have uniform spatial dims
                        all_pv = [inp["pixel_values"] for inp in batch_inputs_list
                                  if "pixel_values" in inp]
                        batched_fwd["pixel_values"] = torch.cat(all_pv, dim=0).to(device)
                    else:  # never run the batch without its images: fall back to bs=1
                        raise RuntimeError(f"pixel_values shapes {pv_shapes} cannot be batched")
                    # Copy other vision keys (image_grid_thw, mm_token_type_ids, etc.)
                    for vk in ["image_grid_thw", "pixel_position_ids", "image_position_ids",
                                "video_grid_thw", "pixel_values_videos"]:
                        vals = [inp[vk] for inp in batch_inputs_list if vk in inp]
                        if vals:
                            batched_fwd[vk] = torch.cat(vals, dim=0).to(device)

                # Per-token modality ids (Qwen3-VL mm_token_type_ids, Gemma token_type_ids),
                # left-padded like input_ids
                for tk in ("mm_token_type_ids", "token_type_ids"):
                    mm_vals = [inp.get(tk) for inp in batch_inputs_list]
                    if any(v is not None for v in mm_vals):
                        padded_mm = torch.zeros((actual_bs, max_len), dtype=torch.long, device=device)
                        for b, mm in enumerate(mm_vals):
                            if mm is not None:
                                sl = mm.shape[1]
                                padded_mm[b, max_len - sl:] = mm[0]
                        batched_fwd[tk] = padded_mm

                # Filter to valid keys, but KEEP image_flags for InternVL forward
                fwd_valid = set(valid_keys)
                if is_internvl_model:
                    fwd_valid.add("image_flags")
                batched_fwd = {k: v for k, v in batched_fwd.items() if k in fwd_valid}

                # Forward pass with hooks
                prefill_bs_active[0] = actual_bs
                batched_prefill[0] = True
                batch_caches.clear()
                hooks_enabled[0] = True

                with torch.no_grad():
                    extractor.model(**batched_fwd)
                batched_prefill[0] = False
                if n_gpus > 1:
                    for gpu_i in range(n_gpus):
                        torch.cuda.synchronize(gpu_i)

                hooks_enabled[0] = False

                # Distribute batch activations to per-sample buffers
                for (layer_idx, comp), acts in batch_caches.items():
                    for b in range(actual_bs):
                        layer_buffers[layer_idx][comp].append(acts[b])

                # Store per-sample info
                for q in batch_queries:
                    all_prepared.append(True)

                n_batched_ok += actual_bs

            except Exception as e:
                # Batch failed (OOM, shape mismatch, etc.) -> fallback to bs=1
                hooks_enabled[0] = False
                batched_prefill[0] = False
                prefill_bs_active[0] = 1
                if n_fallback == 0:
                    print(f"\n  Batch prefill failed at sample {batch_start}: {type(e).__name__}: {e}")
                    print("  Running this batch at bs=1 (later batches are tried batched again)")
                n_fallback += 1
                torch.cuda.empty_cache()

                # Process this batch one by one
                for b, q in enumerate(batch_queries):
                    idx = batch_start + b
                    try:
                        inp = extractor._prepare_inputs(q)
                        fwd_inp = {k: v for k, v in inp.items() if k in valid_keys}
                        all_caches.clear()
                        hooks_enabled[0] = True
                        with torch.no_grad():
                            extractor.model(**fwd_inp)
                        hooks_enabled[0] = False
                        for (li, co), act in all_caches.items():
                            layer_buffers[li][co].append(act)
                        all_prepared.append(True)
                    except Exception as e2:
                        hooks_enabled[0] = False
                        print(f"\n  WARNING: Sample {idx} failed ({type(e2).__name__}: {e2}), skipped")
                        skipped_indices.append(idx)
                        all_prepared.append(None)
                        _zero_activations(idx)
                        torch.cuda.empty_cache()

            if (batch_end) % 200 == 0:
                torch.cuda.empty_cache()

        print(f"  Batched: {n_batched_ok} samples OK, {n_fallback} batch fallbacks")

    n_rows = {len(layer_buffers[li][c]) for li in range(n_layers) for c in hidden_dims}
    if n_rows != {n_samples}:
        raise RuntimeError(f"prefill stored {sorted(n_rows)} activation rows for {n_samples} samples")

    # Remove hooks -- no longer needed
    if not use_split_prefill:
        for hook in hooks:
            hook.remove()

    prefill_time = time.time() - t_phase1
    print(f"  Phase 1 done: {n_samples} samples ({len(skipped_indices)} skipped), {prefill_time:.1f}s")
    if skipped_indices:
        print(f"  Skipped sample indices: {skipped_indices[:20]}{'...' if len(skipped_indices) > 20 else ''}")

    # ---- Phase 2: Generate (activation hooks off; clamps stay) -> responses ----
    per_sample = []
    GEN_TIMEOUT_S = 900
    print(f"  Phase 2: Generate (max_new_tokens={max_new_tokens}, greedy, "
          f"enable_thinking={extractor.thinking_mode})...")
    t_phase2 = time.time()

    for batch_start in tqdm(range(0, n_samples, gen_batch_size), desc="Generate"):
        batch_end = min(batch_start + gen_batch_size, n_samples)
        for idx in range(batch_start, batch_end):
            query = all_queries[idx]
            meta = query.metadata or {}
            response, answer, failed, no_answer = "", "", False, False
            if all_prepared[idx] is None:
                failed = True
            else:
                try:
                    # re-prepared here (not kept on the GPU)
                    inputs = extractor._prepare_inputs(query, generation=True)
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(GEN_TIMEOUT_S)
                    try:
                        gen_ids = extractor.generate_ids(inputs, max_new_tokens=max_new_tokens)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                    response = tokenizer.decode(gen_ids, skip_special_tokens=False)
                    answer = final_answer(gen_ids, tokenizer, thinking,
                                          max_answer_tokens=max_new_tokens)
                    if answer is None:  # no final answer: unlabeled
                        answer, no_answer = "", True
                except Exception as e:
                    failed = True
                    print(f"\n  WARNING: Sample {idx} generation failed: {e}")
                    torch.cuda.empty_cache()

            per_sample.append({
                "index": idx,
                "label": all_labels[idx],
                "harm_category": query.harm_category,
                "source": meta.get("source", ""),
                "source_index": meta.get("source_index", -1),
                "query_text": query.text,
                "response": response,
                "answer": answer,
                "kw_label": "FAILED" if failed else ("NO_ANSWER" if no_answer else kw_label(answer)),
                "failed": failed or no_answer,
            })

        if (batch_start + gen_batch_size) % 200 == 0:
            torch.cuda.empty_cache()

    phase2_time = time.time() - t_phase2
    n_failed = sum(r["kw_label"] == "FAILED" for r in per_sample)
    if n_failed > MAX_FAILED_FRACTION * n_samples:
        raise RuntimeError(f"{n_failed}/{n_samples} samples failed in prefill or generation (see the "
                           "warnings above); stopping instead of detecting on the rest")
    print(f"  Phase 2 done: {n_samples} samples, {phase2_time:.1f}s "
          f"({phase2_time/max(n_samples,1):.2f}s/sample)")

    del all_prepared

    # ---- Save activations (all samples, per-layer) ----
    act_dir = save_dir / "activations"
    if act_dir.is_symlink():  # left by --reuse-round1: never write into the reused run
        act_dir.unlink()
    act_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for layer_idx in tqdm(range(n_layers), desc="Saving layers"):
        layer_data = {}
        for comp in hidden_dims:
            if layer_buffers[layer_idx][comp]:
                arr = np.stack(layer_buffers[layer_idx][comp]).astype(np.float16)
                layer_data[comp] = arr
                total_bytes += arr.nbytes
        np.savez_compressed(act_dir / f"L{layer_idx}.npz", **layer_data)
        del layer_buffers[layer_idx]

    del layer_buffers
    gc.collect()
    print(f"Saved activations: {n_layers} layers, {total_bytes/1e6:.0f} MB -> {act_dir}")

    # ---- Keyword labels: 1 = harmful+REFUSE, 0 = safe+COMPLY, -1 = not in the probe set ----
    probe_labels = []
    for i, rec in enumerate(per_sample):
        if i < n_harmful:
            probe_labels.append(1 if rec["kw_label"] == "REFUSE" else -1)
        else:
            probe_labels.append(0 if rec["kw_label"] == "COMPLY" else -1)
    probe_labels = np.array(probe_labels, dtype=np.int8)
    np.save(save_dir / "labels.npy", probe_labels)

    # Also save group labels (harmful=1, safe=0) for all samples
    group_labels = np.array([1] * n_harmful + [0] * (n_samples - n_harmful), dtype=np.int8)
    np.save(save_dir / "group_labels.npy", group_labels)

    # ---- generation stats ----
    def count(rows, label):
        return sum(1 for r in rows if r["kw_label"] == label)
    harm_rows, safe_rows = per_sample[:n_harmful], per_sample[n_harmful:]
    harm_refuse = count(harm_rows, "REFUSE")
    safe_comply = count(safe_rows, "COMPLY")
    n_no_answer = count(per_sample, "NO_ANSWER")
    baseline_kw = harm_refuse / max(n_harmful, 1)
    n_valid = int((probe_labels >= 0).sum())

    gen_stats = {
        "n_harmful_input": n_harmful,
        "n_safe_input": n_samples - n_harmful,
        "harmful_refuse": harm_refuse,
        "harmful_comply": count(harm_rows, "COMPLY"),
        "safe_comply": safe_comply,
        "safe_refuse": count(safe_rows, "REFUSE"),
        "n_failed": n_failed,
        "n_no_answer": n_no_answer,
        "baseline_kw_refusal": round(baseline_kw, 4),
        "low_baseline": baseline_kw < 0.05,
        "n_probe_samples": n_valid,
    }

    print("  Generation results:")
    print(f"    Harmful: {harm_refuse}/{n_harmful} REFUSE ({baseline_kw*100:.1f}%)")
    print(f"    Safe: {safe_comply}/{n_samples - n_harmful} COMPLY")
    print(f"    Failed: {n_failed}; unlabeled: {n_no_answer}")
    print(f"    Probe samples (label != -1): {n_valid}")
    if gen_stats["low_baseline"]:
        print(f"    baseline keyword refusal {baseline_kw*100:.1f}% < 5%")

    # Save per-sample results and stats
    per_sample_dir = save_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)
    with open(per_sample_dir / "generate.jsonl", "w") as f:
        for rec in per_sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(save_dir / "generation_stats.json", "w") as f:
        json.dump(gen_stats, f, indent=2)

    extract_meta = {"n_layers": n_layers, "hidden_dims": hidden_dims, "n_samples": n_samples}
    return gen_stats, extract_meta


def detect_available_components(extractor: VLMActivationExtractor) -> List[str]:
    """Components of {mlp_gate, mlp_up, mlp_out, self_attn} present on this architecture."""
    return extractor.available_components()


# ---------------------------------------------------------------------------
# Load activations from disk (one layer at a time for probing)
# ---------------------------------------------------------------------------

def load_layer_activations(act_dir: Path, layer_idx: int) -> Dict[str, np.ndarray]:
    """Load activations for a single layer from disk. Returns {comp: array}."""
    npz_path = act_dir / f"L{layer_idx}.npz"
    if not npz_path.exists():
        return {}
    with np.load(npz_path) as data:
        return {comp: data[comp].astype(np.float32) for comp in data.files}


def get_layer_list(act_dir: Path) -> List[int]:
    """Get sorted list of layer indices from saved activation directory."""
    return sorted(int(p.stem[1:]) for p in act_dir.glob("L*.npz"))


def merge_layer_activations(dirs: List[Path], layer_idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load and concatenate a single layer's activations from multiple directories.

    Returns (merged_layer_data, merged_labels).
    Peak RAM = one layer from all runs, not all layers.
    """
    all_layer_data = []
    all_labels = []

    for d in dirs:
        d = Path(d)
        act_dir = d / "activations"
        labels = np.load(d / "labels.npy")
        layer_data = load_layer_activations(act_dir, layer_idx)
        all_layer_data.append(layer_data)
        all_labels.append(labels)

    comps = sorted(all_layer_data[0].keys())
    merged = {}
    for comp in comps:
        arrays = [ld[comp] for ld in all_layer_data if comp in ld]
        merged[comp] = np.concatenate(arrays, axis=0)

    merged_labels = np.concatenate(all_labels)
    return merged, merged_labels


# ---------------------------------------------------------------------------
# Linear probe + Cohen's d screening
# ---------------------------------------------------------------------------

def run_probes_from_disk(
    act_dirs: List[Path],
    labels: np.ndarray,
    layer_list: List[int],
    z_threshold: float = 3.0,
    holdout_ratio: float = 0.2,
    probe_C: float = 1.0,
    max_iter: int = 5000,
    seed: int = 42,
    valid_mask: np.ndarray = None,
) -> Tuple[Dict[Tuple, SafetyNeuron], dict]:
    """Behavioral probes, loading one layer at a time from disk.

    `labels` holds the probe labels (1 = harmful refused, 0 = safe complied) of the rows kept
    by `valid_mask`. One stratified 80/20 split is shared by all (layer, component) probes;
    diagnostics["val_rows"] lists the validation rows as indices into the unmasked samples.

    Phase 1: per-(layer, component) probe -> sign(w) * |Cohen's d| per neuron
    Phase 2: per-layer z-standardization (pooled over components) -> |z| > z_threshold
    """
    print("\n=== Running Linear Probe Screening (disk-backed) ===")
    print(f"  z_threshold={z_threshold}, holdout={holdout_ratio}, C={probe_C}")
    print(f"  n_samples={len(labels)}, n_positives={int(sum(labels))}, n_negatives={int(sum(1-labels))}")
    print(f"  Loading from {len(act_dirs)} activation dir(s), {len(layer_list)} layers")

    y = np.asarray(labels).astype(int)
    train_idx, val_idx = probe_split(y, holdout_ratio, seed)
    rows = np.where(valid_mask)[0] if valid_mask is not None else np.arange(len(y))

    diagnostics = {
        "per_layer": [],
        "total_candidates": 0,
        "components_scanned": 0,
        "val_rows": [int(rows[i]) for i in val_idx],
    }

    raw_scores_all = {}

    for layer_idx in tqdm(layer_list, desc="Probing layers"):
        if len(act_dirs) == 1:
            layer_data = load_layer_activations(act_dirs[0], layer_idx)
        else:
            layer_data, _ = merge_layer_activations([d.parent for d in act_dirs], layer_idx)
        if valid_mask is not None:
            layer_data = {comp: arr[valid_mask] for comp, arr in layer_data.items()}

        for comp, X in layer_data.items():
            key = (layer_idx, comp)
            scores, weights, cohen_d, train_acc, val_acc = probe_scores(
                X[train_idx], y[train_idx], X[val_idx], y[val_idx], probe_C, max_iter, seed)

            neuron_data = {}
            for idx in range(len(scores)):
                neuron_data[idx] = {
                    "raw_score": float(scores[idx]),
                    "weight": float(weights[idx]),
                    "cohen_d": float(cohen_d[idx]),
                    "probe_train_accuracy": train_acc,
                    "probe_val_accuracy": val_acc,
                }
            raw_scores_all[key] = neuron_data

            diagnostics["per_layer"].append({
                "layer": layer_idx, "component": comp,
                "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
                "n_candidates": -1,
            })
            diagnostics["components_scanned"] += 1

            if layer_idx % 5 == 0:
                print(f"  L{layer_idx}/{comp}: train={train_acc:.3f}, val={val_acc:.3f}")

        del layer_data
        gc.collect()

    # Phase 2: Per-layer z-normalization across all components
    print("\n--- Phase 2: Per-layer z-normalization ---")
    candidates = {}

    # Build layer_components mapping from raw_scores_all keys
    layer_components = defaultdict(list)
    for (layer_idx, comp) in raw_scores_all:
        layer_components[layer_idx].append(comp)

    for layer_idx in sorted(layer_components.keys()):
        # Collect ALL raw scores from ALL components in this layer
        all_raw = []
        comp_ranges = {}  # comp -> (start_idx, end_idx) in all_raw

        for comp in layer_components[layer_idx]:
            key = (layer_idx, comp)
            if key not in raw_scores_all:
                continue
            neuron_data = raw_scores_all[key]
            start = len(all_raw)
            for idx in sorted(neuron_data.keys()):
                all_raw.append(neuron_data[idx]["raw_score"])
            comp_ranges[comp] = (start, len(all_raw))

        if len(all_raw) < 10:
            continue

        all_raw = np.array(all_raw)
        # Z-normalize: pool across all components within this layer
        layer_mean = all_raw.mean()
        layer_std = all_raw.std() + 1e-8
        z_scores = (all_raw - layer_mean) / layer_std

        # Select neurons above threshold
        layer_candidates_count = 0
        for comp, (start, end) in comp_ranges.items():
            key = (layer_idx, comp)
            neuron_data = raw_scores_all[key]
            sorted_indices = sorted(neuron_data.keys())

            for i, idx in enumerate(sorted_indices):
                z = float(z_scores[start + i])
                if abs(z) > z_threshold:
                    meta = neuron_data[idx].copy()
                    meta["probe_score"] = meta.pop("raw_score")
                    neuron = SafetyNeuron(
                        layer=layer_idx,
                        component=comp,
                        index=idx,
                        score=z,
                        method="linear_probe",
                        metadata=meta,
                    )
                    candidates[neuron.to_tuple()] = neuron
                    layer_candidates_count += 1

            # Update diagnostics n_candidates
            for d in diagnostics["per_layer"]:
                if d["layer"] == layer_idx and d["component"] == comp:
                    comp_z = z_scores[start:end]
                    d["n_candidates"] = int(np.sum(np.abs(comp_z) > z_threshold))
                    break

        if layer_candidates_count > 0:
            print(f"  L{layer_idx}: {layer_candidates_count} candidates "
                  f"(layer_mean={layer_mean:.4f}, layer_std={layer_std:.4f})")

    diagnostics["total_candidates"] = len(candidates)
    print(f"\nTotal candidates: {len(candidates)} across {diagnostics['components_scanned']} (layer, component) pairs")

    return candidates, diagnostics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Visual safety neuron screening")
    parser.add_argument("--model", type=str, required=True, choices=list(load_registry()["models"]),
                        help="Model key from configs/model_registry.yaml")
    parser.add_argument("--label-file", type=str, default=None,
                        help="ViSafe-Detect label JSONL (default: data/visafe_detect/labels.jsonl)")
    parser.add_argument("--n-harmful", type=int, default=None,
                        help="Cap on harmful images (default: all)")
    parser.add_argument("--n-safe", type=int, default=None,
                        help="Cap on safe images (default: all)")
    parser.add_argument("--z-threshold", type=float, default=None,
                        help="|z| threshold (default: 3.0, from configs/pipeline.yaml)")
    parser.add_argument("--holdout-ratio", type=float, default=None,
                        help="Probe validation fraction (default: 0.2, from configs/pipeline.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test with 20 harmful + 20 safe")
    parser.add_argument("--extract-only", action="store_true",
                        help="Extract and save activations to disk, skip probing")
    parser.add_argument("--probe-only", action="store_true",
                        help="Load saved activations and run probes (no GPU needed)")
    parser.add_argument("--activation-dirs", nargs="+", type=str, default=None,
                        help="Directories with saved activations to merge (for --probe-only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--prefill-bs", dest="gen_bs", type=int, default=4,
                        help="Prefill batch size, at most 8 (default: 4; generation runs one sample at a time)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/stage1/<model>)")
    args = parser.parse_args()

    # Fix all random sources for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- probe-only mode: merge saved activations -> probe -> save results ----
    if args.probe_only:
        if not args.activation_dirs:
            sys.exit("--probe-only requires --activation-dirs")
        pipeline_cfg = load_pipeline_config()
        s1_cfg = pipeline_cfg["detection"]["stage1"]

        act_dirs = [Path(d) / "activations" for d in args.activation_dirs]
        # Load and merge labels, filter out -1
        all_raw_labels = [np.load(Path(d) / "labels.npy") for d in args.activation_dirs]
        raw_labels = np.concatenate(all_raw_labels)
        valid_mask = raw_labels >= 0
        probe_labels = raw_labels[valid_mask].astype(np.float64)
        layer_list = get_layer_list(act_dirs[0])

        n_pos = int(probe_labels.sum())
        n_neg = len(probe_labels) - n_pos
        print(f"Probe-only: {len(args.activation_dirs)} dirs, "
              f"{len(probe_labels)} probe samples ({n_pos} positives / {n_neg} negatives), "
              f"{len(layer_list)} layers")

        run_dir = (Path(args.output_dir) if args.output_dir
                   else REPO_ROOT / "results" / "stage1" / f"{args.model}_merged")
        run_dir.mkdir(parents=True, exist_ok=True)

        z_threshold = args.z_threshold or s1_cfg["z_threshold"]
        candidates, diagnostics = run_probes_from_disk(
            act_dirs=act_dirs,
            labels=probe_labels,
            layer_list=layer_list,
            z_threshold=z_threshold,
            holdout_ratio=args.holdout_ratio or s1_cfg["holdout_ratio"],
            probe_C=s1_cfg["probe_C"],
            max_iter=s1_cfg["max_iter"],
            seed=args.seed,
            valid_mask=valid_mask,
        )

        save_neurons(candidates, run_dir / "stage1_candidates.json")
        with open(run_dir / "probe_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)
        with open(run_dir / "summary.json", "w") as f:
            json.dump({
                "model_key": args.model,
                "mode": "probe_only",
                "activation_dirs": args.activation_dirs,
                "n_positives": n_pos,
                "n_negatives": n_neg,
                "stage1_candidates": len(candidates),
                "z_threshold": z_threshold,
                "timestamp": timestamp,
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Probe-only complete: {len(candidates)} candidates -> {run_dir}")
        print(f"{'='*60}")
        sys.exit(0)

    # ---- normal / extract-only mode ----

    # Load configs
    registry = load_registry()
    pipeline_cfg = load_pipeline_config()

    model_key = args.model
    model_cfg = registry["models"][model_key]

    hf_repo = model_cfg["hf_repo"]
    query_template = pipeline_cfg["data"]["harmful_image"]["query_template"]
    max_side = pipeline_cfg["image"]["max_side"]

    run_dir = (Path(args.output_dir) if args.output_dir
               else REPO_ROOT / "results" / "stage1" / model_key)

    print(f"\n{'='*60}")
    print("Stage 1: Visual Safety Neuron Screening")
    print(f"Model:     {model_key} ({hf_repo})")
    print(f"Output:    {run_dir}")
    print(f"Dry-run:   {args.dry_run}")
    print(f"{'='*60}")

    # Step 1: Load and filter labels
    label_path = Path(args.label_file) if args.label_file else DEFAULT_LABEL_FILE
    print(f"\n[1/6] Loading labels from {label_path}")
    all_labels = load_labels(label_path)
    print(f"  Total labels: {len(all_labels)}")

    harmful_labels = filter_harmful_labels(all_labels)
    safe_labels = filter_safe_labels(all_labels)
    print(f"  Harmful: {len(harmful_labels)}")
    print(f"  Safe: {len(safe_labels)}")

    if args.dry_run:
        args.n_harmful = args.n_harmful or 20
        args.n_safe = args.n_safe or 20

    n_harmful = args.n_harmful or len(harmful_labels)
    n_safe = args.n_safe or len(safe_labels)

    # Shuffle for variety
    rng = random.Random(args.seed)
    rng.shuffle(harmful_labels)
    rng.shuffle(safe_labels)
    harmful_labels = harmful_labels[:n_harmful]
    safe_labels = safe_labels[:n_safe]

    print(f"  Using: {len(harmful_labels)} harmful, {len(safe_labels)} safe")

    if len(harmful_labels) < 20 or len(safe_labels) < 20:
        sys.exit(f"Not enough labeled data: need >= 20 harmful and >= 20 safe, "
                 f"got {len(harmful_labels)} harmful, {len(safe_labels)} safe")

    # Step 2: Load images
    print("\n[2/6] Loading images from data/visafe_detect/images/...")
    harmful_queries = load_images_for_labels(harmful_labels, query_template, max_side)
    safe_queries = load_safe_images_for_screening(safe_labels, query_template, max_side)

    print(f"  Loaded: {len(harmful_queries)} harmful, {len(safe_queries)} safe images")

    if len(harmful_queries) < 10 or len(safe_queries) < 10:
        sys.exit("Too few images loaded: run data/build/fetch_visafe_detect.py first")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Free dataset memory
    del harmful_labels, safe_labels, all_labels
    gc.collect()

    # Step 3: Load model
    print(f"\n[3/6] Loading model {model_key}...")
    model, processor = load_model(hf_repo, model_key)

    extractor = VLMActivationExtractor(
        model, processor, thinking_mode=thinking_setting(model_key),
        thinking_budget=pipeline_cfg.get("generation", {}).get("thinking_budget", 1024))

    # Determine components to scan via auto-detection
    components = detect_available_components(extractor)
    print(f"  Components: {components}")
    print(f"  Layers: {extractor.num_layers}")

    # Forward keys accepted by this model (read from its signature)
    valid_keys = build_valid_keys(model)

    gen_max_tokens = pipeline_cfg.get("generation", {}).get("max_new_tokens", 512)
    if model_cfg.get("size_b", 0) >= 30 and args.gen_bs > 2:
        args.gen_bs = 2  # smaller prefill batches for the 31B/32B models

    print("\n[4/6] Generate + extract activations (merged, single pass)...")
    print(f"  max_new_tokens={gen_max_tokens} (from pipeline.yaml)")

    t0 = time.time()
    gen_stats, extract_meta = generate_and_extract(
        model, processor, extractor,
        harmful_queries, safe_queries,
        components, run_dir,
        max_new_tokens=gen_max_tokens,
        valid_keys=valid_keys,
        gen_batch_size=args.gen_bs,
    )
    extract_time = time.time() - t0
    n_total = len(harmful_queries) + len(safe_queries)
    print(f"  Took {extract_time:.1f}s ({extract_time/n_total:.2f}s/sample)")

    # Free model GPU memory
    del model, processor, extractor
    torch.cuda.empty_cache()
    gc.collect()
    print("  GPU memory freed, running probes on CPU")

    if gen_stats["low_baseline"]:
        print(f"\n  baseline keyword refusal {gen_stats['baseline_kw_refusal']*100:.1f}% < 5%")

    min_positives = DetectionConfig.from_pipeline_yaml(pipeline_cfg).min_positives
    if gen_stats["harmful_refuse"] < min_positives:
        print(f"  Too few probe samples (total={gen_stats['n_probe_samples']}, harmful={gen_stats['harmful_refuse']})")
        print(f"  Fewer than {min_positives} behavioral positives: skipping the probe")
        with open(run_dir / "too_few_positives.md", "w") as f:
            f.write(f"# Too few behavioral positives: {model_key}\n\n")
            f.write(f"Baseline KW refusal: {gen_stats['baseline_kw_refusal']*100:.1f}%\n")
            f.write(f"Harmful REFUSE: {gen_stats['harmful_refuse']}\n")
            f.write(f"Safe COMPLY: {gen_stats['safe_comply']}\n")
            f.write(f"Probe samples: {gen_stats['n_probe_samples']}\n")
        # Save empty candidates
        with open(run_dir / "stage1_candidates.json", "w") as f:
            json.dump([], f)
        print("  --- Done ---")
        sys.exit(0)

    if args.extract_only:
        print(f"\n{'='*60}")
        print(f"Extract-only complete: {n_total} samples -> {run_dir}")
        print("  Activations for ALL samples saved (samples labeled -1 are left out of the probe)")
        print(f"  Use --probe-only --activation-dirs {run_dir} to run probes later")
        print(f"{'='*60}")
        sys.exit(0)

    # Step 5: Run probes (layer-by-layer from disk — low RAM)
    # Load labels; -1 marks samples that are neither positives nor negatives
    labels = np.load(run_dir / "labels.npy")
    valid_mask = labels >= 0
    probe_labels = labels[valid_mask].astype(np.float64)
    print("\n[5/6] Running linear probe screening...")
    s1_cfg = pipeline_cfg["detection"]["stage1"]
    z_threshold = args.z_threshold or s1_cfg["z_threshold"]

    act_dir = run_dir / "activations"
    layer_list = get_layer_list(act_dir)

    t1 = time.time()
    candidates, diagnostics = run_probes_from_disk(
        act_dirs=[act_dir],
        labels=probe_labels,
        layer_list=layer_list,
        z_threshold=z_threshold,
        holdout_ratio=args.holdout_ratio or s1_cfg["holdout_ratio"],
        probe_C=s1_cfg["probe_C"],
        max_iter=s1_cfg["max_iter"],
        seed=args.seed,
        valid_mask=valid_mask,
    )
    probe_time = time.time() - t1
    print(f"  Probing took {probe_time:.1f}s")

    # Step 6: Save results
    print(f"\n[6/6] Saving results to {run_dir}")

    save_neurons(candidates, run_dir / "stage1_candidates.json")

    with open(label_path) as f:
        label_file_lines = sum(1 for _ in f)
    metadata = {
        "model_key": model_key,
        "hf_repo": hf_repo,
        "timestamp": timestamp,
        "label_file": repo_path(label_path),
        "label_file_lines": label_file_lines,
        "n_harmful_input": gen_stats["n_harmful_input"],
        "n_safe_input": gen_stats["n_safe_input"],
        "n_harmful_refuse": gen_stats["harmful_refuse"],
        "n_safe_comply": gen_stats["safe_comply"],
        "n_probe_samples": int(valid_mask.sum()),
        "n_total_with_activations": n_total,
        "stage1_candidates": len(candidates),
        "z_threshold": z_threshold,
        "holdout_ratio": args.holdout_ratio or s1_cfg["holdout_ratio"],
        "probe_C": s1_cfg["probe_C"],
        "generate_time_s": round(extract_time, 1),
        "probe_time_s": round(probe_time, 1),
        "components": components,
        "n_layers": extract_meta["n_layers"],
        "dry_run": args.dry_run,
        "seed": args.seed,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(run_dir / "probe_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    n_harmful_probe = int(probe_labels.sum())
    n_safe_probe = len(probe_labels) - n_harmful_probe
    print(f"\n{'='*60}")
    print("Stage 1 Screening Complete")
    print(f"  Model:       {model_key}")
    print(f"  Activations: {n_total} total (all saved)")
    print(f"  Probe:       {n_harmful_probe} harmful + {n_safe_probe} safe")
    print(f"  Candidates:  {len(candidates)} neurons (|z| > {z_threshold})")
    print(f"  Time:        {extract_time:.0f}s generation + {probe_time:.0f}s probing")
    print(f"  Output:      {run_dir}")

    # Component breakdown
    comp_counts = defaultdict(int)
    layer_counts = defaultdict(int)
    for (layer, comp, idx) in candidates:
        comp_counts[comp] += 1
        layer_counts[layer] += 1

    print("\n  By component:")
    for comp in sorted(comp_counts):
        print(f"    {comp}: {comp_counts[comp]}")

    if layer_counts:
        print("\n  By layer (top 10):")
        for layer in sorted(layer_counts, key=layer_counts.get, reverse=True)[:10]:
            print(f"    L{layer}: {layer_counts[layer]}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
