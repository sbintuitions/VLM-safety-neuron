"""
Safety-neuron detection for vision-language models with PyTorch forward hooks (paper Sec. 3).

A neuron is n = (layer, component, index) with component in
{mlp_gate, mlp_up, mlp_out, self_attn}: the gate_proj and up_proj outputs (d_int channels,
kept separate; a fused gate_up_proj is split into its two halves), the MLP block output and the
attention output projection (d_model channels). a_n(x) is read at the last input-token position.

  - VLMActivationExtractor: input preparation, activation capture, mean-activation clamps
    (temporary and permanent), generation and the first-token logits.
  - VLMSafetyNeuronDetector:
      detect_linear_probe()     behavioral probing (Sec. 3.3), text pipeline
      detect_activation_patch() mean-activation patching against a control-neuron null with
                                BH-FDR (Sec. 3.4), shared by the text and visual pipelines

Decoder layers are found automatically: model.model.layers, model.model.language_model.layers
(the VLM classes of transformers 5.x), model.language_model.model.layers (InternVL) or
model.language_model.layers.
"""

import inspect
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_utils import SafetyQuery
from neuron_types import SafetyNeuron
from refusal import refusal_probability, refusal_token_ids

COMPONENTS = ("mlp_gate", "mlp_up", "mlp_out", "self_attn")

FORWARD_KEYS = {
    "input_ids", "attention_mask", "position_ids",
    "pixel_values", "image_grid_thw", "image_sizes",
    "mm_token_type_ids", "token_type_ids", "image_embeds",
    "image_flags", "pixel_values_videos", "video_grid_thw",
    "pixel_position_ids", "image_position_ids",
}


def probe_split(y: np.ndarray, holdout_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/validation split of sample indices (the 80/20 probe split)."""
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(idx, test_size=holdout_ratio, stratify=y, random_state=seed)
    y_int = np.asarray(y).astype(int)
    for part in (train_idx, val_idx):
        if min(np.bincount(y_int[part], minlength=2)) < 2:
            raise ValueError("probe split: each class needs at least 2 samples in both the training "
                             f"and the validation part (class counts {np.bincount(y_int, minlength=2).tolist()})")
    return np.sort(train_idx), np.sort(val_idx)


def probe_scores(X_train, y_train, X_val, y_val, C: float, max_iter: int, seed: int):
    """Fit the l2 probe on the training split; score_n = sign(w_n) * |Cohen's d_n| on validation.

    Returns (score, weight, cohen_d, train_acc, val_acc).
    """
    clf = LogisticRegression(penalty="l2", C=C, max_iter=max_iter, random_state=seed)
    clf.fit(X_train, y_train)
    w = clf.coef_[0]
    harm, safe = X_val[y_val == 1], X_val[y_val == 0]
    s_pooled = np.sqrt(0.5 * (harm.var(axis=0) + safe.var(axis=0)) + 1e-8)
    d = (harm.mean(axis=0) - safe.mean(axis=0)) / s_pooled
    return (np.sign(w) * np.abs(d), w, d,
            float(clf.score(X_train, y_train)), float(clf.score(X_val, y_val)))


class VLMActivationExtractor:
    """Hooks, inputs and forward passes for one model.

    ``hooks`` are temporary and removed by clear_hooks(); ``permanent_hooks`` hold the
    clamps of neurons confirmed in earlier rounds (the model f_theta^(k)) and survive it.
    ``thinking_mode`` controls enable_thinking in the chat template (None: not passed).
    """

    def __init__(self, model, processor, thinking_mode: Optional[bool] = None,
                 thinking_budget: int = 1024):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.thinking_mode = thinking_mode
        self.thinking_budget = thinking_budget
        self.device = next(model.parameters()).device
        self.hooks = []
        self.permanent_hooks = []
        self._is_internvl = "internvl" in type(model).__name__.lower()
        self._detect_layers()

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _detect_layers(self):
        import torch.nn as nn
        m = self.model
        candidates = [
            lambda: m.model.layers,
            lambda: m.model.language_model.layers,
            lambda: m.language_model.model.layers,
            lambda: m.language_model.layers,
        ]
        layers = None
        for get in candidates:
            try:
                obj = get()
            except AttributeError:
                continue
            if isinstance(obj, nn.ModuleList):
                layers = obj
                break
        if layers is None:
            queue = [m]
            for module in queue:
                for child in module.children():
                    if isinstance(child, nn.ModuleList) and len(child) >= 20 \
                            and hasattr(child[0], "mlp"):
                        layers = child
                        break
                    queue.append(child)
                if layers is not None:
                    break
        if layers is None:
            raise ValueError(f"Cannot detect the decoder layers of {type(m).__name__}")
        self.layers = layers
        self.num_layers = len(layers)
        print(f"Detected {self.num_layers} decoder layers in {type(m).__name__}")

    def _resolve(self, layer_idx: int, component: str):
        """(module, lo, hi): the neuron channels of `component` are module output[..., lo:hi]."""
        layer = self.layers[layer_idx]
        mlp = layer.mlp
        if component in ("mlp_gate", "mlp_up"):
            name = "gate_proj" if component == "mlp_gate" else "up_proj"
            if hasattr(mlp, name):
                mod = getattr(mlp, name)
                return mod, 0, mod.out_features
            if hasattr(mlp, "gate_up_proj"):  # fused [gate | up] (GLM-4.1V)
                mod = mlp.gate_up_proj
                d = mod.out_features // 2
                return (mod, 0, d) if component == "mlp_gate" else (mod, d, 2 * d)
            raise ValueError(f"No {name} on {type(mlp).__name__}")
        if component == "mlp_out":
            return mlp, 0, mlp.down_proj.out_features
        if component == "self_attn":
            if hasattr(layer, "self_attn"):
                mod = layer.self_attn.o_proj
            elif hasattr(layer, "linear_attn"):  # Qwen3.5 linear-attention layers
                mod = layer.linear_attn.out_proj
            else:
                raise ValueError(f"No attention output on {type(layer).__name__}")
            return mod, 0, mod.out_features
        raise ValueError(f"Unknown component: {component}")

    def available_components(self) -> List[str]:
        out = []
        for comp in COMPONENTS:
            try:
                self._resolve(0, comp)
                out.append(comp)
            except (ValueError, AttributeError):
                pass
        return out

    def component_dim(self, component: str) -> int:
        _, lo, hi = self._resolve(0, component)
        return hi - lo

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def capture_hook(self, layer_idx: int, component: str, callback):
        """Register a hook that calls callback(activation[..., lo:hi]); returns the handle."""
        mod, lo, hi = self._resolve(layer_idx, component)

        def hook_fn(module, inputs, output):
            act = output[0] if isinstance(output, tuple) else output
            callback(act[..., lo:hi])

        return mod.register_forward_hook(hook_fn)

    def _clamp_hook(self, layer_idx: int, component: str, index: int, value: float):
        """Clamp a_n to `value` at the last position of every forward pass.

        In a prefill this is the last input token; during generation it is each new token.
        """
        mod, lo, _ = self._resolve(layer_idx, component)
        channel = lo + index

        def hook_fn(module, inputs, output):
            act = output[0] if isinstance(output, tuple) else output
            if act.dim() == 3:
                act[:, -1, channel] = value
            else:
                act[-1, channel] = value
            return output

        return mod.register_forward_hook(hook_fn)

    def register_patch_hook(self, layer_idx: int, component: str, index: int, value: float):
        self.hooks.append(self._clamp_hook(layer_idx, component, index, value))

    def add_permanent_hook(self, layer_idx: int, component: str, index: int, value: float):
        self.permanent_hooks.append(self._clamp_hook(layer_idx, component, index, value))

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clear_permanent_hooks(self):
        for h in self.permanent_hooks:
            h.remove()
        self.permanent_hooks.clear()

    # ------------------------------------------------------------------
    # Inputs, forward, generation
    # ------------------------------------------------------------------

    def _chat_kwargs(self, generation: bool = False):
        """enable_thinking for the chat template."""
        if self.thinking_mode is None:
            return {}
        return {"enable_thinking": bool(self.thinking_mode) and generation}

    def _internvl_prepare_inputs(self, query: SafetyQuery):
        """InternVL: replicate chat() token expansion; text-only queries get no image."""
        question = query.text
        out = {}
        if query.image is not None:
            image_inputs = self.processor.image_processor(
                images=[query.image], crop_to_patches=True, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(device=self.device, dtype=torch.bfloat16)
            num_image_token = getattr(self.model, "num_image_token", 256)
            question = f"<image>\n{query.text}"
            for num_patches in image_inputs["num_patches"]:
                question = question.replace(
                    "<image>", "<img>" + "<IMG_CONTEXT>" * num_image_token * num_patches + "</img>", 1)
            out["pixel_values"] = pixel_values
            out["image_flags"] = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=self.device)
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
        tok = self.tokenizer(text, return_tensors="pt")
        out["input_ids"] = tok["input_ids"].to(self.device)
        out["attention_mask"] = tok["attention_mask"].to(self.device)
        return out

    def _prepare_inputs(self, query: SafetyQuery, generation: bool = False):
        """Chat-template the query (text-only, or one image followed by the text).
        generation=True builds the prompt used to generate the response."""
        if self._is_internvl:
            return self._internvl_prepare_inputs(query)
        if query.image is not None:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": query.text}]}]
        else:
            messages = [{"role": "user", "content": query.text}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self._chat_kwargs(generation))
        if query.image is not None:
            inputs = self.processor(text=[text], images=[query.image], return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        return inputs.to(self.device)

    def _model_inputs(self, inputs) -> dict:
        """Keep the processor outputs the model takes. For text-only inputs, modality-type
        keys the model does not name (e.g. mm_token_type_ids on a text-only Qwen3.5) are dropped."""
        fwd = {k: v for k, v in inputs.items() if k in FORWARD_KEYS}
        if "pixel_values" not in fwd:
            named = set(inspect.signature(self.model.forward).parameters)
            for k in ("mm_token_type_ids", "token_type_ids"):
                if k in fwd and k not in named:
                    fwd.pop(k)
        return fwd

    def forward(self, inputs):
        """Forward pass on prepared inputs (InternVL text-only goes through its language model)."""
        fwd = self._model_inputs(inputs)
        with torch.no_grad():
            if self._is_internvl and "pixel_values" not in fwd:
                return self.model.language_model(
                    input_ids=fwd["input_ids"], attention_mask=fwd.get("attention_mask"))
            return self.model(**fwd)

    def generate_ids(self, inputs, max_new_tokens: int = 512) -> List[int]:
        """Greedy decoding; returns the generated token IDs only."""
        fwd = self._model_inputs(inputs)
        if self.thinking_mode:
            max_new_tokens += self.thinking_budget
        with torch.no_grad():
            if self._is_internvl:
                out = self.model.generate(
                    pixel_values=fwd.get("pixel_values"), input_ids=fwd["input_ids"],
                    attention_mask=fwd.get("attention_mask"),
                    max_new_tokens=max_new_tokens, do_sample=False)
                return out[0].tolist()  # InternVL generates from embeddings: new tokens only
            fwd.pop("image_flags", None)
            out = self.model.generate(**fwd, max_new_tokens=max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None)
        input_len = fwd["input_ids"].shape[1]
        seq = out[0]
        return (seq[input_len:] if seq.shape[0] > input_len else seq).tolist()

    def generate(self, query: SafetyQuery, max_new_tokens: int = 512) -> List[int]:
        return self.generate_ids(self._prepare_inputs(query, generation=True), max_new_tokens)

    def get_last_logits(self, query: SafetyQuery) -> torch.Tensor:
        """Logits at the last input position (the first generated token), shape [vocab]."""
        return self.forward(self._prepare_inputs(query)).logits[0, -1, :].float()

    def extract_last_token(self, queries: List[SafetyQuery], layers: Sequence[int],
                           components: Sequence[str], layers_per_pass: int = 32,
                           desc: str = "Activations") -> Dict[Tuple[int, str], np.ndarray]:
        """a_n(x) at the last input token for all channels of the given layers/components.

        Returns {(layer, component): float32 array [n_queries, dim]}. Layers are hooked in
        chunks of `layers_per_pass` per forward pass (fewer live hooks on multi-GPU models).
        """
        out = {(l, c): np.zeros((len(queries), self.component_dim(c)), dtype=np.float32)
               for l in layers for c in components}
        layers = list(layers)
        chunks = [layers[i:i + layers_per_pass] for i in range(0, len(layers), layers_per_pass)]
        row = [0]

        def make_cb(key):
            def cb(a):
                out[key][row[0]] = a[0, -1].float().cpu().numpy()
            return cb

        for chunk in chunks:
            handles = [self.capture_hook(l, c, make_cb((l, c))) for l in chunk for c in components]
            try:
                for i, q in enumerate(tqdm(queries, desc=desc, leave=False)):
                    row[0] = i
                    self.forward(self._prepare_inputs(q))
            finally:
                for h in handles:
                    h.remove()
        return out

    def __del__(self):
        try:
            self.clear_hooks()
            self.clear_permanent_hooks()
        except Exception:
            pass


class VLMSafetyNeuronDetector:
    """Two-stage detector: behavioral probing, then mean-activation patching."""

    def __init__(self, model, processor, config, thinking_mode: Optional[bool] = None):
        self.model = model
        self.processor = processor
        self.config = config
        self.extractor = VLMActivationExtractor(
            model, processor, thinking_mode=thinking_mode,
            thinking_budget=getattr(config, "thinking_budget", 1024))
        available = self.extractor.available_components()
        self.components = [c for c in getattr(config, "components", available) if c in available]
        self.heldout_safe_queries: List[SafetyQuery] = []
        self.last_patch_results: List[dict] = []

    # ------------------------------------------------------------------
    # Stage 1: behavioral probing
    # ------------------------------------------------------------------

    def detect_linear_probe(
        self,
        positive_queries: List[SafetyQuery],
        negative_queries: List[SafetyQuery],
        exclude: Iterable[Tuple] = (),
    ) -> Dict[Tuple, SafetyNeuron]:
        """Per-(layer, component) probes on harmful-refused (1) vs safe-complied (0) queries.

        Scores are z-standardized within each layer (pooled over its components) and
        neurons with |z| > z_threshold become candidates; `exclude` (neurons confirmed in
        earlier rounds) is removed from the pool. The safe queries of the validation split
        are kept in self.heldout_safe_queries for the safe-mean activations of Stage 2.
        """
        cfg = self.config
        exclude = set(exclude)
        queries = positive_queries + negative_queries
        y = np.array([1] * len(positive_queries) + [0] * len(negative_queries))
        train_idx, val_idx = probe_split(y, cfg.probe_holdout_ratio, cfg.seed)
        self.heldout_safe_queries = [queries[i] for i in val_idx if y[i] == 0]

        print(f"\n=== Behavioral probing: {len(positive_queries)} positives, "
              f"{len(negative_queries)} negatives, components {self.components} ===")
        acts = self.extractor.extract_last_token(
            queries, range(self.extractor.num_layers), self.components, desc="Probe activations")
        candidates = {}
        for layer_idx in tqdm(range(self.extractor.num_layers), desc="Probe layers"):
            scores, meta = {}, {}
            for comp in self.components:
                X = acts.pop((layer_idx, comp))
                s, w, d, tr_acc, va_acc = probe_scores(
                    X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                    cfg.probe_C, cfg.probe_max_iter, cfg.seed)
                scores[comp] = s
                meta[comp] = (w, d, tr_acc, va_acc)
            pooled = np.concatenate([scores[c] for c in self.components])
            mu, sigma = pooled.mean(), pooled.std() + 1e-8

            n_layer = 0
            for comp in self.components:
                z = (scores[comp] - mu) / sigma
                w, d, tr_acc, va_acc = meta[comp]
                for idx in np.where(np.abs(z) > cfg.probe_z_threshold)[0]:
                    key = (layer_idx, comp, int(idx))
                    if key in exclude:
                        continue
                    candidates[key] = SafetyNeuron(
                        layer=layer_idx, component=comp, index=int(idx), score=float(z[idx]),
                        method="linear_probe",
                        metadata={"probe_score": float(scores[comp][idx]),
                                  "weight": float(w[idx]), "cohen_d": float(d[idx]),
                                  "probe_train_accuracy": tr_acc, "probe_val_accuracy": va_acc})
                    n_layer += 1
            if layer_idx % 5 == 0:
                accs = ", ".join(f"{c}={meta[c][3]:.3f}" for c in self.components)
                print(f"  L{layer_idx}: val acc {accs}; {n_layer} candidates")

        print(f"Stage 1: {len(candidates)} candidates with |z| > {cfg.probe_z_threshold}")
        return candidates

    # ------------------------------------------------------------------
    # Stage 2: mean-activation patching
    # ------------------------------------------------------------------

    def _safe_means(self, neurons: List[SafetyNeuron], safe_queries: List[SafetyQuery]) -> Dict[Tuple, float]:
        """a_hat_n: mean of a_n over the held-out safe queries."""
        keys = sorted({(n.layer, n.component) for n in neurons})
        layers = sorted({l for l, _ in keys})
        comps = sorted({c for _, c in keys})
        acts = self.extractor.extract_last_token(safe_queries, layers, comps, desc="Safe means")
        return {n.to_tuple(): float(acts[(n.layer, n.component)][:, n.index].mean()) for n in neurons}

    def _sample_controls(self, candidate_pool: Dict[Tuple, SafetyNeuron], exclude: set,
                         n_controls: int, rng: random.Random) -> List[SafetyNeuron]:
        """Control neurons from outside the candidate pool, allocated to layers in proportion
        to their candidate count and drawn uniformly from the neurons of that layer."""
        layer_counts = defaultdict(int)
        for (layer, _, _) in candidate_pool:
            layer_counts[layer] += 1
        total = sum(layer_counts.values())
        if total == 0:
            return []
        quotas = {l: n_controls * c / total for l, c in layer_counts.items()}
        alloc = {l: int(q) for l, q in quotas.items()}
        for l in sorted(quotas, key=lambda l: quotas[l] - alloc[l], reverse=True)[:n_controls - sum(alloc.values())]:
            alloc[l] += 1

        dims = [(c, self.extractor.component_dim(c)) for c in self.components]
        layer_size = sum(d for _, d in dims)
        taken = set(candidate_pool) | exclude
        controls = []
        for layer in sorted(alloc):
            drawn = 0
            while drawn < alloc[layer]:
                r = rng.randrange(layer_size)
                for comp, d in dims:
                    if r < d:
                        break
                    r -= d
                key = (layer, comp, r)
                if key in taken:
                    continue
                taken.add(key)
                controls.append(SafetyNeuron(layer=layer, component=comp, index=r,
                                             score=0.0, method="control"))
                drawn += 1
        return controls

    def detect_activation_patch(
        self,
        candidate_neurons: Dict[Tuple, SafetyNeuron],
        harmful_queries: List[SafetyQuery],
        safe_queries: List[SafetyQuery],
        n_queries: Optional[int] = None,
        exclude: Iterable[Tuple] = (),
    ) -> Dict[Tuple, SafetyNeuron]:
        """Certify candidates by mean-activation patching.

        harmful_queries: harmful-refused queries; the first n_queries are patched.
        safe_queries:    the held-out safe-query set D_safe for a_hat_n.
        A candidate is confirmed when p_adj < alpha (BH-FDR over candidates), mean dP > 0 and
        the probe sign agrees with the patching sign.
        """
        cfg = self.config
        exclude = set(exclude)
        rng = random.Random(cfg.seed)
        n_queries = n_queries or cfg.patch_n_queries
        test_queries = harmful_queries[:n_queries]

        pool = {k: v for k, v in candidate_neurons.items() if k not in exclude}
        tested = sorted(pool.values(), key=lambda n: abs(n.score), reverse=True)[:cfg.patch_max_candidates]
        if not tested:
            print("\n=== Activation patching: no candidates to test, nothing is confirmed ===")
            self.last_patch_results = []
            return {}
        controls = self._sample_controls(pool, exclude, cfg.patch_n_controls, rng)
        print(f"\n=== Activation patching: {len(tested)} candidates (of {len(pool)}), "
              f"{len(controls)} controls, {len(test_queries)} harmful queries, "
              f"{len(safe_queries)} held-out safe queries ===")

        all_neurons = tested + controls
        means = self._safe_means(all_neurons, safe_queries)

        k_refuse = refusal_token_ids(self.extractor.tokenizer)
        self.extractor.clear_hooks()
        baseline = [refusal_probability(self.extractor.get_last_logits(q), k_refuse)
                    for q in tqdm(test_queries, desc="Clean P_refuse")]
        print(f"Mean clean P_refuse = {np.mean(baseline):.4f} (|K_refuse| = {len(k_refuse)})")

        results = []
        for n in tqdm(all_neurons, desc="Patching"):
            value = means[n.to_tuple()]
            self.extractor.clear_hooks()
            self.extractor.register_patch_hook(n.layer, n.component, n.index, value)
            deltas = [baseline[i] - refusal_probability(self.extractor.get_last_logits(q), k_refuse)
                      for i, q in enumerate(test_queries)]
            self.extractor.clear_hooks()
            results.append({"neuron": n, "mean_delta_p": float(np.mean(deltas)),
                            "patch_value": value, "is_control": n.method == "control"})

        from scipy.stats import norm
        from statsmodels.stats.multitest import multipletests

        ctrl = np.array([r["mean_delta_p"] for r in results if r["is_control"]])
        mu, sigma = float(ctrl.mean()), float(max(ctrl.std(), 1e-12))
        cand = [r for r in results if not r["is_control"]]
        for r in results:
            r["z"] = (r["mean_delta_p"] - mu) / sigma
            r["p"] = float(norm.sf(r["z"]))
        _, p_adj, _, _ = multipletests([r["p"] for r in cand], alpha=cfg.fdr_alpha, method="fdr_bh")
        for r, pa in zip(cand, p_adj):
            r["p_adj"] = float(pa)

        confirmed = {}
        for r in cand:
            n = r["neuron"]
            probe_score = (n.metadata or {}).get("probe_score", n.score)
            signs_agree = np.sign(probe_score) == np.sign(r["mean_delta_p"])
            r["confirmed"] = bool(r["p_adj"] < cfg.fdr_alpha and r["mean_delta_p"] > 0 and signs_agree)
            if r["confirmed"]:
                confirmed[n.to_tuple()] = SafetyNeuron(
                    layer=n.layer, component=n.component, index=n.index,
                    score=r["mean_delta_p"], method="activation_patch",
                    metadata={"patch_value": r["patch_value"],
                              "mean_delta_p_refuse": r["mean_delta_p"],
                              "z_score_vs_control": r["z"], "p_value": r["p"],
                              "p_adj_fdr": r["p_adj"], "probe_z": n.score,
                              "control_mu": mu, "control_sigma": sigma})

        self.last_patch_results = [
            {"layer": r["neuron"].layer, "component": r["neuron"].component,
             "index": r["neuron"].index, "is_control": r["is_control"],
             "probe_z": r["neuron"].score, "patch_value": r["patch_value"],
             "mean_delta_p_refuse": r["mean_delta_p"], "z_score_vs_control": r["z"],
             "p_value": r["p"], "p_adj_fdr": r.get("p_adj"), "confirmed": r.get("confirmed", False)}
            for r in results]
        print(f"Control null: mu={mu:.6f}, sigma={sigma:.6f}; confirmed {len(confirmed)}/{len(tested)}")
        return confirmed


def save_neurons(neurons: Dict[Tuple, SafetyNeuron], path: Path):
    with open(path, "w") as f:
        json.dump([asdict(n) for n in neurons.values()], f, indent=2)
    print(f"Saved {len(neurons)} neurons to {path}")


def clamp_list(neurons: Dict[Tuple, SafetyNeuron]) -> List[list]:
    """The clamps of f_theta^(k) as a sorted list of [layer, component, index, value]."""
    return sorted([l, c, i, float(n.metadata["patch_value"])] for (l, c, i), n in neurons.items())


def repo_path(path) -> str:
    """A data path as recorded in config.json: relative to the repository root when inside it."""
    root = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def check_resume(output_dir: Path, start_round: int, settings: dict, round_file: str):
    """For --start-round k, before the model is loaded: output_dir/config.json was written with the
    same settings and rounds 1..k-1 left their confirmed-neuron files."""
    cfg_path = output_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"--start-round {start_round}: {cfg_path} not found (check --output-dir)")
    with open(cfg_path) as f:
        saved = json.load(f)
    diff = [k for k, v in settings.items() if saved.get(k) != v]
    if diff:
        sys.exit(f"--start-round {start_round}: {output_dir} was run with other settings: "
                         + ", ".join(f"{k} {saved.get(k)!r} (now {settings[k]!r})" for k in diff))
    missing = [str(output_dir / f"round_{r}" / round_file) for r in range(1, start_round)
               if not (output_dir / f"round_{r}" / round_file).exists()]
    if missing:
        sys.exit(f"--start-round {start_round}: missing " + ", ".join(missing))


def install_permanent_hooks(extractor, neurons: Dict[Tuple, SafetyNeuron]) -> int:
    """Clamp each confirmed neuron to its safe mean (metadata['patch_value'])."""
    for (layer, comp, idx), neuron in neurons.items():
        extractor.add_permanent_hook(layer, comp, idx, neuron.metadata["patch_value"])
    return len(neurons)


def read_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]):
    """Written to a temporary file and renamed into place, so an interrupted run leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def save_iterative_summary(output_dir: Path, all_confirmed: Dict[Tuple, SafetyNeuron],
                           round_summaries: List[dict], model_key: str, hf_repo: str):
    """all_confirmed_neurons.json and iterative_summary.json, rewritten after every round."""
    save_neurons(all_confirmed, output_dir / "all_confirmed_neurons.json")
    with open(output_dir / "iterative_summary.json", "w") as f:
        json.dump({"model_key": model_key, "hf_repo": hf_repo,
                   "total_neurons": len(all_confirmed), "rounds": round_summaries}, f, indent=2)


def load_neurons(path: Path) -> Dict[Tuple, SafetyNeuron]:
    neurons = {}
    with open(path) as f:
        for n in json.load(f):
            n.setdefault("score", 0.0)
            n.setdefault("method", "unknown")
            neuron = SafetyNeuron(**n)
            neurons[neuron.to_tuple()] = neuron
    return neurons
