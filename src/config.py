"""
Central configuration for VLM Safety Neuron Detection.

Scanned components, the model registry (configs/model_registry.yaml) and the detection
hyperparameters (read from configs/pipeline.yaml).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_YAML = REPO_ROOT / "configs" / "pipeline.yaml"
MODEL_REGISTRY_YAML = REPO_ROOT / "configs" / "model_registry.yaml"


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

TEXT_COMPONENTS = ("mlp_gate", "mlp_up", "self_attn")          # text scan: three components
VISUAL_COMPONENTS = ("mlp_gate", "mlp_up", "mlp_out", "self_attn")  # visual scan adds mlp_out

with open(MODEL_REGISTRY_YAML) as _f:
    MODELS: Dict[str, dict] = yaml.safe_load(_f)["models"]  # model key -> registry entry
MODEL_REGISTRY: Dict[str, str] = {key: m["hf_repo"] for key, m in MODELS.items()}


def load_pipeline_config() -> dict:
    with open(PIPELINE_YAML) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Detection hyperparameters (paper App. A, Tab. "Detection pipeline hyperparameters")
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    """Two-stage detection inside the iterative loop.

    Stage 1 (behavioral probing): l2 logistic probe per (layer, component) on a stratified
    80/20 split; per-layer z-scores of sign(w) * |Cohen's d|; candidates have |z| > 3.0.
    Stage 2 (activation patching): the top-500 candidates by |z| are clamped to their mean over
    held-out safe queries; dP_refuse on harmful-refused queries is compared with 300
    layer-stratified control neurons and BH-FDR corrected at alpha = 0.05.
    """

    probe_z_threshold: float = 3.0
    probe_C: float = 1.0
    probe_max_iter: int = 5000
    probe_holdout_ratio: float = 0.2

    patch_max_candidates: int = 500
    patch_n_queries: int = 250
    patch_n_controls: int = 300
    fdr_alpha: float = 0.05

    components: Tuple[str, ...] = VISUAL_COMPONENTS
    n_rounds: int = 3
    min_positives: int = 20      # stop when a round has fewer behavioral positives
    max_new_tokens: int = 512    # greedy decoding
    thinking_budget: int = 1024  # thinking models only
    seed: int = 42

    @classmethod
    def from_pipeline_yaml(cls, cfg: dict) -> "DetectionConfig":
        det = cfg["detection"]
        s1, s2, it = det["stage1"], det["stage2"], det.get("iterative", {})
        return cls(
            probe_z_threshold=s1["z_threshold"], probe_C=s1["probe_C"],
            probe_max_iter=s1["max_iter"], probe_holdout_ratio=s1["holdout_ratio"],
            patch_max_candidates=s2["max_candidates"], patch_n_queries=s2["n_queries"],
            patch_n_controls=s2["n_controls"], fdr_alpha=s2["fdr_alpha"],
            n_rounds=it.get("n_rounds", 3), min_positives=it.get("min_positives", 20),
            max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 512),
            thinking_budget=cfg.get("generation", {}).get("thinking_budget", 1024),
        )
