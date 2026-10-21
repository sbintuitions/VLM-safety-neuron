"""
(component x layer) heatmap of safety-neuron counts: one row per module type, one
column per decoder layer, viridis colormap, colorbar labeled "Neuron Count".

Inputs are the `all_confirmed_neurons.json` produced by the iterative detection
pipeline (entries with layer / component / index / score / method / metadata).

Usage:
    # Single run
    python src/plot_neuron_distribution.py \
        --neurons results/text/qwen25vl-7b/all_confirmed_neurons.json \
        --model qwen25vl-7b --title "Qwen2.5-VL-7B (text safety)" \
        --output figures/neuron_dist/qwen25vl-7b_text.pdf

    # Batch over all text and visual results
    python src/plot_neuron_distribution.py --batch
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import MODELS

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Liberation Sans", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Layer counts and display names from configs/model_registry.yaml.
MODEL_NUM_LAYERS = {key: m["num_layers"] for key, m in MODELS.items()}
DISPLAY_NAMES = {key: m["name"] for key, m in MODELS.items()}

# Row order for the heatmap (top to bottom).
COMPONENT_ORDER = ["mlp_gate", "mlp_up", "mlp_out", "self_attn"]
COMPONENT_LABELS = {
    "mlp_gate": "mlp.gate_proj",
    "mlp_up": "mlp.up_proj",
    "mlp_out": "mlp (block output)",
    "self_attn": "self_attn.o_proj",
}


def build_count_matrix(
    neurons: Iterable[dict],
    num_layers: int,
    components: list[str] = COMPONENT_ORDER,
) -> tuple[np.ndarray, list[str]]:
    """Return (rows x layers) integer count matrix and the row labels actually used.

    Rows are the components (in `components` order, then any others) that have at
    least one neuron; with no neurons at all, the first component is kept as one empty row.
    """
    counts = defaultdict(int)
    seen_components: set[str] = set()
    for n in neurons:
        c = n["component"]
        layer = int(n["layer"])
        counts[(c, layer)] += 1
        seen_components.add(c)

    # Canonical order first, then any non-canonical components that showed up.
    ordered = list(components) + [c for c in seen_components if c not in components]
    nonempty = [c for c in ordered if any(counts[(c, l)] > 0 for l in range(num_layers))]
    rows = nonempty if nonempty else ordered[:1]

    mat = np.zeros((len(rows), num_layers), dtype=int)
    for i, c in enumerate(rows):
        for layer in range(num_layers):
            mat[i, layer] = counts[(c, layer)]
    return mat, rows


def plot_heatmap(
    neurons: list[dict],
    num_layers: int,
    title: str,
    output: Path,
    annotate: bool = True,
    cmap: str = "viridis",
    fig_width: float | None = None,
) -> None:
    mat, rows = build_count_matrix(neurons, num_layers)
    n_rows = mat.shape[0]
    total = int(mat.sum())

    if fig_width is None:
        # ~0.18 inches per layer + label margin.
        fig_width = max(6.0, 0.18 * num_layers + 2.0)
    fig_height = 0.55 * n_rows + 1.6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    vmax = max(int(mat.max()), 1)
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    # X axis: layer index, 1-based.
    xticks = list(range(0, num_layers, max(1, num_layers // 12)))
    if (num_layers - 1) not in xticks:
        xticks.append(num_layers - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t + 1) for t in xticks])
    ax.set_xlabel("Layer Index")

    # Y axis: component labels.
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([COMPONENT_LABELS.get(c, c) for c in rows])

    # Annotate non-zero cells with their count when the figure isn't crowded.
    if annotate and num_layers <= 80 and vmax <= 50:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v == 0:
                    continue
                # Pick text color for contrast.
                color = "white" if v < 0.55 * vmax else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_title(f"{title}  (n={total})", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label("Neuron Count")
    # Integer ticks on the colorbar when range is small.
    if vmax <= 20:
        cbar.set_ticks(range(0, vmax + 1))

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}  (n={total}, rows={rows}, layers={num_layers})")


def load_neurons(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "neurons" in data:
        data = data["neurons"]
    if not isinstance(data, list):
        raise ValueError(f"Unexpected neuron JSON shape in {path}: {type(data)}")
    return data


# ---------------------------------------------------------------------------
# Batch mode: walk known result locations and plot each in one go.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def _runs_under(base: Path) -> list[tuple[str, Path]]:
    if not base.exists():
        return []
    return [(child.name, child / "all_confirmed_neurons.json") for child in sorted(base.iterdir())
            if (child / "all_confirmed_neurons.json").exists()]


def batch_text_runs() -> list[tuple[str, Path]]:
    """Text-neuron runs written to results/text/<model>/."""
    return _runs_under(RESULTS / "text")


def batch_visual_runs() -> list[tuple[str, Path]]:
    """Visual-neuron runs written to results/visual/<model>/."""
    return _runs_under(RESULTS / "visual")


def resolve_num_layers(model_key: str, fallback: list[dict]) -> int:
    if model_key in MODEL_NUM_LAYERS:
        return MODEL_NUM_LAYERS[model_key]
    if fallback:
        guess = max(int(n["layer"]) for n in fallback) + 1
        print(f"  [warn] unknown model '{model_key}', falling back to max+1 = {guess}")
        return guess
    raise KeyError(f"Unknown model '{model_key}' and no neurons to infer from")


def run_batch(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stim, runs in [("text", batch_text_runs()), ("visual", batch_visual_runs())]:
        for model_key, neuron_file in runs:
            neurons = load_neurons(neuron_file)
            if not neurons:
                print(f"[skip] {model_key} ({stim}): 0 neurons")
                continue
            num_layers = resolve_num_layers(model_key, neurons)
            disp = DISPLAY_NAMES.get(model_key, model_key)
            title = f"{disp} — {stim} safety neurons"
            output = out_dir / f"{stim}_{model_key}.pdf"
            plot_heatmap(neurons, num_layers, title, output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neurons", type=Path, help="Path to all_confirmed_neurons.json")
    p.add_argument("--model", type=str, help="Model key (for layer count lookup)")
    p.add_argument("--title", type=str, help="Override plot title")
    p.add_argument("--output", type=Path, help="Output figure path (.png/.pdf)")
    p.add_argument("--batch", action="store_true",
                   help="Plot every text and visual result into figures/neuron_dist/")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "figures/neuron_dist",
                   help="Output directory for batch mode")
    p.add_argument("--no-annotate", action="store_true",
                   help="Disable per-cell count annotations")
    args = p.parse_args()

    if args.batch:
        run_batch(args.out_dir)
        return

    if not args.neurons or not args.model or not args.output:
        p.error("--neurons, --model, --output are required (or use --batch)")
    neurons = load_neurons(args.neurons)
    num_layers = resolve_num_layers(args.model, neurons)
    title = args.title or f"{DISPLAY_NAMES.get(args.model, args.model)}"
    plot_heatmap(neurons, num_layers, title, args.output,
                 annotate=not args.no_annotate)


if __name__ == "__main__":
    main()
