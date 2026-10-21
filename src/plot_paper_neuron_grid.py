"""
Paper-ready 2 x N grid of (component x layer) safety-neuron distribution heatmaps.

Top row: text safety neurons; bottom row: visual safety neurons (each read from the
run's all_confirmed_neurons.json). Columns: VLMs of different sizes and families.

Usage:
    python src/plot_paper_neuron_grid.py
    # -> figures/paper/neuron_distribution_grid.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Font for model names and row headers; matplotlib uses the first installed face.
COMIC_FAMILY = ["Comic Sans MS", "Comic Neue", "Chalkboard SE", "DejaVu Sans"]

# Per-model layer counts, display names and the neuron loader from the single-model script.
from plot_neuron_distribution import DISPLAY_NAMES, MODEL_NUM_LAYERS, load_neurons

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Liberation Sans", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#5C4A3A",
    "axes.labelcolor": "#5C4A3A",
    "xtick.color": "#5C4A3A",
    "ytick.color": "#5C4A3A",
    "text.color": "#3D2E22",
})

REPO_ROOT = Path(__file__).resolve().parents[1]
# Expects text runs in results/text/<model>/ and visual runs in results/visual/<model>/
RESULTS = REPO_ROOT / "results"

# Order matters: this is the left-to-right column order in the figure.
SELECTED = [
    {"key": key, "display": DISPLAY_NAMES[key],
     "text_file": RESULTS / "text" / key / "all_confirmed_neurons.json",
     # an empty neuron list draws an empty panel
     "visual_file": RESULTS / "visual" / key / "all_confirmed_neurons.json"}
    for key in ("qwen3vl-2b", "qwen25vl-7b", "qwen3vl-8b", "internvl-8b", "gemma4-31b")
]

# Components shown on each row: the text scan covers three components, the visual
# scan adds mlp_out (Sec. 5.3).
DISPLAY_COMPONENTS_TEXT   = ["mlp_gate", "mlp_up", "self_attn"]
DISPLAY_COMPONENTS_VISUAL = ["mlp_gate", "mlp_up", "mlp_out", "self_attn"]

DISPLAY_LABELS = {c: c for c in DISPLAY_COMPONENTS_VISUAL}

# Custom cartoon-style sequential colormap: cream -> peach -> coral -> dusty red.
CARTOON_PEACH = LinearSegmentedColormap.from_list(
    "cartoon_peach",
    [
        (0.00, "#FFF8F0"),  # cream
        (0.20, "#FFE3CC"),  # warm cream
        (0.45, "#FFC09F"),  # peach
        (0.70, "#FF8B94"),  # coral pink
        (1.00, "#C44D58"),  # dusty red
    ],
    N=256,
)

# Bigger fonts since the figure is denser than the single-model versions.
TITLE_FONT = 11
LABEL_FONT = 9
TICK_FONT = 8
ANNOT_FONT = 7


def num_layers_for(entry: dict) -> int:
    return MODEL_NUM_LAYERS[entry["key"]]


def components_for(stimulus: str) -> list[str]:
    return DISPLAY_COMPONENTS_TEXT if stimulus == "text" else DISPLAY_COMPONENTS_VISUAL


def load_panel(entry: dict, stimulus: str) -> tuple[np.ndarray, list[str], int, int]:
    """Return (count matrix [n_components, n_layers], rows, n_layers, total).

    Component set is `DISPLAY_COMPONENTS_TEXT` for stimulus="text" and
    `DISPLAY_COMPONENTS_VISUAL` for stimulus="visual". The visual row carries
    one extra component (`mlp_out`), which only the visual scan covers.
    """
    path = entry[f"{stimulus}_file"]
    if not path.exists():
        print(f"WARNING: {path} not found; drawing an empty {stimulus} panel for {entry['display']}")
    neurons = load_neurons(path) if path.exists() else []
    n_layers = num_layers_for(entry)
    components = components_for(stimulus)

    counts = np.zeros((len(components), n_layers), dtype=int)
    total_in_scope = 0
    for n in neurons:
        c = n["component"]
        if c not in components:
            continue
        counts[components.index(c), int(n["layer"])] += 1
        total_in_scope += 1
    return counts, [DISPLAY_LABELS.get(c, c) for c in components], n_layers, total_in_scope


def draw_heatmap(ax, mat: np.ndarray, n_layers: int, vmax: int, row_labels: list[str],
                 show_yticks: bool, annotate: bool = False, annot_min: int = 1,
                 aspect: str = "auto"):
    im = ax.imshow(
        mat,
        aspect=aspect,
        cmap=CARTOON_PEACH,
        vmin=0,
        vmax=max(vmax, 1),
        interpolation="nearest",
    )

    # X axis (layer index, 1-based).
    step = max(1, n_layers // 6)
    xticks = list(range(0, n_layers, step))
    if (n_layers - 1) not in xticks:
        xticks.append(n_layers - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t + 1) for t in xticks], fontsize=TICK_FONT)
    ax.set_xlabel("Layer", fontsize=LABEL_FONT)

    ax.set_yticks(range(len(row_labels)))
    if show_yticks:
        ax.set_yticklabels(row_labels, fontsize=TICK_FONT)
    else:
        ax.set_yticklabels([""] * len(row_labels))

    # Cartoon-y separator lines between cells.
    ax.set_xticks(np.arange(-0.5, n_layers, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4, alpha=0.6)
    ax.tick_params(which="minor", length=0)

    # Annotate cells (only when not too crowded); cells below annot_min are left blank.
    if annotate and n_layers <= 80 and vmax <= 30:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v < annot_min:
                    continue
                color = "#3D2E22" if v < 0.6 * max(vmax, 1) else "#FFF8F0"
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        fontsize=ANNOT_FONT, color=color, fontweight="bold")
    return im


def main():
    out_dir = REPO_ROOT / "figures/paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cols = len(SELECTED)
    # Load all panels first.
    panels = {"text": [], "visual": []}
    totals = {"text": [], "visual": []}
    for entry in SELECTED:
        for stim in ("text", "visual"):
            mat, rows, n_layers, total = load_panel(entry, stim)
            panels[stim].append((mat, rows, n_layers, total, entry))
            totals[stim].append(total)
    # One color scale shared by both rows.
    shared_vmax = max(
        max((p[0].max() for p in panels["text"]), default=1),
        max((p[0].max() for p in panels["visual"]), default=1),
    )
    row_vmax = {"text": shared_vmax, "visual": shared_vmax}

    # Two rows of heatmaps plus a colorbar column, ~13 in wide, equal column widths.
    # Both rows have the same height, so the visual row (four components) has slightly
    # shorter cells than the text row (three).
    col_width = 2.35
    cbar_width = 0.30
    fig_width = n_cols * col_width + cbar_width + 1.1
    fig_height = 5.4

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="#FEFDFA")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=n_cols + 1,
        width_ratios=[col_width] * n_cols + [cbar_width],
        height_ratios=[1, 1],
        hspace=0.42,
        wspace=0.18,
        left=0.105, right=0.95, top=0.93, bottom=0.10,
    )

    last_im = {"text": None, "visual": None}
    first_col_axes = {}  # row index -> ax, used later for label alignment
    for col, entry in enumerate(SELECTED):
        for row_idx, stim in enumerate(("text", "visual")):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.set_facecolor("#FFFFFF")
            if col == 0:
                first_col_axes[row_idx] = ax
            mat, row_labels, n_layers, total, _ = panels[stim][col]
            # No per-cell annotations: color encodes the counts.
            # Both rows share the same canvas size (height_ratios=[1,1]); cells
            # stretch to fill, so visual cells (4 components) are slightly
            # shorter than text cells (3 components).
            im = draw_heatmap(
                ax, mat, n_layers, row_vmax[stim], row_labels,
                show_yticks=(col == 0),
                annotate=False,
            )
            last_im[stim] = im

            # Per-panel title: model name on top row; total count under each panel.
            if row_idx == 0:
                ax.set_title(entry["display"], fontsize=TITLE_FONT + 1, pad=8,
                             fontweight="bold", color="#3D2E22",
                             fontfamily=COMIC_FAMILY)
            n_str = f"n = {total}"
            ax.text(0.98, 0.05, n_str,
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=TICK_FONT,
                    color="#3D2E22",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#FFF8F0", edgecolor="#E0BBA0",
                              linewidth=0.6, alpha=0.92))

    # Single shared colorbar spanning both rows (both rows now share vmax).
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im["text"], cax=cax)
    cbar.set_label("Neuron count", fontsize=LABEL_FONT, color="#3D2E22")
    cbar.ax.tick_params(labelsize=TICK_FONT, colors="#5C4A3A")
    if shared_vmax <= 15:
        cbar.set_ticks(range(0, int(shared_vmax) + 1))
    for spine in cax.spines.values():
        spine.set_color("#E0BBA0")

    # Row labels on the far left — vertically centered against each row's
    # heatmap bbox so they stay aligned even if the gridspec margins change.
    fig.canvas.draw()  # force axes positions to materialise before bbox query
    top_pos = first_col_axes[0].get_position()
    bot_pos = first_col_axes[1].get_position()
    top_y = top_pos.y0 + top_pos.height / 2
    bot_y = bot_pos.y0 + bot_pos.height / 2
    fig.text(0.018, top_y, "Text safety\nneurons",
             ha="center", va="center",
             fontsize=TITLE_FONT + 2, fontweight="bold",
             color="#3D2E22", rotation=90,
             fontfamily=COMIC_FAMILY)
    fig.text(0.018, bot_y, "Visual safety\nneurons",
             ha="center", va="center",
             fontsize=TITLE_FONT + 2, fontweight="bold",
             color="#3D2E22", rotation=90,
             fontfamily=COMIC_FAMILY)
    # No suptitle: caption carries the figure description.

    for ext in ("pdf", "png"):
        out = out_dir / f"neuron_distribution_grid.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[saved] {out}")
    plt.close(fig)

    print("\nPer-model totals:")
    for col, entry in enumerate(SELECTED):
        n_text, n_vis = totals['text'][col], totals['visual'][col]
        ratio = f"{n_text / n_vis:.1f}x" if n_vis else "n/a"
        print(f"  {entry['display']:<18}  text={n_text:>3}   visual={n_vis:>3}   ratio={ratio}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
