"""Visualization module — generate all figures for the language production pipeline.

Provides 9 plot functions plus a convenience wrapper:
    - plot_condition_similarities(): SP-DR, DP-SC, DC curves with CI bands
    - plot_rsa_curves(): product/register/within-category RSA per layer
    - plot_probe_curves(): accuracy/selectivity per layer with chance lines
    - plot_zone_comparison(): bar charts per task per zone
    - plot_decomposition(): attention/MLP/residual RSA curves
    - plot_memorization_control(): real vs. fictional RSA overlay
    - plot_rdm_heatmaps(): RDM heatmaps at early/middle/late layers
    - plot_register_confusion(): confusion matrices at selected layers
    - plot_quantization_control(): FP16 vs. quantized RSA overlay
    - generate_all_figures(): loads all data and calls every plot function

All plots are saved as both PNG (300 DPI) and PDF to data/figures/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style & palette constants
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (Wong, 2011 — Nature Methods)
# Order: blue, orange, green, red, purple, brown, pink, gray
CB_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red-orange
    "#CC79A7",  # pink
    "#56B4E9",  # light blue
    "#F0E442",  # yellow
    "#999999",  # gray
]

ZONE_COLORS = {
    "early": "#56B4E9",
    "protocol": "#E69F00",
    "late": "#D55E00",
    "output": "#CC79A7",
}

ZONE_ORDER = ["early", "protocol", "late", "output"]

# Chance levels for probe tasks
CHANCE_LEVELS = {
    "product": 1.0 / 40,
    "category": 1.0 / 8,
    "register": 1.0 / 5,
}

FIGURES_DIR = Path("data/figures")


def _setup_style() -> None:
    """Apply a consistent paper-quality style to matplotlib."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        # Fallback if seaborn style is unavailable
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            pass  # Use default matplotlib style
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _ensure_figures_dir() -> Path:
    """Create the figures directory if it does not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def _save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure as both PNG (300 DPI) and PDF."""
    out_dir = _ensure_figures_dir()
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info("Saved %s (.png + .pdf)", name)


def _get_zone_boundaries(config: dict) -> dict[str, tuple[int, int]] | None:
    """Compute zone boundary layer indices from config.

    Returns a dict mapping zone name to (start_layer, end_layer) inclusive,
    or None if config lacks the needed keys.
    """
    zone_pct = config.get("h3_layer_zone_pct")
    n_layers = config.get("_n_layers")
    if zone_pct is None or n_layers is None:
        return None

    pct_start, pct_end = zone_pct
    last_idx = n_layers - 1
    early_end = int(round(last_idx * pct_start / 100.0))
    protocol_end = int(round(last_idx * pct_end / 100.0))

    return {
        "early": (0, early_end),
        "protocol": (early_end, protocol_end),
        "late": (protocol_end, last_idx),
        "output": (last_idx, last_idx),
    }


def _draw_zone_boundaries(
    ax: plt.Axes,
    config: dict,
    label_y: str = "top",
) -> None:
    """Draw vertical dashed lines at zone boundaries on a per-layer axis.

    Parameters
    ----------
    ax : matplotlib Axes
    config : dict
        Must include ``h3_layer_zone_pct`` and ``_n_layers`` for zone info.
    label_y : str
        Where to place zone labels: "top" or "bottom".
    """
    zones = _get_zone_boundaries(config)
    if zones is None:
        return

    drawn: set[int] = set()
    for zone_name in ZONE_ORDER:
        if zone_name not in zones:
            continue
        start, end = zones[zone_name]
        for boundary in (start, end):
            if boundary not in drawn and boundary > 0:
                ax.axvline(
                    boundary,
                    color="#888888",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=0,
                )
                drawn.add(boundary)

    # Label zones at the midpoint
    y_pos = 0.97 if label_y == "top" else 0.03
    va = "top" if label_y == "top" else "bottom"
    for zone_name in ZONE_ORDER:
        if zone_name not in zones:
            continue
        start, end = zones[zone_name]
        mid = (start + end) / 2.0
        ax.text(
            mid,
            y_pos,
            zone_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va=va,
            fontsize=7,
            color="#666666",
            fontstyle="italic",
        )


def _significance_marker(p: float) -> str:
    """Return a significance marker string for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Plot 1: Condition Similarities
# ---------------------------------------------------------------------------


def plot_condition_similarities(
    condition_data: dict[str, Any],
    config: dict,
) -> plt.Figure:
    """Plot SP-DR, DP-SC, and DC similarity curves per layer with CI bands.

    Parameters
    ----------
    condition_data : dict
        Keys ``"SP-DR"``, ``"DP-SC"``, ``"DC"`` each mapping to a dict with:
          - ``"mean"``: list[float] — per-layer mean cosine similarity
          - ``"ci_lower"``: list[float] — lower 95% CI bound (optional)
          - ``"ci_upper"``: list[float] — upper 95% CI bound (optional)
        Alternatively, each key can map directly to a list[float] of means
        (no CI bands drawn in that case).
    config : dict
        Pipeline configuration.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    condition_labels = ["SP-DR", "DP-SC", "DC"]
    condition_descriptions = {
        "SP-DR": "Same Product, Diff Register",
        "DP-SC": "Diff Product, Same Category",
        "DC": "Different Category",
    }
    colors = [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[3]]

    for i, cond in enumerate(condition_labels):
        if cond not in condition_data:
            continue

        entry = condition_data[cond]

        # Handle both dict-with-ci and plain-list formats
        if isinstance(entry, dict):
            means = np.array(entry.get("mean", entry.get("means", [])))
            ci_lo = entry.get("ci_lower")
            ci_hi = entry.get("ci_upper")
        else:
            means = np.array(entry)
            ci_lo = None
            ci_hi = None

        if len(means) == 0:
            continue

        layers = np.arange(len(means))
        desc = condition_descriptions.get(cond, cond)
        ax.plot(layers, means, color=colors[i], label=f"{cond} ({desc})", linewidth=1.5)

        if ci_lo is not None and ci_hi is not None:
            ax.fill_between(
                layers,
                np.array(ci_lo),
                np.array(ci_hi),
                color=colors[i],
                alpha=0.15,
            )

    # Inject n_layers into config for zone drawing if we can infer it
    config_with_layers = dict(config)
    for cond in condition_labels:
        if cond in condition_data:
            entry = condition_data[cond]
            n = len(entry.get("mean", entry) if isinstance(entry, dict) else entry)
            if n > 0:
                config_with_layers["_n_layers"] = n
                break

    _draw_zone_boundaries(ax, config_with_layers)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Condition Similarities Across Layers")
    ax.legend(loc="best", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()

    _save_figure(fig, "condition_similarities")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: RSA Curves
# ---------------------------------------------------------------------------


def plot_rsa_curves(
    rsa_data: dict[str, np.ndarray | list[float]],
    pvalues: dict[str, Any] | None,
    config: dict,
) -> plt.Figure:
    """Plot RSA correlation per layer for each model RDM, with significance markers.

    Parameters
    ----------
    rsa_data : dict
        Keys like ``"product_identity"``, ``"register_identity"``,
        ``"within_category"`` mapping to per-layer RSA r values (1-D arrays).
    pvalues : dict or None
        If provided, expected keys matching ``rsa_data`` keys, each mapping
        to a dict of ``{layer_idx_str: p_value}`` or a list of per-layer
        p-values.  Used for significance markers.
    config : dict
        Pipeline configuration.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    label_map = {
        "product_identity": "Product Identity",
        "register_identity": "Register Identity",
        "within_category": "Within-Category",
    }
    colors = {
        "product_identity": CB_PALETTE[0],
        "register_identity": CB_PALETTE[1],
        "within_category": CB_PALETTE[2],
    }

    config_with_layers = dict(config)
    color_idx = 0

    for key, values in rsa_data.items():
        values = np.asarray(values)
        if values.size == 0:
            continue

        layers = np.arange(len(values))
        config_with_layers["_n_layers"] = len(values)

        color = colors.get(key, CB_PALETTE[color_idx % len(CB_PALETTE)])
        label = label_map.get(key, key)
        ax.plot(layers, values, color=color, label=label, linewidth=1.5, marker="o",
                markersize=3)

        # Add significance markers
        if pvalues is not None and key in pvalues:
            p_data = pvalues[key]
            for li, layer in enumerate(layers):
                # p_data could be dict {str(layer): p} or list
                if isinstance(p_data, dict):
                    p = p_data.get(str(layer), p_data.get(layer, 1.0))
                elif isinstance(p_data, (list, np.ndarray)) and li < len(p_data):
                    p = p_data[li]
                else:
                    continue

                marker_text = _significance_marker(float(p))
                if marker_text:
                    ax.annotate(
                        marker_text,
                        (layer, values[li]),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=7,
                        color=color,
                    )

        color_idx += 1

    ax.axhline(0, color="#888888", linestyle="-", linewidth=0.5, alpha=0.5)
    _draw_zone_boundaries(ax, config_with_layers)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman r")
    ax.set_title("RSA Correlation Across Layers")
    ax.legend(loc="best", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()

    _save_figure(fig, "rsa_curves")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Probe Curves
# ---------------------------------------------------------------------------


def plot_probe_curves(
    probe_results: dict[str, Any],
    config: dict,
) -> plt.Figure:
    """Plot probe accuracy and selectivity per layer, with chance lines and CIs.

    Parameters
    ----------
    probe_results : dict
        Nested structure.  Expected formats:

        Format A (from train_probes_all_layers, keyed by anisotropy method):
            ``{method: {task: {layer_idx_str: {"macro_f1": float, "bootstrap_ci_95": [lo, hi]}}}``

        Format B (flat, one method already selected):
            ``{task: {layer_idx_str: {"macro_f1": float, ...}}}``

        If selectivity data is present under ``"selectivity"``, it will be
        plotted as dashed lines.
    config : dict
        Pipeline configuration.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()

    # Determine format: if first value is a dict whose values are dicts of
    # layer results, it is Format A; pick the first method.
    tasks_data = _extract_probe_tasks(probe_results)
    selectivity_data = probe_results.get("selectivity", {})

    if not tasks_data:
        logger.warning("No probe task data found — returning empty figure")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No probe data available", ha="center", va="center",
                transform=ax.transAxes)
        _save_figure(fig, "probe_curves")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_acc, ax_sel = axes

    task_colors = {
        "product": CB_PALETTE[0],
        "category": CB_PALETTE[1],
        "register": CB_PALETTE[2],
    }

    config_with_layers = dict(config)

    # ---- Accuracy subplot ----
    for task_name, layer_results in tasks_data.items():
        layer_indices = sorted(int(k) for k in layer_results.keys())
        if not layer_indices:
            continue

        config_with_layers["_n_layers"] = max(layer_indices) + 1

        f1_vals = []
        ci_lo = []
        ci_hi = []
        for li in layer_indices:
            res = layer_results[str(li)] if str(li) in layer_results else layer_results.get(li, {})
            f1_vals.append(res.get("macro_f1", 0.0))
            ci = res.get("bootstrap_ci_95", [None, None])
            ci_lo.append(ci[0])
            ci_hi.append(ci[1])

        f1_vals = np.array(f1_vals)
        color = task_colors.get(task_name, CB_PALETTE[3])
        ax_acc.plot(layer_indices, f1_vals, color=color, label=task_name.capitalize(),
                    linewidth=1.5)

        # CI bands if available
        if ci_lo[0] is not None and ci_hi[0] is not None:
            ax_acc.fill_between(
                layer_indices,
                np.array(ci_lo, dtype=float),
                np.array(ci_hi, dtype=float),
                color=color,
                alpha=0.12,
            )

        # Chance line
        chance = CHANCE_LEVELS.get(task_name)
        if chance is not None:
            ax_acc.axhline(
                chance,
                color=color,
                linestyle=":",
                linewidth=0.8,
                alpha=0.6,
            )
            ax_acc.text(
                max(layer_indices) + 0.5,
                chance,
                f"1/{int(1/chance)}",
                color=color,
                fontsize=7,
                va="center",
            )

    _draw_zone_boundaries(ax_acc, config_with_layers)
    ax_acc.set_xlabel("Layer")
    ax_acc.set_ylabel("Macro F1")
    ax_acc.set_title("Probe Accuracy Across Layers")
    ax_acc.legend(loc="best", framealpha=0.9)
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ---- Selectivity subplot ----
    has_selectivity = False
    if isinstance(selectivity_data, dict) and selectivity_data:
        for task_name, sel_layers in selectivity_data.items():
            if not sel_layers:
                continue
            layer_indices = sorted(int(k) for k in sel_layers.keys())
            sel_vals = [float(sel_layers[str(li)] if str(li) in sel_layers else sel_layers.get(li, 0.0))
                        for li in layer_indices]
            color = task_colors.get(task_name, CB_PALETTE[3])
            ax_sel.plot(layer_indices, sel_vals, color=color, label=task_name.capitalize(),
                        linewidth=1.5, linestyle="--")
            has_selectivity = True

    if has_selectivity:
        _draw_zone_boundaries(ax_sel, config_with_layers)
        ax_sel.set_xlabel("Layer")
        ax_sel.set_ylabel("Selectivity (Real F1 - Control F1)")
        ax_sel.set_title("Probe Selectivity Across Layers")
        ax_sel.legend(loc="best", framealpha=0.9)
        ax_sel.axhline(0, color="#888888", linestyle="-", linewidth=0.5, alpha=0.5)
        ax_sel.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    else:
        ax_sel.text(0.5, 0.5, "Selectivity data\nnot available", ha="center", va="center",
                    transform=ax_sel.transAxes, fontsize=11, color="#999999")
        ax_sel.set_title("Probe Selectivity Across Layers")

    fig.tight_layout()
    _save_figure(fig, "probe_curves")
    return fig


def _extract_probe_tasks(
    probe_results: dict[str, Any],
) -> dict[str, dict]:
    """Extract task-level probe results from potentially nested formats.

    Returns a dict: {task_name: {layer_idx_or_str: result_dict}}.
    """
    if not probe_results:
        return {}

    # Check if top-level keys are anisotropy methods (Format A)
    first_key = next(iter(probe_results))
    first_val = probe_results[first_key]

    # Skip non-dict entries (like "selectivity")
    if not isinstance(first_val, dict):
        return {}

    # Format A: {method: {task: {layer: result}}}
    # Check if the second level contains task names
    if first_key in ("none", "mean_centering", "whitening"):
        # Pick the first method
        return first_val

    # Check if first_val values are dicts with "macro_f1" (Format B direct)
    sample_inner = next(iter(first_val.values()), None)
    if isinstance(sample_inner, dict) and "macro_f1" in sample_inner:
        # Format B: {task: {layer: result}}
        return probe_results

    # Might be {task: {layer: result}} but layer keys are ints
    if isinstance(sample_inner, dict):
        inner_sample = next(iter(sample_inner.values()), None)
        if isinstance(inner_sample, dict) and "macro_f1" in inner_sample:
            return probe_results

    return {}


# ---------------------------------------------------------------------------
# Plot 4: Zone Comparison
# ---------------------------------------------------------------------------


def plot_zone_comparison(
    zone_results: dict[str, Any],
    config: dict,
) -> plt.Figure:
    """Plot bar charts comparing probe performance across zones for each task.

    Parameters
    ----------
    zone_results : dict
        ``{zone_name: {task_name: {"macro_f1": float, "bootstrap_ci_95": [lo, hi]}}}``
    config : dict
        Pipeline configuration.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()

    # Determine tasks and zones present
    tasks: set[str] = set()
    zones_present: list[str] = []
    for zone_name in ZONE_ORDER:
        if zone_name in zone_results:
            zones_present.append(zone_name)
            tasks.update(zone_results[zone_name].keys())

    tasks_sorted = sorted(tasks)

    if not tasks_sorted or not zones_present:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No zone data available", ha="center", va="center",
                transform=ax.transAxes)
        _save_figure(fig, "zone_comparison")
        return fig

    n_tasks = len(tasks_sorted)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5), squeeze=False)
    axes = axes[0]

    bar_width = 0.6
    x = np.arange(len(zones_present))

    task_colors_map = {
        "product": CB_PALETTE[0],
        "category": CB_PALETTE[1],
        "register": CB_PALETTE[2],
    }

    for ti, task_name in enumerate(tasks_sorted):
        ax = axes[ti]
        f1_vals = []
        ci_errors = []

        for zone_name in zones_present:
            res = zone_results.get(zone_name, {}).get(task_name, {})
            f1 = res.get("macro_f1", 0.0)
            f1_vals.append(f1)
            ci = res.get("bootstrap_ci_95", [f1, f1])
            ci_errors.append([f1 - ci[0], ci[1] - f1])

        f1_vals = np.array(f1_vals)
        ci_errors = np.array(ci_errors).T  # shape (2, n_zones)

        bar_colors = [ZONE_COLORS.get(z, "#999999") for z in zones_present]
        ax.bar(x, f1_vals, width=bar_width, color=bar_colors,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(x, f1_vals, yerr=ci_errors, fmt="none", ecolor="#333333",
                     capsize=3, linewidth=1)

        # Chance line
        chance = CHANCE_LEVELS.get(task_name)
        if chance is not None:
            ax.axhline(chance, color="#888888", linestyle=":", linewidth=0.8)
            ax.text(len(x) - 0.5, chance + 0.01, f"chance (1/{int(1/chance)})",
                    fontsize=7, color="#888888")

        ax.set_xticks(x)
        ax.set_xticklabels([z.capitalize() for z in zones_present])
        ax.set_ylabel("Macro F1")
        ax.set_title(f"{task_name.capitalize()} Probe")
        ax.set_ylim(0, 1.05)

    fig.suptitle("Zone-Level Probe Performance", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_figure(fig, "zone_comparison")
    return fig


# ---------------------------------------------------------------------------
# Plot 5: Decomposition
# ---------------------------------------------------------------------------


def plot_decomposition(
    component_rsa: dict[str, np.ndarray | list[float]],
    config: dict,
) -> plt.Figure:
    """Plot RSA curves for attention, MLP, and residual stream components.

    Parameters
    ----------
    component_rsa : dict
        Keys like ``"attention"``, ``"mlp"``, ``"residual"`` mapping to
        per-layer RSA correlation arrays.
    config : dict
        Pipeline configuration.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    component_styles = {
        "attention": {"color": CB_PALETTE[0], "linestyle": "-", "label": "Attention"},
        "mlp": {"color": CB_PALETTE[1], "linestyle": "-", "label": "MLP"},
        "residual": {"color": CB_PALETTE[2], "linestyle": "--", "label": "Residual Stream"},
    }

    config_with_layers = dict(config)
    color_idx = 0

    for comp_name, values in component_rsa.items():
        values = np.asarray(values)
        if values.size == 0:
            continue

        layers = np.arange(len(values))
        config_with_layers["_n_layers"] = len(values)

        style = component_styles.get(comp_name, {
            "color": CB_PALETTE[color_idx % len(CB_PALETTE)],
            "linestyle": "-",
            "label": comp_name.capitalize(),
        })

        ax.plot(layers, values, linewidth=1.5, marker="o", markersize=2, **style)
        color_idx += 1

    ax.axhline(0, color="#888888", linestyle="-", linewidth=0.5, alpha=0.5)
    _draw_zone_boundaries(ax, config_with_layers)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman r")
    ax.set_title("Component-Level RSA Decomposition")
    ax.legend(loc="best", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()

    _save_figure(fig, "decomposition")
    return fig


# ---------------------------------------------------------------------------
# Plot 6: Memorization Control
# ---------------------------------------------------------------------------


def plot_memorization_control(
    real_rsa: np.ndarray | list[float],
    fictional_rsa: np.ndarray | list[float],
    config: dict,
) -> plt.Figure:
    """Overlay real-product and fictional-product RSA curves.

    Parameters
    ----------
    real_rsa : array-like
        Per-layer RSA for real products.
    fictional_rsa : array-like
        Per-layer RSA for fictional products.
    config : dict
        Pipeline configuration.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    real_rsa = np.asarray(real_rsa)
    fictional_rsa = np.asarray(fictional_rsa)

    config_with_layers = dict(config)

    if real_rsa.size > 0:
        layers = np.arange(len(real_rsa))
        config_with_layers["_n_layers"] = len(real_rsa)
        ax.plot(layers, real_rsa, color=CB_PALETTE[0], label="Real Products",
                linewidth=1.5)

    if fictional_rsa.size > 0:
        layers = np.arange(len(fictional_rsa))
        config_with_layers["_n_layers"] = max(
            config_with_layers.get("_n_layers", 0), len(fictional_rsa)
        )
        ax.plot(layers, fictional_rsa, color=CB_PALETTE[3], label="Fictional Products",
                linewidth=1.5, linestyle="--")

    ax.axhline(0, color="#888888", linestyle="-", linewidth=0.5, alpha=0.5)
    _draw_zone_boundaries(ax, config_with_layers)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman r (Product Identity RSA)")
    ax.set_title("Memorization Control: Real vs. Fictional Products")
    ax.legend(loc="best", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()

    _save_figure(fig, "memorization_control")
    return fig


# ---------------------------------------------------------------------------
# Plot 7: RDM Heatmaps
# ---------------------------------------------------------------------------


def plot_rdm_heatmaps(
    rdms: dict[int, np.ndarray],
    layer_indices: list[int],
    config: dict,
) -> plt.Figure:
    """Plot RDM heatmaps at selected layers (e.g., early, middle, late).

    Parameters
    ----------
    rdms : dict
        Mapping from layer index to (N, N) RDM.
    layer_indices : list[int]
        Which layers to plot (typically 3: early, middle, late).
    config : dict
        Pipeline configuration.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()

    # Filter to available layers
    available = [li for li in layer_indices if li in rdms]
    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No RDM data available\nfor selected layers",
                ha="center", va="center", transform=ax.transAxes)
        _save_figure(fig, "rdm_heatmaps")
        return fig

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    for i, li in enumerate(available):
        ax = axes[i]
        rdm = rdms[li]
        im = ax.imshow(rdm, cmap="viridis", aspect="auto", interpolation="nearest")
        ax.set_title(f"Layer {li}")
        ax.set_xlabel("Stimulus")
        ax.set_ylabel("Stimulus")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Dissimilarity")

    fig.suptitle("Representational Dissimilarity Matrices", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_figure(fig, "rdm_heatmaps")
    return fig


# ---------------------------------------------------------------------------
# Plot 8: Register Confusion Matrices
# ---------------------------------------------------------------------------


def plot_register_confusion(
    confusion_data: dict[int, np.ndarray],
    layer_indices: list[int],
    config: dict,
) -> plt.Figure:
    """Plot register confusion matrices at selected layers.

    Parameters
    ----------
    confusion_data : dict
        Mapping from layer index to (n_registers, n_registers) confusion matrix.
    layer_indices : list[int]
        Which layers to plot.
    config : dict
        Pipeline configuration.  ``config["registers"]`` used for axis labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()

    available = [li for li in layer_indices if li in confusion_data]
    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No confusion data available\nfor selected layers",
                ha="center", va="center", transform=ax.transAxes)
        _save_figure(fig, "register_confusion")
        return fig

    registers = config.get("registers", [])

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    for i, li in enumerate(available):
        ax = axes[i]
        cm = confusion_data[li]

        # Normalize rows to get proportions
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
        cm_norm = cm / row_sums

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        # Annotate cells
        n_classes = cm.shape[0]
        for row in range(n_classes):
            for col in range(n_classes):
                val = cm_norm[row, col]
                text_color = "white" if val > 0.5 else "black"
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)

        ax.set_title(f"Layer {li}")

        if registers and len(registers) == n_classes:
            short_labels = [r[:6] for r in registers]
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(n_classes))
            ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")

    fig.suptitle("Register Confusion Matrices", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_figure(fig, "register_confusion")
    return fig


# ---------------------------------------------------------------------------
# Plot 9: Quantization Control
# ---------------------------------------------------------------------------


def plot_quantization_control(
    fp16_rsa: np.ndarray | list[float],
    quant_rsa: np.ndarray | list[float],
    config: dict,
) -> plt.Figure:
    """Overlay FP16 and quantized RSA curves to assess quantization impact.

    Parameters
    ----------
    fp16_rsa : array-like
        Per-layer RSA for the FP16 (unquantized) model.
    quant_rsa : array-like
        Per-layer RSA for the quantized model.
    config : dict
        Pipeline configuration.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_overlay, ax_diff = axes

    fp16_rsa = np.asarray(fp16_rsa)
    quant_rsa = np.asarray(quant_rsa)

    config_with_layers = dict(config)

    # --- Overlay plot ---
    if fp16_rsa.size > 0:
        layers = np.arange(len(fp16_rsa))
        config_with_layers["_n_layers"] = len(fp16_rsa)
        ax_overlay.plot(layers, fp16_rsa, color=CB_PALETTE[0], label="FP16",
                        linewidth=1.5)

    if quant_rsa.size > 0:
        layers = np.arange(len(quant_rsa))
        config_with_layers["_n_layers"] = max(
            config_with_layers.get("_n_layers", 0), len(quant_rsa)
        )
        ax_overlay.plot(layers, quant_rsa, color=CB_PALETTE[3], label="Quantized (Int4)",
                        linewidth=1.5, linestyle="--")

    ax_overlay.axhline(0, color="#888888", linestyle="-", linewidth=0.5, alpha=0.5)
    _draw_zone_boundaries(ax_overlay, config_with_layers)
    ax_overlay.set_xlabel("Layer")
    ax_overlay.set_ylabel("Spearman r")
    ax_overlay.set_title("RSA: FP16 vs. Quantized")
    ax_overlay.legend(loc="best", framealpha=0.9)
    ax_overlay.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- Difference plot ---
    if fp16_rsa.size > 0 and quant_rsa.size > 0:
        min_len = min(len(fp16_rsa), len(quant_rsa))
        diff = fp16_rsa[:min_len] - quant_rsa[:min_len]
        layers = np.arange(min_len)

        ax_diff.plot(layers, diff, color=CB_PALETTE[4], linewidth=1.5)
        ax_diff.fill_between(layers, diff, 0, color=CB_PALETTE[4], alpha=0.15)
        ax_diff.axhline(0, color="#888888", linestyle="-", linewidth=0.5)

        # Correlation between the two
        if min_len > 2:
            from scipy.stats import pearsonr
            r, _ = pearsonr(fp16_rsa[:min_len], quant_rsa[:min_len])
            threshold = config.get("quant_control_threshold", 0.9)
            status = "PASS" if r >= threshold else "INVESTIGATE"
            ax_diff.text(
                0.02, 0.95,
                f"r = {r:.3f} ({status}; threshold = {threshold})",
                transform=ax_diff.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

        _draw_zone_boundaries(ax_diff, config_with_layers)
    else:
        ax_diff.text(0.5, 0.5, "Insufficient data\nfor difference plot",
                     ha="center", va="center", transform=ax_diff.transAxes)

    ax_diff.set_xlabel("Layer")
    ax_diff.set_ylabel("RSA Difference (FP16 - Quantized)")
    ax_diff.set_title("Quantization Effect per Layer")
    ax_diff.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    _save_figure(fig, "quantization_control")
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate_all_figures
# ---------------------------------------------------------------------------


def generate_all_figures(
    config: dict,
    data_dir: str | Path = "data",
) -> dict[str, plt.Figure | None]:
    """Load all available data and generate every figure.

    Gracefully skips any plot whose data files are missing.

    Parameters
    ----------
    config : dict
        Pipeline configuration (typically ``CONFIG`` from run.py).
    data_dir : str or Path
        Root data directory containing analysis/probe outputs.

    Returns
    -------
    dict[str, Figure | None]
        Mapping from figure name to its matplotlib Figure, or None if skipped.
    """
    data_dir = Path(data_dir)
    figures: dict[str, plt.Figure | None] = {}

    logger.info("Generating all figures from %s", data_dir)

    # ---- 1. Condition similarities ----
    cond_path = data_dir / "condition_similarities.json"
    if cond_path.exists():
        try:
            with open(cond_path) as f:
                cond_data = json.load(f)
            figures["condition_similarities"] = plot_condition_similarities(cond_data, config)
        except Exception:
            logger.exception("Failed to generate condition_similarities plot")
            figures["condition_similarities"] = None
    else:
        logger.warning("Skipping condition_similarities — %s not found", cond_path)
        figures["condition_similarities"] = None

    # ---- 2. RSA curves ----
    rsa_files = {
        "product_identity": data_dir / "rsa_product_identity.npy",
        "register_identity": data_dir / "rsa_register_identity.npy",
        "within_category": data_dir / "rsa_within_category.npy",
    }
    rsa_data: dict[str, np.ndarray] = {}
    for key, path in rsa_files.items():
        if path.exists():
            try:
                rsa_data[key] = np.load(path)
            except Exception:
                logger.exception("Failed to load %s", path)

    pvalues_path = data_dir / "rsa_pvalues.json"
    pvalues = None
    if pvalues_path.exists():
        try:
            with open(pvalues_path) as f:
                pvalues = json.load(f)
        except Exception:
            logger.exception("Failed to load %s", pvalues_path)

    if rsa_data:
        try:
            figures["rsa_curves"] = plot_rsa_curves(rsa_data, pvalues, config)
        except Exception:
            logger.exception("Failed to generate rsa_curves plot")
            figures["rsa_curves"] = None
    else:
        logger.warning("Skipping rsa_curves — no RSA .npy files found")
        figures["rsa_curves"] = None

    # ---- 3. Probe curves ----
    probe_path = data_dir / "probe_results.json"
    if probe_path.exists():
        try:
            with open(probe_path) as f:
                probe_data = json.load(f)
            figures["probe_curves"] = plot_probe_curves(probe_data, config)
        except Exception:
            logger.exception("Failed to generate probe_curves plot")
            figures["probe_curves"] = None
    else:
        logger.warning("Skipping probe_curves — %s not found", probe_path)
        figures["probe_curves"] = None

    # ---- 4. Zone comparison ----
    zone_path = data_dir / "zone_results.json"
    if zone_path.exists():
        try:
            with open(zone_path) as f:
                zone_data = json.load(f)
            figures["zone_comparison"] = plot_zone_comparison(zone_data, config)
        except Exception:
            logger.exception("Failed to generate zone_comparison plot")
            figures["zone_comparison"] = None
    else:
        logger.warning("Skipping zone_comparison — %s not found", zone_path)
        figures["zone_comparison"] = None

    # ---- 5. Decomposition ----
    decomp_path = data_dir / "component_rsa.json"
    if decomp_path.exists():
        try:
            with open(decomp_path) as f:
                decomp_data = json.load(f)
            figures["decomposition"] = plot_decomposition(decomp_data, config)
        except Exception:
            logger.exception("Failed to generate decomposition plot")
            figures["decomposition"] = None
    else:
        logger.warning("Skipping decomposition — component data not available")
        figures["decomposition"] = None

    # ---- 6. Memorization control ----
    real_rsa_path = data_dir / "rsa_product_identity.npy"
    fictional_rsa_path = data_dir / "rsa_fictional_product_identity.npy"
    if real_rsa_path.exists() and fictional_rsa_path.exists():
        try:
            real = np.load(real_rsa_path)
            fictional = np.load(fictional_rsa_path)
            figures["memorization_control"] = plot_memorization_control(real, fictional, config)
        except Exception:
            logger.exception("Failed to generate memorization_control plot")
            figures["memorization_control"] = None
    else:
        logger.warning("Skipping memorization_control — real/fictional RSA files not both found")
        figures["memorization_control"] = None

    # ---- 7. RDM heatmaps ----
    rdm_path = data_dir / "rdms.npz"
    if rdm_path.exists():
        try:
            rdm_archive = np.load(rdm_path)
            rdms_loaded: dict[int, np.ndarray] = {}
            for key in rdm_archive.files:
                try:
                    layer_idx = int(key.replace("layer_", ""))
                    rdms_loaded[layer_idx] = rdm_archive[key]
                except ValueError:
                    pass

            if rdms_loaded:
                all_layers = sorted(rdms_loaded.keys())
                # Select early, middle, late layer indices
                early_li = all_layers[0]
                mid_li = all_layers[len(all_layers) // 2]
                late_li = all_layers[-1]
                figures["rdm_heatmaps"] = plot_rdm_heatmaps(
                    rdms_loaded, [early_li, mid_li, late_li], config
                )
            else:
                figures["rdm_heatmaps"] = None
        except Exception:
            logger.exception("Failed to generate rdm_heatmaps plot")
            figures["rdm_heatmaps"] = None
    else:
        logger.warning("Skipping rdm_heatmaps — %s not found", rdm_path)
        figures["rdm_heatmaps"] = None

    # ---- 8. Register confusion ----
    confusion_path = data_dir / "register_confusion.npz"
    if confusion_path.exists():
        try:
            conf_archive = np.load(confusion_path)
            confusion_loaded: dict[int, np.ndarray] = {}
            for key in conf_archive.files:
                try:
                    layer_idx = int(key.replace("layer_", ""))
                    confusion_loaded[layer_idx] = conf_archive[key]
                except ValueError:
                    pass

            if confusion_loaded:
                all_layers = sorted(confusion_loaded.keys())
                early_li = all_layers[0]
                mid_li = all_layers[len(all_layers) // 2]
                late_li = all_layers[-1]
                figures["register_confusion"] = plot_register_confusion(
                    confusion_loaded, [early_li, mid_li, late_li], config
                )
            else:
                figures["register_confusion"] = None
        except Exception:
            logger.exception("Failed to generate register_confusion plot")
            figures["register_confusion"] = None
    else:
        logger.warning("Skipping register_confusion — %s not found", confusion_path)
        figures["register_confusion"] = None

    # ---- 9. Quantization control ----
    quant_rsa_path = data_dir / "rsa_product_identity_quant.npy"
    if real_rsa_path.exists() and quant_rsa_path.exists():
        try:
            fp16 = np.load(real_rsa_path)
            quant = np.load(quant_rsa_path)
            figures["quantization_control"] = plot_quantization_control(fp16, quant, config)
        except Exception:
            logger.exception("Failed to generate quantization_control plot")
            figures["quantization_control"] = None
    else:
        logger.warning("Skipping quantization_control — Tier 4 data not available")
        figures["quantization_control"] = None

    # Close all figures to free memory
    plt.close("all")

    n_generated = sum(1 for v in figures.values() if v is not None)
    logger.info("Generated %d / %d figures", n_generated, len(figures))

    return figures
