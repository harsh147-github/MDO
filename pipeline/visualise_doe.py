"""
CCAV DOE Visual Test — Design Space to DOE
=============================================
Generates a set of diagnostic plots showing how the 36-dimensional
design space gets populated by LHS samples and how the physics
pre-filter separates feasible from infeasible designs.

Run:
    python -m pipeline.visualise_doe
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — works without display server
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.ccav_sampler import (
    get_independent_bounds, get_baseline_vector, generate_doe,
)

OUT_DIR = _REPO_ROOT / "data" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ───────────────────────────────────────────────────
C_FEAS   = "#2ecc71"   # green
C_INFEAS = "#e74c3c"   # red
C_BASE   = "#f39c12"   # gold
C_BOUNDS = "#3498db"   # blue
C_BG     = "#1a1a2e"   # dark background
C_TEXT   = "#ecf0f1"


def _setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": C_BG,
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#444",
        "axes.labelcolor": C_TEXT,
        "text.color": C_TEXT,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
        "grid.color": "#333",
        "grid.alpha": 0.4,
        "font.size": 9,
        "legend.facecolor": "#16213e",
        "legend.edgecolor": "#444",
    })


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 1 — Design Space Bounds vs LHS Samples (strip chart)
# ═══════════════════════════════════════════════════════════════════════
def plot_bounds_vs_samples(all_samples, feasible_mask, bounds, baseline):
    """
    For each of the 28 independent variables, show:
        - The full [lower, upper] range as a blue bar
        - Each LHS sample as a dot (green=feasible, red=infeasible)
        - The baseline as a gold diamond
    """
    keys = sorted(bounds.keys())
    n_vars = len(keys)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle("STAGE 1 → STAGE 2:  Design Space Bounds  vs  LHS Samples",
                 fontsize=14, fontweight="bold", y=0.98)

    for i, k in enumerate(keys):
        lo, base, hi = bounds[k]
        span = hi - lo if hi != lo else 1

        # Normalise everything to [0, 1] within each variable's range
        y_base = (base - lo) / span

        # Plot the bound bar
        ax.barh(i, 1.0, left=0, height=0.6, color=C_BOUNDS, alpha=0.15,
                edgecolor=C_BOUNDS, linewidth=0.5)

        # Collect normalised values for this variable
        feas_vals = []
        infeas_vals = []
        for s, ok in zip(all_samples, feasible_mask):
            if k in s:
                v = (s[k] - lo) / span
                if ok:
                    feas_vals.append(v)
                else:
                    infeas_vals.append(v)

        # Batch scatter (much faster than per-sample)
        if infeas_vals:
            ax.scatter(infeas_vals, [i] * len(infeas_vals),
                       s=4, c=C_INFEAS, alpha=0.25, zorder=2, edgecolors="none")
        if feas_vals:
            ax.scatter(feas_vals, [i] * len(feas_vals),
                       s=4, c=C_FEAS, alpha=0.35, zorder=3, edgecolors="none")

        # Baseline diamond
        ax.scatter(y_base, i, s=80, c=C_BASE, marker="D", zorder=5,
                   edgecolors="white", linewidth=0.8)

    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(keys, fontsize=7)
    ax.set_xlabel("Normalised position within [Lower, Upper] bound")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, n_vars - 0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEAS,
               markersize=8, label=f"Feasible ({sum(feasible_mask)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_INFEAS,
               markersize=8, label=f"Infeasible ({len(feasible_mask)-sum(feasible_mask)})"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_BASE,
               markersize=10, label="Baseline"),
        Rectangle((0, 0), 1, 1, fc=C_BOUNDS, alpha=0.15, ec=C_BOUNDS,
                  label="Design bounds"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT_DIR / "01_bounds_vs_samples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/6] {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 2 — 2D scatter matrix of key wing/mass variables
# ═══════════════════════════════════════════════════════════════════════
def plot_scatter_matrix(all_samples, feasible_mask):
    """
    Pairwise scatter plots of the 6 most important variables,
    coloured by feasibility.
    """
    key_vars = ["wing_span", "wing_root_chord", "wing_tip_chord",
                "mass_fuel_kg", "mass_empty_kg", "cruise_mach"]
    n = len(key_vars)

    fig, axes = plt.subplots(n, n, figsize=(16, 14))
    fig.suptitle("SCATTER MATRIX — Key Variables (green=feasible, red=infeasible)",
                 fontsize=13, fontweight="bold", y=0.99)

    feas_samples = [s for s, ok in zip(all_samples, feasible_mask) if ok]
    infeas_samples = [s for s, ok in zip(all_samples, feasible_mask) if not ok]

    for i, ky in enumerate(key_vars):
        for j, kx in enumerate(key_vars):
            ax = axes[i][j]
            if i == j:
                # Diagonal — histogram
                vals_f = [s[kx] for s in feas_samples if kx in s]
                vals_i = [s[kx] for s in infeas_samples if kx in s]
                ax.hist(vals_f, bins=20, color=C_FEAS, alpha=0.7, density=True)
                ax.hist(vals_i, bins=20, color=C_INFEAS, alpha=0.5, density=True)
            else:
                # Off-diagonal — scatter
                xf = [s[kx] for s in feas_samples if kx in s]
                yf = [s[ky] for s in feas_samples if ky in s]
                xi = [s[kx] for s in infeas_samples if kx in s]
                yi = [s[ky] for s in infeas_samples if ky in s]
                ax.scatter(xi, yi, s=6, c=C_INFEAS, alpha=0.3, edgecolors="none")
                ax.scatter(xf, yf, s=6, c=C_FEAS, alpha=0.4, edgecolors="none")

            if i == n - 1:
                ax.set_xlabel(kx, fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ky, fontsize=6)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=5)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUT_DIR / "02_scatter_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/6] {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 3 — Parallel Coordinates (normalised)
# ═══════════════════════════════════════════════════════════════════════
def plot_parallel_coordinates(all_samples, feasible_mask, bounds):
    """
    Each line is one design sample plotted across all 28 normalised axes.
    Green = feasible, Red = infeasible.
    """
    keys = sorted(bounds.keys())
    n_vars = len(keys)

    lowers = np.array([bounds[k][0] for k in keys])
    uppers = np.array([bounds[k][2] for k in keys])
    spans = uppers - lowers
    spans[spans == 0] = 1

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.suptitle("PARALLEL COORDINATES — 36 Independent Variables (normalised to [0,1])",
                 fontsize=13, fontweight="bold", y=0.98)

    x_pos = np.arange(n_vars)

    # Build arrays for vectorized plotting
    feas_lines = []
    infeas_lines = []
    for s, ok in zip(all_samples, feasible_mask):
        vals = np.array([(s.get(k, bounds[k][1]) - lowers[i]) / spans[i]
                         for i, k in enumerate(keys)])
        if ok:
            feas_lines.append(vals)
        else:
            infeas_lines.append(vals)

    # Plot infeasible first (behind), then feasible
    for vals in infeas_lines:
        ax.plot(x_pos, vals, color=C_INFEAS, alpha=0.04, linewidth=0.4, zorder=2)
    for vals in feas_lines:
        ax.plot(x_pos, vals, color=C_FEAS, alpha=0.12, linewidth=0.4, zorder=3)

    # Baseline on top
    baseline = get_baseline_vector()
    base_vals = np.array([(baseline.get(k, bounds[k][1]) - lowers[i]) / spans[i]
                          for i, k in enumerate(keys)])
    ax.plot(x_pos, base_vals, color=C_BASE, linewidth=2.5, zorder=10,
            marker="D", markersize=5, label="Baseline")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(keys, rotation=65, ha="right", fontsize=6)
    ax.set_ylabel("Normalised value [0 = lower, 1 = upper]")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=C_FEAS, linewidth=2, label="Feasible"),
        Line2D([0], [0], color=C_INFEAS, linewidth=2, label="Infeasible"),
        Line2D([0], [0], color=C_BASE, linewidth=2.5, marker="D",
               markersize=6, label="Baseline"),
    ]
    ax.legend(handles=legend_elems, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT_DIR / "03_parallel_coordinates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/6] {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 4 — Feasibility pie + failure breakdown bar chart
# ═══════════════════════════════════════════════════════════════════════
def plot_filter_summary(feasible_mask, failure_reasons):
    import re

    _CATEGORIES = [
        (re.compile(r"Fuel volume"), "Fuel Volume"),
        (re.compile(r"Taper ratio"), "Taper Ratio"),
        (re.compile(r"Wing loading"), "Wing Loading"),
        (re.compile(r"Aspect ratio"), "Aspect Ratio"),
        (re.compile(r"Inverted taper"), "Inverted Taper"),
        (re.compile(r"GTOW"), "GTOW Check"),
        (re.compile(r"Cruise thrust"), "Thrust Check"),
        (re.compile(r"Cma"), "Stability"),
        (re.compile(r"Fineness"), "Fineness"),
    ]

    counts = {}
    for reasons in failure_reasons:
        for r in reasons:
            categorised = False
            for pat, cat in _CATEGORIES:
                if pat.search(r):
                    counts[cat] = counts.get(cat, 0) + 1
                    categorised = True
                    break
            if not categorised:
                counts["Other"] = counts.get("Other", 0) + 1

    n_feas = sum(feasible_mask)
    n_infeas = len(feasible_mask) - n_feas

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 2])

    # ── Pie chart ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.pie([n_feas, n_infeas],
            labels=[f"Feasible\n{n_feas}", f"Infeasible\n{n_infeas}"],
            colors=[C_FEAS, C_INFEAS],
            autopct="%1.1f%%", startangle=90,
            textprops={"color": C_TEXT, "fontsize": 11},
            wedgeprops={"edgecolor": C_BG, "linewidth": 2})
    ax1.set_title("Physics Pre-Filter Result", fontsize=12, fontweight="bold")

    # ── Bar chart ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    cats = [c[0] for c in sorted_counts]
    vals = [c[1] for c in sorted_counts]
    bars = ax2.barh(cats, vals, color=C_INFEAS, alpha=0.8, edgecolor="#c0392b")
    ax2.set_xlabel("Number of samples failing this check")
    ax2.set_title("Failure Breakdown (samples can fail multiple checks)",
                  fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                 str(v), va="center", fontsize=10, color=C_TEXT)

    fig.tight_layout()
    path = OUT_DIR / "04_filter_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4/6] {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 5 — LHS Coverage Quality (1D marginal histograms)
# ═══════════════════════════════════════════════════════════════════════
def plot_lhs_coverage(all_samples, feasible_mask, bounds):
    """
    Show how uniformly the LHS fills each variable's range.
    Good LHS → flat histograms. Bad → gaps.
    """
    keys = sorted(bounds.keys())
    n_vars = len(keys)
    ncols = 7
    nrows = (n_vars + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.2))
    fig.suptitle("LHS COVERAGE — Marginal Distributions per Variable",
                 fontsize=13, fontweight="bold", y=1.01)
    axes_flat = axes.flatten()

    feas_samples = [s for s, ok in zip(all_samples, feasible_mask) if ok]
    infeas_samples = [s for s, ok in zip(all_samples, feasible_mask) if not ok]

    for i, k in enumerate(keys):
        ax = axes_flat[i]
        lo, _, hi = bounds[k]

        vals_f = [s[k] for s in feas_samples if k in s]
        vals_i = [s[k] for s in infeas_samples if k in s]

        bins = np.linspace(lo, hi, 16)
        ax.hist(vals_f, bins=bins, color=C_FEAS, alpha=0.7, label="Feasible")
        ax.hist(vals_i, bins=bins, color=C_INFEAS, alpha=0.5, label="Infeasible")
        ax.set_title(k, fontsize=6, fontweight="bold")
        ax.tick_params(labelsize=4)
        ax.set_xlim(lo, hi)

    # Hide extra axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Single legend
    axes_flat[0].legend(fontsize=5, loc="upper right")

    fig.tight_layout()
    path = OUT_DIR / "05_lhs_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5/6] {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  PLOT 6 — Key derived variables: wing_area vs mass_GTOW coloured by taper
# ═══════════════════════════════════════════════════════════════════════
def plot_derived_landscape(all_samples, feasible_mask):
    """
    Show how the derived variables (wing_area, mass_GTOW, wing_AR)
    emerge from the independent samples. These are the variables that
    Stage 3 will evaluate aerodynamically.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("DERIVED VARIABLE LANDSCAPE — Computed from Independent Samples",
                 fontsize=13, fontweight="bold", y=1.01)

    feas = [s for s, ok in zip(all_samples, feasible_mask) if ok]
    infeas = [s for s, ok in zip(all_samples, feasible_mask) if not ok]

    # ── Panel 1: Wing Area vs GTOW ───────────────────────────────
    ax = axes[0]
    if infeas:
        ax.scatter([s.get("wing_area", 0) for s in infeas],
                   [s.get("mass_GTOW", 0) for s in infeas],
                   s=12, c=C_INFEAS, alpha=0.2, edgecolors="none", label="_inf")
    if feas:
        ax.scatter([s.get("wing_area", 0) for s in feas],
                   [s.get("mass_GTOW", 0) for s in feas],
                   s=12, c=C_FEAS, alpha=0.5, edgecolors="none", label="_feas")
    ax.set_xlabel("Wing Area [m²]")
    ax.set_ylabel("GTOW [kg]")
    ax.set_title("Wing Area vs GTOW", fontsize=10)
    ax.grid(alpha=0.3)

    # ── Panel 2: Wing Span vs Aspect Ratio ────────────────────────
    ax = axes[1]
    if infeas:
        ax.scatter([s.get("wing_span", 0) for s in infeas],
                   [s.get("wing_AR", 0) for s in infeas],
                   s=12, c=C_INFEAS, alpha=0.2, edgecolors="none")
    if feas:
        ax.scatter([s.get("wing_span", 0) for s in feas],
                   [s.get("wing_AR", 0) for s in feas],
                   s=12, c=C_FEAS, alpha=0.5, edgecolors="none")
    ax.set_xlabel("Wing Span [m]")
    ax.set_ylabel("Aspect Ratio")
    ax.set_title("Span vs Aspect Ratio", fontsize=10)
    ax.grid(alpha=0.3)

    # ── Panel 3: Fuel Mass vs Body Length ────────────────────────
    ax = axes[2]
    if infeas:
        fvn_i = [s.get("mass_fuel_kg", 0) for s in infeas]
        fva_i = [s.get("body_length", 0) for s in infeas]
        ax.scatter(fva_i, fvn_i, s=12, c=C_INFEAS, alpha=0.2, edgecolors="none")
    if feas:
        fvn_f = [s.get("mass_fuel_kg", 0) for s in feas]
        fva_f = [s.get("body_length", 0) for s in feas]
        ax.scatter(fva_f, fvn_f, s=12, c=C_FEAS, alpha=0.5, edgecolors="none")
    ax.set_xlabel("Body Length [m]")
    ax.set_ylabel("Fuel Mass [kg]")
    ax.set_title("Body Length vs Fuel Mass", fontsize=10)
    ax.grid(alpha=0.3)

    # Common legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEAS,
               markersize=8, label="Feasible"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_INFEAS,
               markersize=8, label="Infeasible"),
    ]
    axes[0].legend(handles=legend_elems, fontsize=8, loc="upper left")

    fig.tight_layout()
    path = OUT_DIR / "06_derived_landscape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6/6] {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — Generate all plots
# ═══════════════════════════════════════════════════════════════════════
def main():
    _setup_dark_style()

    print("=" * 60)
    print("  VISUAL TEST — CCAV Design Space → LHS DOE")
    print("=" * 60)

    # Generate fresh samples
    print("\n  Generating 500 LHS samples + physics filter...")
    all_samples, feasible_mask, failure_reasons = generate_doe(
        n_samples=500, seed=42, include_baseline=True, verbose=True
    )

    bounds = get_independent_bounds()
    baseline = get_baseline_vector()

    print(f"\n  Generating 6 diagnostic plots → {OUT_DIR}/\n")

    plot_bounds_vs_samples(all_samples, feasible_mask, bounds, baseline)
    plot_scatter_matrix(all_samples, feasible_mask)
    plot_parallel_coordinates(all_samples, feasible_mask, bounds)
    plot_filter_summary(feasible_mask, failure_reasons)
    plot_lhs_coverage(all_samples, feasible_mask, bounds)
    plot_derived_landscape(all_samples, feasible_mask)

    print(f"\n  All 6 plots saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
