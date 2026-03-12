"""
CCAV Design Space & DOE Sampler — Canonical Design-Space Module
================================================================
Single source of truth for the 42-variable CCAV design space
(36 independent + 6 derived).  Reads from ``config/ccav_design_space.csv``,
generates Latin-Hypercube DOE samples, computes derived variables,
validates physics constraints, and exports results to CSV.

This module **replaces** the former ``stage1_design_space.py`` (xlsx reader)
and ``stage2_doe.py`` (separate DOE generator) with a unified CSV-based flow.

Run standalone:
    python -m pipeline.ccav_sampler                     # default 500 samples
    python -m pipeline.ccav_sampler --samples 720 --seed 42

Pipeline flow:
    ccav_design_space.csv  →  LHS DOE  →  physics filter  →  CSV export

Downstream API:
    load_design_space()       → {key: (lo, base, hi)}   for all 36 indep vars
    compute_derived(sample)   → sample with 6 derived keys filled in
    validate_sample(sample)   → (ok: bool, reasons: list[str])
    get_baseline_vector()     → full 42-var baseline dict
    generate_doe(n)           → (all_samples, feasible_mask, failure_reasons)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── Path to canonical CSV ──────────────────────────────────────────
CSV_PATH = _REPO_ROOT / "config" / "ccav_design_space.csv"


# ═══════════════════════════════════════════════════════════════════
#  CSV READER
# ═══════════════════════════════════════════════════════════════════

def _read_csv(path: Path = CSV_PATH) -> list[dict]:
    """Read the CCAV design-space CSV, returning one dict per variable."""
    import pandas as pd
    df = pd.read_csv(path)
    # Drop empty/summary rows
    df = df.dropna(subset=["ID"]).copy()
    df["ID"] = df["ID"].astype(int)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "id":         int(r["ID"]),
            "category":   str(r["Category"]),
            "name":       str(r["Variable Name"]),
            "python_key": str(r["Python Key"]),
            "description": str(r.get("Description", "")),
            "lower":      float(r["Lower Bound"]) if r["Scale Type"] == "Independent" else None,
            "baseline":   float(r["Baseline"]) if r["Scale Type"] == "Independent" else None,
            "upper":      float(r["Upper Bound"]) if r["Scale Type"] == "Independent" else None,
            "unit":       str(r.get("Unit", "")),
            "scale_type": str(r["Scale Type"]),
            "active":     str(r.get("Active", "YES")).upper() == "YES",
        })
    return rows


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC API — Design-Space Metadata
# ═══════════════════════════════════════════════════════════════════

_CACHE: dict[str, Any] = {}


def _ensure_loaded():
    """Lazy-load and cache the CSV data."""
    if "rows" not in _CACHE:
        _CACHE["rows"] = _read_csv()
    return _CACHE["rows"]


def load_design_space(csv_path: Path | None = None) -> Dict[str, Tuple[float, float, float]]:
    """
    Return all 36 independent variables as ``{python_key: (lower, baseline, upper)}``.

    Parameters
    ----------
    csv_path : Path, optional
        Override the default CSV location.
    """
    if csv_path is not None:
        rows = _read_csv(csv_path)
    else:
        rows = _ensure_loaded()

    return {
        r["python_key"]: (r["lower"], r["baseline"], r["upper"])
        for r in rows
        if r["scale_type"] == "Independent" and r["active"]
    }


def get_independent_bounds() -> Dict[str, Tuple[float, float, float]]:
    """Alias for ``load_design_space()`` — drop-in replacement for Stage 1."""
    return load_design_space()


def get_baseline_vector() -> dict:
    """Return the full 42-variable baseline as a dict (independent + derived)."""
    bounds = load_design_space()
    baseline = {k: b for k, (_, b, _) in bounds.items()}
    return compute_derived(baseline)


def get_derived_keys() -> list[str]:
    """Return the python_keys of the 6 derived variables."""
    return list(DERIVED_FORMULAS.keys())


def get_all_keys() -> list[str]:
    """Return all 42 variable keys (sorted)."""
    rows = _ensure_loaded()
    return [r["python_key"] for r in rows if r["active"]]


def get_design_bounds() -> Dict[str, Tuple[float, float, float]]:
    """Return ALL 42 variables.  Derived vars get their baseline-computed bounds."""
    indep = load_design_space()
    baseline = get_baseline_vector()
    result = dict(indep)
    for k in get_derived_keys():
        v = baseline.get(k, 0.0)
        # Provide ±50% synthetic bounds for derived vars (for UI range sliders)
        lo = v * 0.3 if v > 0 else v * 1.7
        hi = v * 1.7 if v > 0 else v * 0.3
        result[k] = (lo, v, hi)
    return result


# ═══════════════════════════════════════════════════════════════════
#  DERIVED VARIABLE FORMULAS  (K_BLEND = 1.85)
# ═══════════════════════════════════════════════════════════════════

K_BLEND = 1.85   # CCAV blended-wing-body planform correction
K_AR    = 2.143  # Aspect ratio calibration for exposed panel root


def _derive_wing_taper(s: dict) -> float:
    """Taper ratio = tip chord / root chord."""
    rc = s.get("wing_root_chord", 1.0)
    return s.get("wing_tip_chord", 0.0) / rc if rc > 0 else 0.0


def _derive_wing_area(s: dict) -> float:
    """Reference wing area for CCAV blended-wing-body planform.
    S_ref = K_BLEND × span × (root + tip) / 2."""
    return K_BLEND * s["wing_span"] * (s["wing_root_chord"] + s["wing_tip_chord"]) / 2.0


def _derive_wing_AR(s: dict) -> float:
    """Geometric aspect ratio with planform correction.
    AR = K_AR × span² / S_trap."""
    s_trap = s["wing_span"] * (s["wing_root_chord"] + s["wing_tip_chord"]) / 2.0
    return K_AR * s["wing_span"] ** 2 / s_trap if s_trap > 0 else 0.0


def _derive_inlet_area(s: dict) -> float:
    """Inlet capture area = width × height."""
    return s["inlet_width"] * s["inlet_height"]


def _derive_mass_empty(s: dict) -> float:
    """Raymer-class empty weight estimate for a CCAV.
    m_empty = 120 × S_ref^0.5 × span^0.6  (calibrated to baseline ≈ 9000 kg)."""
    S = _derive_wing_area(s)
    span = s["wing_span"]
    return 120.0 * S ** 0.5 * span ** 0.6


def _derive_mass_GTOW(s: dict) -> float:
    """Gross takeoff weight = empty + fuel + payload."""
    m_empty = _derive_mass_empty(s)
    return m_empty + s["mass_fuel_kg"] + s["mass_payload_kg"]


DERIVED_FORMULAS: Dict[str, callable] = {
    "wing_taper":    _derive_wing_taper,
    "wing_area":     _derive_wing_area,
    "wing_AR":       _derive_wing_AR,
    "inlet_area":    _derive_inlet_area,
    "mass_empty_kg": _derive_mass_empty,
    "mass_GTOW":     _derive_mass_GTOW,
}


def compute_derived(sample: dict) -> dict:
    """
    Given a dict of independent variable values, compute all 6 derived
    variables and return the complete 42-variable sample dict.
    """
    out = dict(sample)
    for key, func in DERIVED_FORMULAS.items():
        out[key] = func(out)
    return out


# ═══════════════════════════════════════════════════════════════════
#  PHYSICS VALIDATION  (9 constraint checks)
# ═══════════════════════════════════════════════════════════════════

def validate_sample(sample: dict) -> Tuple[bool, List[str]]:
    """
    Check a complete 42-variable sample for physical consistency.
    Returns ``(is_valid, list_of_failure_reasons)``.
    """
    reasons = []

    # 1. Inverted taper
    if sample.get("wing_tip_chord", 0) > sample.get("wing_root_chord", 1):
        reasons.append(
            f"Inverted taper: tip ({sample['wing_tip_chord']:.2f}) "
            f"> root ({sample['wing_root_chord']:.2f})")

    # 2. Taper ratio range
    taper = sample.get("wing_taper", 0)
    if not (0.08 <= taper <= 0.65):
        reasons.append(f"Taper ratio {taper:.3f} outside [0.08, 0.65]")

    # 3. Aspect ratio range
    ar = sample.get("wing_AR", 0)
    if not (2.0 <= ar <= 16.0):
        reasons.append(f"Aspect ratio {ar:.2f} outside [2, 16]")

    # 4. GTOW > empty
    if sample.get("mass_GTOW", 0) <= sample.get("mass_empty_kg", 0):
        reasons.append("GTOW ≤ empty weight — no fuel/payload margin")

    # 5. Cruise thrust < max thrust
    if sample.get("thrust_cruise", 0) >= sample.get("thrust_max", 0):
        reasons.append(
            f"Cruise thrust ({sample['thrust_cruise']:.0f} kN) "
            f">= max thrust ({sample['thrust_max']:.0f} kN)")

    # 6. Fuel volume feasibility (soft check)
    #    ρ_kerosene ≈ 800 kg/m³.  Estimate available fuel volume from wing area.
    wing_area = sample.get("wing_area", 0)
    avg_chord = (sample.get("wing_root_chord", 0) + sample.get("wing_tip_chord", 0)) / 2.0
    vol_fuel_avail = 0.0167 * wing_area * avg_chord  # K_FUEL estimate
    fuel_vol_needed = sample.get("mass_fuel_kg", 0) / 800.0
    if vol_fuel_avail > 0 and fuel_vol_needed > vol_fuel_avail / 0.60:
        reasons.append(
            f"Fuel volume tight: need {fuel_vol_needed:.2f} m³, "
            f"have {vol_fuel_avail:.2f} m³ (ratio {fuel_vol_needed / vol_fuel_avail:.2f})")

    # 7. Wing loading sanity
    area = sample.get("wing_area", 1)
    gtow = sample.get("mass_GTOW", 0)
    if area > 0:
        wl = gtow / area
        if not (100 <= wl <= 900):
            reasons.append(f"Wing loading {wl:.0f} kg/m² outside [100, 900]")

    # 8. Pitch stability proxy (sweep-based Cma estimate)
    sweep = sample.get("wing_sweep_LE", 35)
    cl = sample.get("CL_cruise", 0.6)
    cma_est = -0.02 * sweep / 35.0 * cl  # simplified negative-Cma proxy
    if cma_est >= 0:
        reasons.append(f"Estimated Cma = {cma_est:.4f} >= 0 — unstable")

    # 9. Body fineness ratio sanity
    bw = sample.get("body_width", 1)
    if bw > 0:
        fin = sample.get("body_length", 0) / bw
        if not (4.0 <= fin <= 25.0):
            reasons.append(f"Body fineness {fin:.1f} outside [4, 25]")

    return (len(reasons) == 0, reasons)


# ═══════════════════════════════════════════════════════════════════
#  LHS DOE GENERATION + FILTERING
# ═══════════════════════════════════════════════════════════════════

def _latin_hypercube(n_samples: int, n_dims: int,
                     seed: int = 42,
                     optimise: str = "maximin") -> np.ndarray:
    """Generate an N×d LHS design in [0,1]^d."""
    from scipy.stats.qmc import LatinHypercube
    if optimise == "maximin":
        sampler = LatinHypercube(d=n_dims, seed=seed, optimization="random-cd")
    else:
        sampler = LatinHypercube(d=n_dims, seed=seed)
    return sampler.random(n=n_samples)


def _scale_to_physical(unit_samples: np.ndarray,
                       keys: list[str],
                       bounds: dict) -> list[dict]:
    """Scale [0,1]^d samples to physical variable ranges."""
    lowers = np.array([bounds[k][0] for k in keys])
    uppers = np.array([bounds[k][2] for k in keys])
    spans = uppers - lowers
    samples = []
    for row in unit_samples:
        physical = lowers + row * spans
        samples.append({k: float(v) for k, v in zip(keys, physical)})
    return samples


def generate_doe(
    n_samples: int = 500,
    seed: int = 42,
    optimise: str = "maximin",
    include_baseline: bool = True,
    verbose: bool = True,
) -> Tuple[List[dict], List[bool], List[List[str]]]:
    """
    Generate a full DOE table with derived variables and feasibility flags.

    Returns
    -------
    all_samples : list of dict
        Every sample (42 variables each), including infeasible ones.
    feasible_mask : list of bool
        True if the sample passes all physics checks.
    failure_reasons : list of list[str]
        For each sample, the list of failed checks (empty if feasible).
    """
    indep_bounds = load_design_space()
    keys = sorted(indep_bounds.keys())
    n_dims = len(keys)

    if verbose:
        print(f"\n  Sampling {n_samples} points in {n_dims}-dimensional space...")

    t0 = time.time()
    unit_lhs = _latin_hypercube(n_samples, n_dims, seed=seed, optimise=optimise)
    t_lhs = time.time() - t0
    if verbose:
        print(f"  LHS generated in {t_lhs:.2f}s  (optimisation: {optimise})")

    raw_samples = _scale_to_physical(unit_lhs, keys, indep_bounds)

    if include_baseline:
        baseline = {k: b for k, (_, b, _) in indep_bounds.items()}
        raw_samples.insert(0, baseline)
        if verbose:
            print("  Baseline injected as sample #0")

    all_samples: list[dict] = []
    feasible_mask: list[bool] = []
    failure_reasons: list[list[str]] = []
    n_feasible = 0
    n_total = len(raw_samples)

    if verbose:
        print(f"  Computing derived vars + physics checks for {n_total} samples...")

    for i, raw in enumerate(raw_samples):
        full = compute_derived(raw)
        all_samples.append(full)
        ok, reasons = validate_sample(full)
        feasible_mask.append(ok)
        failure_reasons.append(reasons)
        if ok:
            n_feasible += 1
        if verbose and (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{n_total} processed  ({n_feasible} feasible so far)")

    if verbose:
        pct = 100.0 * n_feasible / n_total if n_total > 0 else 0
        print(f"\n  RESULT: {n_feasible}/{n_total} feasible ({pct:.1f}%)")

    return all_samples, feasible_mask, failure_reasons


# ═══════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════

def export_csv(
    all_samples: list[dict],
    feasible_mask: list[bool],
    failure_reasons: list[list[str]],
    path: Path,
    feasible_only: bool = False,
) -> Path:
    """Export DOE samples to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not all_samples:
        return path

    all_keys = sorted(all_samples[0].keys())
    fieldnames = ["sample_id", "is_feasible"] + all_keys + ["rejection_reason"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (sample, ok, reasons) in enumerate(
                zip(all_samples, feasible_mask, failure_reasons)):
            if feasible_only and not ok:
                continue
            row = {"sample_id": i, "is_feasible": int(ok)}
            row.update(sample)
            row["rejection_reason"] = "; ".join(reasons) if reasons else ""
            writer.writerow(row)

    return path


# ═══════════════════════════════════════════════════════════════════
#  STATISTICS
# ═══════════════════════════════════════════════════════════════════

def print_filter_statistics(
    all_samples: list[dict],
    feasible_mask: list[bool],
    failure_reasons: list[list[str]],
) -> None:
    """Print a detailed breakdown of constraint filter results."""
    import re

    n_total = len(all_samples)
    n_feas = sum(feasible_mask)
    n_infeas = n_total - n_feas

    print(f"\n{'=' * 65}")
    print("  CCAV DOE — FILTER STATISTICS")
    print(f"{'=' * 65}")
    print(f"\n  Total samples:     {n_total}")
    print(f"  Feasible:          {n_feas}  ({100 * n_feas / n_total:.1f}%)")
    print(f"  Infeasible:        {n_infeas}  ({100 * n_infeas / n_total:.1f}%)")

    _CATS = [
        (re.compile(r"Inverted taper"), "Inverted taper"),
        (re.compile(r"Taper ratio"), "Taper ratio out of range"),
        (re.compile(r"Aspect ratio"), "Aspect ratio out of range"),
        (re.compile(r"GTOW"), "GTOW ≤ empty weight"),
        (re.compile(r"Cruise thrust"), "Cruise thrust >= max thrust"),
        (re.compile(r"Fuel volume"), "Fuel volume tight"),
        (re.compile(r"Wing loading"), "Wing loading out of range"),
        (re.compile(r"Cma|unstable"), "Pitch stability violation"),
        (re.compile(r"fineness", re.I), "Body fineness ratio"),
    ]

    reason_counts: dict[str, int] = {}
    for reasons in failure_reasons:
        for r in reasons:
            categorised = False
            for pat, cat in _CATS:
                if pat.search(r):
                    reason_counts[cat] = reason_counts.get(cat, 0) + 1
                    categorised = True
                    break
            if not categorised:
                reason_counts[r[:40]] = reason_counts.get(r[:40], 0) + 1

    if reason_counts:
        print(f"\n  Failure breakdown:")
        for tag, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / n_total
            bar = "█" * int(pct / 2)
            print(f"    {tag:<35s}  {count:>5d}  ({pct:5.1f}%)  {bar}")

    # Variable coverage in feasible set
    if n_feas > 0:
        feas_samples = [s for s, ok in zip(all_samples, feasible_mask) if ok]
        indep_bounds = load_design_space()
        print(f"\n  Bound coverage (feasible range / design space range):")
        for k in sorted(indep_bounds.keys()):
            blo, _, bhi = indep_bounds[k]
            b_range = bhi - blo
            if b_range <= 0:
                continue
            vals = [s[k] for s in feas_samples]
            f_range = max(vals) - min(vals)
            coverage = f_range / b_range * 100
            bar = "█" * int(coverage / 5)
            print(f"    {k:<25s}  {coverage:5.1f}% {bar}")

    print()


# ═══════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CCAV DOE Sampler — LHS + physics pre-filter")
    parser.add_argument("--samples", "-n", type=int, default=500,
                        help="Number of LHS samples (default: 500)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Don't inject baseline as sample #0")
    args = parser.parse_args()

    print("=" * 65)
    print("  CCAV DOE Sampler — Design of Experiments Generation")
    print("=" * 65)

    t0 = time.time()
    all_samples, feasible_mask, failure_reasons = generate_doe(
        n_samples=args.samples,
        seed=args.seed,
        include_baseline=not args.no_baseline,
        verbose=True,
    )
    t_total = time.time() - t0

    out_dir = _REPO_ROOT / "data"
    out_dir.mkdir(exist_ok=True)

    # CSV — all samples
    csv_all = export_csv(
        all_samples, feasible_mask, failure_reasons,
        out_dir / "ccav_doe_samples.csv", feasible_only=False)
    print(f"\n  CSV (all):      {csv_all}")

    # CSV — feasible only
    csv_feas = export_csv(
        all_samples, feasible_mask, failure_reasons,
        out_dir / "ccav_feasible.csv", feasible_only=True)
    n_feas = sum(feasible_mask)
    print(f"  CSV (feasible): {csv_feas}  ({n_feas} rows)")

    # Statistics
    print_filter_statistics(all_samples, feasible_mask, failure_reasons)

    # Summary
    n_total = len(all_samples)
    print("=" * 65)
    print(f"  COMPLETE  ({t_total:.1f}s)")
    print(f"    {n_total} total samples (LHS, seed={args.seed})")
    print(f"    {n_feas} feasible  ({100 * n_feas / n_total:.1f}%)")
    print(f"    {n_total - n_feas} rejected by physics pre-filter")
    print(f"    Ready for Stage 3 screening")
    print("=" * 65)


if __name__ == "__main__":
    main()
