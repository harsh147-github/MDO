"""
Stage 3 — Low-Fidelity Batch Screening
=======================================
Multi-discipline rapid analysis of feasible CCAV DOE samples.
Evaluates aero (L/D), structures (max stress), and stealth (RCS)
for each design point, then ranks by a composite objective.

Solver priority (per discipline):
    Aero:       AeroSandbox → analytical lifting-line + flat-plate
    Structures: Analytical beam model (Euler-Bernoulli wing box)
    Stealth:    Heuristic RCS from planform alignment + shielding

Run standalone:
    python -m pipeline.stage3_screening                  # all feasible
    python -m pipeline.stage3_screening --samples 50 -v  # quick test

Pipeline flow:
    ccav_feasible.csv  →  per-sample screening  →  ccav_screening_results.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.ccav_sampler import (
    load_design_space,
    compute_derived,
    validate_sample,
    generate_doe,
    get_baseline_vector,
    K_BLEND,
)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS & THRESHOLDS
# ═══════════════════════════════════════════════════════════════════

# Hard constraints for feasibility
STRESS_LIMIT_MPA = 450.0       # Maximum allowable stress
RCS_LIMIT_DBSM   = -20.0      # RCS must be below this (more negative = stealthier)
LD_MIN            = 5.0        # Minimum acceptable L/D

# Objective weights
W_AERO   = 0.4    # L/D penalty weight
W_MASS   = 0.3    # Mass penalty weight
W_RCS    = 0.3    # RCS penalty weight

# Baseline reference values (for normalisation)
BASELINE_LD       = 12.0       # Expected baseline L/D
BASELINE_MASS_KG  = 15500.0   # Expected baseline GTOW
BASELINE_RCS_DBSM = -10.0     # Expected baseline RCS


# ═══════════════════════════════════════════════════════════════════
#  DISCIPLINE EVALUATORS
# ═══════════════════════════════════════════════════════════════════

def evaluate_aero(sample: dict) -> Dict[str, float]:
    """
    Low-fidelity aerodynamic evaluation.

    Uses analytical lifting-line theory + flat-plate drag buildup.
    Returns CL, CD, L/D, and component drag breakdown.
    """
    span     = sample.get("wing_span", 11.9)
    S_ref    = sample.get("wing_area", 80.0)
    AR       = sample.get("wing_AR", 8.5)
    CL       = sample.get("CL_cruise", 0.6)
    mach     = sample.get("cruise_mach", 0.9)
    sweep    = sample.get("wing_sweep_LE", 35.0)
    tc_root  = sample.get("wing_tc_root", 0.12)
    tc_tip   = sample.get("wing_tc_tip", 0.09)

    # Average thickness ratio
    tc_avg = (tc_root + tc_tip) / 2.0

    # Oswald efficiency (Raymer approximation)
    e = 1.78 * (1.0 - 0.045 * AR ** 0.68) - 0.64
    e = max(0.5, min(e, 0.95))

    # Induced drag
    CD_i = CL ** 2 / (math.pi * AR * e)

    # Parasitic drag (flat-plate + form factor)
    Re = 5e6  # Reynolds number estimate at cruise
    Cf = 0.455 / (math.log10(Re) ** 2.58)  # Prandtl-Schlichting
    FF_wing = (1.0 + 0.6 / 0.3 * tc_avg + 100 * tc_avg ** 4) * \
              (1.34 * mach ** 0.18 * math.cos(math.radians(sweep)) ** 0.28)
    S_wet_wing = 2.0 * S_ref * 1.02  # wetted area ≈ 2× reference × 1.02
    CD_0_wing = Cf * FF_wing * S_wet_wing / S_ref

    # Body drag contribution
    body_len = sample.get("body_length", 7.9)
    body_w   = sample.get("body_width", 0.85)
    S_wet_body = math.pi * body_w * body_len * 0.6  # simplified wetted area
    FF_body = 1.0 + 60.0 / (body_len / body_w) ** 3 + 0.0025 * (body_len / body_w)
    CD_0_body = Cf * FF_body * S_wet_body / S_ref

    CD_0 = CD_0_wing + CD_0_body

    # Wave drag (lock-type for transonic)
    M_crit = 0.87 - 0.1 * tc_avg - 0.05 * CL  # simplified Mcrit
    if mach > M_crit:
        dM = mach - M_crit
        CD_wave = 20.0 * dM ** 4  # rapid rise above Mcrit
    else:
        CD_wave = 0.0

    CD_total = CD_0 + CD_i + CD_wave
    L_over_D = CL / CD_total if CD_total > 0 else 0.0

    return {
        "CL":         round(CL, 6),
        "CD_total":   round(CD_total, 6),
        "CD_parasitic": round(CD_0, 6),
        "CD_induced": round(CD_i, 6),
        "CD_wave":    round(CD_wave, 6),
        "L_over_D":   round(L_over_D, 4),
        "oswald_e":   round(e, 4),
    }


def evaluate_structures(sample: dict) -> Dict[str, float]:
    """
    Low-fidelity structural evaluation.

    Analytical Euler-Bernoulli beam model of the wing box.
    Returns max bending stress, tip deflection, and structural mass estimate.
    """
    span      = sample.get("wing_span", 11.9)
    S_ref     = sample.get("wing_area", 80.0)
    root_c    = sample.get("wing_root_chord", 4.5)
    tip_c     = sample.get("wing_tip_chord", 1.5)
    tc_root   = sample.get("wing_tc_root", 0.12)
    n_max     = sample.get("n_max", 5.0)
    gtow      = sample.get("mass_GTOW", 15500.0)

    semi_span = span / 2.0

    # Wing box height at root (fraction of chord × t/c)
    h_spar = root_c * tc_root * 0.6  # 60% of airfoil thickness
    t_skin = 0.003  # 3mm skin thickness (composite)

    # Second moment of area (rectangular box approximation)
    w_box = root_c * 0.4  # 40% chord wing box width
    I_xx = (w_box * h_spar ** 3 / 12.0) - \
           ((w_box - 2 * t_skin) * (h_spar - 2 * t_skin) ** 3 / 12.0)
    I_xx = max(I_xx, 1e-6)

    # Elliptic lift distribution → root bending moment
    W = gtow * 9.81 * n_max  # ultimate load
    L_wing = W / 2.0  # half-span lift
    M_root = L_wing * semi_span * 4.0 / (3.0 * math.pi)  # elliptic BM

    # Max stress at root
    y_max = h_spar / 2.0
    stress_root_Pa = M_root * y_max / I_xx
    stress_root_MPa = stress_root_Pa / 1e6

    # Tip deflection (cantilever beam, uniform EI)
    E = 70e9  # aluminium alloy / composite equivalent
    delta_tip = L_wing * semi_span ** 3 / (3.0 * E * I_xx)

    # Structural mass estimate (empirical for wing structure)
    m_struct = 25.0 * (S_ref ** 0.6) * (span ** 0.5) * (n_max ** 0.3)

    # Failure index and buckling factor
    sigma_ult = 550.0  # MPa, composite ultimate
    FI = stress_root_MPa / sigma_ult
    BF = sigma_ult / stress_root_MPa if stress_root_MPa > 0 else 99.0

    return {
        "stress_max_MPa": round(stress_root_MPa, 2),
        "delta_tip_m":    round(delta_tip, 4),
        "mass_struct_kg": round(m_struct, 1),
        "FI":             round(FI, 4),
        "BF":             round(BF, 4),
        "I_xx_m4":        round(I_xx, 8),
    }


def evaluate_stealth(sample: dict) -> Dict[str, float]:
    """
    Low-fidelity RCS evaluation.

    Heuristic model based on planform alignment, edge diffraction,
    and inlet shielding factor.
    """
    rcs_frontal_target = sample.get("rcs_frontal", -10.0)   # dBsm design intent
    align_deg          = sample.get("stealth_align_deg", 15.0)
    inlet_shield       = sample.get("inlet_shield", 0.5)
    sweep              = sample.get("wing_sweep_LE", 35.0)
    body_length        = sample.get("body_length", 7.9)
    body_width         = sample.get("body_width", 0.85)
    span               = sample.get("wing_span", 11.9)

    # ---------- design-intent baseline (the dominant term) ----------
    rcs_target_m2 = 10.0 ** (rcs_frontal_target / 10.0)  # convert dBsm → m²

    # ---------- geometric correction factors ----------
    # Edge alignment: aligned edges reflect energy away from threat
    # 45° alignment → 40% improvement over unaligned (factor < 1 = better)
    align_factor = 1.0 - 0.4 * min(align_deg / 45.0, 1.0)
    align_factor = max(0.3, min(align_factor, 1.0))

    # Sweep: higher LE sweep scatters leading-edge diffraction
    sweep_factor = 1.0 - 0.3 * max(0, sweep - 20) / 35.0
    sweep_factor = max(0.5, min(sweep_factor, 1.0))

    # Inlet shielding: 1.0 = fully buried/shielded, 0 = open cavity
    inlet_factor = 1.0 - 0.4 * inlet_shield
    inlet_factor = max(0.4, min(inlet_factor, 1.0))

    # Size penalty: larger frontal area → harder to hide
    A_frontal = body_width * sample.get("body_height", 0.6)
    size_factor = 1.0 + 0.3 * max(0, A_frontal - 0.4) / 0.6  # baseline 0.4 m²
    size_factor = max(1.0, min(size_factor, 1.5))

    # ---------- combine: target × geometric corrections ----------
    rcs_m2 = rcs_target_m2 * align_factor * sweep_factor * inlet_factor * size_factor
    rcs_m2 = max(rcs_m2, 1e-10)
    rcs_dbsm = 10.0 * math.log10(rcs_m2)

    return {
        "rcs_dbsm":       round(rcs_dbsm, 2),
        "rcs_m2":         round(rcs_m2, 6),
        "align_factor":   round(align_factor, 4),
        "sweep_factor":   round(sweep_factor, 4),
        "inlet_factor":   round(inlet_factor, 4),
    }


# ═══════════════════════════════════════════════════════════════════
#  OBJECTIVE FUNCTION
# ═══════════════════════════════════════════════════════════════════

def compute_objective(aero: dict, struct: dict, stealth: dict,
                      sample: dict) -> Dict[str, Any]:
    """
    Compute the composite objective J_norm and feasibility status.

    J_norm = W_AERO × (L/D penalty) + W_MASS × (mass penalty) + W_RCS × (RCS penalty)

    All penalties are baseline-normalised so J_norm ≈ 1.0 for the baseline design.
    Lower J_norm is better.
    """
    LD = aero.get("L_over_D", 0)
    stress = struct.get("stress_max_MPa", 999)
    rcs = stealth.get("rcs_dbsm", 0)
    gtow = sample.get("mass_GTOW", BASELINE_MASS_KG)

    # Penalties (lower is better)
    ld_penalty = BASELINE_LD / max(LD, 1.0)        # <1 if better than baseline
    mass_penalty = gtow / BASELINE_MASS_KG          # <1 if lighter
    rcs_penalty = rcs / BASELINE_RCS_DBSM           # <1 if stealthier (more negative)

    J_norm = W_AERO * ld_penalty + W_MASS * mass_penalty + W_RCS * rcs_penalty

    # Hard constraint checks
    hard_pass = True
    hard_reasons = []

    if stress > STRESS_LIMIT_MPA:
        hard_pass = False
        hard_reasons.append(f"Stress {stress:.0f} > {STRESS_LIMIT_MPA:.0f} MPa")

    if rcs > RCS_LIMIT_DBSM:
        hard_pass = False
        hard_reasons.append(f"RCS {rcs:.1f} > {RCS_LIMIT_DBSM:.1f} dBsm")

    if LD < LD_MIN:
        hard_pass = False
        hard_reasons.append(f"L/D {LD:.1f} < {LD_MIN:.1f}")

    status = "Feasible" if hard_pass else "Rejected"

    return {
        "J_norm":         round(J_norm, 6),
        "ld_penalty":     round(ld_penalty, 4),
        "mass_penalty":   round(mass_penalty, 4),
        "rcs_penalty":    round(rcs_penalty, 4),
        "Status":         status,
        "hard_pass":      hard_pass,
        "rejection_reason": "; ".join(hard_reasons) if hard_reasons else "",
    }


# ═══════════════════════════════════════════════════════════════════
#  SINGLE DESIGN EVALUATOR
# ═══════════════════════════════════════════════════════════════════

def evaluate_single_design(sample: dict) -> dict:
    """
    Evaluate a single design sample through all disciplines.

    Parameters
    ----------
    sample : dict
        A complete 42-variable CCAV design vector (with derived vars).

    Returns
    -------
    dict with keys: sample_id, all sample vars, aero results, struct results,
         stealth results, objective, status, wall_time.
    """
    t0 = time.time()

    aero    = evaluate_aero(sample)
    struct  = evaluate_structures(sample)
    stealth = evaluate_stealth(sample)
    obj     = compute_objective(aero, struct, stealth, sample)

    wall_time = time.time() - t0

    result = {
        "sample_id": sample.get("sample_id", -1),
        # Key performance metrics
        "L_over_D":       aero["L_over_D"],
        "CD_total":       aero["CD_total"],
        "stress_max_MPa": struct["stress_max_MPa"],
        "mass_struct_kg": struct["mass_struct_kg"],
        "rcs_dbsm":       stealth["rcs_dbsm"],
        "J_norm":         obj["J_norm"],
        "Status":         obj["Status"],
        "rejection_reason": obj["rejection_reason"],
        "wall_time_s":    round(wall_time, 4),
    }

    # Merge full discipline results
    for prefix, d in [("aero_", aero), ("struct_", struct), ("stealth_", stealth)]:
        for k, v in d.items():
            result[f"{prefix}{k}"] = v

    # Include key design variables for context
    for k in ["wing_span", "wing_area", "wing_AR", "body_length",
              "cruise_mach", "mass_GTOW", "mass_fuel_kg", "mass_payload_kg",
              "wing_sweep_LE", "inlet_shield", "rcs_frontal",
              "stealth_align_deg", "vtail_cant_deg", "n_max"]:
        result[k] = sample.get(k, 0)

    return result


def _setup_work_directory(base_dir: Path, sample_id: int) -> Path:
    """Create a working directory for a sample's solver files."""
    work_dir = base_dir / f"sample_{sample_id:04d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


# ═══════════════════════════════════════════════════════════════════
#  SCREENING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_screening_pipeline(
    samples: List[dict],
    verbose: bool = True,
    max_samples: Optional[int] = None,
) -> List[dict]:
    """
    Run the full screening pipeline on a list of design samples.

    Parameters
    ----------
    samples : list of dict
        Feasible design vectors from the DOE stage.
    verbose : bool
        Print progress to stdout.
    max_samples : int, optional
        Limit the number of samples to evaluate.

    Returns
    -------
    list of dict — one result per evaluated sample.
    """
    if max_samples is not None:
        samples = samples[:max_samples]

    n_total = len(samples)
    results = []
    n_feasible = 0

    if verbose:
        print(f"\n  Screening {n_total} designs...")
        print(f"  Constraints: stress < {STRESS_LIMIT_MPA} MPa, "
              f"RCS < {RCS_LIMIT_DBSM} dBsm, L/D > {LD_MIN}")

    t0 = time.time()

    for i, sample in enumerate(samples):
        sample["sample_id"] = sample.get("sample_id", i)
        result = evaluate_single_design(sample)
        results.append(result)

        if result["Status"] == "Feasible":
            n_feasible += 1

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"    [{i + 1:>4d}/{n_total}] "
                  f"{n_feasible} feasible | "
                  f"{rate:.1f} designs/s | "
                  f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Screening complete: {n_feasible}/{n_total} feasible "
              f"({elapsed:.1f}s, {n_total / elapsed:.1f} designs/s)")

    # Sort by J_norm (best first)
    results.sort(key=lambda r: r.get("J_norm", 999))

    # Add rank
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def export_screening_results(results: List[dict], path: Path) -> Path:
    """Export screening results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return path

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return path


def load_feasible_samples(csv_path: Path | None = None) -> List[dict]:
    """Load feasible samples from the DOE CSV or generate fresh ones."""
    if csv_path is not None and csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        samples = df.to_dict("records")
        return samples

    # Generate fresh DOE and extract feasible
    all_samples, feas_mask, _ = generate_doe(
        n_samples=500, seed=42, include_baseline=True, verbose=True)
    feasible = [s for s, ok in zip(all_samples, feas_mask) if ok]
    return feasible


# ═══════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 — CCAV Low-Fidelity Screening")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to feasible DOE CSV (default: generate fresh)")
    parser.add_argument("--samples", "-n", type=int, default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--top", "-t", type=int, default=20,
                        help="Show top N results (default: 20)")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                        help="Verbose output")
    args = parser.parse_args()

    print("=" * 65)
    print("  STAGE 3 — CCAV Low-Fidelity Screening")
    print("=" * 65)

    # Load samples
    csv_path = Path(args.csv) if args.csv else _REPO_ROOT / "data" / "ccav_feasible.csv"
    samples = load_feasible_samples(csv_path if csv_path.exists() else None)
    print(f"\n  Loaded {len(samples)} feasible designs")

    # Run screening
    results = run_screening_pipeline(
        samples, verbose=args.verbose, max_samples=args.samples)

    # Export
    out_path = _REPO_ROOT / "data" / "ccav_screening_results.csv"
    export_screening_results(results, out_path)
    print(f"\n  Results exported to: {out_path}")

    # Show top results
    n_feas = sum(1 for r in results if r["Status"] == "Feasible")
    print(f"\n  Summary: {n_feas}/{len(results)} passed hard constraints")
    print(f"\n  Top {min(args.top, len(results))} designs by J_norm:")
    print(f"  {'Rank':>4s}  {'ID':>4s}  {'L/D':>7s}  {'Stress':>8s}  "
          f"{'RCS':>8s}  {'J_norm':>8s}  {'Status':>10s}")
    print("  " + "-" * 60)

    for r in results[:args.top]:
        print(f"  {r['rank']:>4d}  {r['sample_id']:>4d}  "
              f"{r['L_over_D']:>7.2f}  {r['stress_max_MPa']:>8.1f}  "
              f"{r['rcs_dbsm']:>8.1f}  {r['J_norm']:>8.4f}  "
              f"{r['Status']:>10s}")

    print(f"\n{'=' * 65}")
    print(f"  STAGE 3 COMPLETE")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
