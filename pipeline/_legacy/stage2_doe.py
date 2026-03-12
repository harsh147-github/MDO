"""
Stage 2 — Design of Experiments (DOE) Generation & Pre-Filtering
=================================================================
Generates a space-filling Latin Hypercube sample across the 28 independent
design variables, computes the 6 derived variables for each point, then
applies the Stage-1 physics pre-filter to reject infeasible designs.

Run standalone:
    python -m pipeline.stage2_doe                   (from repo root)
    python -m pipeline.stage2_doe --samples 1000    (custom count)

Pipeline flow:
    1. Reads 28 independent bounds from Stage 1  (get_independent_bounds)
    2. Generates N samples via LHS in [0,1]^28, maximin-optimised
    3. Scales each sample to physical bounds
    4. Computes 6 derived variables             (compute_derived)
    5. Runs 9 cross-variable physics checks     (validate_sample)
    6. Separates feasible / infeasible samples
    7. Stores ALL samples in SQLite (doe_samples table) + CSV export
    8. Reports statistics: pass/fail ratio, variable coverage, etc.

Downstream API:
    generate_doe(n)         → (all_samples, feasible_mask)
    get_feasible_samples()  → list of dicts from DB (for Stage 3)
    load_doe_csv(path)      → reload from disk
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.db_schema import get_connection, init_db
from pipeline.stage1_design_space import (
    get_independent_bounds,
    get_baseline_vector,
    compute_derived,
    validate_sample,
    get_derived_keys,
)
from pipeline.design_vector import DesignVector


# ═══════════════════════════════════════════════════════════════════════
#  LHS SAMPLING ENGINE
# ═══════════════════════════════════════════════════════════════════════

def _latin_hypercube(n_samples: int, n_dims: int,
                     seed: int = 42,
                     optimise: str = "maximin") -> np.ndarray:
    """
    Generate an n_samples × n_dims LHS design in [0,1]^d.

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    n_dims : int
        Number of dimensions (= number of independent variables).
    seed : int
        RNG seed for reproducibility.
    optimise : str
        "maximin" for scipy's optimised LHS (better space-filling).
        "basic" for plain random LHS (faster, slightly worse coverage).

    Returns
    -------
    np.ndarray of shape (n_samples, n_dims), values in [0, 1].
    """
    from scipy.stats.qmc import LatinHypercube

    if optimise == "maximin":
        # strength=1 + optimization='random-cd' gives good maximin designs
        sampler = LatinHypercube(d=n_dims, seed=seed, optimization="random-cd")
    else:
        sampler = LatinHypercube(d=n_dims, seed=seed)

    return sampler.random(n=n_samples)


def _scale_to_physical(unit_samples: np.ndarray,
                       keys: List[str],
                       bounds: Dict[str, Tuple[float, float, float]]
                       ) -> List[dict]:
    """
    Scale [0,1]^d samples to physical variable ranges.

    Parameters
    ----------
    unit_samples : ndarray (N, d)
    keys : list of python_key names, same order as columns
    bounds : {python_key: (lower, baseline, upper)}

    Returns
    -------
    List of N dicts, each mapping python_key → physical value.
    """
    lowers = np.array([bounds[k][0] for k in keys])
    uppers = np.array([bounds[k][2] for k in keys])
    spans = uppers - lowers

    samples = []
    for row in unit_samples:
        physical = lowers + row * spans
        sample = {k: float(v) for k, v in zip(keys, physical)}
        samples.append(sample)
    return samples


# ═══════════════════════════════════════════════════════════════════════
#  FULL DOE GENERATION + FILTERING
# ═══════════════════════════════════════════════════════════════════════

def generate_doe(
    n_samples: int = 500,
    seed: int = 42,
    optimise: str = "maximin",
    include_baseline: bool = True,
    verbose: bool = True,
) -> Tuple[List[dict], List[bool], List[List[str]]]:
    """
    Generate a full DOE table with derived variables and feasibility flags.

    Parameters
    ----------
    n_samples : int
        Number of LHS samples to generate (before filtering).
    seed : int
        Random seed for reproducibility.
    optimise : str
        LHS optimisation strategy ("maximin" or "basic").
    include_baseline : bool
        If True, inject the baseline design as sample #0.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    all_samples : list of dict
        Every sample (34 variables each), including infeasible ones.
    feasible_mask : list of bool
        True if the sample passes all physics checks.
    failure_reasons : list of list[str]
        For each sample, the list of failed checks (empty if feasible).
    """
    # 1. Get bounds from Stage 1
    indep_bounds = get_independent_bounds()
    keys = sorted(indep_bounds.keys())
    n_dims = len(keys)

    if verbose:
        print(f"\n  Sampling {n_samples} points in {n_dims}-dimensional space...")

    # 2. LHS in unit hypercube
    t0 = time.time()
    unit_lhs = _latin_hypercube(n_samples, n_dims, seed=seed, optimise=optimise)
    t_lhs = time.time() - t0

    if verbose:
        print(f"  LHS generated in {t_lhs:.2f}s  (optimisation: {optimise})")

    # 3. Scale to physical bounds
    raw_samples = _scale_to_physical(unit_lhs, keys, indep_bounds)

    # 4. Optionally prepend baseline
    if include_baseline:
        baseline = get_baseline_vector()
        # Only keep independent keys
        baseline_indep = {k: baseline[k] for k in keys}
        raw_samples.insert(0, baseline_indep)
        if verbose:
            print(f"  Baseline injected as sample #0")

    # 5. Compute derived variables + validate each sample
    all_samples = []
    feasible_mask = []
    failure_reasons = []
    n_feasible = 0
    n_total = len(raw_samples)

    if verbose:
        print(f"  Computing derived vars + physics checks for {n_total} samples...")

    for i, raw in enumerate(raw_samples):
        # Compute 6 derived variables
        full = compute_derived(raw)
        all_samples.append(full)

        # Physics pre-filter
        ok, reasons = validate_sample(full)
        feasible_mask.append(ok)
        failure_reasons.append(reasons)
        if ok:
            n_feasible += 1

        # Progress indicator
        if verbose and (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{n_total} processed  "
                  f"({n_feasible} feasible so far)")

    if verbose:
        pct = 100.0 * n_feasible / n_total if n_total > 0 else 0
        print(f"\n  RESULT: {n_feasible}/{n_total} feasible ({pct:.1f}%)")

    return all_samples, feasible_mask, failure_reasons


# ═══════════════════════════════════════════════════════════════════════
#  DATABASE + CSV STORAGE
# ═══════════════════════════════════════════════════════════════════════

def store_doe(
    all_samples: List[dict],
    feasible_mask: List[bool],
    failure_reasons: List[List[str]],
    run_id: Optional[int] = None,
) -> int:
    """
    Write all samples to the SQLite database and return the run_id.

    The doe_samples table stores the full 34-var vector as JSON plus
    a feasibility flag and failure reasons.
    """
    init_db()
    conn = get_connection()

    # Ensure doe_samples has the extra columns we need
    # (add them if they don't exist — safe ALTER TABLE)
    _ensure_doe_columns(conn)

    # Create a run record
    if run_id is None:
        cur = conn.execute(
            "INSERT INTO runs (status, current_stage, total_samples, notes) "
            "VALUES ('completed', 'stage2_doe', ?, 'LHS DOE generation')",
            (len(all_samples),)
        )
        run_id = cur.lastrowid

    # Insert each sample
    for i, (sample, ok, reasons) in enumerate(
            zip(all_samples, feasible_mask, failure_reasons)):
        conn.execute(
            "INSERT INTO doe_samples "
            "(sample_id, run_id, vector_json, is_feasible, failure_reasons) "
            "VALUES (?, ?, ?, ?, ?)",
            (i, run_id, json.dumps(sample), int(ok),
             json.dumps(reasons) if reasons else None)
        )

    conn.commit()
    conn.close()
    return run_id


def _ensure_doe_columns(conn):
    """Add is_feasible and failure_reasons columns if they don't exist."""
    existing = {row[1] for row in
                conn.execute("PRAGMA table_info(doe_samples)").fetchall()}
    if "is_feasible" not in existing:
        conn.execute(
            "ALTER TABLE doe_samples ADD COLUMN is_feasible INTEGER DEFAULT 1")
    if "failure_reasons" not in existing:
        conn.execute(
            "ALTER TABLE doe_samples ADD COLUMN failure_reasons TEXT")


def export_csv(
    all_samples: List[dict],
    feasible_mask: List[bool],
    failure_reasons: List[List[str]],
    path: Path,
    feasible_only: bool = False,
) -> Path:
    """
    Export DOE samples to a CSV file.

    Parameters
    ----------
    all_samples, feasible_mask, failure_reasons : from generate_doe()
    path : output CSV path
    feasible_only : if True, only write feasible samples

    Returns
    -------
    Path to the written CSV file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine all column keys from first sample
    if not all_samples:
        return path
    all_keys = sorted(all_samples[0].keys())
    fieldnames = ["sample_id", "is_feasible"] + all_keys + ["failure_reasons"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (sample, ok, reasons) in enumerate(
                zip(all_samples, feasible_mask, failure_reasons)):
            if feasible_only and not ok:
                continue
            row = {"sample_id": i, "is_feasible": int(ok)}
            row.update(sample)
            row["failure_reasons"] = "; ".join(reasons) if reasons else ""
            writer.writerow(row)

    return path


# ═══════════════════════════════════════════════════════════════════════
#  QUERY API FOR DOWNSTREAM STAGES
# ═══════════════════════════════════════════════════════════════════════

def get_feasible_samples(run_id: Optional[int] = None) -> List[dict]:
    """
    Retrieve all feasible DOE samples from the database.

    Returns
    -------
    List of dicts — each is a full 34-variable design vector.
    """
    conn = get_connection()
    if run_id is not None:
        rows = conn.execute(
            "SELECT sample_id, vector_json FROM doe_samples "
            "WHERE run_id = ? AND is_feasible = 1 ORDER BY sample_id",
            (run_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT sample_id, vector_json FROM doe_samples "
            "WHERE is_feasible = 1 ORDER BY sample_id"
        ).fetchall()
    conn.close()

    samples = []
    for sid, vjson in rows:
        d = json.loads(vjson)
        d["_sample_id"] = sid
        samples.append(d)
    return samples


def get_all_samples(run_id: Optional[int] = None) -> List[dict]:
    """Retrieve ALL DOE samples (feasible + infeasible) from the database."""
    conn = get_connection()
    if run_id is not None:
        rows = conn.execute(
            "SELECT sample_id, vector_json, is_feasible FROM doe_samples "
            "WHERE run_id = ? ORDER BY sample_id", (run_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT sample_id, vector_json, is_feasible FROM doe_samples "
            "ORDER BY sample_id"
        ).fetchall()
    conn.close()

    samples = []
    for sid, vjson, feas in rows:
        d = json.loads(vjson)
        d["_sample_id"] = sid
        d["_is_feasible"] = bool(feas)
        samples.append(d)
    return samples


# ═══════════════════════════════════════════════════════════════════════
#  STATISTICS & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

def print_filter_statistics(
    all_samples: List[dict],
    feasible_mask: List[bool],
    failure_reasons: List[List[str]],
):
    """
    Print a detailed breakdown of which physics checks pass/fail,
    variable range coverage, and DOE quality metrics.
    """
    n_total = len(all_samples)
    n_feas = sum(feasible_mask)
    n_infeas = n_total - n_feas

    print("\n" + "=" * 65)
    print("  STAGE 2 — DOE FILTER STATISTICS")
    print("=" * 65)

    # ── Overall pass/fail ─────────────────────────────────────────
    print(f"\n  Total samples:     {n_total}")
    print(f"  Feasible:          {n_feas}  ({100*n_feas/n_total:.1f}%)")
    print(f"  Infeasible:        {n_infeas}  ({100*n_infeas/n_total:.1f}%)")

    # ── Failure reason breakdown ──────────────────────────────────
    # Group by check category (strip numeric values to aggregate)
    import re
    reason_counts: Dict[str, int] = {}
    _CATEGORY_PATTERNS = [
        (re.compile(r"Inverted taper"), "Inverted taper"),
        (re.compile(r"Taper ratio"), "Taper ratio out of range"),
        (re.compile(r"Aspect ratio"), "Aspect ratio out of range"),
        (re.compile(r"GTOW"), "GTOW ≤ empty weight"),
        (re.compile(r"Cruise thrust"), "Cruise thrust >= max thrust"),
        (re.compile(r"Fuel volume"), "Fuel volume tight"),
        (re.compile(r"Wing loading"), "Wing loading out of range"),
        (re.compile(r"Cma"), "Cma stability violation"),
        (re.compile(r"Fineness"), "Fuselage fineness ratio"),
    ]

    for reasons in failure_reasons:
        for r in reasons:
            categorised = False
            for pat, cat in _CATEGORY_PATTERNS:
                if pat.search(r):
                    reason_counts[cat] = reason_counts.get(cat, 0) + 1
                    categorised = True
                    break
            if not categorised:
                tag = r.split(":")[0].strip() if ":" in r else r.strip()
                reason_counts[tag] = reason_counts.get(tag, 0) + 1

    if reason_counts:
        print(f"\n  Failure breakdown (a sample can fail multiple checks):")
        for tag, count in sorted(reason_counts.items(),
                                  key=lambda x: -x[1]):
            pct = 100.0 * count / n_total
            bar = "█" * int(pct / 2)
            print(f"    {tag:<35s}  {count:>5d}  ({pct:5.1f}%)  {bar}")

    # ── Variable coverage in feasible set ─────────────────────────
    if n_feas > 0:
        feas_samples = [s for s, ok in zip(all_samples, feasible_mask) if ok]
        keys = sorted(feas_samples[0].keys())

        print(f"\n  Variable ranges in FEASIBLE set ({n_feas} samples):")
        print(f"    {'Variable':<25s}  {'Min':>12s}  {'Max':>12s}  "
              f"{'Mean':>12s}  {'Std':>12s}")
        print("    " + "-" * 75)

        indep_bounds = get_independent_bounds()
        for k in keys:
            vals = [s[k] for s in feas_samples]
            lo, hi, mu, std = min(vals), max(vals), np.mean(vals), np.std(vals)
            # Only show independent + derived (skip internal keys)
            if k.startswith("_"):
                continue
            if abs(mu) < 0.001 and mu != 0:
                print(f"    {k:<25s}  {lo:>12.4e}  {hi:>12.4e}  "
                      f"{mu:>12.4e}  {std:>12.4e}")
            else:
                print(f"    {k:<25s}  {lo:>12.4f}  {hi:>12.4f}  "
                      f"{mu:>12.4f}  {std:>12.4f}")

        # ── Coverage ratio (how much of the original bounds is explored)
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


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 — Generate DOE samples via LHS + physics pre-filter")
    parser.add_argument("--samples", "-n", type=int, default=500,
                        help="Number of LHS samples (default: 500)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Don't inject baseline as sample #0")
    parser.add_argument("--csv-only", action="store_true",
                        help="Export CSV but don't write to database")
    args = parser.parse_args()

    print("=" * 65)
    print("  STAGE 2 — Design of Experiments (DOE) Generation")
    print("=" * 65)

    # ── Generate ──────────────────────────────────────────────────
    t0 = time.time()
    all_samples, feasible_mask, failure_reasons = generate_doe(
        n_samples=args.samples,
        seed=args.seed,
        include_baseline=not args.no_baseline,
        verbose=True,
    )
    t_total = time.time() - t0

    # ── Store ─────────────────────────────────────────────────────
    out_dir = _REPO_ROOT / "data"
    out_dir.mkdir(exist_ok=True)

    if not args.csv_only:
        # Clear previous DOE data
        conn = get_connection()
        conn.execute("DELETE FROM doe_samples")
        conn.commit()
        conn.close()

        run_id = store_doe(all_samples, feasible_mask, failure_reasons)
        print(f"\n  Stored in database (run_id={run_id})")
    else:
        run_id = None

    # CSV — all samples
    csv_all = export_csv(all_samples, feasible_mask, failure_reasons,
                         out_dir / "doe_all_samples.csv",
                         feasible_only=False)
    print(f"  CSV (all):      {csv_all}")

    # CSV — feasible only
    csv_feas = export_csv(all_samples, feasible_mask, failure_reasons,
                          out_dir / "doe_feasible_samples.csv",
                          feasible_only=True)
    n_feas = sum(feasible_mask)
    print(f"  CSV (feasible): {csv_feas}  ({n_feas} rows)")

    # ── Statistics ────────────────────────────────────────────────
    print_filter_statistics(all_samples, feasible_mask, failure_reasons)

    # ── Summary ───────────────────────────────────────────────────
    n_total = len(all_samples)
    print("=" * 65)
    print(f"  STAGE 2 COMPLETE  ({t_total:.1f}s)")
    print(f"    {n_total} total samples generated (LHS, seed={args.seed})")
    print(f"    {n_feas} feasible  ({100*n_feas/n_total:.1f}%)")
    print(f"    {n_total - n_feas} rejected by physics pre-filter")
    print(f"    Ready for Stage 3 (low-fi screening)")
    print("=" * 65)


if __name__ == "__main__":
    main()
