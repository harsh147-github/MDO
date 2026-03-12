"""
CCAV OpenVSP Batch Runner — Geometry + VSPAERO Pipeline
=========================================================
Reads a DOE input CSV, builds parametric OpenVSP geometry for each sample,
runs VSPAERO VLM analysis, and aggregates results into a single output CSV.

Pipeline:
    vsp_doe_vectors.csv → per-sample .vsp3 → VSPAERO VLM → vsp_batch_results.csv

Supports incremental resume: completed samples are skipped on re-run.

Usage:
    python -m pipeline.vsp_batch_runner                          # defaults
    python -m pipeline.vsp_batch_runner --input data/ccav_feasible.csv \\
        --output output/vsp_batch --max-samples 10 --ncpu 4 -v
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.vsp_aero_config import VspAeroConfig, VSP_AERO_KEYS, estimate_mac


# ═════════════════════════════════════════════════════════════════════
#  STATUS FILE I/O  (_status.json per sample directory)
# ═════════════════════════════════════════════════════════════════════

def _write_status(sample_dir: Path, status: str, error: str = "",
                  time_s: float = 0.0, aero: dict | None = None) -> None:
    """Write a JSON status file into the sample directory."""
    data = {"status": status, "error": error, "wall_time_s": round(time_s, 2)}
    if aero:
        data["aero"] = aero
    with open(sample_dir / "_status.json", "w") as f:
        json.dump(data, f, indent=2)


def _read_status(sample_dir: Path) -> dict | None:
    """Read _status.json. Returns dict or None if missing/corrupt."""
    path = sample_dir / "_status.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ═════════════════════════════════════════════════════════════════════
#  CSV READER  (DOE input)
# ═════════════════════════════════════════════════════════════════════

def load_doe_csv(csv_path: Path) -> list[dict]:
    """
    Load the DOE CSV and return a list of sample dicts.
    Each dict has string keys and float values for all numeric columns,
    plus 'sample_id' (int).
    """
    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = {}
            for k, v in row.items():
                if k == "sample_id":
                    sample[k] = int(v)
                elif k in ("is_feasible",):
                    sample[k] = int(float(v))
                elif k == "rejection_reason":
                    sample[k] = v
                else:
                    try:
                        sample[k] = float(v)
                    except (ValueError, TypeError):
                        sample[k] = v
            samples.append(sample)
    return samples


# ═════════════════════════════════════════════════════════════════════
#  SINGLE-SAMPLE PROCESSING
# ═════════════════════════════════════════════════════════════════════

def process_single_sample(
    sample: dict,
    sample_dir: Path,
    config: VspAeroConfig,
    verbose: bool = True,
) -> dict | None:
    """
    Build geometry → compute DegenGeom → run VSPAERO for one sample.

    Returns a result dict with aero coefficients, or None on failure.
    Each call clears the OpenVSP model (singleton) before building.
    """
    import openvsp as vsp
    from pipeline.vsp_geometry import build_ccav_model

    sid = sample.get("sample_id", 0)
    sample_dir.mkdir(parents=True, exist_ok=True)
    vsp3_path = sample_dir / f"sample_{sid:04d}.vsp3"

    t0 = time.time()
    _write_status(sample_dir, "running")

    # ── Step 1: Build geometry ────────────────────────────────────
    try:
        build_ccav_model(sample, vsp3_path, verbose=False)
    except Exception as exc:
        elapsed = time.time() - t0
        err_msg = f"Geometry build error: {exc}"
        _write_status(sample_dir, "failed", error=err_msg, time_s=elapsed)
        if verbose:
            print(f"    FAILED (geometry): {exc}")
        return None

    # ── Step 2: Find reference wing ID ────────────────────────────
    wing_ids = vsp.FindGeomsWithName("CCAV_Wing")
    if not wing_ids:
        elapsed = time.time() - t0
        _write_status(sample_dir, "failed",
                      error="Could not find CCAV_Wing after build", time_s=elapsed)
        if verbose:
            print("    FAILED: CCAV_Wing not found")
        return None
    wing_id = wing_ids[0]

    # ── Step 3: Compute VSPAERO geometry (DegenGeom) ──────────────
    try:
        vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
        vsp.ExecAnalysis("VSPAEROComputeGeometry")
    except Exception as exc:
        elapsed = time.time() - t0
        _write_status(sample_dir, "failed",
                      error=f"DegenGeom error: {exc}", time_s=elapsed)
        if verbose:
            print(f"    FAILED (DegenGeom): {exc}")
        return None

    # ── Step 4: Configure VSPAERO sweep ───────────────────────────
    analysis = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(analysis)

    # Reference quantities from sample
    sref = sample.get("wing_area", 66.0)
    bref = sample.get("wing_span", 11.9)
    cref = estimate_mac(
        sample.get("wing_root_chord", 4.5),
        sample.get("wing_tip_chord", 1.5),
    )
    mach = sample.get("cruise_mach", 0.9)

    vsp.SetDoubleAnalysisInput(analysis, "Sref", [sref])
    vsp.SetDoubleAnalysisInput(analysis, "bref", [bref])
    vsp.SetDoubleAnalysisInput(analysis, "cref", [cref])

    # Mach — single point at cruise Mach
    vsp.SetDoubleAnalysisInput(analysis, "MachStart", [mach])
    vsp.SetDoubleAnalysisInput(analysis, "MachEnd", [mach])
    vsp.SetIntAnalysisInput(analysis, "MachNpts", [1])

    # Alpha sweep
    vsp.SetDoubleAnalysisInput(analysis, "AlphaStart", [config.alpha_start])
    vsp.SetDoubleAnalysisInput(analysis, "AlphaEnd", [config.alpha_end])
    vsp.SetIntAnalysisInput(analysis, "AlphaNpts", [config.alpha_npts])

    # Beta — zero sideslip
    vsp.SetDoubleAnalysisInput(analysis, "BetaStart", [config.beta_start])
    vsp.SetDoubleAnalysisInput(analysis, "BetaEnd", [config.beta_end])
    vsp.SetIntAnalysisInput(analysis, "BetaNpts", [config.beta_npts])

    # Solver settings
    vsp.SetIntAnalysisInput(analysis, "NCPU", [config.ncpu])
    vsp.SetIntAnalysisInput(analysis, "WakeNumIter", [config.wake_iters])

    # Reference wing for stability derivatives
    vsp.SetStringAnalysisInput(analysis, "WingID", [wing_id])

    # ── Step 5: Execute VSPAERO ───────────────────────────────────
    try:
        res_id = vsp.ExecAnalysis(analysis)
    except Exception as exc:
        elapsed = time.time() - t0
        _write_status(sample_dir, "failed",
                      error=f"VSPAERO exec error: {exc}", time_s=elapsed)
        if verbose:
            print(f"    FAILED (VSPAERO): {exc}")
        return None

    elapsed = time.time() - t0

    # ── Step 6: Parse results ─────────────────────────────────────
    aero = _extract_results_api(vsp, res_id)
    if aero is None:
        # Fallback: parse .polar file
        polar_path = vsp3_path.with_suffix(".polar")
        aero = _parse_polar_file(polar_path)

    if aero is None:
        _write_status(sample_dir, "failed",
                      error="No results from VSPAERO (API + polar fallback)",
                      time_s=elapsed)
        if verbose:
            print("    FAILED: no VSPAERO results")
        return None

    # ── Step 7: Sanity check ──────────────────────────────────────
    cl_vals = aero.get("CLtot", [])
    if any(abs(cl) > config.cl_sanity_limit for cl in cl_vals):
        _write_status(sample_dir, "failed",
                      error=f"CL sanity check failed (max |CL|={max(abs(c) for c in cl_vals):.1f})",
                      time_s=elapsed)
        if verbose:
            print(f"    FAILED: CL sanity ({max(abs(c) for c in cl_vals):.1f} > {config.cl_sanity_limit})")
        return None

    # ── Step 8: Extract summary metrics ───────────────────────────
    summary = _compute_summary(aero, config.cruise_alpha_deg)
    summary["sample_id"] = sid
    summary["wall_time_s"] = round(elapsed, 2)
    summary["status"] = "complete"

    _write_status(sample_dir, "complete", time_s=elapsed, aero=summary)

    if verbose:
        ld = summary.get("vsp_LD_at_cruise", 0)
        cl = summary.get("vsp_CL_at_cruise", 0)
        cd = summary.get("vsp_CD_at_cruise", 0)
        print(f"    CL={cl:.4f}  CD={cd:.5f}  L/D={ld:.2f}  ({elapsed:.1f}s)")

    return summary


# ═════════════════════════════════════════════════════════════════════
#  RESULT EXTRACTION
# ═════════════════════════════════════════════════════════════════════

def _extract_results_api(vsp, res_id: str) -> dict | None:
    """Extract polar data from VSPAERO results via the OpenVSP API."""
    if not res_id:
        return None

    top_names = vsp.GetAllDataNames(res_id)
    if "ResultsVec" not in top_names:
        return None

    sub_ids = vsp.GetStringResults(res_id, "ResultsVec")
    for sid in sub_ids:
        rname = vsp.GetResultsName(sid)
        if rname == "VSPAERO_Polar":
            names = vsp.GetAllDataNames(sid)
            result = {}
            for field in ["Alpha", "CLtot", "CDtot", "CDi", "CDo", "L/D", "E",
                          "CMytot", "CMxtot", "CMztot", "CStot", "FC_Mach"]:
                if field in names:
                    result[field] = list(vsp.GetDoubleResults(sid, field))
            if "CLtot" in result and len(result["CLtot"]) > 0:
                return result

    return None


def _parse_polar_file(polar_path: Path) -> dict | None:
    """
    Parse a VSPAERO .polar file (whitespace-delimited table).
    Returns dict of {column_name: [values]} or None if file missing/empty.
    """
    if not polar_path.exists():
        return None

    try:
        with open(polar_path) as f:
            lines = f.readlines()

        # Find the header line (contains 'AoA' and 'CLtot')
        header_idx = None
        for i, line in enumerate(lines):
            if "AoA" in line and "CLtot" in line:
                header_idx = i
                break
        if header_idx is None:
            return None

        headers = lines[header_idx].split()
        data_lines = lines[header_idx + 1:]

        result = {h: [] for h in headers}
        for line in data_lines:
            vals = line.split()
            if len(vals) != len(headers):
                continue
            for h, v in zip(headers, vals):
                try:
                    result[h].append(float(v))
                except ValueError:
                    result[h].append(float("nan"))

        # Rename to match API field names
        rename = {"AoA": "Alpha", "L/D": "L/D"}
        for old, new in rename.items():
            if old in result and old != new:
                result[new] = result.pop(old)

        if "CLtot" in result and len(result["CLtot"]) > 0:
            return result

    except Exception:
        pass

    return None


def _compute_summary(aero: dict, cruise_alpha: float) -> dict:
    """
    Compute summary metrics from the full polar sweep.

    Extracts values at cruise alpha (interpolated), plus
    CL_max, LD_max, CD_min, CLa (lift-curve slope).
    """
    alphas = np.array(aero.get("Alpha", [0]))
    cls = np.array(aero.get("CLtot", [0]))
    cds = np.array(aero.get("CDtot", [1]))
    lds = np.where(cds > 1e-8, cls / cds, 0.0)
    cmys = np.array(aero.get("CMytot", [0] * len(alphas)))

    summary = {}

    # ── Interpolated values at cruise alpha ───────────────────────
    if len(alphas) >= 2:
        summary["vsp_CL_at_cruise"] = float(np.interp(cruise_alpha, alphas, cls))
        summary["vsp_CD_at_cruise"] = float(np.interp(cruise_alpha, alphas, cds))
        summary["vsp_CMy_at_cruise"] = float(np.interp(cruise_alpha, alphas, cmys))
        cd_cruise = summary["vsp_CD_at_cruise"]
        summary["vsp_LD_at_cruise"] = (
            summary["vsp_CL_at_cruise"] / cd_cruise if cd_cruise > 1e-8 else 0.0
        )
    elif len(alphas) == 1:
        summary["vsp_CL_at_cruise"] = float(cls[0])
        summary["vsp_CD_at_cruise"] = float(cds[0])
        summary["vsp_CMy_at_cruise"] = float(cmys[0])
        summary["vsp_LD_at_cruise"] = float(lds[0])
    else:
        summary["vsp_CL_at_cruise"] = float("nan")
        summary["vsp_CD_at_cruise"] = float("nan")
        summary["vsp_CMy_at_cruise"] = float("nan")
        summary["vsp_LD_at_cruise"] = float("nan")

    # ── Aggregate metrics ─────────────────────────────────────────
    summary["vsp_CL_max"] = float(np.max(cls)) if len(cls) > 0 else float("nan")
    summary["vsp_CD_min"] = float(np.min(cds)) if len(cds) > 0 else float("nan")
    summary["vsp_LD_max"] = float(np.max(lds)) if len(lds) > 0 else float("nan")

    # Lift-curve slope (dCL/dalpha in per-degree)
    if len(alphas) >= 2:
        # Use linear regression over the linear region (alpha < 8 deg)
        mask = alphas <= 8.0
        if np.sum(mask) >= 2:
            a_lin = alphas[mask]
            cl_lin = cls[mask]
            cla, _ = np.polyfit(a_lin, cl_lin, 1)
            summary["vsp_CLa_per_deg"] = float(cla)
        else:
            summary["vsp_CLa_per_deg"] = float("nan")
    else:
        summary["vsp_CLa_per_deg"] = float("nan")

    return summary


# ═════════════════════════════════════════════════════════════════════
#  AGGREGATED RESULTS CSV
# ═════════════════════════════════════════════════════════════════════

RESULT_FIELDS = [
    "vsp_CL_at_cruise", "vsp_CD_at_cruise", "vsp_LD_at_cruise",
    "vsp_CMy_at_cruise", "vsp_CL_max", "vsp_CD_min", "vsp_LD_max",
    "vsp_CLa_per_deg", "wall_time_s", "status",
]


def _write_aggregated_csv(
    samples: list[dict],
    results: list[dict | None],
    output_path: Path,
) -> Path:
    """Write the aggregated results CSV (one row per sample)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all DOE keys from the first sample
    doe_keys = sorted(
        k for k in samples[0].keys()
        if k not in ("sample_id", "is_feasible", "rejection_reason")
    )
    fieldnames = ["sample_id"] + doe_keys + RESULT_FIELDS

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for sample, result in zip(samples, results):
            row = {"sample_id": sample.get("sample_id", 0)}
            # DOE variables
            for k in doe_keys:
                row[k] = sample.get(k, "")
            # VSPAERO results
            if result is not None:
                for k in RESULT_FIELDS:
                    row[k] = result.get(k, "")
            else:
                row["status"] = "failed"
                for k in RESULT_FIELDS:
                    if k != "status":
                        row[k] = ""

            writer.writerow(row)

    return output_path


# ═════════════════════════════════════════════════════════════════════
#  BATCH ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════

def run_batch(
    input_csv: Path,
    output_dir: Path,
    config: VspAeroConfig | None = None,
    max_samples: int | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Run the full batch pipeline.

    1. Read DOE CSV
    2. For each sample: build geometry → VSPAERO → extract results
    3. Write aggregated results CSV

    Parameters
    ----------
    input_csv : Path
        DOE input CSV (e.g. data/ccav_feasible.csv).
    output_dir : Path
        Output directory for per-sample files and aggregated CSV.
    config : VspAeroConfig
        Analysis settings. Defaults if None.
    max_samples : int
        Limit number of samples to process (for testing).
    resume : bool
        Skip samples that already have status="complete".
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    Path to the aggregated results CSV.
    """
    if config is None:
        config = VspAeroConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load input CSV ────────────────────────────────────────────
    if verbose:
        print("=" * 72)
        print("  CCAV OpenVSP Batch Runner")
        print("=" * 72)
        print(f"  Input:   {input_csv}")
        print(f"  Output:  {output_dir}")
        print(f"  Config:  alpha=[{config.alpha_start}, {config.alpha_end}] "
              f"x{config.alpha_npts}, ncpu={config.ncpu}")

    samples = load_doe_csv(input_csv)
    total = len(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
        total = len(samples)

    if verbose:
        print(f"  Samples: {total} loaded" +
              (f" (capped from {total})" if max_samples else ""))
        print()

    # ── Resume scan ───────────────────────────────────────────────
    completed_cache: dict[int, dict] = {}
    skipped = 0
    if resume:
        for sample in samples:
            sid = sample.get("sample_id", 0)
            sample_dir = output_dir / f"sample_{sid:04d}"
            status = _read_status(sample_dir)
            if status and status.get("status") == "complete":
                completed_cache[sid] = status.get("aero", {})
                skipped += 1
        if verbose and skipped > 0:
            print(f"  Resume: {skipped}/{total} already complete, skipping.\n")

    # ── Process each sample ───────────────────────────────────────
    results: list[dict | None] = []
    n_ok = 0
    n_fail = 0
    t_batch_start = time.time()

    for idx, sample in enumerate(samples):
        sid = sample.get("sample_id", 0)
        sample_dir = output_dir / f"sample_{sid:04d}"

        # Check resume cache
        if sid in completed_cache:
            results.append(completed_cache[sid])
            n_ok += 1
            if verbose:
                print(f"  [{idx+1:>4d}/{total}]  sample_{sid:04d}  "
                      f"SKIPPED (cached)")
            continue

        if verbose:
            print(f"  [{idx+1:>4d}/{total}]  sample_{sid:04d}  ", end="", flush=True)

        result = process_single_sample(sample, sample_dir, config, verbose=False)

        if result is not None:
            results.append(result)
            n_ok += 1
            if verbose:
                ld = result.get("vsp_LD_at_cruise", 0)
                cl = result.get("vsp_CL_at_cruise", 0)
                cd = result.get("vsp_CD_at_cruise", 0)
                wt = result.get("wall_time_s", 0)
                print(f"COMPLETE   CL={cl:.4f}  CD={cd:.5f}  "
                      f"L/D={ld:.2f}   ({wt:.1f}s)")
        else:
            results.append(None)
            n_fail += 1
            if verbose:
                status = _read_status(sample_dir)
                err = status.get("error", "unknown") if status else "unknown"
                print(f"FAILED     {err}")

        # Progress ETA
        if verbose and (idx + 1) % 5 == 0:
            elapsed_batch = time.time() - t_batch_start
            processed = idx + 1 - skipped
            if processed > 0:
                avg_s = elapsed_batch / processed
                remaining = total - (idx + 1)
                eta_s = avg_s * remaining
                eta_m = eta_s / 60
                print(f"  --- progress: {idx+1}/{total} | "
                      f"{n_ok} ok, {n_fail} failed | "
                      f"ETA: {eta_m:.0f}m ---")

    # ── Write aggregated CSV ──────────────────────────────────────
    results_csv = output_dir / "vsp_batch_results.csv"
    _write_aggregated_csv(samples, results, results_csv)

    elapsed_total = time.time() - t_batch_start

    if verbose:
        print()
        print("=" * 72)
        print(f"  BATCH COMPLETE")
        print(f"    Processed: {total} samples in {elapsed_total:.0f}s "
              f"({elapsed_total/60:.1f}m)")
        print(f"    Success:   {n_ok}")
        print(f"    Failed:    {n_fail}")
        print(f"    Results:   {results_csv}")
        print("=" * 72)

    return results_csv


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CCAV OpenVSP Batch Runner — geometry + VSPAERO pipeline")

    parser.add_argument("--input", "-i", type=Path,
                        default=_REPO_ROOT / "data" / "ccav_feasible.csv",
                        help="DOE input CSV (default: data/ccav_feasible.csv)")
    parser.add_argument("--output", "-o", type=Path,
                        default=_REPO_ROOT / "output" / "vsp_batch",
                        help="Output directory (default: output/vsp_batch)")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Max samples to process (default: all)")

    # Analysis settings
    parser.add_argument("--alpha-start", type=float, default=-2.0)
    parser.add_argument("--alpha-end", type=float, default=10.0)
    parser.add_argument("--alpha-npts", type=int, default=7)
    parser.add_argument("--ncpu", type=int, default=4)
    parser.add_argument("--wake-iters", type=int, default=5)
    parser.add_argument("--cruise-alpha", type=float, default=3.0,
                        help="Reference alpha for single-point extraction (deg)")

    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run all samples even if previously completed")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    config = VspAeroConfig(
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_npts=args.alpha_npts,
        ncpu=args.ncpu,
        wake_iters=args.wake_iters,
        cruise_alpha_deg=args.cruise_alpha,
    )

    run_batch(
        input_csv=args.input,
        output_dir=args.output,
        config=config,
        max_samples=args.max_samples,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
