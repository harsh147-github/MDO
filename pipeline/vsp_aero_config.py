"""
VSPAERO Batch Analysis Configuration
=====================================
Dataclass for analysis settings and constants that define which
design-vector keys are geometry vs. flight-condition parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VspAeroConfig:
    """Analysis settings for VSPAERO batch sweep runs."""

    # ── Alpha sweep ──────────────────────────────────────────────
    alpha_start: float = -2.0
    alpha_end: float = 10.0
    alpha_npts: int = 7

    # ── Beta ─────────────────────────────────────────────────────
    beta_start: float = 0.0
    beta_end: float = 0.0
    beta_npts: int = 1

    # ── Solver controls ──────────────────────────────────────────
    ncpu: int = 4
    wake_iters: int = 5
    num_wake_nodes: int = 8

    # ── Freestream defaults (overridden per-sample where needed) ─
    re_cref: float = 1e7
    vinf: float = 100.0

    # ── Robustness ───────────────────────────────────────────────
    max_wall_time_s: float = 120.0
    cl_sanity_limit: float = 50.0

    # ── Reference alpha for single-point extraction ──────────────
    cruise_alpha_deg: float = 3.0


# ═════════════════════════════════════════════════════════════════════
#  DESIGN-VECTOR KEY GROUPS
# ═════════════════════════════════════════════════════════════════════

# Keys consumed by vsp_geometry.py to build the OpenVSP model
VSP_GEOMETRY_KEYS: list[str] = [
    # Wing (10)
    "wing_span", "wing_root_chord", "wing_tip_chord", "wing_sweep_LE",
    "wing_dihedral", "wing_twist_root", "wing_twist_tip", "wing_kink_eta",
    "wing_tc_root", "wing_tc_tip",
    # Body (5)
    "body_length", "body_width", "body_height",
    "body_nose_fineness", "body_tail_fineness",
    # V-Tail (6)
    "vtail_cant_deg", "vtail_span_frac", "vtail_root_chord_frac",
    "vtail_sweep", "vtail_taper", "vtail_tc",
    # Inlet (3 geometric + 1 non-geometric kept for traceability)
    "inlet_width", "inlet_height", "inlet_x_frac",
]

# Keys needed from the DOE vector to configure each VSPAERO run
VSP_AERO_KEYS: list[str] = [
    "cruise_mach",
    "wing_area",           # Sref (derived)
    "wing_span",           # bref
    "wing_root_chord",     # for MAC calculation
    "wing_tip_chord",      # for MAC calculation
    "wing_taper",          # for MAC calculation (derived)
]

# All 42 variable keys (written to output CSV for traceability)
ALL_DOE_KEYS: list[str] = [
    "CL_cruise", "TSFC",
    "body_height", "body_length", "body_nose_fineness", "body_tail_fineness", "body_width",
    "cruise_mach", "design_range",
    "inlet_area", "inlet_height", "inlet_shield", "inlet_width", "inlet_x_frac",
    "mass_GTOW", "mass_empty_kg", "mass_fuel_kg", "mass_payload_kg",
    "n_max", "rcs_frontal", "stealth_align_deg",
    "thrust_cruise", "thrust_max",
    "vtail_cant_deg", "vtail_root_chord_frac", "vtail_span_frac",
    "vtail_sweep", "vtail_taper", "vtail_tc",
    "wing_AR", "wing_area", "wing_dihedral", "wing_kink_eta",
    "wing_root_chord", "wing_span", "wing_sweep_LE", "wing_taper",
    "wing_tc_root", "wing_tc_tip", "wing_tip_chord",
    "wing_twist_root", "wing_twist_tip",
]


def estimate_mac(root_chord: float, tip_chord: float) -> float:
    """Mean aerodynamic chord for a trapezoidal wing."""
    taper = tip_chord / root_chord if root_chord > 0 else 0
    return (2 / 3) * root_chord * (1 + taper + taper**2) / (1 + taper) if (1 + taper) > 0 else root_chord
