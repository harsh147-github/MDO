"""
CCAV Parametric Geometry Builder — OpenVSP Backend
====================================================
Converts the 42-variable CCAV design vector into an OpenVSP .vsp3 model
with 4 components: wing, fuselage, V-tail, and inlet.

Usage
-----
    # Build baseline model
    python -m pipeline.vsp_geometry

    # Build from a specific DOE sample dict
    from pipeline.vsp_geometry import build_ccav_model
    model_path = build_ccav_model(sample_dict, "output/my_design.vsp3")

Components created
------------------
- **Wing**: 2-section cranked planform (root→kink→tip), NACA 4-series airfoils
- **Fuselage**: 7-section body with elliptical cross-sections, nose/tail fineness
- **V-Tail**: Single wing component with cant angle as dihedral, symmetric pair
- **Inlet**: Stack component with 3 elliptical cross-sections on fuselage underside
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import openvsp as vsp

# ── Repo root (for default output paths) ─────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ═════════════════════════════════════════════════════════════════════

def _set(geom_id: str, parm: str, group: str, val: float) -> None:
    """Set a parameter value (silent no-op if parm not found)."""
    pid = vsp.GetParm(geom_id, parm, group)
    if pid:
        vsp.SetParmVal(pid, val)


def _set_xform(geom_id: str, x: float = 0, y: float = 0, z: float = 0,
               xrot: float = 0, yrot: float = 0, zrot: float = 0) -> None:
    """Set position and rotation of a component."""
    _set(geom_id, "X_Rel_Location", "XForm", x)
    _set(geom_id, "Y_Rel_Location", "XForm", y)
    _set(geom_id, "Z_Rel_Location", "XForm", z)
    _set(geom_id, "X_Rel_Rotation", "XForm", xrot)
    _set(geom_id, "Y_Rel_Rotation", "XForm", yrot)
    _set(geom_id, "Z_Rel_Rotation", "XForm", zrot)


# ═════════════════════════════════════════════════════════════════════
#  WING BUILDER
# ═════════════════════════════════════════════════════════════════════

def _build_wing(s: dict) -> str:
    """
    Build the main wing: 2-section cranked planform.

    Section 1: root → kink
    Section 2: kink → tip

    Parameters used: wing_span, wing_root_chord, wing_tip_chord,
    wing_sweep_LE, wing_dihedral, wing_twist_root, wing_twist_tip,
    wing_kink_eta, wing_tc_root, wing_tc_tip
    """
    wing_id = vsp.AddGeom("WING")
    vsp.SetGeomName(wing_id, "CCAV_Wing")

    semi_span = s["wing_span"] / 2.0
    kink_eta = s["wing_kink_eta"]

    # Span splits
    inner_span = semi_span * kink_eta
    outer_span = semi_span * (1.0 - kink_eta)

    # Kink chord — linear interpolation between root and tip
    kink_chord = s["wing_root_chord"] + kink_eta * (s["wing_tip_chord"] - s["wing_root_chord"])

    # Kink t/c — linear interpolation
    kink_tc = s["wing_tc_root"] + kink_eta * (s["wing_tc_tip"] - s["wing_tc_root"])

    # Kink twist — linear interpolation
    kink_twist = s["wing_twist_root"] + kink_eta * (s["wing_twist_tip"] - s["wing_twist_root"])

    # ── Section 1: root → kink ────────────────────────────────────
    # The wing starts with one XSec by default; we need to ensure
    # we have 2 sections (= 3 airfoil stations: root, kink, tip).
    # OpenVSP wing starts with 2 sections by default (XSec_0 = root,
    # XSec_1 = first panel). We insert one more to get root→kink→tip.
    vsp.InsertXSec(wing_id, 1, vsp.XS_FOUR_SERIES)

    # Section 1 (index 1): root → kink
    _set(wing_id, "Span", "XSec_1", inner_span)
    _set(wing_id, "Root_Chord", "XSec_1", s["wing_root_chord"])
    _set(wing_id, "Tip_Chord", "XSec_1", kink_chord)
    _set(wing_id, "Sweep", "XSec_1", s["wing_sweep_LE"])
    _set(wing_id, "Sweep_Location", "XSec_1", 0.0)        # sweep at LE
    _set(wing_id, "Dihedral", "XSec_1", s["wing_dihedral"])
    _set(wing_id, "Twist", "XSec_1", s["wing_twist_root"])

    # Section 2 (index 2): kink → tip
    _set(wing_id, "Span", "XSec_2", outer_span)
    _set(wing_id, "Root_Chord", "XSec_2", kink_chord)
    _set(wing_id, "Tip_Chord", "XSec_2", s["wing_tip_chord"])
    _set(wing_id, "Sweep", "XSec_2", s["wing_sweep_LE"])
    _set(wing_id, "Sweep_Location", "XSec_2", 0.0)
    _set(wing_id, "Dihedral", "XSec_2", s["wing_dihedral"])
    _set(wing_id, "Twist", "XSec_2", kink_twist)

    # ── Airfoil thickness ──────────────────────────────────────────
    # Set NACA 4-series t/c at each station
    xsurf_id = vsp.GetXSecSurf(wing_id, 0)

    # Station 0 = root
    vsp.ChangeXSecShape(xsurf_id, 0, vsp.XS_FOUR_SERIES)
    xsec_0 = vsp.GetXSec(xsurf_id, 0)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_0, "ThickChord"), s["wing_tc_root"])
    vsp.SetParmVal(vsp.GetXSecParm(xsec_0, "Camber"), 0.02)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_0, "CamberLoc"), 0.4)

    # Station 1 = kink
    vsp.ChangeXSecShape(xsurf_id, 1, vsp.XS_FOUR_SERIES)
    xsec_1 = vsp.GetXSec(xsurf_id, 1)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_1, "ThickChord"), kink_tc)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_1, "Camber"), 0.02)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_1, "CamberLoc"), 0.4)

    # Station 2 = tip
    vsp.ChangeXSecShape(xsurf_id, 2, vsp.XS_FOUR_SERIES)
    xsec_2 = vsp.GetXSec(xsurf_id, 2)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_2, "ThickChord"), s["wing_tc_tip"])
    vsp.SetParmVal(vsp.GetXSecParm(xsec_2, "Camber"), 0.01)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_2, "CamberLoc"), 0.4)

    # Position wing at ~25% body length, vertically centred
    wing_x = s.get("body_length", 7.9) * 0.30
    _set_xform(wing_id, x=wing_x, z=0.0)

    vsp.Update()
    return wing_id


# ═════════════════════════════════════════════════════════════════════
#  FUSELAGE BUILDER
# ═════════════════════════════════════════════════════════════════════

def _build_fuselage(s: dict) -> str:
    """
    Build the CCAV fuselage with 7 cross-sections.

    Sections: nose tip → nose mid → forward body → max section →
              aft body → tail start → tail tip.

    Parameters used: body_length, body_width, body_height,
    body_nose_fineness, body_tail_fineness
    """
    fuse_id = vsp.AddGeom("FUSELAGE")
    vsp.SetGeomName(fuse_id, "CCAV_Body")

    L = s["body_length"]
    W = s["body_width"]
    H = s["body_height"]
    nose_fin = s["body_nose_fineness"]   # nose_length / body_width
    tail_fin = s["body_tail_fineness"]   # tail_length / body_width

    _set(fuse_id, "Length", "Design", L)

    nose_len = nose_fin * W
    tail_len = tail_fin * W

    # Clamp so nose + tail don't exceed body length
    total_end = nose_len + tail_len
    if total_end > L * 0.85:
        scale = (L * 0.85) / total_end
        nose_len *= scale
        tail_len *= scale

    # Fractional positions along body (0.0 → 1.0)
    x_nose_mid = (nose_len * 0.5) / L
    x_fwd = nose_len / L
    x_max = 0.40
    x_aft = 1.0 - tail_len / L
    x_tail_start = 1.0 - (tail_len * 0.3) / L

    # Cross-section definitions: (x_frac, width_frac, height_frac)
    sections = [
        (0.0,            0.01,  0.01),   # nose tip
        (x_nose_mid,     0.40,  0.50),   # nose widening
        (x_fwd,          0.80,  0.85),   # forward body
        (x_max,          1.00,  1.00),   # max section
        (x_aft,          0.85,  0.90),   # aft body
        (x_tail_start,   0.45,  0.50),   # tail taper
        (1.0,            0.05,  0.05),   # tail tip
    ]

    # Get the XSecSurf — default fuselage has 5 sections
    xsurf_id = vsp.GetXSecSurf(fuse_id, 0)

    # Adjust section count to match our 7 sections
    while vsp.GetNumXSec(xsurf_id) > len(sections):
        vsp.CutXSec(fuse_id, vsp.GetNumXSec(xsurf_id) - 2)
    while vsp.GetNumXSec(xsurf_id) < len(sections):
        vsp.InsertXSec(fuse_id, vsp.GetNumXSec(xsurf_id) - 2, vsp.XS_ELLIPSE)

    # Configure each cross-section via XSec parm API
    for i, (x_frac, w_frac, h_frac) in enumerate(sections):
        vsp.ChangeXSecShape(xsurf_id, i, vsp.XS_SUPER_ELLIPSE)
        xsec = vsp.GetXSec(xsurf_id, i)

        # Dimensions
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "Super_Width"), W * w_frac)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "Super_Height"), H * h_frac)

        # Super-ellipse exponents (2.5 = slightly boxy for stealth)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "Super_M"), 2.5)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "Super_N"), 2.5)

        # X position along body (XLocPercent is 0–1 fraction, not percent)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "XLocPercent"), x_frac)

    _set_xform(fuse_id, x=0.0, z=0.0)

    vsp.Update()
    return fuse_id


# ═════════════════════════════════════════════════════════════════════
#  V-TAIL BUILDER
# ═════════════════════════════════════════════════════════════════════

def _build_vtail(s: dict) -> str:
    """
    Build the V-tail as a symmetric wing pair with cant angle.

    Parameters used: vtail_cant_deg, vtail_span_frac, vtail_root_chord_frac,
    vtail_sweep, vtail_taper, vtail_tc
    """
    vtail_id = vsp.AddGeom("WING")
    vsp.SetGeomName(vtail_id, "CCAV_VTail")

    # V-tail dimensions
    vtail_span = s["vtail_span_frac"] * s["wing_span"] / 2.0
    vtail_root_c = s["vtail_root_chord_frac"] * s["wing_root_chord"]
    vtail_tip_c = s["vtail_taper"] * vtail_root_c

    # Section 1 (the only panel)
    _set(vtail_id, "Span", "XSec_1", vtail_span)
    _set(vtail_id, "Root_Chord", "XSec_1", vtail_root_c)
    _set(vtail_id, "Tip_Chord", "XSec_1", vtail_tip_c)
    _set(vtail_id, "Sweep", "XSec_1", s["vtail_sweep"])
    _set(vtail_id, "Sweep_Location", "XSec_1", 0.0)
    _set(vtail_id, "Dihedral", "XSec_1", s["vtail_cant_deg"])

    # Airfoil — symmetric NACA for V-tail
    xsurf_id = vsp.GetXSecSurf(vtail_id, 0)

    for i in range(vsp.GetNumXSec(xsurf_id)):
        vsp.ChangeXSecShape(xsurf_id, i, vsp.XS_FOUR_SERIES)
        xsec = vsp.GetXSec(xsurf_id, i)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "ThickChord"), s["vtail_tc"])
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "Camber"), 0.0)       # symmetric
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "CamberLoc"), 0.4)

    # Position at rear of fuselage (~85% body length)
    vtail_x = s.get("body_length", 7.9) * 0.85
    _set_xform(vtail_id, x=vtail_x, z=s.get("body_height", 0.6) * 0.3)

    vsp.Update()
    return vtail_id


# ═════════════════════════════════════════════════════════════════════
#  INLET BUILDER
# ═════════════════════════════════════════════════════════════════════

def _build_inlet(s: dict) -> str:
    """
    Build the dorsal inlet as a Stack component with 3 cross-sections.

    Parameters used: inlet_width, inlet_height, inlet_x_frac, body_length
    """
    inlet_id = vsp.AddGeom("STACK")
    vsp.SetGeomName(inlet_id, "CCAV_Inlet")

    iw = s["inlet_width"]
    ih = s["inlet_height"]
    inlet_length = max(iw, ih) * 2.5   # inlet duct length ≈ 2.5× opening size

    # Stack design policy — open (not looped)
    _set(inlet_id, "OrderPolicy", "Design", 0)

    # Get the XSecSurf — default Stack has 5 sections
    xsurf_id = vsp.GetXSecSurf(inlet_id, 0)

    # Trim to 3 cross-sections
    while vsp.GetNumXSec(xsurf_id) > 3:
        vsp.CutXSec(inlet_id, vsp.GetNumXSec(xsurf_id) - 2)
    while vsp.GetNumXSec(xsurf_id) < 3:
        vsp.InsertXSec(inlet_id, vsp.GetNumXSec(xsurf_id) - 1, vsp.XS_ELLIPSE)

    # Section definitions: (width, height, x_delta)
    sec_defs = [
        (iw,                ih,                0.0),                  # opening
        (iw * 0.85,         ih * 0.85,         inlet_length * 0.5),  # mid-duct
        ((iw+ih)/2*0.80,    (iw+ih)/2*0.80,    inlet_length * 0.5),  # engine face
    ]

    for i, (w, h, xd) in enumerate(sec_defs):
        vsp.ChangeXSecShape(xsurf_id, i, vsp.XS_ELLIPSE)
        xsec = vsp.GetXSec(xsurf_id, i)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "Ellipse_Width"), w)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "Ellipse_Height"), h)
        # Set XDelta via XSec parm (Stack uses delta positioning)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "XDelta"), xd)

    # Position inlet on top of fuselage (dorsal intake for stealth)
    inlet_x = s["inlet_x_frac"] * s.get("body_length", 7.9)
    inlet_z = s.get("body_height", 0.6) * 0.45
    _set_xform(inlet_id, x=inlet_x, y=0.0, z=inlet_z)

    vsp.Update()
    return inlet_id


# ═════════════════════════════════════════════════════════════════════
#  MAIN BUILDER
# ═════════════════════════════════════════════════════════════════════

def build_ccav_model(
    sample: dict,
    output_path: str | Path = "output/ccav_baseline.vsp3",
    *,
    export_stl: bool = False,
    export_degengeom: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Build a complete CCAV OpenVSP model from a design vector.

    Parameters
    ----------
    sample : dict
        Design vector with all 36+ keys from ccav_design_space.csv.
    output_path : str or Path
        Where to save the .vsp3 file.
    export_stl : bool
        Also export an STL mesh alongside the .vsp3.
    export_degengeom : bool
        Also compute and export DegenGeom CSV.
    verbose : bool
        Print progress messages.

    Returns
    -------
    Path to the saved .vsp3 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Initialise OpenVSP ────────────────────────────────────────
    vsp.VSPCheckSetup()
    vsp.VSPRenew()

    if verbose:
        print(f"  Building CCAV model ({vsp.GetVSPVersion()})...")

    # ── Build components ──────────────────────────────────────────
    if verbose:
        print("    [1/4] Wing — cranked planform, NACA 4-series")
    wing_id = _build_wing(sample)

    if verbose:
        print("    [2/4] Fuselage — 7-section super-ellipse body")
    fuse_id = _build_fuselage(sample)

    if verbose:
        print("    [3/4] V-Tail — symmetric pair with cant angle")
    vtail_id = _build_vtail(sample)

    if verbose:
        print("    [4/4] Inlet — dorsal stack duct")
    inlet_id = _build_inlet(sample)

    # ── Final update & save ───────────────────────────────────────
    vsp.Update()
    vsp.WriteVSPFile(str(output_path))

    if verbose:
        print(f"\n  Saved: {output_path}")

    # ── Optional exports ──────────────────────────────────────────
    if export_stl:
        stl_path = output_path.with_suffix(".stl")
        vsp.ExportFile(str(stl_path), vsp.SET_ALL, vsp.EXPORT_STL)
        if verbose:
            print(f"  STL:   {stl_path}")

    if export_degengeom:
        vsp.ComputeDegenGeom(vsp.SET_ALL, vsp.DEGEN_GEOM_CSV_TYPE)
        if verbose:
            print("  DegenGeom CSV computed.")

    # ── Report component summary ──────────────────────────────────
    if verbose:
        print(f"\n  Component summary:")
        print(f"    Wing:     span={sample['wing_span']:.1f}m, "
              f"root={sample['wing_root_chord']:.2f}m, "
              f"sweep={sample['wing_sweep_LE']:.1f}°")
        print(f"    Body:     L={sample['body_length']:.1f}m, "
              f"W={sample['body_width']:.2f}m, "
              f"H={sample['body_height']:.2f}m")
        print(f"    V-Tail:   cant={sample['vtail_cant_deg']:.1f}°, "
              f"sweep={sample['vtail_sweep']:.1f}°, "
              f"taper={sample['vtail_taper']:.2f}")
        print(f"    Inlet:    {sample['inlet_width']:.2f}×{sample['inlet_height']:.2f}m "
              f"@ {sample['inlet_x_frac']*100:.0f}% body")

    return output_path


# ═════════════════════════════════════════════════════════════════════
#  CLI: BUILD BASELINE
# ═════════════════════════════════════════════════════════════════════

def main():
    """Build the CCAV baseline model from the design space CSV."""
    # Import from sibling module
    from pipeline.ccav_sampler import get_baseline_vector

    baseline = get_baseline_vector()

    print("=" * 65)
    print("  CCAV Parametric Geometry — OpenVSP Builder")
    print("=" * 65)

    out_dir = _REPO_ROOT / "output"
    out_path = out_dir / "ccav_baseline.vsp3"

    build_ccav_model(
        baseline,
        out_path,
        export_stl=True,
        verbose=True,
    )

    print("\n" + "=" * 65)
    print("  COMPLETE — Open in OpenVSP GUI or use for VSPAERO analysis")
    print("=" * 65)


if __name__ == "__main__":
    main()
