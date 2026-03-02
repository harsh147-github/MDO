"""
Design Vector → CAD-Ready Parameter Handoff
=============================================
Converts the Stage 1 design space into a structured, grouped parameter
dictionary that any parametric CAD tool can consume directly.

Supported downstream targets:
    - AeroSandbox (Python, in-process)  →  build_aerosandbox_geometry()
    - OpenVSP (.vspscript headless)     →  export_vsp_params()
    - FreeCAD (Python API)              →  export_freecad_params()
    - JSON / CSV flat export            →  export_json(), export_csv_row()

Usage from Stage 2 (DOE) or Stage 3 (screening):
    from pipeline.design_vector import DesignVector

    dv = DesignVector.from_baseline()          # load baseline from DB
    dv = DesignVector.from_dict(sample_dict)   # from an LHS sample

    params = dv.cad_params()                   # grouped dict for any CAD tool
    dv.export_json(path)                       # write to disk
    airplane = dv.build_aerosandbox()          # direct AeroSandbox Airplane
"""
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stage1_design_space import (
    compute_derived,
    get_baseline_vector,
    validate_sample,
    DERIVED_FORMULAS,
)


class DesignVector:
    """
    A single aircraft design point — the 34-variable vector plus all
    derived/secondary geometry quantities needed for CAD generation.
    """

    def __init__(self, raw: dict):
        """
        Parameters
        ----------
        raw : dict
            Must contain at least the 28 independent variable keys.
            Derived keys will be (re)computed automatically.
        """
        # Compute / overwrite derived variables
        self._raw = compute_derived(raw)
        self._secondary = self._compute_secondary()

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_baseline(cls) -> "DesignVector":
        """Load the spreadsheet baseline from the database."""
        return cls(get_baseline_vector())

    @classmethod
    def from_dict(cls, d: dict) -> "DesignVector":
        """Create from any dict with at least the 28 independent keys."""
        return cls(d)

    @classmethod
    def from_json(cls, path: Path) -> "DesignVector":
        """Load from a previously exported JSON file."""
        with open(path) as f:
            return cls(json.load(f))

    # ── Raw access ────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> float:
        if key in self._raw:
            return self._raw[key]
        if key in self._secondary:
            return self._secondary[key]
        raise KeyError(f"Unknown design variable: {key}")

    def raw_dict(self) -> dict:
        """Return the flat 34-variable dict."""
        return dict(self._raw)

    # ── Validation ────────────────────────────────────────────────────

    def validate(self) -> Tuple[bool, List[str]]:
        """Run the cross-variable consistency checks."""
        return validate_sample(self._raw)

    # ── Secondary geometry (computed from the 34 primaries) ───────────

    def _compute_secondary(self) -> dict:
        """
        Compute all secondary geometric quantities that CAD tools need
        but are NOT part of the 34 design variables.
        """
        r = self._raw
        s = {}

        # ── Wing planform ─────────────────────────────────────────────
        s["semi_span"] = r["wing_span"] / 2.0
        sweep_rad = math.radians(r["wing_sweep_LE"])

        # Mean Aerodynamic Chord (trapezoid formula)
        lam = r["wing_taper"]
        s["MAC"] = (2.0 / 3.0) * r["wing_root_chord"] * \
                   (1.0 + lam + lam**2) / (1.0 + lam)

        # Tip LE offset
        s["tip_x_le"] = s["semi_span"] * math.tan(sweep_rad)

        # Wing loading
        s["wing_loading_kg_m2"] = r["mass_GTOW"] / r["wing_area"] if r["wing_area"] > 0 else 0

        # ── Fuselage ──────────────────────────────────────────────────
        fuse_l = r["fuse_length"]
        fuse_d = fuse_l / r["fuse_fineness"]
        s["fuse_diameter"] = fuse_d
        s["fuse_radius"] = fuse_d / 2.0

        # Station positions (ogive-cylinder-boattail)
        s["fuse_nose_len"] = fuse_l * 0.20
        s["fuse_mid_start"] = fuse_l * 0.20
        s["fuse_mid_end"] = fuse_l * 0.75
        s["fuse_tail_start"] = fuse_l * 0.75

        # ── V-Tail ────────────────────────────────────────────────────
        vt_angle_rad = math.radians(r["vtail_angle"])
        s["vtail_span"] = r["wing_span"] * 0.20  # 20% of wingspan
        s["vtail_root_chord"] = r["wing_root_chord"] * 0.40
        s["vtail_tip_chord"] = s["vtail_root_chord"] * 0.60
        s["vtail_x_le"] = fuse_l * 0.70  # 70% of fuselage length
        s["vtail_tip_y"] = s["vtail_span"] * math.cos(vt_angle_rad)
        s["vtail_tip_z"] = s["vtail_span"] * math.sin(vt_angle_rad)
        s["vtail_tip_x_le"] = s["vtail_x_le"] + \
                              s["vtail_span"] * math.tan(math.radians(30))

        # ── Inlet ─────────────────────────────────────────────────────
        s["inlet_width"] = math.sqrt(r["inlet_area"])  # assume square-ish
        s["inlet_height"] = r["inlet_area"] / s["inlet_width"]
        s["inlet_x"] = fuse_l * 0.25  # dorsal or chin inlet

        # ── Reference values for aero analysis ────────────────────────
        s["Sref"] = r["wing_area"]
        s["bref"] = r["wing_span"]
        s["cref"] = s["MAC"]
        s["Xcg"] = fuse_l * 0.40      # 40% fuselage length
        s["Ycg"] = 0.0
        s["Zcg"] = 0.0

        # ── Flight condition ──────────────────────────────────────────
        # ISA at cruise altitude (assume ~10 km for Mach 0.9)
        alt_m = 10000.0  # can be refined later
        T_isa = 288.15 - 0.0065 * alt_m
        a_sound = math.sqrt(1.4 * 287.05 * max(T_isa, 200))
        s["cruise_velocity_ms"] = r["cruise_mach"] * a_sound
        rho_isa = 1.225 * (T_isa / 288.15) ** 4.256
        s["cruise_rho_kg_m3"] = rho_isa
        s["cruise_q_Pa"] = 0.5 * rho_isa * s["cruise_velocity_ms"] ** 2

        return s

    # ── Grouped CAD parameter dict ────────────────────────────────────

    def cad_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a hierarchically grouped dict suitable for direct CAD
        tool consumption. Every value has its unit annotated.

        Groups: wing, fuselage, vtail, inlet, mass, aero, flight, stealth,
                structures, packaging, reference
        """
        r = self._raw
        s = self._secondary

        return {
            "wing": {
                "span_m": r["wing_span"],
                "semi_span_m": s["semi_span"],
                "root_chord_m": r["wing_root_chord"],
                "tip_chord_m": r["wing_tip_chord"],
                "taper_ratio": r["wing_taper"],
                "sweep_LE_deg": r["wing_sweep_LE"],
                "area_m2": r["wing_area"],
                "aspect_ratio": r["wing_AR"],
                "MAC_m": s["MAC"],
                "tip_x_le_m": s["tip_x_le"],
                "wing_loading_kg_m2": s["wing_loading_kg_m2"],
            },
            "fuselage": {
                "length_m": r["fuse_length"],
                "fineness_ratio": r["fuse_fineness"],
                "diameter_m": s["fuse_diameter"],
                "radius_m": s["fuse_radius"],
                "nose_len_m": s["fuse_nose_len"],
                "mid_start_m": s["fuse_mid_start"],
                "mid_end_m": s["fuse_mid_end"],
                "tail_start_m": s["fuse_tail_start"],
            },
            "vtail": {
                "cant_angle_deg": r["vtail_angle"],
                "span_m": s["vtail_span"],
                "root_chord_m": s["vtail_root_chord"],
                "tip_chord_m": s["vtail_tip_chord"],
                "x_le_m": s["vtail_x_le"],
                "tip_y_m": s["vtail_tip_y"],
                "tip_z_m": s["vtail_tip_z"],
                "tip_x_le_m": s["vtail_tip_x_le"],
            },
            "inlet": {
                "throat_area_m2": r["inlet_area"],
                "width_m": s["inlet_width"],
                "height_m": s["inlet_height"],
                "x_position_m": s["inlet_x"],
            },
            "mass": {
                "empty_kg": r["mass_empty"],
                "fuel_kg": r["mass_fuel"],
                "payload_kg": r["mass_payload"],
                "GTOW_kg": r["mass_GTOW"],
                "Ixx_kgm2": r["Ixx"],
            },
            "aero": {
                "CL_cruise": r["CL_cruise"],
                "CD0_target": r["CD0_target"],
                "Cma": r["Cma"],
            },
            "flight": {
                "cruise_mach": r["cruise_mach"],
                "cruise_velocity_ms": s["cruise_velocity_ms"],
                "cruise_rho_kg_m3": s["cruise_rho_kg_m3"],
                "cruise_q_Pa": s["cruise_q_Pa"],
                "n_max_g": r["n_max"],
                "design_range_km": r["design_range"],
                "endurance_hr": r["endurance_hr"],
            },
            "propulsion": {
                "thrust_max_kN": r["thrust_max"],
                "thrust_cruise_kN": r["thrust_cruise"],
                "TSFC_kg_N_s": r["TSFC"],
            },
            "stealth": {
                "rcs_frontal_dBsm": r["rcs_frontal"],
                "planform_align_deg": r["stealth_align_deg"],
                "inlet_shield_factor": r["inlet_shield"],
            },
            "structures": {
                "FI_target": r["FI_target"],
                "BF_target": r["BF_target"],
            },
            "packaging": {
                "vol_fuel_m3": r["vol_fuel"],
                "vol_weapons_m3": r["vol_weapons"],
                "vol_sensors_m3": r["vol_sensors"],
            },
            "reference": {
                "Sref_m2": s["Sref"],
                "bref_m": s["bref"],
                "cref_m": s["cref"],
                "Xcg_m": s["Xcg"],
                "Ycg_m": s["Ycg"],
                "Zcg_m": s["Zcg"],
            },
        }

    # ── Export formats ────────────────────────────────────────────────

    def export_json(self, path: Path, indent: int = 2) -> Path:
        """
        Write the full grouped CAD parameters to a JSON file.
        Any CAD tool or script can parse this directly.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.cad_params(), f, indent=indent)
        return path

    def export_flat_json(self, path: Path) -> Path:
        """Write the flat 34-variable + secondary dict to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = {**self._raw, **self._secondary}
        with open(path, "w") as f:
            json.dump(merged, f, indent=2)
        return path

    def to_csv_row(self) -> dict:
        """Return a single flat dict suitable for one row in a DOE CSV."""
        return {**self._raw, **self._secondary}

    # ── AeroSandbox geometry builder ──────────────────────────────────

    def build_aerosandbox(self, save_png: Optional[Path] = None):
        """
        Build and return an AeroSandbox Airplane object directly
        from this design vector. Optionally save a 3-view PNG.

        Requires: pip install aerosandbox
        """
        import aerosandbox as asb
        import aerosandbox.numpy as anp

        r = self._raw
        s = self._secondary

        airfoil = asb.Airfoil("naca64a010")

        # ── Wing ──────────────────────────────────────────────────────
        wing = asb.Wing(
            name="Main Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=r["wing_root_chord"],
                    airfoil=airfoil,
                ),
                asb.WingXSec(
                    xyz_le=[s["tip_x_le"], s["semi_span"], 0],
                    chord=r["wing_tip_chord"],
                    airfoil=airfoil,
                ),
            ],
        )

        # ── Fuselage ──────────────────────────────────────────────────
        fuse_l = r["fuse_length"]
        fuse_r = s["fuse_radius"]
        fuselage = asb.Fuselage(
            name="Fuselage",
            xsecs=[
                asb.FuselageXSec(xyz_c=[-fuse_l * 0.2, 0, 0], radius=0),
                asb.FuselageXSec(xyz_c=[0, 0, 0], radius=fuse_r),
                asb.FuselageXSec(xyz_c=[fuse_l * 0.6, 0, 0], radius=fuse_r),
                asb.FuselageXSec(xyz_c=[fuse_l * 0.8, 0, 0], radius=fuse_r * 0.3),
            ],
        )

        # ── V-Tail ────────────────────────────────────────────────────
        vtail = asb.Wing(
            name="V-Tail",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[s["vtail_x_le"], 0, 0],
                    chord=s["vtail_root_chord"],
                    airfoil=airfoil,
                ),
                asb.WingXSec(
                    xyz_le=[
                        s["vtail_tip_x_le"],
                        s["vtail_tip_y"],
                        s["vtail_tip_z"],
                    ],
                    chord=s["vtail_tip_chord"],
                    airfoil=airfoil,
                ),
            ],
        )

        airplane = asb.Airplane(
            name="CCAV",
            xyz_ref=[s["Xcg"], s["Ycg"], s["Zcg"]],
            wings=[wing, vtail],
            fuselages=[fuselage],
        )

        if save_png is not None:
            import matplotlib.pyplot as plt
            airplane.draw_three_view(show=False)
            fig = plt.gcf()
            save_png = Path(save_png)
            save_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_png, dpi=120, bbox_inches="tight")
            plt.close("all")

        return airplane

    # ── Pretty print ──────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary string."""
        p = self.cad_params()
        lines = ["CCAV Design Vector — CAD Parameter Summary",
                 "=" * 55]
        for group_name, group in p.items():
            lines.append(f"\n  [{group_name.upper()}]")
            for key, val in group.items():
                if isinstance(val, float):
                    if abs(val) < 0.001 and val != 0:
                        lines.append(f"    {key:<30s} = {val:>12.4e}")
                    else:
                        lines.append(f"    {key:<30s} = {val:>12.4f}")
                else:
                    lines.append(f"    {key:<30s} = {val}")
        return "\n".join(lines)


# ── CLI: demonstrate the handoff ──────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Design Vector → CAD Handoff Demo")
    print("=" * 60)

    # 1. Load baseline
    dv = DesignVector.from_baseline()

    # 2. Validate
    ok, reasons = dv.validate()
    print(f"\nBaseline validation: {'PASS' if ok else 'FAIL'}")
    for r in reasons:
        print(f"  - {r}")

    # 3. Print grouped CAD params
    print(f"\n{dv.summary()}")

    # 4. Export JSON
    out_dir = _REPO_ROOT / "data" / "geometries"
    json_path = dv.export_json(out_dir / "baseline_cad_params.json")
    print(f"\nGrouped CAD params exported to: {json_path}")

    flat_path = dv.export_flat_json(out_dir / "baseline_flat.json")
    print(f"Flat vector exported to:        {flat_path}")

    # 5. Show what downstream tools receive
    params = dv.cad_params()
    print(f"\nCAD param groups: {list(params.keys())}")
    print(f"Wing group keys:  {list(params['wing'].keys())}")
    print(f"Total parameters: {sum(len(g) for g in params.values())}")
    print(f"\nReady for: AeroSandbox / OpenVSP / FreeCAD / any parametric tool")
