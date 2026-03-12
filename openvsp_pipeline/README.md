# OpenVSP Parametric Geometry & VSPAERO Pipeline

Automated pipeline that converts CCAV design vectors (from LHS DOE sampling) into
OpenVSP 3D geometry models and runs VSPAERO VLM analysis for rapid low-fidelity
aerodynamic screening.

## Pipeline Flow

```
config/ccav_design_space.csv      42-variable CCAV design space definition
        |
        v
scripts/ccav_sampler.py           LHS sampling + physics validation
        |
        v
data/ccav_feasible.csv            Feasible DOE samples (input to batch runner)
        |
        v
scripts/vsp_batch_runner.py       Per-sample: geometry build + VSPAERO VLM
  ├── scripts/vsp_geometry.py       Parametric .vsp3 model (wing, body, vtail, inlet)
  ├── scripts/vsp_aero_config.py    Analysis config (alpha sweep, solver settings)
  └── VSPAERO (VLM)                 Alpha sweep → CL, CD, L/D, CMy
        |
        v
results/vsp_batch_results.csv    Aggregated: 42 design vars + 8 aero metrics per sample
```

## Components Built per Sample

| Component | OpenVSP Type | Parameterisation |
|-----------|-------------|------------------|
| Wing      | WING        | 2-section cranked planform, NACA 4-series, kink at η |
| Fuselage  | FUSELAGE    | 7 super-ellipse cross-sections, nose/tail fineness |
| V-Tail    | WING        | Symmetric pair, cant angle = dihedral |
| Inlet     | STACK       | 3 elliptical cross-sections, dorsal placement |

## VSPAERO Output Metrics

| Metric | Description |
|--------|-------------|
| `vsp_CL_at_cruise` | Lift coefficient at cruise alpha (3°) |
| `vsp_CD_at_cruise` | Total drag coefficient at cruise alpha |
| `vsp_LD_at_cruise` | Lift-to-drag ratio at cruise alpha |
| `vsp_CMy_at_cruise` | Pitching moment at cruise alpha |
| `vsp_CL_max` | Maximum CL across alpha sweep |
| `vsp_CD_min` | Minimum CD across alpha sweep |
| `vsp_LD_max` | Maximum L/D across alpha sweep |
| `vsp_CLa_per_deg` | Lift-curve slope (linear region) |

## Requirements

- **OpenVSP 3.48.2** (download from https://openvsp.org/download.php)
- **Python 3.13** (must match OpenVSP build)
- Packages: `openvsp`, `openvsp_config`, `degen_geom`, `utilities`, `vsp_airfoils`, `numpy`, `pandas`

## Usage

```bash
# Generate DOE samples
python scripts/ccav_sampler.py --samples 500 --seed 42

# Run batch (all feasible samples)
python scripts/vsp_batch_runner.py --input data/ccav_feasible.csv --output ../output/vsp_batch

# Quick test (10 samples)
python scripts/vsp_batch_runner.py --max-samples 10 --ncpu 4 -v

# Custom alpha sweep
python scripts/vsp_batch_runner.py --alpha-start -2 --alpha-end 12 --alpha-npts 8
```

## Folder Structure

```
openvsp_pipeline/
├── README.md
├── scripts/
│   ├── ccav_sampler.py          # DOE sampling + physics validation
│   ├── vsp_geometry.py          # Parametric OpenVSP model builder
│   ├── vsp_aero_config.py       # Analysis config dataclass
│   └── vsp_batch_runner.py      # Batch orchestrator (CSV → geometry → VSPAERO → results)
├── config/
│   └── ccav_design_space.csv    # 42-variable design space (source of truth)
├── data/
│   └── ccav_feasible.csv        # Feasible DOE samples (batch runner input)
└── results/
    └── vsp_batch_results.csv    # Aggregated VSPAERO results (3 test samples)
```
