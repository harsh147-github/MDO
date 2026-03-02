# MDO Lab 4 — CCAV Aerostructural Design

Multi-disciplinary Design Optimisation pipeline for a Cooperative Combat Air Vehicle (CCAV).
V-tail, single-engine, stealth-aligned, Mach 0.8-1.5 flight regime.

## Current Status

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Design Space Definition (34 variables, 6 derived) | Complete |
| 2 | DOE / LHS Sampling + Physics Pre-filter | Complete |
| 3 | Low-fi Screening (AeroSandbox VLM) | Planned |
| 4 | Mid-fi CFD (OpenFOAM RANS) | Planned |
| 5 | Structural Analysis (TACS FEM) | Planned |
| 6 | Coupled Aerostructural Optimisation | Planned |

## Quick Start

```bash
pip install -r requirements.txt
python explorer_app.py          # opens http://127.0.0.1:5050
```

## Stage 1 — Design Space

34 design variables covering wing geometry, fuselage, propulsion, mass budget,
stealth signatures, and stability derivatives. 6 additional variables are derived
via calibrated K-factor formulas (K_BLEND=2.24, K_AR=2.143, K_IXX=0.00455,
K_FUEL=0.0167). Source of truth: `config/design_space.xlsx`.

Key modules:
- `pipeline/stage1_design_space.py` — bounds, derived formulas, 9 physics validation checks
- `pipeline/design_vector.py` — DesignVector class wrapping 34-var sample into 63 grouped CAD params

## Stage 2 — DOE & Sampling

500 Latin-Hypercube samples (scipy.stats.qmc, optimized), physics pre-filtered
through 9 constraint checks -> ~283 feasible (56.5% pass rate).

Key modules:
- `pipeline/stage2_doe.py` — LHS generation + validation
- `pipeline/visualise_doe.py` — 6 diagnostic PNG plots

## Interactive Explorer

`explorer_app.py` — local Python HTTP server serving a single-page 3D Plotly app:
- 3D scatter plot with axis/preset/colour controls
- Translucent feasible-region cloud (mesh3d convex hull, toggleable)
- Live-editable spreadsheet tab — cell edits recompute derived vars, re-validate, and update the plot instantly
- Point inspector with full CAD parameter breakdown

## Repository Structure

```
config/design_space.xlsx        # 34-variable design space definition
pipeline/
  stage1_design_space.py        # bounds, derived formulas, validation
  stage2_doe.py                 # LHS DOE + physics filter
  design_vector.py              # CAD parameter grouping
  visualise_doe.py              # diagnostic plots
  db_schema.py                  # SQLite schema utilities
explorer_app.py                 # interactive 3D explorer (local server)
requirements.txt
```
