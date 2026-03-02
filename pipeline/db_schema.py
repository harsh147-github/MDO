"""
Central SQLite database for the MDO pipeline.
ALL stages write results here. Dashboards / post-processing read from here.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pipeline.db"

SCHEMA = """
-- ── Pipeline run metadata ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config_hash     TEXT,
    status          TEXT DEFAULT 'running',   -- running | completed | failed
    current_stage   TEXT,
    current_sample  INTEGER,
    total_samples   INTEGER,
    notes           TEXT
);

-- ── Stage 1: Design variable definitions ───────────────────────────
CREATE TABLE IF NOT EXISTS design_variables (
    var_id          INTEGER PRIMARY KEY,
    category        TEXT,
    name            TEXT,
    python_key      TEXT UNIQUE,
    unit            TEXT,
    lower_bound     REAL,
    baseline        REAL,
    upper_bound     REAL,
    scale_type      TEXT,       -- Independent | Derived
    active          INTEGER DEFAULT 1
);

-- ── Stage 2: DOE sample vectors ────────────────────────────────────
CREATE TABLE IF NOT EXISTS doe_samples (
    sample_id       INTEGER PRIMARY KEY,
    run_id          INTEGER REFERENCES runs(run_id),
    vector_json     TEXT,       -- full {python_key: value} dict as JSON
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Stage 3: Low-fidelity screening results ────────────────────────
CREATE TABLE IF NOT EXISTS lofi_results (
    sample_id               INTEGER PRIMARY KEY REFERENCES doe_samples(sample_id),
    run_id                  INTEGER REFERENCES runs(run_id),

    -- 3.1 Geometry
    geometry_status         TEXT,
    geometry_file           TEXT,
    wing_area_actual        REAL,
    aspect_ratio_actual     REAL,

    -- 3.2 Aero
    CL                      REAL,
    CD_total                REAL,
    CD_induced              REAL,
    CD_parasitic            REAL,
    CD_wave                 REAL,
    L_over_D                REAL,
    Cma                     REAL,
    aero_status             TEXT,

    -- 3.3 Stealth
    rcs_penalty             REAL,
    rcs_frontal_dbsm        REAL,
    rcs_side_dbsm           REAL,
    stealth_status          TEXT,

    -- 3.4 Packaging
    vol_internal_m3         REAL,
    vol_fuel_required_m3    REAL,
    vol_weapons_required_m3 REAL,
    vol_penalty             REAL,
    packaging_status        TEXT,

    -- 3.5 Structural (coarse)
    FI_max                  REAL,
    BF_min                  REAL,
    mass_structural_kg      REAL,
    structural_status       TEXT,

    -- 3.6 Objective
    J_norm_lofi             REAL,

    -- 3.7 Filter
    is_feasible             INTEGER,
    infeasibility_reasons   TEXT,
    rank_in_feasible        INTEGER,

    evaluated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Stage 4: High-fidelity batch results ───────────────────────────
CREATE TABLE IF NOT EXISTS hifi_results (
    sample_id               INTEGER PRIMARY KEY,
    run_id                  INTEGER REFERENCES runs(run_id),
    CD_hifi                 REAL,
    Cma_hifi                REAL,
    rcs_penalty_hifi        REAL,
    mass_optimized_kg       REAL,
    inner_loop_iterations   INTEGER,
    inner_loop_converged    INTEGER,
    J_norm_hifi             REAL,
    is_feasible_hifi        INTEGER,
    is_baseline             INTEGER DEFAULT 0,
    evaluated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Stage 5: Optimisation history ──────────────────────────────────
CREATE TABLE IF NOT EXISTS optimization_history (
    iteration               INTEGER PRIMARY KEY,
    run_id                  INTEGER REFERENCES runs(run_id),
    x_vector_json           TEXT,
    J_norm                  REAL,
    CD                      REAL,
    rcs_penalty             REAL,
    mass_kg                 REAL,
    vol_penalty             REAL,
    is_feasible             INTEGER,
    is_improvement          INTEGER,
    delta_J                 REAL,
    cumulative_best_J       REAL,
    wall_time_seconds       REAL,
    evaluated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Stage 6: 6-DOF validation ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS sixdof_results (
    design_id               INTEGER PRIMARY KEY,
    run_id                  INTEGER REFERENCES runs(run_id),
    trim_alpha_deg          REAL,
    trim_delta_e_deg        REAL,
    short_period_damping    REAL,
    short_period_freq_hz    REAL,
    dutch_roll_damping      REAL,
    dutch_roll_freq_hz      REAL,
    roll_time_constant_s    REAL,
    max_load_factor         REAL,
    control_saturation      INTEGER,
    verdict                 TEXT,       -- CERTIFIED | REJECTED
    rejection_reason        TEXT,
    evaluated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Progress log (real-time dashboard feed) ────────────────────────
CREATE TABLE IF NOT EXISTS progress_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES runs(run_id),
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    stage       TEXT,
    sample_id   INTEGER,
    message     TEXT,
    level       TEXT DEFAULT 'info'   -- info | warning | error | success
);
"""


def init_db():
    """Create the database and all tables (idempotent)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)
    conn.close()
    return DB_PATH


def get_connection() -> sqlite3.Connection:
    """Return a connection to the pipeline database (creates it if missing)."""
    if not DB_PATH.exists():
        init_db()
    return sqlite3.connect(str(DB_PATH))


if __name__ == "__main__":
    path = init_db()
    print(f"Database initialised at {path}")
