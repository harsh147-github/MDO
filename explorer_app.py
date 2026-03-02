"""
Local MDO Design Explorer — CCAV 3D Interactive Visualization
==============================================================
Two-phase workflow:
  Phase 1 — Browse the 34-variable design space (bounds, constraints, baseline)
  Phase 2 — Click "Generate DOE" → live-streamed LHS sampling with animation

Run:   python explorer_app.py
Opens: http://127.0.0.1:5050   (Ctrl+C to stop)
"""
import json, sys, time, threading, webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.stage1_design_space import (
    get_independent_bounds, get_baseline_vector,
    compute_derived, validate_sample, get_derived_keys,
)
from pipeline.design_vector import DesignVector

PORT = 5050
HOST = "127.0.0.1"

# ═══════════════════════════════════════════════════════════════════
#  LOAD DESIGN SPACE  (no DOE yet — generated on demand)
# ═══════════════════════════════════════════════════════════════════
print("[1/2] Loading design space ...")
BOUNDS       = get_independent_bounds()       # {key: (lo, base, hi)}
BASELINE_RAW = get_baseline_vector()          # all 34 keys from DB
BL_FULL      = compute_derived({k: BASELINE_RAW[k] for k in BOUNDS})
for k, v in BASELINE_RAW.items():
    if k not in BL_FULL:
        BL_FULL[k] = v

ALL_KEYS   = sorted(BL_FULL.keys())
DERIVED    = get_derived_keys()
INDEP_KEYS = sorted(BOUNDS.keys())

bounds_dict = {k: [lo, base, hi] for k, (lo, base, hi) in BOUNDS.items()}

CONSTRAINTS = [
    {"id":"inverted_taper","name":"Inverted Taper","rule":"tip chord \u2264 root chord"},
    {"id":"taper_range","name":"Taper Ratio","rule":"\u03bb \u2208 [0.08, 0.65]"},
    {"id":"ar_range","name":"Aspect Ratio","rule":"AR \u2208 [2, 16]"},
    {"id":"gtow","name":"GTOW > Empty","rule":"GTOW > mass_empty"},
    {"id":"thrust","name":"Thrust Budget","rule":"T_cruise < T_max"},
    {"id":"fuel_vol","name":"Fuel Volume","rule":"V_need \u2264 V_avail / 0.6"},
    {"id":"wing_loading","name":"Wing Loading","rule":"W/S \u2208 [100, 900] kg/m\u00b2"},
    {"id":"stability","name":"Pitch Stability","rule":"Cm\u03b1 < 0"},
    {"id":"fineness","name":"Fuse. Fineness","rule":"FR \u2208 [5, 20]"},
]

INIT_JSON = json.dumps({
    "keys": ALL_KEYS,
    "indep": INDEP_KEYS,
    "derived": DERIVED,
    "bounds": bounds_dict,
    "baseline": {k: BL_FULL.get(k, 0) for k in ALL_KEYS},
    "constraints": CONSTRAINTS,
})

print(f"      {len(ALL_KEYS)} vars ({len(BOUNDS)} indep + {len(DERIVED)} derived), "
      f"{len(CONSTRAINTS)} constraints")

# ── server-side DOE store (populated during streaming) ────────────
DOE_STORE = {"samples": {k: [] for k in ALL_KEYS}, "feas": [], "reasons": [], "n": 0}
DOE_LOCK  = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
#  HTML PAGE
# ═══════════════════════════════════════════════════════════════════
PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>MDO Design Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{background:#1a1d23;color:#c8ccd4;font-family:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;
  font-weight:400;-webkit-font-smoothing:antialiased}

/* ── header ── */
.hdr{background:#1e2128;border-bottom:1px solid #2a2e36;padding:10px 24px;
  display:flex;justify-content:space-between;align-items:center;height:52px}
.hdr h1{font-size:1rem;font-weight:600;letter-spacing:.6px;color:#d4d8e0}
.hdr h1 span{color:#7b8494;font-weight:400}
.sub{font-size:.68rem;color:#5a6070;margin-top:1px;font-weight:300}
.stats{display:flex;gap:12px}
.st{text-align:center;padding:3px 12px}
.st .v{font-size:1rem;font-weight:500;font-family:'IBM Plex Mono',monospace}
.st .l{font-size:.58rem;color:#5a6070;text-transform:uppercase;letter-spacing:.8px}

/* ── grid ── */
.main{display:grid;grid-template-columns:240px 1fr 340px;height:calc(100vh - 52px)}
.pnl{background:#1e2128;padding:14px 16px;overflow-y:auto;
  scrollbar-width:thin;scrollbar-color:#2a2e36 transparent}
.pnl:first-child{border-right:1px solid #2a2e36}
.pnl:last-child{border-left:1px solid #2a2e36}
.pnl h3{font-size:.64rem;font-weight:500;letter-spacing:1px;text-transform:uppercase;
  margin-bottom:10px;color:#5a6070;padding-bottom:5px;border-bottom:1px solid #2a2e36}

/* ── controls ── */
.cg{margin-bottom:11px}
.cg label{display:block;font-size:.64rem;color:#5a6070;margin-bottom:3px;
  text-transform:uppercase;letter-spacing:.4px}
select{width:100%;padding:6px 8px;background:#262a33;color:#b0b6c2;
  border:1px solid #2a2e36;border-radius:4px;font-size:.76rem;
  font-family:'IBM Plex Sans',sans-serif;cursor:pointer;outline:none;transition:.15s}
select:hover{border-color:#3e4450}select:focus{border-color:#5a6070}
.tg{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px}
.tb{padding:4px 10px;border-radius:3px;border:1px solid #2a2e36;background:transparent;
  color:#6b7280;font-size:.68rem;font-family:'IBM Plex Sans',sans-serif;cursor:pointer;transition:.15s}
.tb:hover{border-color:#3e4450;color:#9ca3af}
.tb.ag{border-color:#3a7a5e;color:#6dba8a;background:rgba(109,186,138,.06)}
.tb.ar{border-color:#7a3a3a;color:#d47272;background:rgba(212,114,114,.06)}
.tb.ay{border-color:#7a6a3a;color:#c9a84e;background:rgba(201,168,78,.06)}
.tb.ab{border-color:#3a5a7a;color:#72a8d4;background:rgba(114,168,212,.06)}
.preset{padding:3px 8px;border-radius:3px;border:1px solid #2a2e36;background:transparent;
  color:#6b7280;font-size:.64rem;cursor:pointer;transition:.15s;font-family:'IBM Plex Sans',sans-serif}
.preset:hover{border-color:#3e4450;color:#9ca3af;background:rgba(255,255,255,.02)}
.slider{width:100%;-webkit-appearance:none;height:3px;border-radius:1.5px;
  background:#2a2e36;outline:none;margin-top:4px}
.slider::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;
  background:#7b8494;cursor:pointer;border:2px solid #1e2128}
hr.d{border:none;border-top:1px solid #2a2e36;margin:10px 0}
#plot3d{width:100%;height:100%}

/* ── tabs ── */
.tabs{display:flex;gap:0;margin-bottom:10px;border-bottom:1px solid #2a2e36}
.tab{padding:6px 14px;font-size:.68rem;font-weight:500;color:#5a6070;cursor:pointer;
  border-bottom:2px solid transparent;transition:.15s;text-transform:uppercase;letter-spacing:.5px}
.tab:hover{color:#9ca3af}.tab.act{color:#b0b6c2;border-bottom-color:#6dba8a}
.tab.dis{opacity:.3;pointer-events:none}
.tab-body{display:none}.tab-body.act{display:block}

/* ── design-space tab ── */
.ds-row{display:flex;gap:6px;margin-bottom:12px}
.ds-card{flex:1;text-align:center;padding:8px 4px;background:#262a33;border-radius:4px;
  border:1px solid #2a2e36}
.ds-card .n{font-size:1.1rem;font-weight:500;font-family:'IBM Plex Mono',monospace}
.ds-card .l{font-size:.54rem;color:#5a6070;text-transform:uppercase;letter-spacing:.7px;margin-top:2px}

/* bounds table */
.bt-wrap{max-height:180px;overflow-y:auto;margin-bottom:6px;border:1px solid #2a2e36;border-radius:4px;
  scrollbar-width:thin;scrollbar-color:#2a2e36 transparent}
.bt{width:100%;border-collapse:collapse;font-size:.62rem}
.bt th{position:sticky;top:0;background:#262a33;color:#7b8494;font-weight:500;padding:3px 5px;
  text-align:left;font-size:.58rem;text-transform:uppercase;letter-spacing:.5px;z-index:2}
.bt td{padding:2px 5px;border-bottom:1px solid rgba(255,255,255,.02);
  font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#b0b6c2}
.bt tr:hover td{background:rgba(255,255,255,.02)}
.bt .der td{color:#c9a84e;font-style:italic}

/* constraint list */
.con{display:flex;align-items:center;gap:6px;padding:5px 8px;margin-bottom:3px;
  border-radius:4px;background:#262a33;border:1px solid #2a2e36;font-size:.68rem;transition:.2s}
.con-icon{width:18px;height:18px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:.56rem;flex-shrink:0;transition:.2s}
.con.idle .con-icon{background:rgba(114,168,212,.08);color:#72a8d4;border:1px solid rgba(114,168,212,.15)}
.con.pass .con-icon{background:rgba(109,186,138,.1);color:#6dba8a;border:1px solid rgba(109,186,138,.2)}
.con.fail .con-icon{background:rgba(212,114,114,.1);color:#d47272;border:1px solid rgba(212,114,114,.2)}
.con.fail{border-color:rgba(212,114,114,.15)}
.con-name{flex:1;color:#b0b6c2}.con-rule{color:#5a6070;font-size:.58rem}
.con-ct{font-family:'IBM Plex Mono';font-size:.62rem;color:#d47272;min-width:28px;text-align:right}

/* generate DOE */
.gen{margin-top:6px;padding-top:10px;border-top:1px solid #2a2e36}
.gen-row{display:flex;gap:8px;align-items:flex-end;margin-bottom:10px}
.gen-lbl{font-size:.58rem;color:#5a6070;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}
.gen-inp{width:72px;padding:5px 7px;background:#262a33;color:#b0b6c2;border:1px solid #2a2e36;
  border-radius:4px;font-size:.76rem;font-family:'IBM Plex Mono';text-align:center;outline:none}
.gen-inp:focus{border-color:#5a6070}
.gen-btn{width:100%;padding:11px;border:1px solid #3a7a5e;background:rgba(109,186,138,.06);
  color:#6dba8a;font-size:.84rem;font-weight:500;border-radius:5px;cursor:pointer;
  font-family:'IBM Plex Sans';transition:.2s;letter-spacing:.4px}
.gen-btn:hover{background:rgba(109,186,138,.14);border-color:#5aaa7e;box-shadow:0 0 20px rgba(109,186,138,.08)}
.gen-btn:disabled{opacity:.35;cursor:default;border-color:#2a2e36;background:transparent;color:#5a6070;box-shadow:none}
.gen-btn:active:not(:disabled){transform:scale(0.98)}
.gen-pg{margin-top:10px}
.gen-st{font-size:.68rem;color:#7b8494;margin-bottom:6px;display:flex;align-items:center}
.gen-track{height:3px;background:#2a2e36;border-radius:2px;overflow:hidden}
.gen-bar{height:100%;background:linear-gradient(90deg,#6dba8a,#4ecdc4);border-radius:2px;
  transition:width .35s ease-out;width:0%}
.gen-pct{font-size:.6rem;color:#5a6070;margin-top:4px;text-align:right;font-family:'IBM Plex Mono'}

@keyframes spin{to{transform:rotate(360deg)}}
.spinner{display:inline-block;width:12px;height:12px;border:2px solid #2a2e36;
  border-top-color:#6dba8a;border-radius:50%;animation:spin .7s linear infinite;margin-right:6px;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:.6}50%{opacity:1}}
.gen-st.active{animation:pulse 1.5s ease-in-out infinite}

/* ── inspector ── */
.empty{color:#3e4450;font-size:.78rem;margin-top:20px;text-align:center;line-height:1.6}
.ph{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:8px;padding-bottom:7px;border-bottom:1px solid #2a2e36}
.ph .id{font-size:.9rem;font-weight:500;color:#b0b6c2;font-family:'IBM Plex Mono',monospace}
.badge{padding:2px 8px;border-radius:3px;font-size:.62rem;font-weight:500;letter-spacing:.3px}
.bp{background:rgba(109,186,138,.08);color:#6dba8a;border:1px solid rgba(109,186,138,.2)}
.bf{background:rgba(212,114,114,.08);color:#d47272;border:1px solid rgba(212,114,114,.2)}
.fr{background:rgba(212,114,114,.04);border:1px solid rgba(212,114,114,.1);border-radius:4px;
  padding:6px 8px;margin-bottom:10px;font-size:.68rem;color:#d47272;line-height:1.5}
.pg{margin-bottom:8px}
.pgh{font-size:.62rem;font-weight:500;color:#7b8494;text-transform:uppercase;letter-spacing:.8px;
  margin-bottom:3px;padding-bottom:2px;border-bottom:1px solid #2a2e36}
.pr{display:flex;justify-content:space-between;padding:1.5px 0;font-size:.68rem;
  border-bottom:1px solid rgba(255,255,255,.02)}
.pr .k{color:#5a6070}.pr .vl{color:#b0b6c2;font-family:'IBM Plex Mono',monospace;font-size:.66rem}

/* ── spreadsheet ── */
.ss-wrap{max-height:calc(100vh - 180px);overflow:auto;scrollbar-width:thin;scrollbar-color:#2a2e36 transparent}
.ss{width:100%;border-collapse:collapse;font-size:.68rem}
.ss th{position:sticky;top:0;background:#262a33;color:#7b8494;font-weight:500;padding:4px 6px;
  text-align:left;font-size:.62rem;text-transform:uppercase;letter-spacing:.5px;
  border-bottom:1px solid #2a2e36;z-index:2}
.ss td{padding:3px 6px;border-bottom:1px solid rgba(255,255,255,.02);
  font-family:'IBM Plex Mono',monospace;font-size:.66rem;color:#b0b6c2;white-space:nowrap}
.ss tr:hover td{background:rgba(255,255,255,.02)}
.ss tr.feas-row td:first-child{border-left:2px solid rgba(109,186,138,.4)}
.ss tr.infeas-row td:first-child{border-left:2px solid rgba(212,114,114,.3)}
.ss td[contenteditable]{cursor:text;outline:none;border-radius:2px;transition:.1s}
.ss td[contenteditable]:focus{background:rgba(109,186,138,.08);outline:1px solid rgba(109,186,138,.25)}
.ss td.edited{color:#6dba8a}.ss td.id-col{color:#5a6070;font-weight:500;cursor:default}
.ss td.feas-col{font-size:.6rem}
.ss-info{font-size:.62rem;color:#5a6070;margin-bottom:6px;font-style:italic}
.ss-toolbar{display:flex;gap:6px;margin-bottom:8px;align-items:center;flex-wrap:wrap}
.ss-toolbar select{width:auto;padding:4px 6px;font-size:.66rem}
.ss-toolbar .ss-btn{padding:3px 10px;border-radius:3px;border:1px solid #2a2e36;background:transparent;
  color:#6dba8a;font-size:.64rem;cursor:pointer;transition:.15s;font-family:'IBM Plex Sans',sans-serif}
.ss-toolbar .ss-btn:hover{background:rgba(109,186,138,.08);border-color:#3a7a5e}
.ss-toolbar .ss-btn.warn{color:#d47272}
.ss-toolbar .ss-btn.warn:hover{background:rgba(212,114,114,.08);border-color:#7a3a3a}
</style></head><body>

<!-- ═══ HEADER ═══ -->
<div class="hdr">
  <div>
    <h1>Design Explorer <span>/ CCAV</span></h1>
    <div class="sub" id="subtitle">34-variable design space &middot; 9 physics constraints &middot; ready to generate</div>
  </div>
  <div class="stats">
    <div class="st" id="st-v"><div class="v" style="color:#72a8d4">34</div><div class="l">Variables</div></div>
    <div class="st hs" id="st-n"><div class="v" style="color:#b0b6c2" id="hN">0</div><div class="l">Samples</div></div>
    <div class="st hs" id="st-f"><div class="v" style="color:#6dba8a" id="hF">0</div><div class="l">Feasible</div></div>
    <div class="st hs" id="st-i"><div class="v" style="color:#d47272" id="hI">0</div><div class="l">Rejected</div></div>
    <div class="st hs" id="st-p"><div class="v" style="color:#c9a84e" id="hP">&mdash;</div><div class="l">Pass Rate</div></div>
  </div>
</div>

<div class="main">
<!-- ═══ LEFT PANEL ═══ -->
<div class="pnl">
  <h3>Axes</h3>
  <div class="cg"><label>X Axis</label><select id="sx"></select></div>
  <div class="cg"><label>Y Axis</label><select id="sy"></select></div>
  <div class="cg"><label>Z Axis</label><select id="sz"></select></div>
  <hr class="d">
  <h3>Presets</h3>
  <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:10px">
    <button class="preset" onclick="ax('wing_span','wing_area','mass_GTOW')">Wing</button>
    <button class="preset" onclick="ax('cruise_mach','CL_cruise','CD0_target')">Aero</button>
    <button class="preset" onclick="ax('mass_fuel','vol_fuel','design_range')">Fuel / Range</button>
    <button class="preset" onclick="ax('wing_span','wing_AR','wing_taper')">Planform</button>
    <button class="preset" onclick="ax('thrust_max','mass_GTOW','n_max')">Loads</button>
    <button class="preset" onclick="ax('rcs_frontal','stealth_align_deg','inlet_shield')">Stealth</button>
    <button class="preset" onclick="ax('fuse_length','fuse_fineness','wing_sweep_LE')">Fuselage</button>
    <button class="preset" onclick="ax('Ixx','mass_GTOW','wing_span')">Inertia</button>
  </div>
  <hr class="d">
  <h3>Visibility</h3>
  <div class="tg">
    <button id="bF" class="tb ag" onclick="tv('f')">Feasible</button>
    <button id="bI" class="tb ar" onclick="tv('i')">Infeasible</button>
    <button id="bB" class="tb ay" onclick="tv('b')">Baseline</button>
    <button id="bC" class="tb" onclick="tv('c')">Feasible Cloud</button>
  </div>
  <hr class="d">
  <h3>Colour By</h3>
  <div class="cg"><select id="sc" onchange="draw()"></select></div>
  <hr class="d">
  <div class="cg">
    <label>Point Size &mdash; <span id="sv">5</span></label>
    <input type="range" min="2" max="14" value="5" class="slider" id="ssSlider"
           oninput="document.getElementById('sv').textContent=this.value;draw()">
  </div>
  <div class="cg">
    <label>Cloud Opacity &mdash; <span id="co">12</span>%</label>
    <input type="range" min="2" max="40" value="12" class="slider" id="cslider"
           oninput="document.getElementById('co').textContent=this.value;draw()">
  </div>
</div>

<!-- ═══ CENTER: 3-D PLOT ═══ -->
<div id="plot3d"></div>

<!-- ═══ RIGHT PANEL ═══ -->
<div class="pnl" id="rp">
  <div class="tabs">
    <div class="tab act" onclick="switchTab('design')">Design Space</div>
    <div class="tab dis" id="tbtn-insp" onclick="switchTab('inspector')">Inspector</div>
    <div class="tab dis" id="tbtn-ss" onclick="switchTab('spreadsheet')">Spreadsheet</div>
  </div>

  <!-- ── Design Space tab ── -->
  <div class="tab-body act" id="tab-design">
    <div class="ds-row">
      <div class="ds-card"><div class="n" style="color:#72a8d4">28</div><div class="l">Independent</div></div>
      <div class="ds-card"><div class="n" style="color:#c9a84e">6</div><div class="l">Derived</div></div>
      <div class="ds-card"><div class="n" style="color:#d47272">9</div><div class="l">Constraints</div></div>
    </div>
    <h3>Variable Bounds <span style="float:right;font-weight:300;letter-spacing:0">28 sampled</span></h3>
    <div class="bt-wrap" id="bt-wrap"></div>
    <hr class="d">
    <h3>Constraint Verification</h3>
    <div id="con-list"></div>
    <hr class="d">
    <div class="gen">
      <h3>Generate DOE</h3>
      <div class="gen-row">
        <div><div class="gen-lbl">Samples</div><input class="gen-inp" id="gen-n" type="number" value="500" min="50" max="5000" step="50"></div>
        <div><div class="gen-lbl">Seed</div><input class="gen-inp" id="gen-seed" type="number" value="42" min="1" max="9999"></div>
      </div>
      <button class="gen-btn" id="gen-btn" onclick="startDOE()">&#9654;&ensp;Generate DOE</button>
      <div class="gen-pg" id="gen-pg" style="display:none">
        <div class="gen-st" id="gen-st"><span class="spinner"></span>Preparing&hellip;</div>
        <div class="gen-track"><div class="gen-bar" id="gen-bar"></div></div>
        <div class="gen-pct" id="gen-pct"></div>
      </div>
    </div>
  </div>

  <!-- ── Inspector tab ── -->
  <div class="tab-body" id="tab-inspector">
    <div class="empty" id="ie">Click any point to inspect<br>its design vector</div>
    <div id="ic" style="display:none"></div>
  </div>

  <!-- ── Spreadsheet tab ── -->
  <div class="tab-body" id="tab-spreadsheet">
    <div class="ss-toolbar">
      <select id="ss-cols" multiple size="4" style="width:140px;font-size:.66rem" onchange="buildTable()"></select>
      <div style="display:flex;flex-direction:column;gap:3px">
        <button class="ss-btn" onclick="addAxisCols()">Show X/Y/Z</button>
        <button class="ss-btn warn" onclick="resetEdits()">Reset Edits</button>
      </div>
    </div>
    <div class="ss-info">Double-click cells to edit &mdash; changes update the 3D plot live.</div>
    <div class="ss-wrap" id="ss-wrap"></div>
  </div>
</div>
</div>

<script>
/* ═══════════════════════════════════════════════════════════════
   DATA & STATE
   ═══════════════════════════════════════════════════════════════ */
const D = __INIT_JSON__;
let STATE = 'idle'; // idle | generating | complete

/* DOE data — populated during streaming */
const DOE = { samples:{}, feas:[], reasons:[], n:0, nFeas:0, details:{} };
D.keys.forEach(k => DOE.samples[k] = []);

/* constraint failure counters */
const CF = {};
D.constraints.forEach(c => CF[c.id] = 0);

/* visibility */
let sF=true, sI=true, sB=true, sC=false;
let xK='wing_span', yK='wing_area', zK='mass_GTOW';

function nice(k){ return k.replace(/_/g,' ').replace(/\b\w/g,l=>l.toUpperCase()) }
function fmt(v){ return typeof v==='number'?(Math.abs(v)<.001&&v!==0?v.toExponential(3):
  Math.abs(v)>=1e4?v.toFixed(0):v.toFixed(4)):v }

/* ═══════ INIT UI ═══════ */
/* header: hide DOE stats initially */
document.querySelectorAll('.hs').forEach(e=>e.style.display='none');

/* dropdowns */
['sx','sy','sz','sc'].forEach(id=>{
  const sel=document.getElementById(id);
  if(id==='sc'){const o=document.createElement('option');o.value='_feas';o.textContent='Feasibility';sel.appendChild(o)}
  D.keys.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=nice(k);sel.appendChild(o)});
});
document.getElementById('sx').value=xK;
document.getElementById('sy').value=yK;
document.getElementById('sz').value=zK;
document.getElementById('sx').onchange=function(){xK=this.value;draw()};
document.getElementById('sy').onchange=function(){yK=this.value;draw()};
document.getElementById('sz').onchange=function(){zK=this.value;draw()};

/* spreadsheet column selector */
const ssCols=document.getElementById('ss-cols');
D.keys.forEach(k=>{
  const o=document.createElement('option');o.value=k;o.textContent=nice(k);
  if(k===xK||k===yK||k===zK)o.selected=true;ssCols.appendChild(o);
});

/* bounds table */
(function(){
  let h='<table class="bt"><thead><tr><th>Variable</th><th>Lower</th><th>Base</th><th>Upper</th></tr></thead><tbody>';
  D.indep.forEach(k=>{
    const b=D.bounds[k];
    h+='<tr><td>'+k+'</td><td>'+fmt(b[0])+'</td><td>'+fmt(b[1])+'</td><td>'+fmt(b[2])+'</td></tr>';
  });
  D.derived.forEach(k=>{
    h+='<tr class="der"><td>'+k+'</td><td colspan="3" style="text-align:center;font-size:.56rem">derived &mdash; baseline '+fmt(D.baseline[k])+'</td></tr>';
  });
  h+='</tbody></table>';
  document.getElementById('bt-wrap').innerHTML=h;
})();

/* constraint list */
(function(){
  let h='';
  D.constraints.forEach(c=>{
    h+='<div class="con idle" id="con-'+c.id+'">'+
       '<div class="con-icon">\u25CB</div>'+
       '<div class="con-name">'+c.name+'</div>'+
       '<div class="con-rule">'+c.rule+'</div>'+
       '<div class="con-ct" id="cc-'+c.id+'"></div></div>';
  });
  document.getElementById('con-list').innerHTML=h;
})();

/* ═══════════════════════════════════════════════════════════════
   K-FACTOR DERIVED FORMULAS (JS port for spreadsheet edits)
   ═══════════════════════════════════════════════════════════════ */
const K_BLEND=2.24, K_AR=2.143, K_IXX=0.00455, K_FUEL=0.0167;
function recompute(s){
  s.wing_taper=s.wing_tip_chord/s.wing_root_chord;
  s.wing_area=K_BLEND*s.wing_span*(s.wing_root_chord+s.wing_tip_chord)/2;
  const strap=s.wing_span*(s.wing_root_chord+s.wing_tip_chord)/2;
  s.wing_AR=strap>0?K_AR*s.wing_span*s.wing_span/strap:0;
  s.mass_GTOW=s.mass_empty+s.mass_fuel+s.mass_payload;
  s.Ixx=K_IXX*s.mass_GTOW*Math.pow(s.wing_span/2,2);
  s.vol_fuel=K_FUEL*s.wing_area*((s.wing_root_chord+s.wing_tip_chord)/2);
}
function jsValidate(s){
  const r=[];
  if(s.wing_tip_chord>s.wing_root_chord)r.push('Inverted taper');
  const t=s.wing_taper;if(t<0.08||t>0.65)r.push('Taper '+t.toFixed(3)+' outside [0.08,0.65]');
  const ar=s.wing_AR;if(ar<2||ar>16)r.push('AR '+ar.toFixed(2)+' outside [2,16]');
  if(s.mass_GTOW<=s.mass_empty)r.push('GTOW <= empty');
  if(s.thrust_cruise>=s.thrust_max)r.push('Cruise thrust >= max');
  const fvn=s.mass_fuel/800,fva=s.vol_fuel;
  if(fva>0&&fvn>fva/0.6)r.push('Fuel vol tight');
  const a=s.wing_area,g=s.mass_GTOW;
  if(a>0){const wl=g/a;if(wl<100||wl>900)r.push('Wing loading '+wl.toFixed(0))}
  if(s.Cma>=0)r.push('Cma>=0');
  const fi=s.fuse_fineness;if(fi<5||fi>20)r.push('Fuse fineness '+fi.toFixed(1));
  return r;
}

/* ═══════════════════════════════════════════════════════════════
   3-D PLOT
   ═══════════════════════════════════════════════════════════════ */
let drawPending=false;
function requestDraw(){if(!drawPending){drawPending=true;requestAnimationFrame(()=>{draw();drawPending=false})}}

function draw(){
  const ps=+document.getElementById('ssSlider').value;
  const cb=document.getElementById('sc').value;
  const cop=(+document.getElementById('cslider').value)/100;
  const traces=[];

  if(DOE.n>0){
    const xs=DOE.samples[xK],ys=DOE.samples[yK],zs=DOE.samples[zK];
    const fI=[],iI=[];
    for(let j=0;j<DOE.n;j++){if(DOE.feas[j])fI.push(j);else iI.push(j)}

    function cvals(idx){if(cb==='_feas')return null;return idx.map(j=>DOE.samples[cb][j])}

    if(sF&&fI.length){
      const cv=cvals(fI);
      const mk=cv
        ?{size:ps,color:cv,colorscale:'Viridis',opacity:.92,
          colorbar:{title:{text:nice(cb),font:{size:10,color:'#9ca3af'}},thickness:10,len:.5,
            tickfont:{color:'#9ca3af',size:9},outlinewidth:0,bgcolor:'rgba(30,33,40,0.6)'},
          line:{width:1,color:'rgba(109,186,138,0.35)'}}
        :{size:ps,color:'rgba(80,220,140,0.92)',opacity:.92,
          line:{width:1,color:'rgba(120,255,170,0.3)'}};
      traces.push({type:'scatter3d',mode:'markers',
        x:fI.map(j=>xs[j]),y:fI.map(j=>ys[j]),z:fI.map(j=>zs[j]),
        marker:mk,name:'Feasible ('+fI.length+')',customdata:fI,
        hovertemplate:'<b>#%{customdata}</b><br>'+nice(xK)+': %{x:.4g}<br>'+nice(yK)+': %{y:.4g}<br>'+nice(zK)+': %{z:.4g}<br>Feasible<extra></extra>'
      });
    }
    if(sI&&iI.length){
      const cv=cvals(iI);
      const mk=cv
        ?{size:ps*.7,color:cv,colorscale:'Viridis',opacity:.35,showscale:false,
          line:{width:.5,color:'rgba(212,114,114,0.2)'}}
        :{size:ps*.7,color:'rgba(230,100,100,0.55)',opacity:.4,
          line:{width:.5,color:'rgba(255,130,130,0.25)'}};
      traces.push({type:'scatter3d',mode:'markers',
        x:iI.map(j=>xs[j]),y:iI.map(j=>ys[j]),z:iI.map(j=>zs[j]),
        marker:mk,name:'Infeasible ('+iI.length+')',customdata:iI,
        hovertemplate:'<b>#%{customdata}</b><br>'+nice(xK)+': %{x:.4g}<br>'+nice(yK)+': %{y:.4g}<br>'+nice(zK)+': %{z:.4g}<br>Rejected<extra></extra>'
      });
    }
    if(sC&&fI.length>10){
      traces.push({type:'mesh3d',
        x:fI.map(j=>xs[j]),y:fI.map(j=>ys[j]),z:fI.map(j=>zs[j]),
        opacity:cop,color:'rgba(109,186,138,0.6)',alphahull:7,
        name:'Feasible Region',hoverinfo:'skip',showlegend:true,
        lighting:{ambient:.9,diffuse:.3,specular:.05,roughness:.9},flatshading:true
      });
    }
  }

  if(sB){
    traces.push({type:'scatter3d',mode:'markers+text',
      x:[D.baseline[xK]],y:[D.baseline[yK]],z:[D.baseline[zK]],
      marker:{size:ps+6,color:'rgba(230,195,70,0.95)',symbol:'diamond',
        line:{width:2,color:'rgba(255,230,120,0.5)'}},
      text:['Baseline'],textposition:'top center',
      textfont:{size:9,color:'#c9a84e',family:'IBM Plex Mono'},name:'Baseline',
      hovertemplate:'<b>Baseline</b><br>'+nice(xK)+': %{x:.4g}<br>'+nice(yK)+': %{y:.4g}<br>'+nice(zK)+': %{z:.4g}<extra></extra>'
    });
  }

  const layout={
    paper_bgcolor:'#1a1d23',
    font:{color:'#7b8494',family:'IBM Plex Sans, Segoe UI, sans-serif',size:11},
    margin:{l:0,r:0,t:0,b:0},
    scene:{
      xaxis:{title:{text:nice(xK),font:{size:11,color:'#9ca3af'}},
        backgroundcolor:'#1a1d23',gridcolor:'#303440',gridwidth:1,
        showbackground:true,zerolinecolor:'#3a3e4a',
        tickfont:{size:10,color:'#9ca3af'},linecolor:'#3a3e4a',mirror:true},
      yaxis:{title:{text:nice(yK),font:{size:11,color:'#9ca3af'}},
        backgroundcolor:'#1a1d23',gridcolor:'#303440',gridwidth:1,
        showbackground:true,zerolinecolor:'#3a3e4a',
        tickfont:{size:10,color:'#9ca3af'},linecolor:'#3a3e4a',mirror:true},
      zaxis:{title:{text:nice(zK),font:{size:11,color:'#9ca3af'}},
        backgroundcolor:'#1a1d23',gridcolor:'#303440',gridwidth:1,
        showbackground:true,zerolinecolor:'#3a3e4a',
        tickfont:{size:10,color:'#9ca3af'},linecolor:'#3a3e4a',mirror:true},
      bgcolor:'#1a1d23',aspectmode:'cube',camera:{eye:{x:1.5,y:1.5,z:0.85}},
    },
    legend:{x:.01,y:.99,bgcolor:'rgba(26,29,35,0.9)',
      bordercolor:'#2a2e36',borderwidth:1,font:{size:10,color:'#7b8494'}},
    showlegend:true,
  };

  Plotly.react('plot3d',traces,layout,{
    displayModeBar:true,modeBarButtonsToRemove:['resetCameraDefault3d'],
    displaylogo:false,responsive:true,
    toImageButtonOptions:{format:'png',width:1920,height:1080,scale:2}
  });

  const el=document.getElementById('plot3d');
  el.removeAllListeners&&el.removeAllListeners('plotly_click');
  el.on('plotly_click',onPt);
}

/* ═══════════════════════════════════════════════════════════════
   DOE STREAMING
   ═══════════════════════════════════════════════════════════════ */
let totalExpected=0;

function startDOE(){
  const n=parseInt(document.getElementById('gen-n').value)||500;
  const seed=parseInt(document.getElementById('gen-seed').value)||42;

  STATE='generating';
  totalExpected=n+1;

  /* reset DOE data */
  D.keys.forEach(k=>DOE.samples[k]=[]);
  DOE.feas=[];DOE.reasons=[];DOE.n=0;DOE.nFeas=0;DOE.details={};
  D.constraints.forEach(c=>{CF[c.id]=0});
  resetConstraintUI();

  /* UI: show progress, swap header stats */
  document.getElementById('gen-btn').disabled=true;
  document.getElementById('gen-btn').textContent='Generating\u2026';
  document.getElementById('gen-pg').style.display='block';
  document.getElementById('st-v').style.display='none';
  document.querySelectorAll('.hs').forEach(e=>e.style.display='');
  document.getElementById('subtitle').textContent=
    'Stage 2 DOE \u2014 generating '+n+' LHS samples\u2026';

  const source=new EventSource('/api/stream-doe?n='+n+'&seed='+seed);

  source.addEventListener('phase',function(e){
    const info=JSON.parse(e.data);
    const st=document.getElementById('gen-st');
    st.className='gen-st active';
    st.innerHTML='<span class="spinner"></span>'+info.msg;
    if(info.total) totalExpected=info.total;
  });

  source.onmessage=function(e){
    const batch=JSON.parse(e.data);
    batch.forEach(s=>{
      D.keys.forEach(k=>DOE.samples[k].push(s.v[k]||0));
      DOE.feas.push(s.f);
      DOE.reasons.push((s.r||[]).join('; '));
      DOE.n++;
      if(s.f) DOE.nFeas++;
      if(!s.f && s.r && s.r.length) catFails(s.r);
    });
    updateProgress();
    requestDraw();
  };

  source.addEventListener('done',function(e){
    source.close();
    STATE='complete';
    onDOEComplete();
  });

  source.onerror=function(){
    source.close();
    if(DOE.n>0){STATE='complete';onDOEComplete()}
  };
}

function catFails(reasons){
  reasons.forEach(r=>{
    if(/inverted taper/i.test(r)) CF.inverted_taper++;
    else if(/taper ratio/i.test(r)) CF.taper_range++;
    else if(/aspect ratio/i.test(r)) CF.ar_range++;
    else if(/gtow|empty/i.test(r)) CF.gtow++;
    else if(/cruise thrust|thrust/i.test(r)) CF.thrust++;
    else if(/fuel vol/i.test(r)) CF.fuel_vol++;
    else if(/wing load/i.test(r)) CF.wing_loading++;
    else if(/cma|unstable/i.test(r)) CF.stability++;
    else if(/fineness/i.test(r)) CF.fineness++;
  });
  updateConstraintUI();
}

function resetConstraintUI(){
  D.constraints.forEach(c=>{
    const el=document.getElementById('con-'+c.id);
    el.className='con idle';
    el.querySelector('.con-icon').textContent='\u25CB';
    document.getElementById('cc-'+c.id).textContent='';
  });
}

function updateConstraintUI(){
  D.constraints.forEach(c=>{
    const el=document.getElementById('con-'+c.id);
    const cc=document.getElementById('cc-'+c.id);
    if(CF[c.id]>0){
      el.className='con fail';
      el.querySelector('.con-icon').textContent='\u2717';
      cc.textContent=CF[c.id]+' fail';
    } else {
      el.className='con pass';
      el.querySelector('.con-icon').textContent='\u2713';
      cc.textContent='';
    }
  });
}

function updateProgress(){
  const pct=Math.min(100,(DOE.n/totalExpected)*100);
  document.getElementById('gen-bar').style.width=pct+'%';
  document.getElementById('gen-pct').textContent=DOE.n+' / '+totalExpected+' samples';
  document.getElementById('hN').textContent=DOE.n;
  document.getElementById('hF').textContent=DOE.nFeas;
  document.getElementById('hI').textContent=DOE.n-DOE.nFeas;
  document.getElementById('hP').textContent=DOE.n>0?(100*DOE.nFeas/DOE.n).toFixed(1)+'%':'\u2014';
}

function onDOEComplete(){
  /* progress area */
  const st=document.getElementById('gen-st');
  st.className='gen-st';
  st.innerHTML='\u2714 Complete \u2014 '+DOE.nFeas+' / '+DOE.n+' feasible ('+
    (100*DOE.nFeas/DOE.n).toFixed(1)+'%)';
  document.getElementById('gen-bar').style.width='100%';
  document.getElementById('gen-btn').textContent='\u25B6\u2002Regenerate DOE';
  document.getElementById('gen-btn').disabled=false;

  document.getElementById('subtitle').textContent=
    'Stage 2 DOE \u2014 '+DOE.n+' samples \u00b7 '+DOE.nFeas+' feasible \u00b7 edit live';

  /* enable tabs */
  document.getElementById('tbtn-insp').classList.remove('dis');
  document.getElementById('tbtn-ss').classList.remove('dis');

  /* turn cloud ON */
  sC=true;
  document.getElementById('bC').classList.add('ab');

  /* final constraint tally — mark zero-fail constraints green */
  updateConstraintUI();

  draw();
}

/* ═══════════════════════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════════════════════ */
function switchTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('act'));
  document.querySelectorAll('.tab-body').forEach(t=>t.classList.remove('act'));
  if(name==='design'){
    document.querySelectorAll('.tab')[0].classList.add('act');
    document.getElementById('tab-design').classList.add('act');
  } else if(name==='inspector'){
    document.querySelectorAll('.tab')[1].classList.add('act');
    document.getElementById('tab-inspector').classList.add('act');
  } else {
    document.querySelectorAll('.tab')[2].classList.add('act');
    document.getElementById('tab-spreadsheet').classList.add('act');
    if(STATE==='complete') buildTable();
  }
}

/* ═══════════════════════════════════════════════════════════════
   INSPECTOR (on-demand detail fetch)
   ═══════════════════════════════════════════════════════════════ */
function onPt(ev){
  if(!ev||!ev.points||!ev.points.length)return;
  const idx=ev.points[0].customdata;if(idx===undefined||idx===null)return;
  if(STATE!=='complete')return;
  switchTab('inspector');showInspector(idx);
}

async function showInspector(idx){
  document.getElementById('ie').style.display='none';
  const c=document.getElementById('ic');c.style.display='block';

  if(!DOE.details[idx]){
    c.innerHTML='<div class="empty">Loading\u2026</div>';
    const sample={};D.keys.forEach(k=>sample[k]=DOE.samples[k][idx]);
    try{
      const resp=await fetch('/api/inspect',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({idx,sample})});
      DOE.details[idx]=await resp.json();
    }catch(e){c.innerHTML='<div class="empty">Failed to load</div>';return}
  }

  const f=DOE.feas[idx],r=DOE.reasons[idx],det=DOE.details[idx];
  let h='<div class="ph"><span class="id">#'+idx+'</span>'+
    (f?'<span class="badge bp">Feasible</span>':'<span class="badge bf">Infeasible</span>')+'</div>';
  if(r)h+='<div class="fr">'+r.replace(/;/g,'<br>')+'</div>';
  for(const[g,ps]of Object.entries(det)){
    h+='<div class="pg"><div class="pgh">'+g+'</div>';
    for(const[k,v]of Object.entries(ps)){
      const fv=typeof v==='number'?(Math.abs(v)<.001&&v!==0?v.toExponential(3):v.toFixed(4)):v;
      h+='<div class="pr"><span class="k">'+k+'</span><span class="vl">'+fv+'</span></div>';
    }h+='</div>';
  }c.innerHTML=h;
}

/* ═══════════════════════════════════════════════════════════════
   SPREADSHEET
   ═══════════════════════════════════════════════════════════════ */
function getSelectedCols(){return Array.from(document.getElementById('ss-cols').selectedOptions).map(o=>o.value)}
function addAxisCols(){
  Array.from(document.getElementById('ss-cols').options).forEach(o=>{
    if(o.value===xK||o.value===yK||o.value===zK)o.selected=true;
  });buildTable();
}

let tableBuilt=false;
function buildTable(){
  if(STATE!=='complete')return;
  const cols=getSelectedCols();
  if(!cols.length){document.getElementById('ss-wrap').innerHTML='<div class="empty">Select columns above</div>';return}
  const isDerived=new Set(D.derived||[]);
  let h='<table class="ss"><thead><tr><th>#</th><th>F</th>';
  cols.forEach(c=>h+='<th'+(isDerived.has(c)?' style="color:#c9a84e"':'')+'>'+nice(c)+'</th>');
  h+='</tr></thead><tbody>';
  for(let i=0;i<DOE.n;i++){
    const f=DOE.feas[i];
    h+='<tr class="'+(f?'feas-row':'infeas-row')+'">';
    h+='<td class="id-col">'+i+'</td>';
    h+='<td class="feas-col" style="color:'+(f?'#6dba8a':'#d47272')+'">'+(f?'\u2713':'\u2717')+'</td>';
    cols.forEach(c=>{
      const v=DOE.samples[c][i];
      const editable=!isDerived.has(c);
      const fv=typeof v==='number'?(Math.abs(v)<.001&&v!==0?v.toExponential(3):v.toFixed(4)):v;
      h+='<td'+(editable?' contenteditable="true" data-r="'+i+'" data-c="'+c+'"':' style="color:#7b8494"')+'>'+fv+'</td>';
    });h+='</tr>';
  }
  h+='</tbody></table>';
  document.getElementById('ss-wrap').innerHTML=h;
  document.querySelectorAll('.ss td[contenteditable]').forEach(td=>{
    td.addEventListener('blur',onCellEdit);
    td.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();td.blur()}});
  });
  tableBuilt=true;
}

function onCellEdit(e){
  const td=e.target,row=+td.dataset.r,col=td.dataset.c;
  const val=parseFloat(td.textContent.trim());if(isNaN(val))return;
  DOE.samples[col][row]=val;td.classList.add('edited');
  const s={};D.keys.forEach(k=>s[k]=DOE.samples[k][row]);
  recompute(s);D.keys.forEach(k=>DOE.samples[k][row]=s[k]);
  const reasons=jsValidate(s);DOE.feas[row]=reasons.length===0?1:0;DOE.reasons[row]=reasons.join('; ');
  let nf=0;for(let j=0;j<DOE.n;j++)nf+=DOE.feas[j];DOE.nFeas=nf;
  document.getElementById('hF').textContent=nf;
  document.getElementById('hI').textContent=DOE.n-nf;
  document.getElementById('hP').textContent=(100*nf/DOE.n).toFixed(1)+'%';
  draw();
  const tr=td.closest('tr');if(tr){
    tr.className=DOE.feas[row]?'feas-row':'infeas-row';
    const ft=tr.querySelectorAll('td')[1];
    if(ft){ft.textContent=DOE.feas[row]?'\u2713':'\u2717';ft.style.color=DOE.feas[row]?'#6dba8a':'#d47272'}
    tr.querySelectorAll('td[data-r]').forEach(cell=>{
      const ck=cell.dataset.c;
      if((D.derived||[]).includes(ck)){const v=DOE.samples[ck][row];
        cell.textContent=Math.abs(v)<.001&&v!==0?v.toExponential(3):v.toFixed(4)}
    });
  }
  fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({row,col,val})}).catch(()=>{});
}
function resetEdits(){location.reload()}

/* ═══════════════════════════════════════════════════════════════
   TOGGLES / PRESETS
   ═══════════════════════════════════════════════════════════════ */
function tv(w){
  if(w==='f'){sF=!sF;document.getElementById('bF').classList.toggle('ag',sF)}
  else if(w==='i'){sI=!sI;document.getElementById('bI').classList.toggle('ar',sI)}
  else if(w==='b'){sB=!sB;document.getElementById('bB').classList.toggle('ay',sB)}
  else if(w==='c'){sC=!sC;document.getElementById('bC').classList.toggle('ab',sC)}
  draw();
}
function ax(x,y,z){
  xK=x;yK=y;zK=z;
  document.getElementById('sx').value=x;document.getElementById('sy').value=y;document.getElementById('sz').value=z;
  draw();if(tableBuilt)buildTable();
}

/* ═══════ INITIAL DRAW (baseline only) ═══════ */
draw();
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════════
#  SSE STREAMING DOE GENERATOR
# ═══════════════════════════════════════════════════════════════════
def stream_doe(handler, n_samples, seed):
    """Stream DOE samples to the client via Server-Sent Events."""
    from scipy.stats.qmc import LatinHypercube

    keys    = sorted(BOUNDS.keys())
    n_dims  = len(keys)
    lowers  = np.array([BOUNDS[k][0] for k in keys])
    uppers  = np.array([BOUNDS[k][2] for k in keys])
    spans   = uppers - lowers

    def send(event_type, data):
        msg = ""
        if event_type:
            msg += f"event: {event_type}\n"
        msg += f"data: {json.dumps(data)}\n\n"
        handler.wfile.write(msg.encode("utf-8"))
        handler.wfile.flush()

    try:
        total = n_samples + 1  # +1 for baseline

        # Reset server store
        with DOE_LOCK:
            DOE_STORE["samples"] = {k: [] for k in ALL_KEYS}
            DOE_STORE["feas"]    = []
            DOE_STORE["reasons"] = []
            DOE_STORE["n"]       = 0

        # ── Phase 1: Baseline ────────────────────────────────────
        send("phase", {"phase": "baseline",
                        "msg": "Injecting baseline design\u2026",
                        "total": total})

        bl_indep = {k: BASELINE_RAW[k] for k in keys}
        bl_full  = compute_derived(bl_indep)
        for k, v in BASELINE_RAW.items():
            if k not in bl_full:
                bl_full[k] = v
        ok, reasons = validate_sample(bl_full)

        send(None, [{"v": {k: bl_full.get(k, 0) for k in ALL_KEYS},
                      "f": int(ok),
                      "r": [str(r) for r in reasons],
                      "bl": 1}])

        with DOE_LOCK:
            for k in ALL_KEYS:
                DOE_STORE["samples"][k].append(bl_full.get(k, 0))
            DOE_STORE["feas"].append(int(ok))
            DOE_STORE["reasons"].append("; ".join(str(r) for r in reasons))
            DOE_STORE["n"] += 1

        # ── Phase 2: LHS optimisation ────────────────────────────
        send("phase", {"phase": "lhs",
                        "msg": f"Optimizing {n_samples}-point Latin Hypercube ({n_dims}D)\u2026",
                        "total": total})

        sampler  = LatinHypercube(d=n_dims, seed=seed, optimization="random-cd")
        unit_lhs = sampler.random(n=n_samples)

        # ── Phase 3: Stream validation ───────────────────────────
        send("phase", {"phase": "validate",
                        "msg": "Running physics validation\u2026",
                        "total": total})

        BATCH = 5
        batch = []
        for i in range(n_samples):
            row = unit_lhs[i]
            physical = {k: float(lowers[j] + row[j] * spans[j])
                        for j, k in enumerate(keys)}
            full = compute_derived(physical)
            ok, reasons = validate_sample(full)

            with DOE_LOCK:
                for k in ALL_KEYS:
                    DOE_STORE["samples"][k].append(full.get(k, 0))
                DOE_STORE["feas"].append(int(ok))
                DOE_STORE["reasons"].append("; ".join(str(r) for r in reasons))
                DOE_STORE["n"] += 1

            batch.append({"v": {k: full.get(k, 0) for k in ALL_KEYS},
                           "f": int(ok),
                           "r": [str(r) for r in reasons]})

            if len(batch) >= BATCH or i == n_samples - 1:
                send(None, batch)
                batch = []
                time.sleep(0.012)  # pace for smooth animation

        # ── Done ─────────────────────────────────────────────────
        nf = sum(DOE_STORE["feas"])
        send("done", {"total": DOE_STORE["n"], "feasible": nf})

    except (BrokenPipeError, ConnectionResetError,
            ConnectionAbortedError, OSError):
        pass  # client disconnected


# ═══════════════════════════════════════════════════════════════════
#  HTTP HANDLER
# ═══════════════════════════════════════════════════════════════════
FULL_PAGE = None  # built once on first request

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        global FULL_PAGE
        parsed = urlparse(self.path)

        if parsed.path == "/":
            if FULL_PAGE is None:
                FULL_PAGE = PAGE.replace("__INIT_JSON__", INIT_JSON).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(FULL_PAGE)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(FULL_PAGE)

        elif parsed.path == "/api/stream-doe":
            params = parse_qs(parsed.query)
            n    = int(params.get("n", ["500"])[0])
            seed = int(params.get("seed", ["42"])[0])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            stream_doe(self, n, seed)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        parsed = urlparse(self.path)

        if parsed.path == "/api/inspect":
            try:
                payload = json.loads(body)
                dv      = DesignVector.from_dict(payload["sample"])
                result  = dv.cad_params()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif parsed.path == "/api/update":
            try:
                payload = json.loads(body)
                row, col, val = payload["row"], payload["col"], payload["val"]
                with DOE_LOCK:
                    if col in DOE_STORE["samples"] and 0 <= row < DOE_STORE["n"]:
                        DOE_STORE["samples"][col][row] = val
                        s = {k: DOE_STORE["samples"][k][row] for k in ALL_KEYS}
                        s = compute_derived(s)
                        for k in ALL_KEYS:
                            DOE_STORE["samples"][k][row] = s[k]
                        ok, reasons = validate_sample(s)
                        DOE_STORE["feas"][row] = int(ok)
                        DOE_STORE["reasons"][row] = "; ".join(str(r) for r in reasons)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ═══════════════════════════════════════════════════════════════════
#  LAUNCH
# ═══════════════════════════════════════════════════════════════════
def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")

print(f"\n{'='*56}")
print(f"  MDO DESIGN EXPLORER")
print(f"  -> http://{HOST}:{PORT}")
print(f"  {len(ALL_KEYS)} variables | {len(CONSTRAINTS)} constraints")
print(f"  Press Ctrl+C to stop")
print(f"{'='*56}\n")

print("[2/2] Server starting ...")
threading.Timer(1.0, open_browser).start()

try:
    ThreadedServer((HOST, PORT), Handler).serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
