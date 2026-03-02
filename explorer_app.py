"""
Local MDO Design Explorer — CCAV 3D Interactive Visualization
==============================================================
Run:   python explorer_app.py
Opens: http://127.0.0.1:5050   (Ctrl+C to stop)
Features:
  - 3-D scatter (Feasible / Infeasible / Baseline)
  - Translucent feasible-region cloud (mesh3d alphahull) ON by default
  - Live-editable spreadsheet tab — edits update the 3-D plot instantly
  - Server-side persistence via POST /api/update
"""
import json, sys, threading, webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.stage1_design_space import (
    get_independent_bounds, get_baseline_vector,
    compute_derived, validate_sample, get_derived_keys,
)
from pipeline.stage2_doe import generate_doe
from pipeline.design_vector import DesignVector

PORT = 5050
HOST = "127.0.0.1"

# ═══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("[1/3] Generating 500 LHS samples ...")
ALL, FEAS, REASONS = generate_doe(
    n_samples=500, seed=42, include_baseline=True, verbose=False
)
BOUNDS   = get_independent_bounds()
BASELINE = get_baseline_vector()
ALL_KEYS = sorted(ALL[0].keys())
DERIVED  = get_derived_keys()
N        = len(ALL)
N_FEAS   = int(sum(FEAS))

print(f"      {N_FEAS}/{N} feasible  ({100*N_FEAS/N:.1f} %)")

# ── blob ──────────────────────────────────────────────────────────
print("[2/3] Building data blob ...")

bounds_dict = {}
for k, (lo, base, hi) in BOUNDS.items():
    bounds_dict[k] = [lo, base, hi]

blob = {
    "keys":     ALL_KEYS,
    "derived":  DERIVED,
    "bounds":   bounds_dict,
    "n":        N,
    "nFeas":    N_FEAS,
    "baseline": {k: BASELINE.get(k, 0) for k in ALL_KEYS},
    "feas":     [int(f) for f in FEAS],
    "reasons":  ["; ".join(r) if r else "" for r in REASONS],
    "samples":  {k: [s.get(k, 0) for s in ALL] for k in ALL_KEYS},
}

details = []
for s in ALL:
    try:
        dv = DesignVector.from_dict(s)
        details.append(dv.cad_params())
    except Exception:
        details.append({"raw": s})
blob["details"] = details

DATA_JSON = json.dumps(blob)


# ═══════════════════════════════════════════════════════════════════
#  HTML PAGE
# ═══════════════════════════════════════════════════════════════════
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MDO Design Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{background:#1a1d23;color:#c8ccd4;font-family:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;
  font-weight:400;-webkit-font-smoothing:antialiased}

.hdr{background:#1e2128;border-bottom:1px solid #2a2e36;padding:10px 24px;
  display:flex;justify-content:space-between;align-items:center;height:52px}
.hdr h1{font-size:1rem;font-weight:600;letter-spacing:.6px;color:#d4d8e0}
.hdr h1 span{color:#7b8494;font-weight:400}
.hdr .sub{font-size:.68rem;color:#5a6070;margin-top:1px;font-weight:300}
.stats{display:flex;gap:12px}
.st{text-align:center;padding:3px 12px}
.st .v{font-size:1rem;font-weight:500;font-family:'IBM Plex Mono',monospace}
.st .l{font-size:.58rem;color:#5a6070;text-transform:uppercase;letter-spacing:.8px}

.main{display:grid;grid-template-columns:240px 1fr 340px;height:calc(100vh - 52px)}

.pnl{background:#1e2128;padding:14px 16px;overflow-y:auto;
  scrollbar-width:thin;scrollbar-color:#2a2e36 transparent}
.pnl:first-child{border-right:1px solid #2a2e36}
.pnl:last-child{border-left:1px solid #2a2e36}
.pnl h3{font-size:.64rem;font-weight:500;letter-spacing:1px;text-transform:uppercase;
  margin-bottom:10px;color:#5a6070;padding-bottom:5px;border-bottom:1px solid #2a2e36}

.cg{margin-bottom:11px}
.cg label{display:block;font-size:.64rem;color:#5a6070;margin-bottom:3px;
  text-transform:uppercase;letter-spacing:.4px}
select{width:100%;padding:6px 8px;background:#262a33;color:#b0b6c2;
  border:1px solid #2a2e36;border-radius:4px;font-size:.76rem;
  font-family:'IBM Plex Sans',sans-serif;cursor:pointer;outline:none;transition:.15s}
select:hover{border-color:#3e4450}
select:focus{border-color:#5a6070}

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

/* ── TABS ── */
.tabs{display:flex;gap:0;margin-bottom:10px;border-bottom:1px solid #2a2e36}
.tab{padding:6px 14px;font-size:.68rem;font-weight:500;color:#5a6070;cursor:pointer;
  border-bottom:2px solid transparent;transition:.15s;text-transform:uppercase;letter-spacing:.5px}
.tab:hover{color:#9ca3af}
.tab.act{color:#b0b6c2;border-bottom-color:#6dba8a}
.tab-body{display:none}
.tab-body.act{display:block}

/* ── INSPECTOR ── */
.ins .empty{color:#3e4450;font-size:.78rem;margin-top:20px;text-align:center;line-height:1.6}
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
.pr .k{color:#5a6070}
.pr .vl{color:#b0b6c2;font-family:'IBM Plex Mono',monospace;font-size:.66rem}

/* ── SPREADSHEET ── */
.ss-wrap{max-height:calc(100vh - 180px);overflow:auto;
  scrollbar-width:thin;scrollbar-color:#2a2e36 transparent}
.ss{width:100%;border-collapse:collapse;font-size:.68rem}
.ss th{position:sticky;top:0;background:#262a33;color:#7b8494;font-weight:500;
  padding:4px 6px;text-align:left;font-size:.62rem;text-transform:uppercase;
  letter-spacing:.5px;border-bottom:1px solid #2a2e36;z-index:2}
.ss td{padding:3px 6px;border-bottom:1px solid rgba(255,255,255,.02);
  font-family:'IBM Plex Mono',monospace;font-size:.66rem;color:#b0b6c2;white-space:nowrap}
.ss tr:hover td{background:rgba(255,255,255,.02)}
.ss tr.feas-row td:first-child{border-left:2px solid rgba(109,186,138,.4)}
.ss tr.infeas-row td:first-child{border-left:2px solid rgba(212,114,114,.3)}
.ss td[contenteditable]{cursor:text;outline:none;border-radius:2px;transition:.1s}
.ss td[contenteditable]:focus{background:rgba(109,186,138,.08);outline:1px solid rgba(109,186,138,.25)}
.ss td.edited{color:#6dba8a}
.ss td.id-col{color:#5a6070;font-weight:500;cursor:default}
.ss td.feas-col{font-size:.6rem}
.ss-info{font-size:.62rem;color:#5a6070;margin-bottom:6px;font-style:italic}
.ss-toolbar{display:flex;gap:6px;margin-bottom:8px;align-items:center;flex-wrap:wrap}
.ss-toolbar select{width:auto;padding:4px 6px;font-size:.66rem}
.ss-toolbar .ss-btn{padding:3px 10px;border-radius:3px;border:1px solid #2a2e36;
  background:transparent;color:#6dba8a;font-size:.64rem;cursor:pointer;transition:.15s;
  font-family:'IBM Plex Sans',sans-serif}
.ss-toolbar .ss-btn:hover{background:rgba(109,186,138,.08);border-color:#3a7a5e}
.ss-toolbar .ss-btn.warn{color:#d47272}
.ss-toolbar .ss-btn.warn:hover{background:rgba(212,114,114,.08);border-color:#7a3a3a}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <h1>Design Explorer <span>/ CCAV</span></h1>
    <div class="sub">Stage 2 DOE &mdash; 34-variable design space &middot; edit live</div>
  </div>
  <div class="stats">
    <div class="st"><div class="v" style="color:#b0b6c2" id="hN"></div><div class="l">Samples</div></div>
    <div class="st"><div class="v" style="color:#6dba8a" id="hF"></div><div class="l">Feasible</div></div>
    <div class="st"><div class="v" style="color:#d47272" id="hI"></div><div class="l">Rejected</div></div>
    <div class="st"><div class="v" style="color:#c9a84e" id="hP"></div><div class="l">Pass Rate</div></div>
  </div>
</div>

<div class="main">
<!-- LEFT PANEL -->
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
    <button id="bC" class="tb ab" onclick="tv('c')">Feasible Cloud</button>
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

<!-- CENTER: 3-D PLOT -->
<div id="plot3d"></div>

<!-- RIGHT PANEL -->
<div class="pnl" id="rp">
  <div class="tabs">
    <div class="tab act" onclick="switchTab('inspector')">Inspector</div>
    <div class="tab" onclick="switchTab('spreadsheet')">Spreadsheet</div>
  </div>

  <div class="tab-body act" id="tab-inspector">
    <div class="empty" id="ie">Click any point to inspect<br>its design vector</div>
    <div id="ic" style="display:none"></div>
  </div>

  <div class="tab-body" id="tab-spreadsheet">
    <div class="ss-toolbar">
      <select id="ss-cols" multiple size="4" style="width:140px;font-size:.66rem"
              onchange="buildTable()"></select>
      <div style="display:flex;flex-direction:column;gap:3px">
        <button class="ss-btn" onclick="addAxisCols()">Show X/Y/Z</button>
        <button class="ss-btn warn" onclick="resetEdits()">Reset Edits</button>
      </div>
    </div>
    <div class="ss-info">Double-click cells to edit. Changes update the 3D plot live.</div>
    <div class="ss-wrap" id="ss-wrap"></div>
  </div>
</div>
</div>

<script>
/* ═══════ DATA ═══════ */
const D = __DATA_JSON__;

document.getElementById('hN').textContent=D.n;
document.getElementById('hF').textContent=D.nFeas;
document.getElementById('hI').textContent=D.n-D.nFeas;
document.getElementById('hP').textContent=(100*D.nFeas/D.n).toFixed(1)+'%';

/* ═══════ STATE ═══════ */
let sF=true,sI=true,sB=true,sC=true;
let xK='wing_span',yK='wing_area',zK='mass_GTOW';

function nice(k){return k.replace(/_/g,' ').replace(/\b\w/g,l=>l.toUpperCase())}

/* dropdowns */
['sx','sy','sz','sc'].forEach(id=>{
  const sel=document.getElementById(id);
  if(id==='sc'){const o=document.createElement('option');o.value='_feas';o.textContent='Feasibility';sel.appendChild(o)}
  D.keys.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=nice(k);sel.appendChild(o)});
});
document.getElementById('sx').value=xK;
document.getElementById('sy').value=yK;
document.getElementById('sz').value=zK;
document.getElementById('sx').onchange=function(){xK=this.value;draw();if(tableBuilt)buildTable()};
document.getElementById('sy').onchange=function(){yK=this.value;draw();if(tableBuilt)buildTable()};
document.getElementById('sz').onchange=function(){zK=this.value;draw();if(tableBuilt)buildTable()};

/* spreadsheet column selector */
const ssCols=document.getElementById('ss-cols');
D.keys.forEach(k=>{
  const o=document.createElement('option');o.value=k;o.textContent=nice(k);
  if(k===xK||k===yK||k===zK)o.selected=true;
  ssCols.appendChild(o);
});

/* ═══════ DERIVED + VALIDATION (JS port) ═══════ */
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

function validate(s){
  const r=[];
  if(s.wing_tip_chord>s.wing_root_chord)r.push('Inverted taper');
  const t=s.wing_taper;if(t<0.08||t>0.65)r.push('Taper '+t.toFixed(3)+' outside [0.08,0.65]');
  const ar=s.wing_AR;if(ar<2||ar>16)r.push('AR '+ar.toFixed(2)+' outside [2,16]');
  if(s.mass_GTOW<=s.mass_empty)r.push('GTOW <= empty');
  if(s.thrust_cruise>=s.thrust_max)r.push('Cruise thrust >= max');
  const fvn=s.mass_fuel/800, fva=s.vol_fuel;
  if(fva>0&&fvn>fva/0.6)r.push('Fuel vol tight');
  const a=s.wing_area, g=s.mass_GTOW;
  if(a>0){const wl=g/a;if(wl<100||wl>900)r.push('Wing loading '+wl.toFixed(0))}
  if(s.Cma>=0)r.push('Cma>=0');
  const fi=s.fuse_fineness;if(fi<5||fi>20)r.push('Fuse fineness '+fi.toFixed(1));
  return r;
}

/* ═══════ DRAW ═══════ */
function draw(){
  const xs=D.samples[xK],ys=D.samples[yK],zs=D.samples[zK];
  const ps=+document.getElementById('ssSlider').value;
  const cb=document.getElementById('sc').value;
  const cop=(+document.getElementById('cslider').value)/100;
  const fI=[],iI=[];
  for(let j=0;j<D.n;j++){if(D.feas[j])fI.push(j);else iI.push(j)}

  function cvals(idx){if(cb==='_feas')return null;return idx.map(j=>D.samples[cb][j])}
  const traces=[];

  /* feasible */
  if(sF&&fI.length){
    const cv=cvals(fI);
    const mk=cv
      ?{size:ps,color:cv,colorscale:'Viridis',opacity:0.92,
        colorbar:{title:{text:nice(cb),font:{size:10,color:'#9ca3af'}},thickness:10,len:0.5,
          tickfont:{color:'#9ca3af',size:9},outlinewidth:0,bgcolor:'rgba(30,33,40,0.6)'},
        line:{width:1,color:'rgba(109,186,138,0.35)'}}
      :{size:ps,color:'rgba(80,220,140,0.92)',opacity:0.92,
        line:{width:1,color:'rgba(120,255,170,0.3)'}};
    traces.push({type:'scatter3d',mode:'markers',
      x:fI.map(j=>xs[j]),y:fI.map(j=>ys[j]),z:fI.map(j=>zs[j]),
      marker:mk,name:'Feasible ('+fI.length+')',customdata:fI,
      hovertemplate:'<b>#%{customdata}</b><br>'+nice(xK)+': %{x:.4g}<br>'+nice(yK)+': %{y:.4g}<br>'+nice(zK)+': %{z:.4g}<br>Feasible<extra></extra>'
    });
  }

  /* infeasible */
  if(sI&&iI.length){
    const cv=cvals(iI);
    const mk=cv
      ?{size:ps*0.7,color:cv,colorscale:'Viridis',opacity:0.35,symbol:'circle',showscale:false,
        line:{width:0.5,color:'rgba(212,114,114,0.2)'}}
      :{size:ps*0.7,color:'rgba(230,100,100,0.55)',opacity:0.4,symbol:'circle',
        line:{width:0.5,color:'rgba(255,130,130,0.25)'}};
    traces.push({type:'scatter3d',mode:'markers',
      x:iI.map(j=>xs[j]),y:iI.map(j=>ys[j]),z:iI.map(j=>zs[j]),
      marker:mk,name:'Infeasible ('+iI.length+')',customdata:iI,
      hovertemplate:'<b>#%{customdata}</b><br>'+nice(xK)+': %{x:.4g}<br>'+nice(yK)+': %{y:.4g}<br>'+nice(zK)+': %{z:.4g}<br>Rejected<extra></extra>'
    });
  }

  /* baseline */
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

  /* translucent feasible cloud */
  if(sC&&fI.length>10){
    traces.push({type:'mesh3d',
      x:fI.map(j=>xs[j]),y:fI.map(j=>ys[j]),z:fI.map(j=>zs[j]),
      opacity:cop,color:'rgba(109,186,138,0.6)',alphahull:7,
      name:'Feasible Region',hoverinfo:'skip',showlegend:true,
      lighting:{ambient:0.9,diffuse:0.3,specular:0.05,roughness:0.9},
      flatshading:true
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
      bgcolor:'#1a1d23',aspectmode:'cube',
      camera:{eye:{x:1.5,y:1.5,z:0.85}},
    },
    legend:{x:0.01,y:0.99,bgcolor:'rgba(26,29,35,0.9)',
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

/* ═══════ INSPECTOR ═══════ */
function onPt(ev){
  if(!ev||!ev.points||!ev.points.length)return;
  const idx=ev.points[0].customdata;if(idx===undefined||idx===null)return;
  switchTab('inspector');showInspector(idx);
}
function showInspector(idx){
  document.getElementById('ie').style.display='none';
  const c=document.getElementById('ic');c.style.display='block';
  const f=D.feas[idx],r=D.reasons[idx],det=D.details[idx];
  let h='<div class="ph"><span class="id">#'+idx+'</span>'+
    (f?'<span class="badge bp">Feasible</span>':'<span class="badge bf">Infeasible</span>')+'</div>';
  if(r)h+='<div class="fr">'+r.replace(/;/g,'<br>')+'</div>';
  for(const[g,ps]of Object.entries(det)){
    h+='<div class="pg"><div class="pgh">'+g+'</div>';
    for(const[k,v]of Object.entries(ps)){
      const fv=typeof v==='number'?(Math.abs(v)<0.001&&v!==0?v.toExponential(3):v.toFixed(4)):v;
      h+='<div class="pr"><span class="k">'+k+'</span><span class="vl">'+fv+'</span></div>';
    }h+='</div>';
  }c.innerHTML=h;
}

/* ═══════ TABS ═══════ */
function switchTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('act'));
  document.querySelectorAll('.tab-body').forEach(t=>t.classList.remove('act'));
  if(name==='inspector'){
    document.querySelectorAll('.tab')[0].classList.add('act');
    document.getElementById('tab-inspector').classList.add('act');
  } else {
    document.querySelectorAll('.tab')[1].classList.add('act');
    document.getElementById('tab-spreadsheet').classList.add('act');
    buildTable();
  }
}

/* ═══════ SPREADSHEET ═══════ */
function getSelectedCols(){
  const sel=document.getElementById('ss-cols');
  return Array.from(sel.selectedOptions).map(o=>o.value);
}
function addAxisCols(){
  const sel=document.getElementById('ss-cols');
  Array.from(sel.options).forEach(o=>{
    if(o.value===xK||o.value===yK||o.value===zK)o.selected=true;
  });
  buildTable();
}

let tableBuilt=false;
function buildTable(){
  const cols=getSelectedCols();
  if(!cols.length){document.getElementById('ss-wrap').innerHTML='<div class="empty">Select columns above</div>';return}
  const isDerived=new Set(D.derived||[]);

  let h='<table class="ss"><thead><tr><th>#</th><th>F</th>';
  cols.forEach(c=>h+='<th'+(isDerived.has(c)?' style="color:#c9a84e"':'')+'>'+nice(c)+'</th>');
  h+='</tr></thead><tbody>';

  for(let i=0;i<D.n;i++){
    const f=D.feas[i];
    h+='<tr class="'+(f?'feas-row':'infeas-row')+'">';
    h+='<td class="id-col">'+i+'</td>';
    h+='<td class="feas-col" style="color:'+(f?'#6dba8a':'#d47272')+'">'+(f?'\u2713':'\u2717')+'</td>';
    cols.forEach(c=>{
      const v=D.samples[c][i];
      const editable=!isDerived.has(c);
      const fv=typeof v==='number'?(Math.abs(v)<0.001&&v!==0?v.toExponential(3):v.toFixed(4)):v;
      h+='<td'+(editable?' contenteditable="true" data-r="'+i+'" data-c="'+c+'"':' style="color:#7b8494"')+'>'+fv+'</td>';
    });
    h+='</tr>';
  }
  h+='</tbody></table>';
  document.getElementById('ss-wrap').innerHTML=h;

  /* attach edit listeners */
  document.querySelectorAll('.ss td[contenteditable]').forEach(td=>{
    td.addEventListener('blur',onCellEdit);
    td.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();td.blur()}});
  });
  tableBuilt=true;
}

function onCellEdit(e){
  const td=e.target;
  const row=+td.dataset.r, col=td.dataset.c;
  const raw=td.textContent.trim();
  const val=parseFloat(raw);
  if(isNaN(val))return;

  /* update data */
  D.samples[col][row]=val;
  td.classList.add('edited');

  /* rebuild sample dict and recompute derived */
  const s={};
  D.keys.forEach(k=>{s[k]=D.samples[k][row]});
  recompute(s);

  /* write derived back */
  D.keys.forEach(k=>{D.samples[k][row]=s[k]});

  /* re-validate */
  const reasons=validate(s);
  D.feas[row]=reasons.length===0?1:0;
  D.reasons[row]=reasons.join('; ');

  /* update header stats */
  let nf=0;for(let j=0;j<D.n;j++)nf+=D.feas[j];
  D.nFeas=nf;
  document.getElementById('hF').textContent=nf;
  document.getElementById('hI').textContent=D.n-nf;
  document.getElementById('hP').textContent=(100*nf/D.n).toFixed(1)+'%';

  /* redraw plot */
  draw();

  /* refresh table row visuals (derived values + feasibility) */
  const tr=td.closest('tr');
  if(tr){
    tr.className=D.feas[row]?'feas-row':'infeas-row';
    const feasTd=tr.querySelectorAll('td')[1];
    if(feasTd){feasTd.textContent=D.feas[row]?'\u2713':'\u2717';feasTd.style.color=D.feas[row]?'#6dba8a':'#d47272'}
    /* update derived cells in row */
    tr.querySelectorAll('td[data-r]').forEach(cell=>{
      const ck=cell.dataset.c;
      if((D.derived||[]).includes(ck)){
        const v=D.samples[ck][row];
        cell.textContent=Math.abs(v)<0.001&&v!==0?v.toExponential(3):v.toFixed(4);
      }
    });
  }

  /* persist back to server */
  fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({row:row,col:col,val:val})
  }).catch(function(){});
}

function resetEdits(){
  location.reload();
}

/* ═══════ TOGGLES ═══════ */
function tv(w){
  if(w==='f'){sF=!sF;document.getElementById('bF').classList.toggle('ag',sF)}
  else if(w==='i'){sI=!sI;document.getElementById('bI').classList.toggle('ar',sI)}
  else if(w==='b'){sB=!sB;document.getElementById('bB').classList.toggle('ay',sB)}
  else if(w==='c'){sC=!sC;document.getElementById('bC').classList.toggle('ab',sC)}
  draw();
}
function ax(x,y,z){
  xK=x;yK=y;zK=z;
  document.getElementById('sx').value=x;
  document.getElementById('sy').value=y;
  document.getElementById('sz').value=z;
  draw();if(tableBuilt)buildTable();
}

draw();
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
#  INJECT & SERVE
# ═══════════════════════════════════════════════════════════════════
FULL_PAGE = PAGE.replace("__DATA_JSON__", DATA_JSON).encode("utf-8")

print(f"[3/3] Page ready  ({len(FULL_PAGE)/1024:.0f} KB)")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(FULL_PAGE)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(FULL_PAGE)

    def do_POST(self):
        """Handle live edits from the spreadsheet."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
            row = payload["row"]
            col = payload["col"]
            val = payload["val"]
            # Update the in-memory data
            if col in blob["samples"] and 0 <= row < N:
                blob["samples"][col][row] = val
                # Recompute derived for this sample via Python
                s = {k: blob["samples"][k][row] for k in ALL_KEYS}
                s = compute_derived(s)
                for k in ALL_KEYS:
                    blob["samples"][k][row] = s[k]
                ok, reasons = validate_sample(s)
                blob["feas"][row] = int(ok)
                blob["reasons"][row] = "; ".join(reasons)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, fmt, *args):
        pass


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


print(f"\n{'='*56}")
print(f"  MDO DESIGN EXPLORER")
print(f"  -> http://{HOST}:{PORT}")
print(f"  {N} samples | {N_FEAS} feasible | 34 variables")
print(f"  Live editing enabled")
print(f"  Press Ctrl+C to stop")
print(f"{'='*56}\n")

threading.Timer(1.0, open_browser).start()

try:
    HTTPServer((HOST, PORT), Handler).serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
