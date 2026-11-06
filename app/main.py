"""
BF-TWIN backend - FastAPI service for the Blast Furnace Digital Twin,
Process Simulator and Supervisory Process-Control System (Extractmet Private Limited).

Run locally:   uvicorn app.main:app --reload --port 8000
Render:        uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from functools import lru_cache
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .core import materials as MAT
from .core import thermo as TH
from .core import slag as SL
from .core.geometry import Geometry
from .core.heatloss import WallModel
from .core.simulate import solve_steady, full_report, run_dynamic, SCENARIOS, kpi_block
from .core.economics import economics

VERSION = "1.0.0"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

app = FastAPI(title="BF-TWIN API", version=VERSION,
              description="Hybrid physics-AI blast furnace digital twin, simulator and supervisory control (Extractmet).")
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=False)
app.add_middleware(GZipMiddleware, minimum_size=1000)

_lock = threading.Lock()
_cache: dict = {}


def _clean(o):
    """Make numpy types JSON-safe."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    if name not in _cache:
        with open(path) as fh:
            _cache[name] = json.load(fh)
    return _cache[name]


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------
class SteadyReq(BaseModel):
    preset: str = MAT.DEFAULT_PRESET
    geometry: Optional[dict] = None
    operation: Optional[dict] = None
    materials: Optional[dict] = None
    lining: Optional[dict] = None
    cooling: Optional[dict] = None
    utilities: Optional[dict] = None
    closure: bool = True
    ncell: int = Field(36, ge=20, le=60)


class DynamicReq(SteadyReq):
    scenario: str = "baseline"
    hours: float = Field(24.0, ge=2.0, le=72.0)
    events: Optional[list] = None


class GeomReq(BaseModel):
    preset: str = MAT.DEFAULT_PRESET
    geometry: Optional[dict] = None
    lining: Optional[dict] = None


class OptReq(BaseModel):
    bounds: Optional[dict] = None
    limits: Optional[dict] = None
    pop: int = Field(80, ge=20, le=200)
    gens: int = Field(80, ge=10, le=250)
    seed: int = 3
    hmt_target: float = 1500.0


class PredictReq(BaseModel):
    x: list


class HeatLossReq(BaseModel):
    preset: str = MAT.DEFAULT_PRESET
    lining: Optional[dict] = None
    cooling: Optional[dict] = None
    T_process: Optional[dict] = None


class SlagReq(BaseModel):
    composition: dict
    T_C: float = 1500.0


class RacewayReq(BaseModel):
    operation: dict
    preset: str = MAT.DEFAULT_PRESET


class AdviceReq(BaseModel):
    hmt: float
    si: float
    pci: float = 170.0
    moisture: float = 12.0
    o2_enrich: float = 4.5
    blast_temp: float = 1170.0
    blast_volume: float = 4850.0
    coke_rate: float = 355.0
    sp_hmt: float = 1500.0
    sp_si: float = 0.45


def _inputs(req: SteadyReq) -> dict:
    d = {"preset": req.preset}
    base = MAT.default_inputs(req.preset)
    for k in ("geometry", "operation", "materials", "lining", "cooling", "utilities"):
        v = getattr(req, k)
        if v:
            d[k] = v
    if req.geometry:
        base_g = dict(base["geometry"])
        base_g.update(req.geometry)
        d["geometry"] = base_g
    return d


def _is_default(req: SteadyReq) -> bool:
    return (req.preset in MAT.PRESETS and not any([req.geometry, req.operation, req.materials, req.lining,
                                                    req.cooling, req.utilities]) and req.closure and req.ncell == 36)


def _coke_guess(inp: dict):
    """Surrogate thermal closure as the first guess for the coke rate (default furnace)."""
    try:
        if inp.get("preset", MAT.DEFAULT_PRESET) != MAT.DEFAULT_PRESET or inp.get("materials") or inp.get("geometry"):
            return None
        from .ai.surrogate import Surrogate
        sur = _surrogate()
        ops = MAT.default_inputs(MAT.DEFAULT_PRESET)["operation"]
        ops.update(inp.get("operation") or {})
        x6 = np.array([[ops[k] for k in ["blast_volume", "blast_temp", "o2_enrich", "pci", "moisture", "pellet_frac"]]])
        if np.any(x6 < sur.xlo[:6] - 1e-6) or np.any(x6 > sur.xhi[:6] + 1e-6):
            return None
        c, _ = sur.closure(x6, target=ops.get("hmt_target", 1500.0))
        return float(c[0]) / 1.018          # actual coke per tHM -> charge set-point
    except Exception:
        return None


@lru_cache(maxsize=1)
def _surrogate():
    from .ai.surrogate import Surrogate
    return Surrogate()


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
# Optional: serve the frontend from the same process for a one-command local run
#   FRONTEND_DIR=../bf-digital-twin-frontend uvicorn app.main:app --port 8000   ->  http://localhost:8000/ui/
_FRONT = os.environ.get("FRONTEND_DIR")
if _FRONT and os.path.isdir(_FRONT):
    from fastapi.staticfiles import StaticFiles
    app.mount("/ui", StaticFiles(directory=_FRONT, html=True), name="ui")


@app.get("/")
def root():
    return {"service": "BF-TWIN API", "version": VERSION, "docs": "/docs", "company": "Extractmet Private Limited",
            "ui": "/ui/" if (_FRONT and os.path.isdir(_FRONT)) else None}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": VERSION, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifacts": {f: os.path.exists(os.path.join(DATA, f)) for f in
                          ["base_report.json", "study_results.json", "surrogate_pinn.npz", "gru_thermal.npz",
                           "softsensor_gbm.json"]}}


@app.get("/api/meta")
def meta():
    from .ai.physics import DEC, X_UNITS
    return _clean({"version": VERSION, "catalogue": MAT.catalogue(), "default_inputs": MAT.default_inputs(MAT.DEFAULT_PRESET),
                   "presets_inputs": {k: MAT.default_inputs(k)["operation"] for k in MAT.PRESETS},
                   "scenarios": {k: v["label"] for k, v in SCENARIOS.items()}, "decision_variables": DEC,
                   "decision_units": X_UNITS[:6], "elements": ["Fe", "C", "O", "H", "N", "Si", "Ca", "Mg", "Al", "Mn", "P", "S", "Ti"]})


@app.post("/api/geometry")
def geometry(req: GeomReq):
    inp = MAT.default_inputs(req.preset)
    g = dict(inp["geometry"])
    if req.geometry:
        g.update(req.geometry)
    try:
        geo = Geometry(g, 36)
    except Exception as e:
        raise HTTPException(400, f"invalid geometry: {e}")
    return _clean(geo.drawing(req.lining or inp["lining"]))


@app.get("/api/base")
def base(preset: str = MAT.DEFAULT_PRESET):
    rep = _load(f"presets/{preset}.json") if preset != MAT.DEFAULT_PRESET else _load("base_report.json")
    if rep is None:
        rep = _clean(full_report(solve_steady({"preset": preset})))
    return rep


@app.post("/api/steady")
def steady(req: SteadyReq):
    if _is_default(req):
        return base(req.preset)
    inp = _inputs(req)
    t0 = time.time()
    try:
        guess = _coke_guess(inp) if req.closure else None
        with _lock:
            bf = solve_steady(inp, ncell=req.ncell, closure=req.closure, coke=guess)
            rep = full_report(bf)
    except Exception as e:
        raise HTTPException(500, f"model error: {e}")
    rep["solve_info"]["api_s"] = round(time.time() - t0, 2)
    rep["solve_info"]["coke_guess"] = guess
    return _clean(rep)


@app.get("/api/scenarios")
def scenarios():
    return {k: v for k, v in SCENARIOS.items()}


@app.post("/api/dynamic")
def dynamic(req: DynamicReq):
    if _is_default(req) and req.events is None:
        cached = _load(f"dynamic/{req.scenario}.json")
        if cached is not None and abs(float(cached.get("hours", 0)) - req.hours) < 1e-6:
            return cached
    inp = _inputs(req)
    try:
        with _lock:
            guess = _coke_guess(inp)
            bf0 = solve_steady(inp, ncell=req.ncell, closure=True, coke=guess)
            bf, res = run_dynamic(bf=bf0, scenario=req.scenario, hours=req.hours, events=req.events, record_every=2)
    except Exception as e:
        raise HTTPException(500, f"model error: {e}")
    return _clean(res)


@app.get("/api/study")
def study():
    s = _load("study_results.json")
    if s is None:
        raise HTTPException(404, "study results not built yet (run tools/train_all.py)")
    return s


@app.post("/api/optimise")
def optimise(req: OptReq):
    try:
        sur = _surrogate()
    except Exception as e:
        raise HTTPException(503, f"surrogate not available: {e}")
    g = Geometry(MAT.default_inputs(MAT.DEFAULT_PRESET)["geometry"], 36)
    base_x = np.array([MAT.default_inputs(MAT.DEFAULT_PRESET)["operation"][k]
                       for k in ["blast_volume", "blast_temp", "o2_enrich", "pci", "moisture", "pellet_frac"]])
    res = sur.optimise(g.V_work, bounds=req.bounds, limits=req.limits, pop=req.pop, gens=req.gens, seed=req.seed,
                       target=req.hmt_target, base_x=base_x)
    Y = res["Y"]
    yi = sur.yidx
    front = [{"x": res["X"][i].round(3).tolist(), "coke": round(float(res["coke"][i]), 2),
              "co2": round(float(res["F"][i, 0]), 2), "cost": round(float(res["F"][i, 1]), 1),
              "prod": round(float(-res["F"][i, 2]), 4), "si": round(float(Y[i, yi["si_pct"]]), 4),
              "raft": round(float(Y[i, yi["raft"]]), 1), "ttop": round(float(Y[i, yi["ttop"]]), 1),
              "dP": round(float(Y[i, yi["dP"]]), 1), "hmt": round(float(Y[i, yi["hmt"]]), 1)} for i in range(len(res["F"]))]
    return _clean({"front": front, "knee": res["knee"], "cpu_s": round(res["cpu_s"], 2), "history": res["history"][::4],
                   "dec_names": ["blast_volume", "blast_temp", "o2_enrich", "pci", "moisture", "pellet_frac"],
                   "bounds_lo": res["lo"], "bounds_hi": res["hi"]})


@app.post("/api/surrogate/predict")
def surrogate_predict(req: PredictReq):
    sur = _surrogate()
    X = np.atleast_2d(np.array(req.x, float))
    if X.shape[1] == 6:
        c, Y = sur.closure(X)
        X = np.hstack([X, c[:, None]])
    else:
        Y = sur.predict(X)
    names = sur.meta["y_names"]
    return _clean({"x": X.tolist(), "y": [dict(zip(names, row)) for row in Y.round(5).tolist()]})


@app.post("/api/mpc/advice")
def mpc_advice(req: AdviceReq):
    """One NMPC move from a steady history at the current operating point (demo advisory)."""
    from .ai.gru import GRUModel
    from .ai.control import NMPC, GRU_IN
    path = os.path.join(DATA, "gru_thermal.npz")
    if not os.path.exists(path):
        raise HTTPException(503, "GRU model not built")
    gm = GRUModel.from_npz(path)
    st = _load("study_results.json") or {}
    coef = np.array(st.get("control", {}).get("raft_coef", [1559.0, -2.0, -5.0, 50.0, 0.0]))
    raft_fn = lambda p, m, o: float(coef[0] + coef[1] * p + coef[2] * m + coef[3] * o + coef[4] * p * o)
    u = [getattr(req, k) for k in GRU_IN]
    hist = np.tile(u, (96, 1))
    mpc = NMPC(gm, raft_fn, {"pci": req.pci, "moisture": req.moisture, "o2_enrich": req.o2_enrich},
               sp=(req.sp_hmt, req.sp_si))
    mv = mpc.step(hist, np.array([req.hmt, req.si]))
    pr = np.array(mpc.prediction)
    from .ai.control import SAMPLE
    pred = {"hmt": pr[:, 0].round(2).tolist(), "si": pr[:, 1].round(4).tolist(), "dt_h": SAMPLE / 3600.0}
    return _clean({"advice": mv, "prediction": pred, "cpu_s": round(mpc.last_cpu, 3),
                   "raft_at_advice": raft_fn(mv["pci"], mv["moisture"], mv["o2_enrich"])})


@app.get("/api/thermo")
def thermo():
    from .core.furnace import PARAMS as P
    kin = [
        {"step": "Fe2O3 -> Fe3O4 (CO, H2)", "law": "r = k(T) * p_CO * (1 - X) * a_s, irreversible", "k_ref": P["k_hem"], "E_kJ": P["E_hem"] / 1e3, "T_ref_K": P["T_hem"] + 273.15, "units": "1/(s bar)"},
        {"step": "Fe3O4 -> FeO (CO, H2)", "law": "r = k(T) * (p_CO - p_CO2/K) * n_Fe3O4", "k_ref": P["k_mag"], "E_kJ": P["E_mag"] / 1e3, "T_ref_K": P["T_mag"] + 273.15, "units": "1/(s bar)"},
        {"step": "FeO -> Fe (CO)", "law": "r = k(T) * RI * (p_CO - p_CO2/K_w) * n_FeO, approach to Fe/FeO equilibrium", "k_ref": P["k_wus"], "E_kJ": P["E_wus"] / 1e3, "T_ref_K": P["T_wus"] + 273.15, "units": "1/(s bar)"},
        {"step": "FeO -> Fe (H2)", "law": "same as CO path, rate x h2_factor", "k_ref": P["k_wus"] * P["h2_factor"], "E_kJ": P["E_wus"] / 1e3, "T_ref_K": P["T_wus"] + 273.15, "units": "1/(s bar)"},
        {"step": "C + CO2 = 2CO (solution loss)", "law": "r = k(T) * (CRI/25) * m_C * p_CO2 (gasification of coke)", "k_ref": P["k_bou"], "E_kJ": P["E_bou"] / 1e3, "T_ref_K": 1373.15, "units": "kg/(s kg bar)"},
        {"step": "C + H2O = CO + H2", "law": "Boudouard rate x wg_factor with p_H2O", "k_ref": P["k_bou"] * P["wg_factor"], "E_kJ": P["E_bou"] / 1e3, "T_ref_K": 1373.15, "units": "kg/(s kg bar)"},
        {"step": "PCI char + CO2", "law": "Boudouard rate x char_factor (high-surface char)", "k_ref": P["k_bou"] * P["char_factor"], "E_kJ": P["E_bou"] / 1e3, "T_ref_K": 1373.15, "units": "kg/(s kg bar)"},
        {"step": "CaCO3 / MgCO3 calcination", "law": "r = k(T) * n_carbonate (T > 1000 K)", "k_ref": P["k_cal"], "E_kJ": P["E_cal"] / 1e3, "T_ref_K": 1173.15, "units": "1/s"},
        {"step": "CO + H2O = CO2 + H2 (shift)", "law": "approach to K_shift(T) with k(T)", "k_ref": P["k_shift"], "E_kJ": P["E_shift"] / 1e3, "T_ref_K": 1173.15, "units": "1/s"},
        {"step": "FeO(l) + C = Fe + CO (direct reduction of dripping slag)", "law": "r = k(T) * a_ls * x_FeO", "k_ref": P["k_dr"], "E_kJ": P["E_dr"] / 1e3, "T_ref_K": 1773.15, "units": "kmol/(s m3)"},
        {"step": "Carburisation C(coke) -> [C]", "law": "r = k * a_ls * (C_sat(T) - [C]) (dissolution, mass-transfer controlled)", "k_ref": P["k_carb"], "E_kJ": 0.0, "T_ref_K": None, "units": "m/s"},
        {"step": "SiO(g) + [C] = [Si] + CO", "law": "capture: k(T) per m of dripping height", "k_ref": P["k_sicap"], "E_kJ": P["E_sicap"] / 1e3, "T_ref_K": 1773.15, "units": "1/m"},
        {"step": "Raceway: PCI burnout", "law": "X = 1 - exp(-k exp(-E/RT) tau)", "k_ref": P["burnout_k"], "E_kJ": P["burnout_E"] / 1e3, "T_ref_K": None, "units": "-"},
        {"step": "Raceway: SiO2(ash) + C = SiO + CO", "law": "fraction phi_SiO * exp(-E/R (1/T - 1/T_ref))", "k_ref": P["phi_SiO"], "E_kJ": P["E_SiO"] / 1e3, "T_ref_K": 2373.15, "units": "-"},
        {"step": "Hearth: (FeS)/(CaS) desulphurisation", "law": "d[S]/dt = k_S (L_S,eq [S] - (S)) mass transfer to Young sulphide capacity", "k_ref": P["k_S"], "E_kJ": 0.0, "T_ref_K": None, "units": "1/s"},
        {"step": "Hearth: [Si] + 2(FeO) / (MnO) reoxidation", "law": "desiliconisation k(T) with partition equilibria", "k_ref": P["k_desi"], "E_kJ": P["E_desi"] / 1e3, "T_ref_K": 1773.15, "units": "1/s"},
    ]
    return _clean({"reactions": TH.reaction_table(), "baur_glaessner": TH.baur_glaessner(), "ellingham": TH.ellingham(),
                   "kinetics": kin, "params": P})


@app.post("/api/heatloss")
def heatloss(req: HeatLossReq):
    inp = MAT.default_inputs(req.preset)
    lining = copy.deepcopy(inp["lining"])
    if req.lining:
        lining.update(req.lining)
    cooling = dict(inp["cooling"])
    if req.cooling:
        cooling.update(req.cooling)
    g = Geometry(inp["geometry"], 36)
    w = WallModel(lining, cooling, g)
    T = {"throat": 200, "upper_stack": 650, "lower_stack": 1000, "belly": 1350, "bosh": 1500, "tuyere_zone": 1530,
         "hearth_wall": 1500, "hearth_bottom": 1490}
    if req.T_process:
        T.update(req.T_process)
    rep = w.report({k: v + 273.15 for k, v in T.items()})
    curves = {}
    for name, z in w.zones.items():
        Tg, qs = z.flux_table()
        curves[name] = {"T_C": (Tg - 273.15).round(0).tolist(), "q_kW_m2": (qs / 1000).round(2).tolist()}
    return _clean({"zones": rep, "curves": curves, "total_MW": round(sum(r["loss_MW"] for r in rep), 3)})


@app.post("/api/slag")
def slag(req: SlagReq):
    return _clean(SL.slag_report(req.composition, req.T_C + 273.15))


@app.post("/api/raceway")
def raceway(req: RacewayReq):
    rep = base(req.preset)
    from .core.furnace import BlastFurnace
    bf = BlastFurnace({"preset": req.preset}, ncell=36)
    ops = dict(bf.ops)
    ops.update(req.operation)
    P = rep["kpi"]["production_tph"] / 3.6
    T_coke = (rep["profiles"]["T_solid_C"][0] or 1650.0) + 273.15
    rw = bf.raceway(ops, T_coke, ops["pci"] * P / 1000.0)
    keys = ["burnout", "phi_SiO", "nSiO", "char", "C_coke", "depth", "v_tuyere", "KE_tuyere_kW", "bosh_gas_Nm3_min", "oc_ratio"]
    out = {k: rw[k] for k in keys}
    out["T_raft_C"] = rw["T_raft"] - 273.15
    g = rw["gas"]
    tot = g[0] + g[2] + g[4]
    out["bosh_comp"] = {"CO": g[0] / tot * 100, "H2": g[2] / tot * 100, "N2": g[4] / tot * 100}
    return _clean(out)


@app.get("/api/validation")
def validation():
    v = _load("validation.json")
    if v is None:
        raise HTTPException(404, "validation not built")
    return v


@app.get("/api/presets/summary")
def presets_summary():
    out = {}
    for k in MAT.PRESETS:
        rep = _load(f"presets/{k}.json") if k != MAT.DEFAULT_PRESET else _load("base_report.json")
        if rep:
            kp = rep["kpi"]
            out[k] = {kk: kp.get(kk) for kk in ["production_tpd", "productivity_work", "coke_rate", "pci_rate", "fuel_rate",
                                                 "raft_C", "eta_CO", "hmt_C", "hm_Si", "co2_kg_t", "cost_inr_t",
                                                 "top_gas_T_C", "pressure_drop_kPa", "heat_loss_GJ_t"]}
            out[k]["geometry"] = rep["geometry"]["summary"]
    return out
