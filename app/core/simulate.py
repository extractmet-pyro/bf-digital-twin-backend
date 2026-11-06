"""
Steady-state solver (with hot-metal-temperature thermal closure on the coke rate),
dynamic scenario runner and the full result report used by the API and GUI.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from . import thermo as TH
from . import slag as SL
from .furnace import (BlastFurnace, rescale_coke, courant_dt, IS, ELEM, EI, NE, GAS8, iCO, iCO2, iH2, iH2O, iN2,
                      ORE_IDX, _elem_vec, _FORMULA)
from .economics import economics

N_DEFAULT = 36


# ---------------------------------------------------------------------------
# convergence & closure
# ---------------------------------------------------------------------------
def converge(bf: BlastFurnace, dt=None, max_steps=600, min_steps=80, tol_T=0.6, tol_hmt=0.4, window=40):
    """Pseudo-transient run until the condensed temperature profile and the hearth
    temperature are stationary over `window` steps."""
    dt = dt or courant_dt(bf)
    Th, Ts = [], []
    for k in range(max_steps):
        bf.step(dt, steady=True)
        Th.append(bf.T_h)
        Ts.append(bf.T_s.copy())
        if k % 25 == 24:
            Pk = float(bf.liq_out["m"] @ TH.MW_METAL)
            if Pk > 1.0:
                bf.P_nom = 0.5 * bf.P_nom + 0.5 * Pk      # keep PCI kg/tHM consistent with production
        if k >= min_steps and k % 10 == 0 and k >= window:
            if (np.max(np.abs(Ts[-1] - Ts[-1 - window])) < tol_T * 4
                    and max(Th[-window:]) - min(Th[-window:]) < tol_hmt * 4
                    and abs(Th[-1] - Th[-11]) < tol_hmt):
                return k + 1
    return max_steps


def _fast_hearth(bf: BlastFurnace, on: bool):
    """Steady solves use a small deadman thermal mass (fast hearth response);
    dynamic runs restore the full deadman inventory at the current temperature."""
    T = bf.T_h
    if on:
        if not getattr(bf, "_dm_full", None):
            bf._dm_full = bf.M_dm
        bf.M_dm = 0.04 * bf._dm_full
    else:
        if getattr(bf, "_dm_full", None):
            bf.M_dm = bf._dm_full
    bf.H_h = bf._hearth_H(T)


def solve_steady(inputs=None, ncell=N_DEFAULT, closure=True, coke=None, params=None, bf=None,
                 max_iter=10, tol=2.5, verbose=False):
    """Converged steady state. If closure=True the coke rate is iterated (bracketed
    secant / bisection) until the hot-metal temperature equals operation['hmt_target']."""
    t0 = time.time()
    if bf is None:
        bf = BlastFurnace(inputs, ncell=ncell, params=params)
    _fast_hearth(bf, True)
    if coke is not None and abs(coke - bf.recipe["coke"]) > 1e-6:
        rescale_coke(bf, float(coke))
    dt = courant_dt(bf)
    target = bf.ops["hmt_target"] + 273.15
    pts = []
    steps = 0
    c = bf.recipe["coke"]
    lo = hi = None
    for it in range(max_iter if closure else 1):
        steps += converge(bf, dt, max_steps=700 if it == 0 else 450)
        T = bf.T_h
        pts.append((c, T))
        if verbose:
            print(f"  closure it{it}: coke={c:.2f} HMT={T - 273.15:.1f}")
        if not closure or abs(T - target) < tol:
            break
        if T < target:
            lo = (c, T) if lo is None or c > lo[0] else lo
        else:
            hi = (c, T) if hi is None or c < hi[0] else hi
        if lo and hi:
            (ca, ta), (cb, tb) = lo, hi
            cn = ca + (target - ta) * (cb - ca) / (tb - ta) if tb != ta else 0.5 * (ca + cb)
            # keep inside bracket, avoid stagnation at the ends
            w = cb - ca
            cn = min(max(cn, ca + 0.1 * w), cb - 0.1 * w)
        else:
            if len(pts) >= 2 and abs(pts[-1][0] - pts[-2][0]) > 0.2:
                s = max((pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0]), 1.0)
            else:
                s = 6.0
            cn = c - (T - target) / s
            cn = float(np.clip(cn, c - 30.0, c + 30.0))
        c = float(np.clip(cn, 200.0, 650.0))
        rescale_coke(bf, c)
    bf.solve_info = {"closure": closure, "iterations": len(pts), "steps": steps,
                     "coke_path": [[round(a, 2), round(b - 273.15, 1)] for a, b in pts],
                     "cpu_s": round(time.time() - t0, 2)}
    return bf


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
def _r(x, n=3):
    try:
        return round(float(x), n)
    except Exception:
        return x


def _wt(nvec, mw, names):
    m = nvec * mw
    tot = m.sum()
    return {k: _r(v / tot * 100.0, 4) for k, v in zip(names, m)} if tot > 0 else {k: 0.0 for k in names}


def _stream_comp_solid(bf, nvec):
    """Element/compound wt% of a condensed stream given species kmol (with ASH)."""
    out = {}
    tot = 0.0
    for i, s in enumerate(bf_SOL()):
        if nvec[i] <= 0:
            continue
        m = nvec[i] * bf.MWS[i]
        tot += m
        out[s] = out.get(s, 0.0) + m
    return {k: _r(v / tot * 100, 3) for k, v in out.items()} if tot > 0 else {}


def bf_SOL():
    from .furnace import SOL
    return SOL


def kpi_block(bf: BlastFurnace) -> dict:
    g = bf.geom
    ops = bf.ops
    lo = bf.liq_out
    P_kg_s = float(lo["m"] @ TH.MW_METAL)          # HM production kg/s (steady: inflow = tapping)
    tph = P_kg_s * 3.6
    tpd = P_kg_s * 86.4
    per_t = 1000.0 / max(P_kg_s, 1e-9)              # multiply a per-second rate by this to get per tHM
    rw = bf.rw
    gas = bf.top["gas"]
    dry = gas[iCO] + gas[iCO2] + gas[iH2] + gas[iN2]
    wet = dry + gas[iH2O]
    eta_co = gas[iCO2] / max(gas[iCO] + gas[iCO2], 1e-12)
    eta_h2 = gas[iH2O] / max(gas[iH2] + gas[iH2O], 1e-12)
    hm_mass = float(bf.hm @ TH.MW_METAL)
    sl_mass = float(bf.sl @ TH.MW_SLAG)
    wm = bf.hm * TH.MW_METAL / hm_mass * 100
    ws = bf.sl * TH.MW_SLAG / sl_mass * 100
    slag_rate = float(lo["s"] @ TH.MW_SLAG) * per_t + sum(v * TH.MW[k] for k, v in rw["slag_ox"].items()) * per_t
    blast_nm3_t = ops["blast_volume"] * 60.0 / max(tph, 1e-9)
    o2_pure = ops["blast_volume"] * ops["o2_enrich"] / 100.0 / bf.inp["utilities"]["oxygen_purity"] * 60.0 / max(tph, 1e-9)
    cv = (gas[iCO] * 12.633 + gas[iH2] * 10.783) / max(dry, 1e-12)   # MJ/Nm3 dry (LHV)
    ext = bf.ext
    sol_loss = (ext["bou"].sum() + ext["wg"].sum()) * 12.011 * per_t
    # direct reduction (FeO reduced by carbon in dripping zone + hearth) as carbon
    Fe_t = wm[0] * 10.0 / 55.845      # kmol Fe/tHM
    # wall losses
    Q_shaft = float(ext["Qw"].sum())
    Q_tuy = bf.wall.tuyere_loss(rw["T_raft"])
    Q_h = bf.Q_hearth
    Q_tot = Q_shaft + Q_tuy + Q_h
    # cohesive zone and TRZ
    z = g.z_mid
    Tsoft, Tmelt = bf.charge["T_soft"], bf.charge["T_melt"]
    Ts = bf.T_s
    cz_top = _interp_level(z, Ts, Tsoft)
    cz_bot = _interp_level(z, Ts, Tmelt)
    trz = _trz(bf)
    blast_p_g = bf.P_blast - 1.01325
    dP = (bf.P_blast - (ops["top_pressure"] + 1.01325)) * 100.0
    Vb = ops["blast_volume"]
    bosh = rw["bosh_gas_Nm3_min"]
    K_perm = ((bf.P_blast) ** 2 - (ops["top_pressure"] + 1.01325) ** 2) / max(bosh / 1000.0, 1e-9) ** 1.7
    vdot = bf.charged_m3 / max(bf._dt_last_step, 1e-9)
    coke_t = bf.charge["coke_kg_per_m3"] * vdot * per_t
    pci_t = rw["pci_dry"] / (1 - bf.coal_moist) * per_t
    fuel = coke_t + pci_t
    return {
        "production_tpd": _r(tpd, 0), "production_tph": _r(tph, 2),
        "productivity_work": _r(tpd / g.V_work, 3), "productivity_inner": _r(tpd / g.V_inner, 3),
        "hearth_productivity_t_m2_d": _r(tpd / g.A_hearth, 1),
        "coke_rate": _r(coke_t, 1), "pci_rate": _r(pci_t, 1), "fuel_rate": _r(fuel, 1),
        "coke_rate_setpoint": _r(bf.recipe["coke"], 1), "pci_setpoint": _r(ops["pci"], 1),
        "carbon_rate": _r(coke_t * bf.inp["materials"]["coke"]["C"] / 100 + pci_t * bf.coal["C"] * (1 - bf.coal_moist), 1),
        "ore_coke_ratio": _r(bf.recipe["ore_total"] / coke_t, 3),
        "blast_nm3_per_t": _r(blast_nm3_t, 1), "oxygen_nm3_per_t": _r(o2_pure, 1),
        "blast_o2_pct": _r(21.0 + ops["o2_enrich"], 2),
        "raft_C": _r(rw["T_raft"] - 273.15, 1), "bosh_gas_nm3_min": _r(bosh, 0),
        "bosh_gas_nm3_t": _r(bosh * 60.0 / max(tph, 1e-9), 0),
        "pci_burnout_pct": _r(rw["burnout"] * 100, 1), "raceway_depth_m": _r(rw["depth"], 2),
        "tuyere_velocity_m_s": _r(rw["v_tuyere"], 1), "tuyere_KE_kW": _r(rw["KE_tuyere_kW"], 1),
        "top_gas_T_C": _r(bf.top["T"] - 273.15, 1),
        "top_gas_nm3_t_dry": _r(dry * TH.NM3_PER_KMOL * per_t, 0),
        "top_gas_CO": _r(gas[iCO] / dry * 100, 2), "top_gas_CO2": _r(gas[iCO2] / dry * 100, 2),
        "top_gas_H2": _r(gas[iH2] / dry * 100, 2), "top_gas_N2": _r(gas[iN2] / dry * 100, 2),
        "top_gas_H2O_wet": _r(gas[iH2O] / wet * 100, 2),
        "eta_CO": _r(eta_co, 4), "eta_H2": _r(eta_h2, 4), "top_gas_LHV_MJ_nm3": _r(cv, 3),
        "blast_pressure_barg": _r(blast_p_g, 3), "top_pressure_barg": _r(ops["top_pressure"], 2),
        "pressure_drop_kPa": _r(dP, 1), "permeability_K": _r(K_perm, 3),
        "hmt_C": _r(bf.T_h - 273.15, 1),
        "hm_C": _r(wm[1], 3), "hm_Si": _r(wm[2], 4), "hm_Mn": _r(wm[3], 3), "hm_P": _r(wm[4], 4),
        "hm_S": _r(wm[5], 4), "hm_Fe": _r(wm[0], 3),
        "slag_rate": _r(slag_rate, 1),
        "slag_B2": _r(ws[TH.LI["CaO"]] / ws[TH.LI["SiO2"]], 3),
        "slag_Al2O3": _r(ws[TH.LI["Al2O3"]], 2), "slag_MgO": _r(ws[TH.LI["MgO"]], 2),
        "slag_FeO": _r(ws[TH.LI["FeO"]], 3),
        "solution_loss_C": _r(sol_loss, 1),
        "solution_loss_total_C": _r(((gas[iCO] + gas[iCO2]) - rw["C_coke"] - rw["coal_C_burnt"]) * 12.011 * per_t
                                    - (bf.charge["n_per_m3"][IS["CaCO3"]] + bf.charge["n_per_m3"][IS["MgCO3"]]) * bf.charge["m3_per_tHM"] * 12.011, 1),
        "heat_loss_MW": _r(Q_tot, 2), "heat_loss_GJ_t": _r(Q_tot / max(P_kg_s, 1e-9), 3),
        "heat_loss_shaft_MW": _r(Q_shaft, 2), "heat_loss_hearth_MW": _r(Q_h, 2), "heat_loss_tuyere_MW": _r(Q_tuy, 2),
        "cz_top_m": _r(cz_top, 2), "cz_bottom_m": _r(cz_bot, 2),
        "trz_T_C": _r(trz[0], 0), "trz_from_m": _r(trz[1], 2), "trz_to_m": _r(trz[2], 2),
        "liquid_T_tuyere_C": _r(lo["T"] - 273.15, 1),
        "burden_descent_m_h": _r(bf.charge["m3_per_tHM"] * tph / (math.pi * g.r_t ** 2), 3),
        "charge_volume_m3_h": _r(bf.charge["m3_per_tHM"] * tph, 1),
        "solids_residence_h": _r(g.V_work / max(bf.charge["m3_per_tHM"] * tph, 1e-9), 2),
        "LS": _r(getattr(bf, "LS_eq", 0.0), 2),
    }


def _interp_level(z, T, Tlev):
    """Height where the (decreasing upward) temperature crosses Tlev."""
    for j in range(len(z) - 1):
        if (T[j] - Tlev) * (T[j + 1] - Tlev) <= 0 and T[j] != T[j + 1]:
            return z[j] + (Tlev - T[j]) / (T[j + 1] - T[j]) * (z[j + 1] - z[j])
    return float(z[0]) if T[0] < Tlev else float(z[-1])


def _trz(bf):
    """Thermal reserve zone: longest run of cells with |dT/dz| small around 850-1050 C."""
    z = bf.geom.z_mid
    T = bf.T_s - 273.15
    grad = np.abs(np.gradient(T, z))
    mask = (grad < 25.0) & (T > 780) & (T < 1080)
    best = (0, 0)
    j = 0
    while j < len(mask):
        if mask[j]:
            k = j
            while k + 1 < len(mask) and mask[k + 1]:
                k += 1
            if k - j > best[1] - best[0]:
                best = (j, k)
            j = k + 1
        else:
            j += 1
    if best == (0, 0) and not mask[0]:
        i = int(np.argmin(np.abs(T - 950)))
        return float(T[i]), float(z[i]), float(z[i])
    a, b = best
    return float(T[a:b + 1].mean()), float(z[a]), float(z[b])


def profiles(bf: BlastFurnace) -> dict:
    g = bf.geom
    n = bf.n
    fe = n[:, IS["Fe"]] + n[:, IS["FeO"]] + 2 * n[:, IS["Fe2O3"]] + 3 * n[:, IS["Fe3O4"]]
    O = n[:, IS["FeO"]] + 3 * n[:, IS["Fe2O3"]] + 4 * n[:, IS["Fe3O4"]]
    # reduction degree relative to Fe2O3 (O/Fe = 1.5)
    rd = np.where(fe > 1e-9, 1.0 - (O / np.maximum(fe, 1e-9)) / 1.5, np.nan)
    met = np.where(fe > 1e-9, n[:, IS["Fe"]] / np.maximum(fe, 1e-9), np.nan)
    frac = lambda k, mult=1: np.where(fe > 1e-9, mult * n[:, IS[k]] / np.maximum(fe, 1e-9), np.nan)
    y = bf.y_g
    per = lambda a: [None if (isinstance(v, float) and math.isnan(v)) else _r(v, 4) for v in a]
    qwall = bf.ext["Qw"] / np.maximum(g.A_wall, 1e-9) * 1000.0   # kW/m2
    Pcum = bf.P - 1.01325
    soft = np.clip((bf.T_s - bf.charge["T_soft"]) / max(bf.charge["T_melt"] - bf.charge["T_soft"], 1.0), 0, 1)
    zone = []
    for j in range(bf.N):
        if bf.V_ore[j] / g.V_cell[j] < 0.02:
            zone.append("dripping")
        elif soft[j] > 0:
            zone.append("cohesive")
        else:
            zone.append("granular")
    return {
        "z": per(g.z_mid), "r": per(g.radius(g.z_mid)), "zone_geom": g.zones, "zone_process": zone,
        "T_solid_C": per(bf.T_s - 273.15), "T_gas_C": per(bf.T_g - 273.15),
        "CO": per(y[:, 0] * 100), "CO2": per(y[:, 1] * 100), "H2": per(y[:, 2] * 100),
        "H2O": per(y[:, 3] * 100), "N2": per(y[:, 4] * 100),
        "eta_CO": per(y[:, 1] / np.maximum(y[:, 0] + y[:, 1], 1e-12)),
        "pressure_barg": per(Pcum), "dP_kPa_per_m": per(bf.dP_cell * 100 / g.dz),
        "reduction_degree": per(rd * 100), "metallisation": per(met * 100),
        "Fe2O3_frac": per(frac("Fe2O3", 2) * 100), "Fe3O4_frac": per(frac("Fe3O4", 3) * 100),
        "FeO_frac": per(frac("FeO") * 100), "Fe_frac": per(frac("Fe") * 100),
        "ore_volfrac": per(bf.V_ore / g.V_cell), "coke_volfrac": per(bf.V_coke / g.V_cell),
        "softening": per(soft),
        "rate_hematite": per(bf.ext["hem"]), "rate_magnetite": per(bf.ext["mag"]),
        "rate_wustite_CO": per(bf.ext["wus"]), "rate_wustite_H2": per(bf.ext["wusH"]),
        "rate_solution_loss": per(bf.ext["bou"]), "rate_water_gas": per(bf.ext["wg"]),
        "rate_char": per(bf.ext["char"]), "rate_calcination": per(bf.ext["cal"]),
        "rate_evaporation": per(bf.ext["evap"]), "rate_SiO_capture": per(bf.ext["sicap"]),
        "Q_gas_solid_MW": per(bf.ext["Qgs"]), "wall_flux_kW_m2": per(qwall),
        "wall_loss_MW": per(bf.ext["Qw"]),
        "eq_CO_wustite": per([TH.co_fraction_wustite(T) * 100 if T > 843 else None for T in bf.T_s]),
    }


def streams(bf: BlastFurnace) -> dict:
    """Input and output streams per tHM and per day with compositions (final weights)."""
    mat = bf.inp["materials"]
    r = bf.recipe
    ops = bf.ops
    lo = bf.liq_out
    rw = bf.rw
    P_kg_s = float(lo["m"] @ TH.MW_METAL)
    tpd = P_kg_s * 86.4
    per_t = 1000.0 / max(P_kg_s, 1e-9)
    ins = []

    def add(name, kg_t, comp, kind, T=None):
        ins.append({"stream": name, "kind": kind, "kg_per_tHM": _r(kg_t, 2), "t_per_day": _r(kg_t * tpd / 1000.0, 1),
                    "composition_wt_pct": {k: _r(v, 3) for k, v in comp.items()}, "T_C": T})
    for b in ["sinter", "pellet", "lump"]:
        if r[b] > 0:
            add(mat["burden"][b]["label"], r[b], mat["burden"][b]["comp"], "ferrous burden", 30)
    for f in ["limestone", "dolomite", "quartzite"]:
        if r[f] > 0.01:
            add(mat["flux"][f]["label"], r[f], mat["flux"][f]["comp"], "flux", 30)
    ck = mat["coke"]
    add("Coke (dry)", r["coke"], {"C": ck["C"], "ash": ck["ash"], "S": ck["S"], "VM": ck["VM"]}, "fuel", 30)
    pc = mat["pci"]
    add("PCI coal", ops["pci"], {k: pc[k] for k in ["C", "H", "O", "N", "S", "ash"]}, "fuel (injected)", bf.p["T_coal"] - 273.15)
    tph = P_kg_s * 3.6
    Vb = ops["blast_volume"] * 60.0 / max(tph, 1e-9)        # Nm3/tHM dry
    yO2 = (21.0 + ops["o2_enrich"]) / 100.0
    air = Vb * (1 - (ops["o2_enrich"] / 100.0) / 0.79)
    o2 = Vb - air
    add("Hot blast air", air * 1.293, {"O2": 23.2, "N2": 76.8}, "blast", ops["blast_temp"])
    add("Oxygen enrichment", o2 * 1.429, {"O2": 99.5, "N2/Ar": 0.5}, "blast", ops["blast_temp"])
    add("Blast moisture (steam)", ops["moisture"] * Vb / 1000.0, {"H2O": 100.0}, "blast", ops["blast_temp"])
    # moisture with burden & coke
    w = sum(r[b] * mat["burden"][b]["moisture"] / (100 - mat["burden"][b]["moisture"]) for b in ["sinter", "pellet", "lump"])
    w += r["coke"] * ck["moisture"] / (100 - ck["moisture"])
    add("Charged moisture", w, {"H2O": 100.0}, "moisture", 30)
    # outputs
    outs = []

    def addo(name, kg_t, comp, kind, T=None):
        outs.append({"stream": name, "kind": kind, "kg_per_tHM": _r(kg_t, 2), "t_per_day": _r(kg_t * tpd / 1000.0, 1),
                     "composition_wt_pct": comp, "T_C": T})
    hm_mass = float(bf.hm @ TH.MW_METAL)
    wm = bf.hm * TH.MW_METAL / hm_mass * 100
    addo("Hot metal", 1000.0, {k: _r(v, 4) for k, v in zip(TH.METAL, wm)}, "product", _r(bf.T_h - 273.15, 1))
    sl_mass = float(bf.sl @ TH.MW_SLAG)
    ws = bf.sl * TH.MW_SLAG / sl_mass * 100
    slag_kg = float(lo["s"] @ TH.MW_SLAG) * per_t + sum(v * TH.MW[k] for k, v in rw["slag_ox"].items()) * per_t
    addo("Slag", slag_kg, {k: _r(v, 3) for k, v in zip(TH.SLAG, ws)}, "by-product", _r(bf.T_h - 273.15 + 15, 1))
    gas = bf.top["gas"]
    mw = [28.010, 44.009, 2.016, 18.015, 28.014, 44.085, 12.011, 32.06]
    gmass = sum(gas[k] * mw[k] for k in range(6)) * per_t
    gdry = gas[iCO] + gas[iCO2] + gas[iH2] + gas[iN2]
    comp = {"CO": _r(gas[iCO] / gdry * 100, 2), "CO2": _r(gas[iCO2] / gdry * 100, 2),
            "H2": _r(gas[iH2] / gdry * 100, 2), "N2": _r(gas[iN2] / gdry * 100, 2),
            "H2O (wet basis)": _r(gas[iH2O] / (gdry + gas[iH2O]) * 100, 2)}
    addo("Top gas (BF gas)", gmass, comp, "by-product gas", _r(bf.top["T"] - 273.15, 1))
    dust_n = bf.charge["dust_per_m3"] * bf.charge["m3_per_tHM"]
    dust_kg = float(dust_n @ bf.MWS) + gas[6] * 12.011 * per_t
    addo("Flue dust (incl. unburnt char)", dust_kg, {"C (char+coke fines)": _r((dust_n[IS["C"]] * 12.011 + gas[6] * 12.011 * per_t) / max(dust_kg, 1e-9) * 100, 1)}, "dust", _r(bf.top["T"] - 273.15, 1))
    total_in = sum(x["kg_per_tHM"] for x in ins)
    total_out = sum(x["kg_per_tHM"] for x in outs)
    return {"inputs": ins, "outputs": outs, "total_in_kg_t": _r(total_in, 1), "total_out_kg_t": _r(total_out, 1),
            "mass_closure_pct": _r((total_in - total_out) / total_in * 100, 3)}


def balances(bf: BlastFurnace, window_acc=None) -> dict:
    """Element and enthalpy balance closure (from the last step and the accumulator)."""
    last = bf.last
    E_in = last["E_in"]
    rel = last["resid_E"] / np.maximum(np.abs(E_in), 1e-12)
    elem = []
    acc = bf.acc
    for e in ELEM:
        i = EI[e]
        elem.append({"element": e, "in_kmol": _r(acc["in_E"][i], 3), "out_kmol": _r(acc["out_E"][i], 3),
                     "step_residual_rel": float(f"{rel[i]:.2e}")})
    return {
        "elements": elem,
        "enthalpy_step_residual_MJ": float(f"{last['resid_H']:.3e}"),
        "enthalpy_step_input_MJ": _r(last["H_in"], 2),
        "max_rel_element_residual": float(f"{np.max(np.abs(rel)):.3e}"),
    }


def heat_balance(bf: BlastFurnace) -> dict:
    """Practical BF heat balance (MJ/tHM, 25 C reference, Hess-law categories)."""
    lo = bf.liq_out
    rw = bf.rw
    ops = bf.ops
    P = float(lo["m"] @ TH.MW_METAL)
    k = 1000.0 / max(P, 1e-9)              # per-second -> per-tHM
    T0 = TH.T0
    gas = bf.top["gas"]
    # blast sensible
    Tb = ops["blast_temp"] + 273.15
    blast_sens = (rw["nO2"] * (TH.h_gas_exact("O2", Tb) - TH.h_gas_exact("O2", T0)) +
                  rw["nN2"] * (bf._hg(iN2, Tb) - bf._hg(iN2, T0)) + rw["nH2Ob"] * (bf._hg(iH2O, Tb) - bf._hg(iH2O, T0))) * k
    # carbon gasified (all carbon leaving as CO+CO2) -> C to CO
    C_gas = (gas[iCO] + gas[iCO2]) * k
    co_comb = C_gas * 110.53
    co_to_co2 = gas[iCO2] * k * 282.98
    # H2 oxidation: H2O in top gas minus water that entered as H2O (blast, coal, charge moisture)
    h2o_in = (rw["nH2Ob"]) * k
    burden_w = bf.charge["n_per_m3"][IS["H2Ol"]] * bf.charge["m3_per_tHM"]
    h2_ox = max(gas[iH2O] * k - burden_w - h2o_in * 0.0, 0.0) * 241.83
    # demand side
    wm = bf.hm * TH.MW_METAL / float(bf.hm @ TH.MW_METAL) * 100
    nFe = wm[0] * 10 / 55.845
    ch = bf.charge["n_per_m3"] * bf.charge["m3_per_tHM"]
    fe2o3 = ch[IS["Fe2O3"]]
    feo = ch[IS["FeO"]]
    diss_fe = fe2o3 * 824.2 + feo * 272.0
    diss_si = wm[2] * 10 / 28.086 * (910.7 - 81.3)
    diss_mn = wm[3] * 10 / 54.938 * (385.2 - 16.2)
    diss_p = wm[4] * 10 / 30.974 * (1504.9 / 2 + 50.0)
    calc = ch[IS["CaCO3"]] * 178.3 + ch[IS["MgCO3"]] * 100.6
    moist_dec = rw["nH2Ob"] * k * 241.83 + burden_w * 44.0
    coal_dec = -bf.coal_hf * rw["pci_dry"] * k
    hm_sens = TH.h_metal(list(np.array([wm[0], wm[1], wm[2], wm[3], wm[4], wm[5]]) * 10 / TH.MW_METAL), bf.T_h) - \
        sum(np.array([0, 22.6, -81.3, 16.2, -50.0, -70.8]) * np.array([wm[0], wm[1], wm[2], wm[3], wm[4], wm[5]]) * 10 / TH.MW_METAL)
    slag_kg = (float(lo["s"] @ TH.MW_SLAG) + sum(v * TH.MW[kk] for kk, v in rw["slag_ox"].items())) * k
    ws = bf.sl / bf.sl.sum()
    nsl = ws * slag_kg / float(ws @ TH.MW_SLAG)
    slag_sens = TH.h_slag(list(nsl), bf.T_h + 15) - TH.h_slag(list(nsl), T0) + slag_kg * TH.L_SLAG_MELT * 0
    Ttop = bf.top["T"]
    top_sens = sum(gas[i] * (bf._hg(i, Ttop) - bf._hg(i, T0)) for i in range(8)) * k
    losses = (float(bf.ext["Qw"].sum()) + bf.Q_hearth + bf.wall.tuyere_loss(rw["T_raft"])) * k
    supply = {"Hot blast sensible heat": blast_sens, "Combustion C -> CO (tuyeres + solution loss)": co_comb,
              "Oxidation CO -> CO2 (indirect reduction)": co_to_co2, "Oxidation H2 -> H2O": h2_ox}
    demand = {"Dissociation of iron oxides": diss_fe, "Reduction of SiO2 / MnO / P2O5": diss_si + diss_mn + diss_p,
              "Carbonate decomposition": calc, "Moisture decomposition & evaporation": moist_dec,
              "PCI coal decomposition": coal_dec, "Hot metal sensible + latent": hm_sens,
              "Slag sensible + latent": slag_sens, "Top gas sensible heat": top_sens, "Wall, hearth & tuyere losses": losses}
    S = sum(supply.values())
    D = sum(demand.values())
    return {"supply_MJ_t": {a: _r(b, 1) for a, b in supply.items()}, "demand_MJ_t": {a: _r(b, 1) for a, b in demand.items()},
            "total_supply": _r(S, 1), "total_demand": _r(D, 1), "unaccounted_MJ_t": _r(S - D, 1),
            "unaccounted_pct": _r((S - D) / S * 100, 2),
            "note": "Categories follow the classical Hess-law BF heat balance; the model itself closes "
                    "enthalpy exactly on absolute enthalpies every time step (see balances)."}


def rist(bf: BlastFurnace) -> dict:
    """Rist operating diagram data (O/C vs O/Fe)."""
    k_fe = None
    lo = bf.liq_out
    P = float(lo["m"] @ TH.MW_METAL)
    k = 1000.0 / P
    wm = bf.hm * TH.MW_METAL / float(bf.hm @ TH.MW_METAL) * 100
    nFe = wm[0] * 10 / 55.845
    gas = bf.top["gas"]
    C = (gas[iCO] + gas[iCO2]) * k
    mu = C / nFe                                   # C/Fe
    eta = gas[iCO2] / (gas[iCO] + gas[iCO2])
    ch = bf.charge["n_per_m3"] * bf.charge["m3_per_tHM"]
    OFe_burden = (3 * ch[IS["Fe2O3"]] + ch[IS["FeO"]]) / nFe
    x_top = 1 + eta
    y_top = OFe_burden
    # line y = mu*(x - x0) ... passes through top point with slope mu
    x0 = x_top - y_top / mu
    yb = -mu * x0 if x0 > 0 else 0.0
    T_trz = _trz(bf)[0] + 273.15
    xw = 1 + (1 - TH.co_fraction_wustite(T_trz))
    yw = 1.05
    line = [[0.0, _r(mu * (0 - x0), 4)], [_r(x_top, 4), _r(y_top, 4)]]
    eff = (mu * (xw - x0)) / yw if yw else None
    return {"C_per_Fe": _r(mu, 4), "eta_CO": _r(eta, 4), "operating_line": line,
            "W_point": [_r(xw, 4), yw], "top_point": [_r(x_top, 4), _r(y_top, 4)],
            "trz_T_C": _r(T_trz - 273.15, 1), "y_at_W": _r(mu * (xw - x0), 4),
            "shaft_efficiency": _r(min(1.2, (y_top - (mu * (x_top - xw))) / max(yw, 1e-9)) if mu else 0, 3),
            "direct_reduction_O_per_Fe": _r(mu * (1 - x0), 4)}


def pseudo2d(bf: BlastFurnace, nr=21, nz=60, center_index=None) -> dict:
    """Pseudo-2D temperature field reconstructed from the 1-D solution with a radial
    gas-distribution (burden distribution) index: centre-working furnaces have
    an inverted-V cohesive zone. Isotherms are shifted upward in the centre."""
    g = bf.geom
    ci = center_index if center_index is not None else bf.ops.get("burden_distribution", 1.0)
    zs = np.linspace(0, g.H, nz)
    T1 = np.interp(zs, g.z_mid, bf.T_s)
    out = []
    shift_max = 0.32 * g.H * ci
    for zz in zs:
        R = float(g.radius(zz))
        row = []
        for i in range(nr):
            rr = i / (nr - 1)
            # isotherm shift: centre (rr=0) up by shift_max, wall (rr=1) down slightly
            sh = shift_max * (1 - rr ** 1.6) - 0.08 * g.H * rr ** 4
            Tz = float(np.interp(zz - sh, g.z_mid, bf.T_s))
            row.append(round(Tz - 273.15, 1))
        out.append(row)
    return {"z": zs.round(3).tolist(), "r_norm": np.linspace(0, 1, nr).round(3).tolist(),
            "T_C": out, "T_soft_C": _r(bf.charge["T_soft"] - 273.15, 1), "T_melt_C": _r(bf.charge["T_melt"] - 273.15, 1),
            "radius_m": [round(float(g.radius(z)), 3) for z in zs]}


def heatloss_report(bf: BlastFurnace) -> list:
    g = bf.geom
    T_by_zone = {}
    cnt = {}
    Tmix = 0.5 * (bf.T_s + bf.T_g)
    for z, T in zip(g.zones, Tmix):
        T_by_zone[z] = T_by_zone.get(z, 0.0) + T
        cnt[z] = cnt.get(z, 0) + 1
    for z in T_by_zone:
        T_by_zone[z] /= cnt[z]
    T_by_zone["hearth_wall"] = bf.T_h
    T_by_zone["hearth_bottom"] = bf.T_h
    T_by_zone["tuyere_zone"] = bf.T_h + 30
    rep = bf.wall.report(T_by_zone)
    # replace shaft zone losses by the exact cell sums
    sums = {}
    for z, q in zip(g.zones, bf.ext["Qw"]):
        sums[z] = sums.get(z, 0.0) + q
    for r_ in rep:
        if r_["zone"] in sums:
            r_["loss_MW"] = round(float(sums[r_["zone"]]), 3)
    rep.append({"zone": "tuyeres (coolers)", "loss_MW": round(bf.wall.tuyere_loss(bf.rw["T_raft"]), 3),
                "area_m2": None, "q_total_kW_m2": None, "profile": [], "layers": []})
    return rep


def hearth_report(bf: BlastFurnace) -> dict:
    g = bf.geom
    p = bf.p
    hm_mass = float(bf.hm @ TH.MW_METAL)
    sl_mass = float(bf.sl @ TH.MW_SLAG)
    A = g.A_hearth * p["deadman_void"]
    sump = g.A_hearth * g.h_sump * p["deadman_void"] * p["rho_hm"] * (1 - p["sump_active"])
    lvl_hm = (hm_mass + sump) / p["rho_hm"] / A - g.h_sump
    lvl_sl = lvl_hm + sl_mass / p["rho_slag"] / A
    ws = {s: v for s, v in zip(TH.SLAG, bf.sl * TH.MW_SLAG / sl_mass * 100)}
    wsl = {k: ws[k] for k in ["SiO2", "Al2O3", "CaO", "MgO", "MnO", "FeO", "TiO2"]}
    sp = SL.slag_report(wsl, bf.T_h)
    return {"hm_inventory_t": _r(hm_mass / 1000 + sump / 1000, 1), "slag_inventory_t": _r(sl_mass / 1000, 1),
            "hm_level_above_taphole_m": _r(lvl_hm, 3), "slag_level_above_taphole_m": _r(lvl_sl, 3),
            "taphole_to_tuyere_m": g.h_hearth, "T_hearth_C": _r(bf.T_h - 273.15, 1),
            "deadman_coke_t": _r((getattr(bf, "_dm_full", None) or bf.M_dm) * 12.011 / 1000 / p["deadman_participation"], 1),
            "hearth_loss_MW": _r(bf.Q_hearth, 3), "slag_properties": sp}


def full_report(bf: BlastFurnace) -> dict:
    k = kpi_block(bf)
    econ = economics(bf, k)
    k.update({"co2_kg_t": econ["co2"]["total_kg_t"], "cost_inr_t": econ["cost"]["total_inr_t"]})
    rw = bf.rw
    ops = bf.ops
    return {
        "kpi": k,
        "profiles": profiles(bf),
        "streams": streams(bf),
        "balances": balances(bf),
        "heat_balance": heat_balance(bf),
        "economics": econ,
        "rist": rist(bf),
        "raceway": {key: _r(rw[key], 4) for key in ["burnout", "phi_SiO", "nSiO", "char", "C_coke", "depth", "v_tuyere",
                                                   "KE_tuyere_kW", "bosh_gas_Nm3_min", "oc_ratio"]} |
        {"T_raft_C": _r(rw["T_raft"] - 273.15, 1),
         "bosh_gas_comp": _bosh_comp(rw)},
        "hearth": hearth_report(bf),
        "heat_loss": heatloss_report(bf),
        "pseudo2d": pseudo2d(bf),
        "geometry": bf.geom.drawing(bf.inp["lining"]),
        "recipe": {kk: (_r(v, 2) if isinstance(v, (int, float)) else v) for kk, v in bf.recipe.items()},
        "operation": ops,
        "solve_info": getattr(bf, "solve_info", {}),
        "charge": {"m3_per_tHM": _r(bf.charge["m3_per_tHM"], 4), "ore_volfrac": _r(bf.charge["fore"], 4),
                   "T_soft_C": _r(bf.charge["T_soft"] - 273.15, 1), "T_melt_C": _r(bf.charge["T_melt"] - 273.15, 1),
                   "RI": _r(bf.charge["RI"], 3)},
    }


def _bosh_comp(rw):
    g = rw["gas"]
    tot = g[0] + g[2] + g[4] + g[5]
    return {"CO": _r(g[0] / tot * 100, 2), "H2": _r(g[2] / tot * 100, 2), "N2": _r(g[4] / tot * 100, 2),
            "SiO": _r(g[5] / tot * 100, 4)}


# ---------------------------------------------------------------------------
# dynamic scenarios
# ---------------------------------------------------------------------------
SCENARIOS = {
    "baseline": {"label": "Baseline (no disturbance)", "events": []},
    "coke_moisture": {"label": "Coke moisture +1.5 %-pt (uncompensated) at t = 2 h",
                      "events": [{"t": 2.0, "var": "coke_moisture", "delta": 1.5}]},
    "ore_reducibility": {"label": "Ore reducibility -8 % at t = 2 h",
                         "events": [{"t": 2.0, "var": "ore_reducibility", "value": 0.92}]},
    "pci_trip": {"label": "PCI trip for 1.5 h at t = 4 h",
                 "events": [{"t": 4.0, "var": "pci", "value": 0.0}, {"t": 5.5, "var": "pci", "restore": True}]},
    "wind_cut": {"label": "Wind cut -20 % blast for 2 h at t = 4 h",
                 "events": [{"t": 4.0, "var": "blast_volume", "scale": 0.8}, {"t": 6.0, "var": "blast_volume", "restore": True}]},
    "hot_blast_drop": {"label": "Hot-blast temperature -50 C for 3 h at t = 3 h",
                       "events": [{"t": 3.0, "var": "blast_temp", "delta": -50.0}, {"t": 6.0, "var": "blast_temp", "restore": True}]},
    "step_pci": {"label": "Step test: PCI +10 kg/tHM at t = 2 h", "events": [{"t": 2.0, "var": "pci", "delta": 10.0}]},
    "step_moisture": {"label": "Step test: blast moisture +5 g/Nm3 at t = 2 h",
                      "events": [{"t": 2.0, "var": "moisture", "delta": 5.0}]},
    "step_oxygen": {"label": "Step test: O2 enrichment +1 %-pt at t = 2 h",
                    "events": [{"t": 2.0, "var": "o2_enrich", "delta": 1.0}]},
    "step_coke": {"label": "Step test: coke rate +5 kg/tHM at t = 2 h",
                  "events": [{"t": 2.0, "var": "coke_rate", "delta": 5.0}]},
    "step_blast_temp": {"label": "Step test: blast temperature +30 C at t = 2 h",
                        "events": [{"t": 2.0, "var": "blast_temp", "delta": 30.0}]},
}


def apply_event(bf: BlastFurnace, ev: dict, base_ops: dict):
    v = ev["var"]
    if ev.get("restore"):
        val = base_ops[v]
    elif "value" in ev:
        val = ev["value"]
    elif "delta" in ev:
        val = bf.ops[v] + ev["delta"]
    elif "scale" in ev:
        val = bf.ops[v] * ev["scale"]
    else:
        return
    if v == "coke_rate":
        bf.ops["coke_rate"] = val
        bf.build_recipe(val)
    elif v == "coke_moisture":
        bf.ops["coke_moisture_assumed"] = base_ops.get("coke_moisture", bf.ops["coke_moisture"])
        bf.ops["coke_moisture"] = val
        bf._recipe_per_m3()
    elif v in ("pellet_frac", "lump_frac", "basicity_target", "mgo_target"):
        bf.ops[v] = val
        bf.build_recipe(bf.recipe["coke"])
    else:
        bf.ops[v] = val


def snapshot(bf: BlastFurnace) -> dict:
    lo = bf.liq_out
    P = float(lo["m"] @ TH.MW_METAL)
    gas = bf.top["gas"]
    dry = gas[iCO] + gas[iCO2] + gas[iH2] + gas[iN2]
    hm_mass = float(bf.hm @ TH.MW_METAL)
    wm = bf.hm * TH.MW_METAL / hm_mass * 100
    sl_mass = float(bf.sl @ TH.MW_SLAG)
    ws = bf.sl * TH.MW_SLAG / max(sl_mass, 1e-9) * 100
    g = bf.geom
    p = bf.p
    A = g.A_hearth * p["deadman_void"]
    sump = g.A_hearth * g.h_sump * p["deadman_void"] * p["rho_hm"] * (1 - p["sump_active"])
    lvl_hm = (hm_mass + sump) / p["rho_hm"] / A - g.h_sump
    lvl_sl = lvl_hm + sl_mass / p["rho_slag"] / A
    Q_tot = float(bf.ext["Qw"].sum()) + bf.Q_hearth + bf.wall.tuyere_loss(bf.rw["T_raft"])
    rel = bf.last["resid_E"] / np.maximum(np.abs(bf.last["E_in"]), 1e-12)
    return {
        "t_h": round(bf.t / 3600.0, 4),
        "prod_tph": round(P * 3.6, 2), "tapped_tph": round(float(bf.tapped["m"] @ TH.MW_METAL) / bf._dt_last * 3.6, 2),
        "hmt_C": round(bf.T_h - 273.15, 2), "Si": round(wm[2], 4), "S": round(wm[5], 4), "C": round(wm[1], 3),
        "Mn": round(wm[3], 3), "slag_B2": round(ws[TH.LI["CaO"]] / max(ws[TH.LI["SiO2"]], 1e-9), 3),
        "slag_FeO": round(ws[TH.LI["FeO"]], 3),
        "top_T_C": round(bf.top["T"] - 273.15, 1), "CO": round(gas[iCO] / dry * 100, 2),
        "CO2": round(gas[iCO2] / dry * 100, 2), "H2": round(gas[iH2] / dry * 100, 2),
        "eta_CO": round(gas[iCO2] / (gas[iCO] + gas[iCO2]), 4),
        "raft_C": round(bf.rw["T_raft"] - 273.15, 1), "blast_p_barg": round(bf.P_blast - 1.01325, 3),
        "dP_kPa": round((bf.P_blast - bf.ops["top_pressure"] - 1.01325) * 100, 1),
        "pci": round(bf.ops["pci"], 2), "o2_enrich": round(bf.ops["o2_enrich"], 3),
        "moisture": round(bf.ops["moisture"], 2), "blast_temp": round(bf.ops["blast_temp"], 1),
        "blast_volume": round(bf.ops["blast_volume"], 1), "coke_rate_set": round(bf.recipe["coke"], 2),
        "coke_moisture": round(bf.ops.get("coke_moisture", 4.5), 2),
        "heat_loss_MW": round(Q_tot, 3), "stave_load_MW": round(float(bf.ext["Qw"].sum()), 3),
        "hm_level_m": round(lvl_hm, 3), "slag_level_m": round(lvl_sl, 3),
        "cast_open": bool(bf.cast.get("open")), "taphole": bf.cast.get("hole"),
        "cz_top_m": round(_interp_level(g.z_mid, bf.T_s, bf.charge["T_soft"]), 3),
        "cz_bot_m": round(_interp_level(g.z_mid, bf.T_s, bf.charge["T_melt"]), 3),
        "burden_descent_m_h": round(bf.charged_m3 / bf._dt_last * 3600.0 / (math.pi * g.r_t ** 2), 3),
        "bal_max_rel": float(f"{np.max(np.abs(rel)):.2e}"), "bal_H_MJ": float(f"{bf.last['resid_H']:.2e}"),
        "liquid_T_C": round(lo["T"] - 273.15, 1),
    }


def run_dynamic(inputs=None, scenario="baseline", hours=36.0, dt=None, events=None, bf=None, params=None,
                ncell=N_DEFAULT, field_every_h=0.5, controller=None, record_every=1):
    """Initialise at steady state (thermal closure) and integrate in time with casting."""
    t0 = time.time()
    if bf is None:
        bf = solve_steady(inputs, ncell=ncell, closure=True, params=params)
    _fast_hearth(bf, False)
    base_ops = copy.deepcopy(bf.ops)
    bf.t = 0.0
    bf.casts = []
    bf.cast = {"open": False, "t_next": 10.0, "t_close": 0.0, "hole": 0, "count": 0}
    bf.acc = bf._zero_acc()
    evs = sorted(events if events is not None else SCENARIOS.get(scenario, SCENARIOS["baseline"])["events"],
                 key=lambda e: e["t"])
    dt = dt or min(300.0, courant_dt(bf))
    bf._dt_last = dt
    nsteps = int(round(hours * 3600.0 / dt))
    traj = []
    fields = {"t_h": [], "T_s": [], "rd": []}
    next_field = 0.0
    ei = 0
    for k in range(nsteps):
        th = bf.t / 3600.0
        while ei < len(evs) and evs[ei]["t"] <= th + 1e-9:
            apply_event(bf, evs[ei], base_ops)
            ei += 1
        if controller is not None:
            controller(bf, th)
        bf.step(dt, steady=False)
        if k % record_every == 0:
            traj.append(snapshot(bf))
        if th >= next_field:
            n = bf.n
            fe = n[:, IS["Fe"]] + n[:, IS["FeO"]] + 2 * n[:, IS["Fe2O3"]] + 3 * n[:, IS["Fe3O4"]]
            O = n[:, IS["FeO"]] + 3 * n[:, IS["Fe2O3"]] + 4 * n[:, IS["Fe3O4"]]
            rd = np.where(fe > 1e-9, (1 - O / np.maximum(fe, 1e-9) / 1.5) * 100, np.nan)
            fields["t_h"].append(round(th, 3))
            fields["T_s"].append(np.round(bf.T_s - 273.15, 1).tolist())
            fields["rd"].append([None if math.isnan(v) else round(float(v), 2) for v in rd])
            next_field += field_every_h
    return bf, {"trajectory": traj, "casts": bf.casts, "fields": fields, "z": bf.geom.z_mid.round(3).tolist(),
                "scenario": scenario, "events": evs, "dt_s": dt, "hours": hours,
                "cpu_s": round(time.time() - t0, 2)}
