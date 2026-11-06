"""
BF-TWIN first-principles blast-furnace model.

Three coupled sub-models (as in the NeuroMet 2027 abstract):

1. One-dimensional counter-current SHAFT model (tuyere level -> stock line)
   - Eulerian condensed phase (burden + coke) descending with a velocity set by
     the volume freed through raceway combustion, coke gasification and melting
   - quasi-steady gas phase marched upward every time step
   - indirect reduction Fe2O3->Fe3O4->FeO->Fe by CO and H2 (equilibrium-limited
     first-order kinetics, ore reducibility multiplier), solution-loss and
     water-gas reactions (coke reactivity), PCI-char gasification, water-gas
     shift, carbonate calcination, moisture evaporation, SiO capture/re-oxidation
   - gas-solid heat transfer (Ranz-Marshall with intraparticle resistance),
     wall heat loss (coupled conduction/convection/radiation lining model),
     Ergun pressure drop for layered ore/coke with cohesive-zone and
     dripping-zone corrections
   - softening/melting (cohesive zone), liquid march through the dripping zone:
     liquid-gas heat exchange, direct reduction of FeO by coke, carburisation
     (carbon dissolution kinetics), MnO and P2O5 reduction
2. PCI RACEWAY model: coke/char combustion to CO, blast moisture decomposition,
   coal devolatilisation (formation enthalpy from GCV), PCI burnout kinetics,
   SiO generation from ash, raceway adiabatic flame temperature (RAFT),
   tuyere velocity, raceway depth.
3. HEARTH hot-metal/slag model: liquid inventories, deadman coke, slag-metal
   kinetics towards equilibrium (S via sulphide capacity, Si, Mn), carbon
   dissolution to saturation, hearth wall/bottom heat loss, cast schedule.

Every time step closes element balances (Fe, C, O, H, N, Si, Ca, Mg, Al, Mn,
P, S, Ti) and the enthalpy balance on absolute (formation-based) enthalpies.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from . import thermo as TH
from .geometry import Geometry
from .heatloss import WallModel
from .materials import default_inputs, deep_update, pci_gcv
from . import slag as SL

RJ = TH.R_J
SOL = TH.SOLID + ["Sb", "ASH"]               # condensed species in shaft cells
NS = len(SOL)
IS = {s: i for i, s in enumerate(SOL)}
GAS8 = ["CO", "CO2", "H2", "H2O", "N2", "SiO", "Cchar", "Sg"]
NG = len(GAS8)
iCO, iCO2, iH2, iH2O, iN2, iSiO, iCH, iSG = range(8)
ELEM = ["Fe", "C", "O", "H", "N", "Si", "Ca", "Mg", "Al", "Mn", "P", "S", "Ti"]
NE = len(ELEM)
EI = {e: i for i, e in enumerate(ELEM)}
ORE_SPECIES = ["Fe2O3", "Fe3O4", "FeO", "Fe", "SiO2", "Al2O3", "CaO", "MgO", "MnO", "TiO2", "P2O5",
               "CaCO3", "MgCO3", "H2Ol", "Sb"]
ORE_IDX = np.array([IS[s] for s in ORE_SPECIES])
COKE_IDX = np.array([IS["C"], IS["S"], IS["ASH"]])
ASH_OX = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "TiO2", "P2O5", "MnO"]

_FORMULA = {
    "Fe2O3": {"Fe": 2, "O": 3}, "Fe3O4": {"Fe": 3, "O": 4}, "FeO": {"Fe": 1, "O": 1}, "Fe": {"Fe": 1},
    "C": {"C": 1}, "SiO2": {"Si": 1, "O": 2}, "Al2O3": {"Al": 2, "O": 3}, "CaO": {"Ca": 1, "O": 1},
    "MgO": {"Mg": 1, "O": 1}, "MnO": {"Mn": 1, "O": 1}, "TiO2": {"Ti": 1, "O": 2}, "P2O5": {"P": 2, "O": 5},
    "CaCO3": {"Ca": 1, "C": 1, "O": 3}, "MgCO3": {"Mg": 1, "C": 1, "O": 3}, "H2Ol": {"H": 2, "O": 1},
    "S": {"S": 1}, "Sb": {"S": 1}, "CO": {"C": 1, "O": 1}, "CO2": {"C": 1, "O": 2}, "H2": {"H": 2},
    "H2O": {"H": 2, "O": 1}, "N2": {"N": 2}, "SiO": {"Si": 1, "O": 1}, "Cchar": {"C": 1}, "Sg": {"S": 1},
    "O2": {"O": 2}, "CaS": {"Ca": 1, "S": 1}, "Si": {"Si": 1}, "Mn": {"Mn": 1}, "P": {"P": 1},
}


def _elem_vec(formula):
    v = np.zeros(NE)
    for e, c in formula.items():
        v[EI[e]] += c
    return v


# ---------------------------------------------------------------------------
# default model / calibration parameters
# ---------------------------------------------------------------------------
PARAMS = {
    # indirect reduction (1/(s bar)), Arrhenius about Tref
    "k_hem": 6.0e-3, "E_hem": 60e3, "T_hem": 800.0,
    "k_mag": 2.5e-3, "E_mag": 70e3, "T_mag": 1000.0,
    "k_wus": 0.5e-4, "E_wus": 90e3, "T_wus": 1200.0,
    "h2_factor": 3.0,
    # solution loss: kg C/(s kg C bar) at 1373 K for CRI = 25
    "k_bou": 3.0e-4, "E_bou": 230e3, "wg_factor": 3.0, "char_factor": 25.0,
    # carbonates, moisture, shift
    "k_cal": 4e-3, "E_cal": 180e3, "k_evap": 4e-3,
    "k_shift": 2.0, "E_shift": 80e3,
    # cohesive zone / melting
    "melt_band": 60.0, "c_initial": 1.5,
    # dripping zone liquids
    "h_lg": 180.0, "a_lg": 2.0, "h_ls": 300.0, "a_ls": 25.0, "tau_l": 35.0, "central_path": 0.0,
    "k_dr": 8.0e-2, "E_dr": 160e3, "k_carb": 1.2e-2, "k_mn": 4.0e-3, "k_p": 8.0e-2,
    # raceway
    "burnout_k": 0.82, "burnout_E": 50e3, "phi_SiO": 0.42, "E_SiO": 350e3, "f_Sg": 0.07,
    "pci_carrier_nm3_per_kg": 0.03,
    # SiO capture in dripping zone (1/m) and hearth desiliconisation
    "k_sicap": 0.55, "E_sicap": 150e3,
    "k_desi": 1.1e-4, "E_desi": -60e3,
    # hearth
    "k_S": 1.0e-2, "k_hcarb": 6.0e-4, "k_hfeo": 4.0e-4, "deadman_participation": 0.35,
    "rho_hm": 6900.0, "rho_slag": 2700.0, "deadman_void": 0.38, "sump_active": 0.30,
    # heat transfer / flow
    "hv_factor": 1.0, "hv_drip": 0.05, "hv_cz": 1.0, "wall_gas_share": 0.5, "dp_factor": 2.8,
    "eps_ore": 0.40, "eps_coke": 0.50, "holdup": 0.035,
    # charge
    "T_charge": 303.15, "T_coal": 333.15,
    # cast schedule
    "cast_duration_min": 125.0, "cast_gap_min": 20.0, "tap_rate_factor": 1.32,
    "lab_delay_min": 20.0,
}


class BlastFurnace:
    """Dynamic first-principles blast furnace (also used for steady states)."""

    def __init__(self, inputs: dict | None = None, ncell: int = 36, params: dict | None = None):
        preset = (inputs or {}).get("preset", "BF-2800")
        self.inp = deep_update(default_inputs(preset), inputs or {})
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.geom = Geometry(self.inp["geometry"], ncell)
        self.N = ncell
        self.wall = WallModel(self.inp["lining"], self.inp["cooling"], self.geom)
        self._setup_species()
        self.ops = dict(self.inp["operation"])
        self.t = 0.0
        self.hist = []
        self.mode = "dynamic"
        self.recipe = None
        self._est_hm = {"C": 4.6, "Si": 0.45, "Mn": 0.25, "P": 0.10, "S": 0.03, "slag_FeO": 0.5, "dust_Fe": 4.0}
        self.P_nom = None
        self.build_recipe(self.ops["coke_rate"])
        self._init_state()

    # ------------------------------------------------------------------
    # species tables
    # ------------------------------------------------------------------
    def _setup_species(self):
        coke = self.inp["materials"]["coke"]
        aw = coke["ash_comp"]
        tot = sum(aw.values())
        self.ash_w = {k: aw.get(k, 0.0) / tot for k in ASH_OX}
        # ASH enthalpy per kg (MJ/kg) and cp
        Hash = np.zeros(TH.NGRID)
        Cash = np.zeros(TH.NGRID)
        for k, w in self.ash_w.items():
            Hash += w / TH.MW[k] * TH.HTAB[k]
            Cash += w / TH.MW[k] * TH.CPTAB[k]
        self.SOLH = np.vstack([TH.SOL_H, TH.HTAB["S"][None, :], Hash[None, :]])
        self.SOLCP = np.vstack([TH.SOL_CP, TH.CPTAB["S"][None, :], Cash[None, :]])
        self.MWS = np.concatenate([TH.MW_SOLID, [TH.MW["S"], 1.0]])
        # element matrices
        self.ES = np.zeros((NS, NE))
        for s in SOL:
            if s == "ASH":
                for k, w in self.ash_w.items():
                    self.ES[IS[s]] += w / TH.MW[k] * _elem_vec(_FORMULA[k])
            else:
                self.ES[IS[s]] = _elem_vec(_FORMULA[s])
        self.EG = np.array([_elem_vec(_FORMULA[g]) for g in GAS8])
        self.EM = np.array([_elem_vec({"Fe": 1}), _elem_vec({"C": 1}), _elem_vec({"Si": 1}), _elem_vec({"Mn": 1}),
                            _elem_vec({"P": 1}), _elem_vec({"S": 1})])
        self.EL = np.array([_elem_vec(_FORMULA[s]) for s in TH.SLAG])
        # gas tables (python lists)
        gH = [TH.HTAB["g_" + g] for g in ["CO", "CO2", "H2", "H2O", "N2", "SiO"]] + [TH.HTAB["C"], TH.HTAB["g_Sg"]]
        gC = [TH.CPTAB["g_" + g] for g in ["CO", "CO2", "H2", "H2O", "N2", "SiO"]] + [TH.CPTAB["C"], TH.CPTAB["g_Sg"]]
        self._GH = [list(r) for r in gH]
        self._GC = [list(r) for r in gC]
        self._HC = list(TH.HTAB["C"])
        self.GH = np.array(gH)
        # PCI coal
        pc = self.inp["materials"]["pci"]
        dry = {k: pc[k] / 100.0 for k in ["C", "H", "O", "N", "S", "ash"]}
        s = sum(dry.values())
        self.coal = {k: v / s for k, v in dry.items()}
        self.coal_moist = pc["moisture"] / 100.0
        gcv = pci_gcv(pc)
        c = self.coal
        h_prod = (c["C"] / 12.011 * -393.52 + c["H"] / 2.016 * -285.83 + c["S"] / 32.06 * -296.81)
        paw = pc["ash_comp"]
        ptot = sum(paw.values())
        self.pci_ash_w = {k: paw.get(k, 0.0) / ptot for k in ASH_OX if paw.get(k, 0.0) > 0}
        h_ash = sum(w / TH.MW[k] * TH.COND[k][0] for k, w in self.pci_ash_w.items())
        self.coal_hf = h_prod + gcv + c["ash"] * h_ash         # MJ/kg dry coal (formation)
        self.coal_gcv = gcv
        self.coal_cp = 1.30e-3                                # MJ/(kg K)

    def h_cells(self, n, T):
        Hs = TH.interp_tab(self.SOLH, T)
        return np.einsum("cs,sc->c", n, Hs)

    def cp_cells(self, n, T):
        Cs = TH.interp_tab(self.SOLCP, T)
        return np.einsum("cs,sc->c", n, Cs)

    def T_cells(self, n, H, Tg):
        T = np.array(Tg, dtype=float).copy()
        for _ in range(30):
            h = self.h_cells(n, T)
            c = np.maximum(self.cp_cells(n, T), 1e-9)
            dT = np.clip((H - h) / c, -250.0, 250.0)
            T = np.clip(T + dT, 270.0, 3300.0)
            if np.max(np.abs(dT)) < 2e-3:
                break
        return T

    # gas helpers (scalar, python lists)
    def _hg(self, k, T):
        x = T - 250.0
        if x < 0:
            x = 0.0
        elif x > 3349.0:
            x = 3349.0
        i = int(x)
        f = x - i
        r = self._GH[k]
        return r[i] + f * (r[i + 1] - r[i])

    def _hmix(self, g, T):
        x = T - 250.0
        if x < 0:
            x = 0.0
        elif x > 3349.0:
            x = 3349.0
        i = int(x)
        f = x - i
        s = 0.0
        GH = self._GH
        for k in range(NG):
            gk = g[k]
            if gk != 0.0:
                r = GH[k]
                s += gk * (r[i] + f * (r[i + 1] - r[i]))
        return s

    def _cpmix(self, g, T):
        x = T - 250.0
        if x < 0:
            x = 0.0
        elif x > 3349.0:
            x = 3349.0
        i = int(x)
        f = x - i
        s = 0.0
        GC = self._GC
        for k in range(NG):
            gk = g[k]
            if gk != 0.0:
                r = GC[k]
                s += gk * (r[i] + f * (r[i + 1] - r[i]))
        return s

    def _Tgas(self, g, H, T):
        for _ in range(25):
            h = self._hmix(g, T)
            c = self._cpmix(g, T)
            if c <= 1e-12:
                return T
            dT = (H - h) / c
            if dT > 300:
                dT = 300.0
            elif dT < -300:
                dT = -300.0
            T += dT
            if T < 270.0:
                T = 270.0
            elif T > 3500.0:
                T = 3500.0
            if -1e-3 < dT < 1e-3:
                break
        return T

    # ------------------------------------------------------------------
    # burden (charge) calculation
    # ------------------------------------------------------------------
    def build_recipe(self, coke_rate: float):
        """Per-tHM charge: ferrous burden from Fe balance; limestone/dolomite/
        quartzite from slag basicity and MgO targets."""
        mat = self.inp["materials"]
        op = self.ops
        pf = op["pellet_frac"] / 100.0
        lf = op["lump_frac"] / 100.0
        sf = max(0.0, 1.0 - pf - lf)
        mix = {"sinter": sf, "pellet": pf, "lump": lf}
        est = self._est_hm
        coke = mat["coke"]
        pci = op["pci"]
        # ferrous blend composition (wt fractions)
        blend = {}
        for b, f in mix.items():
            comp = mat["burden"][b]["comp"]
            tot = sum(comp.values())
            for k, v in comp.items():
                blend[k] = blend.get(k, 0.0) + f * v / tot
        fe_blend = blend.get("Fe2O3", 0) * 111.69 / 159.69 + blend.get("FeO", 0) * 55.845 / 71.844
        coke_dry = coke_rate
        ash = coke_dry * coke["ash"] / 100.0
        fe_coke = ash * self.ash_w["Fe2O3"] * 111.69 / 159.69
        fe_pci = pci * self.coal["ash"] * self.pci_ash_w.get("Fe2O3", 0) * 111.69 / 159.69
        hm_other = est["C"] + est["Si"] + est["Mn"] + est["P"] + est["S"]
        fe_hm = 1000.0 * (1.0 - hm_other / 100.0)
        slag_guess = 300.0
        fe_slag = slag_guess * est["slag_FeO"] / 100.0 * 55.845 / 71.844
        fe_dust = est["dust_Fe"]
        flux = {"limestone": 0.0, "dolomite": 0.0, "quartzite": 0.0}
        for _ in range(30):
            fe_flux = sum(flux[f] * mat["flux"][f]["comp"].get("Fe2O3", 0.0) / 100.0 * 111.69 / 159.69 for f in flux)
            fe_need = fe_hm + fe_slag + fe_dust - fe_coke - fe_pci - fe_flux
            ore = fe_need / fe_blend
            # slag oxides (kg/tHM)
            ox = {k: 0.0 for k in ["SiO2", "Al2O3", "CaO", "MgO", "MnO", "TiO2"]}
            for k in ox:
                ox[k] += ore * blend.get(k, 0.0)
                ox[k] += ash * self.ash_w.get(k, 0.0)
                ox[k] += pci * self.coal["ash"] * self.pci_ash_w.get(k, 0.0)
            for f, m in flux.items():
                comp = mat["flux"][f]["comp"]
                ox["CaO"] += m * (comp.get("CaCO3", 0) / 100 * 56.077 / 100.09 + comp.get("CaO", 0) / 100)
                ox["MgO"] += m * (comp.get("MgCO3", 0) / 100 * 40.304 / 84.313 + comp.get("MgO", 0) / 100)
                ox["SiO2"] += m * comp.get("SiO2", 0) / 100
                ox["Al2O3"] += m * comp.get("Al2O3", 0) / 100
            ox["SiO2"] -= est["Si"] * 10 * 60.084 / 28.086
            ox["MnO"] -= est["Mn"] * 10 * 70.937 / 54.938
            slag = sum(ox.values()) + 0.5 + slag_guess * est["slag_FeO"] / 100
            slag_guess = slag
            # targets
            B = op["basicity_target"]
            mg = op["mgo_target"] / 100.0
            dCaO = B * ox["SiO2"] - ox["CaO"]
            dMgO = mg * slag - ox["MgO"]
            # dolomite first for MgO, then limestone/quartzite for basicity
            dl = mat["flux"]["dolomite"]["comp"]
            mgo_per_dol = dl["MgCO3"] / 100 * 40.304 / 84.313
            cao_per_dol = dl["CaCO3"] / 100 * 56.077 / 100.09
            ls = mat["flux"]["limestone"]["comp"]
            cao_per_ls = ls["CaCO3"] / 100 * 56.077 / 100.09
            qz = mat["flux"]["quartzite"]["comp"]
            new = dict(flux)
            new["dolomite"] = max(0.0, flux["dolomite"] + 0.8 * dMgO / mgo_per_dol)
            dCaO2 = dCaO - (new["dolomite"] - flux["dolomite"]) * cao_per_dol
            if flux["quartzite"] > 0 or (dCaO2 < 0 and flux["limestone"] <= 0):
                new["quartzite"] = max(0.0, flux["quartzite"] - 0.8 * dCaO2 / B / (qz["SiO2"] / 100))
                new["limestone"] = 0.0
            else:
                new["limestone"] = max(0.0, flux["limestone"] + 0.8 * dCaO2 / cao_per_ls)
            if max(abs(new[k] - flux[k]) for k in flux) < 0.01:
                flux = new
                break
            flux = new
        self.recipe = {
            "coke": coke_rate, "pci": pci, "ore_total": ore, "mix": mix,
            "sinter": ore * sf, "pellet": ore * pf, "lump": ore * lf,
            "limestone": flux["limestone"], "dolomite": flux["dolomite"], "quartzite": flux["quartzite"],
            "slag_est": slag_guess, "fe_blend": fe_blend,
        }
        self._recipe_per_m3()
        return self.recipe

    def _recipe_per_m3(self):
        """Convert the per-tHM recipe to species kmol per m3 of mixed charge."""
        mat = self.inp["materials"]
        r = self.recipe
        op = self.ops
        n = np.zeros(NS)
        dust = np.zeros(NS)
        vol_ore = 0.0
        vol_coke = 0.0
        ri_vol = 0.0
        tsoft = tmelt = mass_ore = 0.0
        for b in ["sinter", "pellet", "lump"]:
            m = r[b]
            if m <= 0:
                continue
            spec = mat["burden"][b]
            comp = spec["comp"]
            tot = sum(comp.values())
            fd = spec["fines_to_dust"]
            for k, v in comp.items():
                key = "Sb" if k == "S" else k
                amt = m * v / tot / TH.MW["S" if k == "S" else k]
                n[IS[key]] += amt * (1 - fd)
                dust[IS[key]] += amt * fd
            n[IS["H2Ol"]] += m * spec["moisture"] / (100 - spec["moisture"]) / TH.MW["H2O"]
            v = m * (1 - fd) / spec["bulk_density"]
            vol_ore += v
            ri_vol += v * spec["reducibility"]
            tsoft += m * spec["T_soft"]
            tmelt += m * spec["T_melt"]
            mass_ore += m
        for f in ["limestone", "dolomite", "quartzite"]:
            m = r[f]
            if m <= 0:
                continue
            spec = mat["flux"][f]
            for k, v in spec["comp"].items():
                n[IS[k]] += m * v / 100.0 / TH.MW[k]
            n[IS["H2Ol"]] += m * spec["moisture"] / 100.0 / TH.MW["H2O"]
            vol_ore += m / spec["bulk_density"]
        coke = mat["coke"]
        cm = r["coke"]
        fdc = coke["fines_to_dust"]
        cs = coke["C"] + coke["ash"] + coke["S"]
        n[IS["C"]] += cm * (1 - fdc) * coke["C"] / cs / TH.MW["C"]
        n[IS["S"]] += cm * (1 - fdc) * coke["S"] / cs / TH.MW["S"]
        n[IS["ASH"]] += cm * (1 - fdc) * coke["ash"] / cs
        dust[IS["C"]] += cm * fdc * coke["C"] / cs / TH.MW["C"]
        dust[IS["ASH"]] += cm * fdc * coke["ash"] / cs
        # coke moisture: actual vs assumed (weighing on wet basis)
        assumed = op.get("coke_moisture_assumed", coke["moisture"]) / 100.0
        actual = op.get("coke_moisture", coke["moisture"]) / 100.0
        wet = cm / (1 - assumed)
        dry_actual = wet * (1 - actual)
        scale = dry_actual / cm
        for k in ["C", "S", "ASH"]:
            n[IS[k]] *= scale
        n[IS["H2Ol"]] += wet * actual / TH.MW["H2O"]
        vol_coke = dry_actual * (1 - fdc) / coke["bulk_density"]
        V = vol_ore + vol_coke
        self.charge = {
            "coke_kg_per_m3": dry_actual / V, "ore_kg_per_m3": (r["sinter"] + r["pellet"] + r["lump"]) / V,
            "n_per_m3": n / V, "dust_per_m3": dust / V, "fore": vol_ore / V, "fcoke": vol_coke / V,
            "m3_per_tHM": V, "RI": ri_vol / max(vol_ore, 1e-9),
            "T_soft": tsoft / max(mass_ore, 1e-9) + 273.15, "T_melt": tmelt / max(mass_ore, 1e-9) + 273.15,
        }
        Tch = self.p["T_charge"]
        self.charge["H_per_m3"] = float(np.dot(self.charge["n_per_m3"], TH.interp_tab(self.SOLH, [Tch])[:, 0]))
        self.charge["Hdust_per_m3"] = float(np.dot(self.charge["dust_per_m3"], TH.interp_tab(self.SOLH, [Tch])[:, 0]))

    # ------------------------------------------------------------------
    # initial state
    # ------------------------------------------------------------------
    def _init_state(self):
        N = self.N
        g = self.geom
        z = g.z_mid / g.H
        Tprof = 1820.0 - 1400.0 * np.clip(z, 0, 1) ** 0.75
        Tprof = np.where(z > 0.82, 950 - (z - 0.82) / 0.18 * 650, Tprof)
        Tprof = np.clip(Tprof, 380.0, 1820.0)
        nchg = self.charge["n_per_m3"]
        n = np.zeros((N, NS))
        Vore = np.zeros(N)
        Vcoke = np.zeros(N)
        Tm = self.charge["T_melt"]
        for j in range(N):
            V = g.V_cell[j]
            if Tprof[j] >= Tm:
                # dripping zone: coke only
                cfrac = 1.0
                n[j, COKE_IDX] = nchg[COKE_IDX] / self.charge["fcoke"] * V
                Vcoke[j] = V
            else:
                n[j] = nchg * V
                Vore[j] = V * self.charge["fore"]
                Vcoke[j] = V * self.charge["fcoke"]
                # pre-reduce according to temperature
                fe = 2 * n[j, IS["Fe2O3"]] + n[j, IS["FeO"]]
                T = Tprof[j]
                n[j, IS["H2Ol"]] = n[j, IS["H2Ol"]] if T < 380 else 0.0
                if T > 1050:
                    met = min(0.7, (T - 1050) / 300 * 0.7)
                    n[j, IS["Fe2O3"]] = 0
                    n[j, IS["Fe3O4"]] = 0
                    n[j, IS["Fe"]] = fe * met
                    n[j, IS["FeO"]] = fe * (1 - met)
                elif T > 850:
                    n[j, IS["Fe2O3"]] = 0
                    n[j, IS["Fe3O4"]] = 0
                    n[j, IS["FeO"]] = fe
                elif T > 650:
                    n[j, IS["Fe2O3"]] = 0
                    n[j, IS["Fe3O4"]] = fe / 3.0
                    n[j, IS["FeO"]] = 0
                if T > 1100:
                    for s, s2 in [("CaCO3", "CaO"), ("MgCO3", "MgO")]:
                        n[j, IS[s2]] += n[j, IS[s]]
                        n[j, IS[s]] = 0
        self.n = n
        self.T_s = Tprof.copy()
        self.H_s = self.h_cells(n, self.T_s)
        self.V_ore = Vore
        self.V_ore0 = Vore.copy()
        self.V_coke = Vcoke
        self.RIf = Vore * self.charge["RI"] * self.ops.get("ore_reducibility", 1.0)
        self.CRIf = Vcoke * self.ops.get("coke_reactivity", 1.0)
        self.T_g = np.clip(Tprof + 30, 300, 2300)
        self.P = np.linspace(self.ops["top_pressure"] + 1.013 + 1.6, self.ops["top_pressure"] + 1.013, N)
        self.liq_buf = [None] * N      # melt waiting for the liquid march
        self.si_buf = np.zeros(N)      # Si captured (kmol) waiting for liquids
        self.si_buf_h = np.zeros(N)    # its enthalpy (MJ)
        self.pend_gas = np.zeros((N, NG))   # gas generated in condensed update (kmol)
        self.pend_H = np.zeros(N)
        self.Q_lg = np.zeros(N)
        self.gas_src = np.zeros((N, NG))
        self.gas_srcH = np.zeros(N)
        self.y_g = np.zeros((N, 6))
        self.hearth_coke_demand = 0.0   # kmol C/s drawn by hearth reactions
        # hearth inventory (kmol) at nominal level
        hm_mass = self.geom.A_hearth * (self.p["sump_active"] * self.geom.h_sump + 0.30 * self.geom.h_hearth) * self.p["deadman_void"] * self.p["rho_hm"]
        slag_mass = self.geom.A_hearth * 0.35 * self.geom.h_hearth * self.p["deadman_void"] * self.p["rho_slag"]
        self.hm = np.array([hm_mass * 0.946 / 55.845, hm_mass * 0.046 / 12.011, hm_mass * 0.0045 / 28.086,
                            hm_mass * 0.0025 / 54.938, hm_mass * 0.001 / 30.974, hm_mass * 0.0003 / 32.06])
        sw = {"SiO2": 0.335, "Al2O3": 0.175, "CaO": 0.36, "MgO": 0.085, "MnO": 0.004, "FeO": 0.005,
              "TiO2": 0.01, "P2O5": 0.0005, "CaS": 0.025}
        self.sl = np.array([slag_mass * sw[s] / TH.MW[s] for s in TH.SLAG])
        Vh = self.geom.V_hearth
        self.M_dm = Vh * (1 - self.p["deadman_void"]) * 1000.0 * self.p["deadman_participation"] / 12.011  # kmol C
        self.T_h = 1773.15
        self.H_h = self._hearth_H(self.T_h)
        self.cast = {"open": False, "t_next": 0.0, "t_close": 0.0, "hole": 0, "count": 0}
        self.measurements = []
        self.casts = []
        self.acc = self._zero_acc()
        self.last = {}
        self.top = {}
        self._dt_last = 300.0

    def _hearth_H(self, T):
        return TH.h_metal(list(self.hm), T) + TH.h_slag(list(self.sl), T) + self.M_dm * TH.h_cond_species("C", T)

    def _hearth_T(self, H, Tg):
        hm = list(self.hm)
        sl = list(self.sl)
        Mdm = self.M_dm

        def extra(T):
            return Mdm * TH.h_cond_species("C", T), Mdm * TH.cp_cond_species("C", T)
        return TH.T_from_h_liquids(hm, sl, H, extra_h_fn=extra, Tguess=Tg)

    def _zero_acc(self):
        return {"in_E": np.zeros(NE), "out_E": np.zeros(NE), "in_H": 0.0, "out_H": 0.0, "loss_H": 0.0,
                "hm_kg": 0.0, "slag_kg": 0.0, "time": 0.0}

    # ------------------------------------------------------------------
    # inventory (elements & enthalpy) for balance checks
    # ------------------------------------------------------------------
    def inventory(self):
        E = self.n.sum(axis=0) @ self.ES
        H = float(self.H_s.sum())
        for j in range(self.N):
            b = self.liq_buf[j]
            if b is not None:
                E = E + b["m"] @ self.EM + b["s"] @ self.EL
                H += b["H"]
        E = E + self.si_buf.sum() * self.EM[2]
        H += float(self.si_buf_h.sum())
        E = E + self.pend_gas.sum(axis=0) @ self.EG
        H += float(self.pend_H.sum())
        E = E + self.hm @ self.EM + self.sl @ self.EL + self.M_dm * self.EG[iCH]
        H += self.H_h
        return E, H

    # ------------------------------------------------------------------
    # raceway sub-model
    # ------------------------------------------------------------------
    def raceway(self, ops, coke_T, pci_kg_s, ash_per_C=None, S_per_C=None):
        p = self.p
        V = ops["blast_volume"] / 60.0 / TH.NM3_PER_KMOL         # kmol/s dry blast
        yO2 = (21.0 + ops["o2_enrich"]) / 100.0
        nO2 = V * yO2
        nN2 = V * (1 - yO2)
        nH2Ob = ops["moisture"] * ops["blast_volume"] / 60.0 / 1000.0 / 18.015
        Tb = ops["blast_temp"] + 273.15
        dry = pci_kg_s * (1 - self.coal_moist)
        c = self.coal
        nCc = dry * c["C"] / 12.011
        nH2c = dry * c["H"] / 2.016
        nOc = dry * c["O"] / 15.999
        nN2c = dry * c["N"] / 28.014 + pci_kg_s * p["pci_carrier_nm3_per_kg"] / TH.NM3_PER_KMOL
        nSc = dry * c["S"] / 32.06
        nH2Oc = pci_kg_s * self.coal_moist / 18.015
        # burnout
        oc = (2 * nO2 + nH2Ob) / max(nCc, 1e-9)
        vm = self.inp["materials"]["pci"]["VM"]
        kb = p["burnout_k"] * oc * math.sqrt(max(vm, 5.0) / 20.0) * math.exp(-p["burnout_E"] / RJ * (1 / Tb - 1 / 1443.0))
        Xb = min(0.97, 1 - math.exp(-kb)) if nCc > 0 else 1.0
        Oat = 2 * nO2 + nH2Ob + nOc + nH2Oc
        cburn_coal = min(Xb * nCc, 0.95 * Oat)
        char = nCc - cburn_coal
        C_comb = Oat - cburn_coal               # coke carbon burnt to CO
        # ash associated with coke reaching the raceway
        a = ash_per_C if ash_per_C is not None else self.recipe_ash_per_C()
        sC = S_per_C if S_per_C is not None else self.recipe_S_per_C()
        wS = self.ash_w["SiO2"] / TH.MW["SiO2"]
        wF = self.ash_w["Fe2O3"] / TH.MW["Fe2O3"]
        pash = dry * c["ash"]
        pS = pash * self.pci_ash_w.get("SiO2", 0) / TH.MW["SiO2"]
        pF = pash * self.pci_ash_w.get("Fe2O3", 0) / TH.MW["Fe2O3"]

        def solve(T_raft):
            phi = min(0.9, p["phi_SiO"] * math.exp(-p["E_SiO"] / RJ * (1 / T_raft - 1 / 2400.0)))
            # C_total = C_comb + C_total*a*(phi*wS + 3*wF) + phi*pS + 3*pF
            frac = a * (phi * wS + 3 * wF)
            Ct = (C_comb + phi * pS + 3 * pF) / max(1e-6, 1 - frac)
            ash = Ct * a
            nSiO = phi * (ash * wS + pS)
            nFe = 2 * (ash * wF + pF)
            Scoke = Ct * sC
            return phi, Ct, ash, nSiO, nFe, Scoke

        # energy in
        H_blast = nO2 * TH.h_gas_exact("O2", Tb) + nN2 * self._hg(iN2, Tb) + nH2Ob * self._hg(iH2O, Tb)
        Tco = p["T_coal"]
        H_coal = dry * (self.coal_hf + self.coal_cp * (Tco - 298.15)) + nH2Oc * TH.h_cond_species("H2Ol", Tco) \
            + pci_kg_s * p["pci_carrier_nm3_per_kg"] / TH.NM3_PER_KMOL * self._hg(iN2, Tco)
        T = 2300.0
        for it in range(40):
            phi, Ct, ash, nSiO, nFe, Scoke = solve(T)
            nCO = Oat + nSiO + 3 * (nFe / 2.0)  # O from H2O/O2/coal + SiO2 + Fe2O3 reduction
            # oxygen check: SiO2 + C -> SiO + CO gives one CO per SiO; Fe2O3 + 3C -> 2Fe + 3CO
            nCO = Oat + nSiO + 1.5 * nFe
            H_coke = Ct * TH.h_cond_species("C", coke_T) + ash * self._ash_h(coke_T) + Scoke * TH.h_cond_species("S", coke_T)
            Hin = H_blast + H_coal + H_coke
            Sg = p["f_Sg"] * (Scoke + nSc)
            gas = [nCO, 0.0, nH2c + nH2Ob + nH2Oc, 0.0, nN2 + nN2c, nSiO, char, Sg]
            # condensed products at RAFT: ash oxides minus SiO2/Fe2O3 consumed, Fe liquid, S
            slag_ox = self._raceway_slag(ash, pash, nSiO, nFe)
            S_liq = (1 - p["f_Sg"]) * (Scoke + nSc)

            def hout(Tx):
                h = self._hmix(gas, Tx)
                h += sum(v * TH.h_cond_species(k, Tx) for k, v in slag_ox.items()) + self._slag_mass(slag_ox) * TH.L_SLAG_MELT
                h += nFe * (TH.h_cond_species("Fe", Tx) + TH.L_FE_MELT) + S_liq * TH.h_cond_species("dS", Tx)
                return h
            h1 = hout(T)
            h2 = hout(T + 1.0)
            dT = (Hin - h1) / max(h2 - h1, 1e-9)
            dT = max(-300, min(300, dT))
            T += dT
            if abs(dT) < 0.01:
                break
        T_raft = T
        H_gas = self._hmix(gas, T_raft)
        H_cond = Hin - H_gas
        # tuyere velocity & raceway depth
        g = self.geom
        P_blast = self.P[0] if hasattr(self, "P") else 4.0
        Vact = (V + nH2Ob) * TH.NM3_PER_KMOL * Tb / 273.15 * 1.01325 / max(P_blast, 1.2)
        A_t = g.n_tuyere * math.pi * g.d_tuyere ** 2 / 4
        v_t = Vact / A_t
        rho_b = (nO2 * 32 + nN2 * 28 + nH2Ob * 18) / max(Vact, 1e-9)
        KE = 0.5 * rho_b * Vact / g.n_tuyere * v_t ** 2      # W per tuyere
        depth = 0.30 + 0.85 * math.sqrt(KE / 1.2e5)
        return {
            "T_raft": T_raft, "gas": gas, "H_gas": H_gas, "H_cond": H_cond, "slag_ox": slag_ox,
            "nFe": nFe, "S_liq": S_liq, "C_coke": Ct, "ash": ash, "S_coke": Scoke, "char": char,
            "burnout": Xb, "coal_C_burnt": cburn_coal, "phi_SiO": phi, "nSiO": nSiO,
            "v_tuyere": v_t, "KE_tuyere_kW": KE / 1000.0, "depth": depth,
            "nO2": nO2, "nN2": nN2, "nH2Ob": nH2Ob, "H_blast": H_blast, "H_coal": H_coal,
            "bosh_gas_Nm3_min": (nCO + gas[2] + gas[4] + nSiO) * TH.NM3_PER_KMOL * 60.0,
            "oc_ratio": oc, "pci_dry": dry, "pci_ash": pash, "nSc": nSc, "H_coke": H_coke,
        }

    def _ash_h(self, T):
        i, f = TH._idx(T)
        r = self.SOLH[IS["ASH"]]
        return r[i] + f * (r[i + 1] - r[i])

    def _raceway_slag(self, coke_ash, pci_ash, nSiO, nFe):
        ox = {}
        for k, w in self.ash_w.items():
            ox[k] = ox.get(k, 0.0) + coke_ash * w / TH.MW[k]
        for k, w in self.pci_ash_w.items():
            ox[k] = ox.get(k, 0.0) + pci_ash * w / TH.MW[k]
        ox["SiO2"] = ox.get("SiO2", 0.0) - nSiO
        ox["Fe2O3"] = ox.get("Fe2O3", 0.0) - nFe / 2.0
        ox.pop("Fe2O3", None)
        return {k: max(0.0, v) for k, v in ox.items()}

    def _slag_mass(self, ox):
        return sum(v * TH.MW[k] for k, v in ox.items())

    def recipe_ash_per_C(self):
        n = self.charge["n_per_m3"]
        return n[IS["ASH"]] / max(n[IS["C"]], 1e-9)

    def recipe_S_per_C(self):
        n = self.charge["n_per_m3"]
        return n[IS["S"]] / max(n[IS["C"]], 1e-9)

    # ------------------------------------------------------------------
    # liquid march (dripping zone), top -> bottom
    # ------------------------------------------------------------------
    def _liquid_march(self, dt):
        """Dripping-zone liquid march (top -> bottom).

        Two percolation paths: a peripheral path through the active coke zone next to
        the raceways (heat exchange with gas and coke) and a central path through the
        low-gas-flow deadman region (no sensible-heat exchange).  Their mixture enters
        the hearth.  Reactions (direct reduction, carburisation, MnO/P2O5 reduction) act
        on the whole stream; their enthalpy effects are shared by both paths."""
        N = self.N
        p = self.p
        fc = p["central_path"]
        m = np.zeros(6)
        s = np.zeros(len(TH.SLAG))
        Hp = 0.0            # peripheral enthalpy flow (MW)
        Hc = 0.0            # central enthalpy flow
        Tl = 1700.0
        Q_lg = np.zeros(N)
        Q_ls = np.zeros(N)
        gsrc = np.zeros((N, NG))
        gsrcH = np.zeros(N)
        dC = np.zeros(N)
        present = np.zeros(N, dtype=bool)
        hA_l = p["h_lg"] * p["a_lg"] * self.geom.V_cell * 1e-6 * p["hv_factor"]
        hA_ls = p["h_ls"] * p["a_ls"] * self.geom.V_cell * 1e-6
        Ts = self.T_s
        nC = self.n[:, IS["C"]]
        any_liq = False
        for j in range(N - 1, -1, -1):
            b = self.liq_buf[j]
            if b is not None:
                m = m + b["m"] / dt
                s = s + b["s"] / dt
                Hp += (1 - fc) * b["H"] / dt
                Hc += fc * b["H"] / dt
                self.liq_buf[j] = None
                any_liq = True
            if self.si_buf[j] > 0:
                m[2] += self.si_buf[j] / dt
                Hp += (1 - fc) * self.si_buf_h[j] / dt
                Hc += fc * self.si_buf_h[j] / dt
                self.si_buf[j] = 0.0
                self.si_buf_h[j] = 0.0
            if not any_liq or (m.sum() + s.sum()) <= 1e-12:
                continue
            present[j] = True
            mp = list(m * (1 - fc))
            sp = list(s * (1 - fc))
            Tp = TH.T_from_h_liquids(mp, sp, Hp, Tguess=Tl)
            Clp = TH.cp_metal(mp, Tp) + TH.cp_slag(sp, Tp)
            if Clp > 1e-12:
                eps = 1 - math.exp(-hA_l[j] / Clp)
                Q = eps * Clp * (self.T_g[j] - Tp)
                Hp += Q
                Q_lg[j] += Q
                Tp += Q / Clp
                if nC[j] > 1e-6:
                    eps2 = 1 - math.exp(-hA_ls[j] / Clp)
                    Q2 = eps2 * Clp * (Ts[j] - Tp)
                    Hp += Q2
                    Q_ls[j] += Q2
                    Tp += Q2 / Clp
            ml = list(m)
            sl = list(s)
            Tl = TH.T_from_h_liquids(ml, sl, Hp + Hc, Tguess=Tp)
            Tsj = Ts[j]
            if nC[j] <= 1e-6:
                continue
            Cavail = 0.4 * nC[j] / dt
            tau = p["tau_l"]
            hC = TH.h_cond_species("C", Tsj)
            hCO = self._hg(iCO, Tl)
            dH = 0.0
            iFeO = TH.LI["FeO"]
            if s[iFeO] > 0:
                k = p["k_dr"] * math.exp(-p["E_dr"] / RJ * (1 / Tl - 1 / 1723.0))
                x = min(s[iFeO] * (1 - math.exp(-k * tau)), Cavail)
                s[iFeO] -= x
                m[0] += x
                dC[j] += x
                Cavail -= x
                gsrc[j, iCO] += x
                gsrcH[j] += x * hCO
                dH += x * hC - x * hCO
            iMn = TH.LI["MnO"]
            if s[iMn] > 0 and Cavail > 0:
                k = p["k_mn"] * math.exp(-200e3 / RJ * (1 / Tl - 1 / 1773.0))
                x = min(0.6 * s[iMn] * (1 - math.exp(-k * tau)), Cavail)
                s[iMn] -= x
                m[3] += x
                dC[j] += x
                Cavail -= x
                gsrc[j, iCO] += x
                gsrcH[j] += x * hCO
                dH += x * hC - x * hCO
            iP = TH.LI["P2O5"]
            if s[iP] > 0 and Cavail > 0:
                x = min(s[iP] * (1 - math.exp(-p["k_p"] * tau)), Cavail / 5)
                s[iP] -= x
                m[4] += 2 * x
                dC[j] += 5 * x
                Cavail -= 5 * x
                gsrc[j, iCO] += 5 * x
                gsrcH[j] += 5 * x * hCO
                dH += 5 * x * hC - 5 * x * hCO
            massm = m[0] * 55.845 + m[1] * 12.011 + m[2] * 28.086 + m[3] * 54.938 + m[4] * 30.974 + m[5] * 32.06
            if massm > 0 and Cavail > 0:
                wC = m[1] * 12.011 / massm * 100
                wSi = m[2] * 28.086 / massm * 100
                Csat = SL.carbon_saturation(Tl - 273.15, Si=wSi)
                if Csat > wC:
                    k = p["k_carb"] * self.geom.dz
                    dw = (Csat - wC) * (1 - math.exp(-k * tau))
                    x = min(dw / 100 * massm / 12.011, Cavail)
                    m[1] += x
                    dC[j] += x
                    dH += x * hC
            Hp += (1 - fc) * dH
            Hc += fc * dH
        H = Hp + Hc
        Tl = TH.T_from_h_liquids(list(m), list(s), H, Tguess=Tl) if (m.sum() + s.sum()) > 0 else Tl
        self.liq_out = {"m": m, "s": s, "H": H, "T": Tl}
        self.Q_lg = Q_lg
        self.Q_ls = Q_ls
        self.gas_src = gsrc
        self.gas_srcH = gsrcH
        self.liq_present = present
        return dC

    # ------------------------------------------------------------------
    # gas march, bottom -> top
    # ------------------------------------------------------------------
    def _gas_march(self, dt, rw):
        N = self.N
        p = self.p
        g = self.geom
        ops = self.ops
        n = self.n
        Sl = n.tolist()
        Ts = self.T_s.tolist()
        P = self.P.tolist()
        Vc = g.V_cell
        RI = (self.RIf / np.maximum(self.V_ore, 1e-9)).tolist()
        CRI = (self.CRIf / np.maximum(self.V_coke, 1e-9)).tolist()
        cri = self.inp["materials"]["coke"]["CRI"] / 25.0
        dS = [[0.0] * NS for _ in range(N)]
        dHs = [0.0] * N
        gas = list(rw["gas"])
        Hg = rw["H_gas"] - self.wall.tuyere_loss(rw["T_raft"])
        Tg = self._Tgas(gas, Hg, rw["T_raft"] - 50)
        Tg_out = [0.0] * N
        ycomp = [None] * N
        dP = [0.0] * N
        ext = {k: [0.0] * N for k in ["hem", "mag", "wus", "wusH", "bou", "wg", "char", "cal", "evap", "sicap", "sire", "Qgs", "Qw"]}
        # wall loss per cell (MW) using mixed temperature
        Qw_cell = self.wall.cell_losses(0.5 * (self.T_s + self.T_g)).tolist()
        qshare = p["wall_gas_share"]
        hv = self._hv_cells()
        dp_cell = self._dp_coeffs()
        liq = self.liq_present
        T_soft = self.charge["T_soft"]
        T_melt = self.charge["T_melt"]
        exp = math.exp
        kh0, Eh, Th0 = p["k_hem"], p["E_hem"], p["T_hem"]
        km0, Em, Tm0 = p["k_mag"], p["E_mag"], p["T_mag"]
        kw0, Ew, Tw0 = p["k_wus"], p["E_wus"], p["T_wus"]
        fH = p["h2_factor"]
        kB0, EB = p["k_bou"], p["E_bou"]
        sicap_done = False
        for j in range(N):
            # sources from liquid march & pending gas
            if self.gas_src[j].any():
                for k in range(NG):
                    gas[k] += self.gas_src[j][k]
                Hg += self.gas_srcH[j]
            if self.pend_gas[j].any():
                for k in range(NG):
                    gas[k] += self.pend_gas[j][k] / dt
                Hg += self.pend_H[j] / dt
                self.pend_gas[j] = 0.0
                self.pend_H[j] = 0.0
            Hg -= self.Q_lg[j]
            Tg = self._Tgas(gas, Hg, Tg)
            Tsj = Ts[j]
            Pj = P[j]
            s = Sl[j]
            dsj = dS[j]
            hcum = 0.0
            # ---- PCI char gasification (entrained, at gas temperature)
            ntot = gas[0] + gas[1] + gas[2] + gas[3] + gas[4] + gas[5]
            if gas[iCH] > 1e-12 and Tg > 1000.0:
                kc = kB0 * p["char_factor"] * cri * exp(-EB / RJ * (1 / Tg - 1 / 1373.0)) * 12.011
                pco2 = gas[1] / ntot * Pj
                ph2o = gas[3] / ntot * Pj
                tau = 0.8
                X = 1 - exp(-kc * (pco2 + p["wg_factor"] * ph2o) * tau)
                x = gas[iCH] * X
                if x > 0:
                    w = pco2 / max(pco2 + p["wg_factor"] * ph2o, 1e-12)
                    x1 = min(x * w, 0.9 * gas[1])
                    x2 = min(x * (1 - w), 0.9 * gas[3])
                    gas[iCH] -= x1 + x2
                    gas[1] -= x1
                    gas[0] += 2 * x1 + x2
                    gas[3] -= x2
                    gas[2] += x2
                    ext["char"][j] += x1 + x2
            ntot = gas[0] + gas[1] + gas[2] + gas[3] + gas[4] + gas[5]
            # ---- coke solution loss and water-gas (at solid temperature)
            nCj = s[IS["C"]]
            if nCj > 1e-9 and Tsj > 900.0:
                kB = kB0 * cri * CRI[j] * exp(-EB / RJ * (1 / Tsj - 1 / 1373.0))
                mC = nCj * 12.011
                # Boudouard
                if gas[1] > 1e-12:
                    K = TH.K_boudouard(Tsj)
                    a_, b_ = gas[0], gas[1]
                    nt = ntot
                    A = 4 * Pj + K
                    B = 4 * a_ * Pj - K * b_ + K * nt
                    C = a_ * a_ * Pj - K * b_ * nt
                    disc = B * B - 4 * A * C
                    xeq = (-B + math.sqrt(max(disc, 0.0))) / (2 * A) if disc > 0 else 0.0
                    xeq = min(max(xeq, 0.0), b_)
                    if xeq > 1e-12:
                        r0 = kB * mC * b_ / nt * Pj / 12.011
                        x = xeq * (1 - exp(-r0 / xeq))
                        x = min(x, 0.5 * nCj / dt)
                        h_in = x * self._hg(iCO2, Tg)
                        h_out = 2 * x * self._hg(iCO, Tsj)
                        gas[1] -= x
                        gas[0] += 2 * x
                        Hg += h_out - h_in
                        dsj[IS["C"]] -= x
                        dHs[j] += h_in - h_out
                        ext["bou"][j] += x
                        ntot += x
                # water-gas
                if gas[3] > 1e-12:
                    K = TH.K_watergas(Tsj)
                    # (CO+x)(H2+x)/((H2O-x)(nt+x)) P = K   -> solve numerically (few iterations)
                    a_, h_, w_ = gas[0], gas[2], gas[3]
                    nt = ntot
                    lo, hi = 0.0, w_
                    for _ in range(30):
                        mid = 0.5 * (lo + hi)
                        f = (a_ + mid) * (h_ + mid) * Pj - K * (w_ - mid) * (nt + mid)
                        if f > 0:
                            hi = mid
                        else:
                            lo = mid
                    xeq = 0.5 * (lo + hi)
                    if xeq > 1e-12:
                        r0 = p["wg_factor"] * kB * mC * w_ / nt * Pj / 12.011
                        x = xeq * (1 - exp(-r0 / xeq))
                        x = min(x, 0.25 * nCj / dt)
                        h_in = x * self._hg(iH2O, Tg)
                        h_out = x * (self._hg(iCO, Tsj) + self._hg(iH2, Tsj))
                        gas[3] -= x
                        gas[0] += x
                        gas[2] += x
                        Hg += h_out - h_in
                        dsj[IS["C"]] -= x
                        dHs[j] += h_in - h_out
                        ext["wg"][j] += x
                        ntot += x
            # ---- wustite reduction (CO, H2)
            nF = s[IS["FeO"]]
            if nF > 1e-9 and Tsj > 843.0:
                k = kw0 * RI[j] * exp(-Ew / RJ * (1 / Tsj - 1 / Tw0))
                K1 = TH.K_wustite_CO(Tsj)
                xeq = (K1 * gas[0] - gas[1]) / (1 + K1)
                if xeq > 1e-12:
                    r0 = k * nF * (gas[0] - gas[1] / K1) / ntot * Pj
                    x = xeq * (1 - exp(-r0 / xeq)) if r0 > 0 else 0.0
                    x = min(x, 0.9 * nF / dt)
                    h_in = x * self._hg(iCO, Tg)
                    h_out = x * self._hg(iCO2, Tsj)
                    gas[0] -= x
                    gas[1] += x
                    Hg += h_out - h_in
                    dsj[IS["FeO"]] -= x
                    dsj[IS["Fe"]] += x
                    dHs[j] += h_in - h_out
                    ext["wus"][j] += x
                    nF -= x * dt
                K3 = TH.K_wustite_H2(Tsj)
                xeq = (K3 * gas[2] - gas[3]) / (1 + K3)
                if xeq > 1e-12 and nF > 1e-9:
                    r0 = fH * k * nF * (gas[2] - gas[3] / K3) / ntot * Pj
                    x = xeq * (1 - exp(-r0 / xeq)) if r0 > 0 else 0.0
                    x = min(x, 0.9 * nF / dt)
                    h_in = x * self._hg(iH2, Tg)
                    h_out = x * self._hg(iH2O, Tsj)
                    gas[2] -= x
                    gas[3] += x
                    Hg += h_out - h_in
                    dsj[IS["FeO"]] -= x
                    dsj[IS["Fe"]] += x
                    dHs[j] += h_in - h_out
                    ext["wusH"][j] += x
            # ---- magnetite -> wustite
            nM = s[IS["Fe3O4"]]
            if nM > 1e-9:
                k = km0 * RI[j] * exp(-Em / RJ * (1 / Tsj - 1 / Tm0))
                for (ired, iox, Kf, fac) in ((iCO, iCO2, TH.K_magnetite_CO, 1.0), (iH2, iH2O, TH.K_magnetite_H2, fH)):
                    K2 = Kf(Tsj)
                    xeq = (K2 * gas[ired] - gas[iox]) / (1 + K2)
                    if xeq > 1e-12 and nM > 1e-9:
                        r0 = fac * k * nM * (gas[ired] - gas[iox] / K2) / ntot * Pj
                        x = xeq * (1 - exp(-r0 / xeq)) if r0 > 0 else 0.0
                        x = min(x, 0.9 * nM / dt)
                        h_in = x * self._hg(ired, Tg)
                        h_out = x * self._hg(iox, Tsj)
                        gas[ired] -= x
                        gas[iox] += x
                        Hg += h_out - h_in
                        dsj[IS["Fe3O4"]] -= x
                        dsj[IS["FeO"]] += 3 * x
                        dHs[j] += h_in - h_out
                        ext["mag"][j] += x
                        nM -= x * dt
            # ---- hematite -> magnetite (irreversible)
            nH = s[IS["Fe2O3"]]
            if nH > 1e-9:
                k = kh0 * RI[j] * exp(-Eh / RJ * (1 / Tsj - 1 / Th0))
                for (ired, iox, fac) in ((iCO, iCO2, 1.0), (iH2, iH2O, fH)):
                    if gas[ired] <= 1e-12 or nH <= 1e-9:
                        continue
                    r0 = fac * k * nH * gas[ired] / ntot * Pj / 3.0
                    x = min(r0, 0.95 * gas[ired], 0.9 * nH / 3.0 / dt)
                    h_in = x * self._hg(ired, Tg)
                    h_out = x * self._hg(iox, Tsj)
                    gas[ired] -= x
                    gas[iox] += x
                    Hg += h_out - h_in
                    dsj[IS["Fe2O3"]] -= 3 * x
                    dsj[IS["Fe3O4"]] += 2 * x
                    dHs[j] += h_in - h_out
                    ext["hem"][j] += 3 * x
                    nH -= 3 * x * dt
            # ---- carbonate decomposition
            for sp, ox, peq in (("CaCO3", "CaO", TH.p_eq_CaCO3), ("MgCO3", "MgO", TH.p_eq_MgCO3)):
                nc = s[IS[sp]]
                if nc > 1e-9 and Tsj > 700.0:
                    pe = peq(Tsj)
                    pco2 = gas[1] / ntot * Pj
                    if pe > pco2:
                        k = p["k_cal"] * exp(-p["E_cal"] / RJ * (1 / Tsj - 1 / 1173.0))
                        x = min(nc * k * (1 - pco2 / pe), 0.9 * nc / dt)
                        h_out = x * self._hg(iCO2, Tsj)
                        gas[1] += x
                        Hg += h_out
                        dsj[IS[sp]] -= x
                        dsj[IS[ox]] += x
                        dHs[j] -= h_out
                        ext["cal"][j] += x
                        ntot += x
            # ---- moisture evaporation
            nw = s[IS["H2Ol"]]
            if nw > 1e-9 and Tsj > 345.0:
                x = min(nw * p["k_evap"] * min(1.0, (Tsj - 345.0) / 40.0), 0.95 * nw / dt)
                h_out = x * self._hg(iH2O, Tsj)
                gas[3] += x
                Hg += h_out
                dsj[IS["H2Ol"]] -= x
                dHs[j] -= h_out
                ext["evap"][j] += x
                ntot += x
            # ---- SiO capture by dripping metal / re-oxidation above
            if gas[iSiO] > 1e-12:
                if liq[j] and Tg > 1500.0:
                    phi = 1 - exp(-p["k_sicap"] * g.dz * exp(-p["E_sicap"] / RJ * (1 / Tg - 1 / 1873.0)))
                    x = gas[iSiO] * phi
                    cav = 0.2 * s[IS["C"]] / dt
                    x = min(x, cav)
                    if x > 0:
                        hSiO = self._hg(iSiO, Tg)
                        hCO = self._hg(iCO, Tg)
                        hC = TH.h_cond_species("C", Tsj)
                        gas[iSiO] -= x
                        gas[0] += x
                        Hg += x * (hCO - hSiO)
                        dsj[IS["C"]] -= x
                        dHs[j] -= x * hC
                        self.si_buf[j] += x * dt
                        self.si_buf_h[j] += x * dt * (hSiO + hC - hCO)
                        ext["sicap"][j] += x
                elif not liq[j]:
                    x = gas[iSiO]
                    xc = min(x, gas[0])
                    hSiO = self._hg(iSiO, Tg)
                    hCO = self._hg(iCO, Tg)
                    gas[iSiO] -= xc
                    gas[0] -= xc
                    Hg -= xc * (hSiO + hCO)
                    dsj[IS["SiO2"]] += xc
                    dsj[IS["C"]] += xc
                    dHs[j] += xc * (hSiO + hCO)
                    ext["sire"][j] += xc
            # ---- water-gas shift (gas phase, at Tg)
            if gas[0] > 1e-9 and gas[3] > 1e-9 and Tg > 700.0:
                Ks = TH.K_shift(Tg)
                a_, b_, h_, w_ = gas[0], gas[1], gas[2], gas[3]
                # (b+x)(h+x) = Ks (a-x)(w-x)
                A = 1 - Ks
                B = b_ + h_ + Ks * (a_ + w_)
                C = b_ * h_ - Ks * a_ * w_
                if abs(A) < 1e-9:
                    xeq = -C / B
                else:
                    disc = B * B - 4 * A * C
                    if disc < 0:
                        disc = 0.0
                    r1 = (-B + math.sqrt(disc)) / (2 * A)
                    r2 = (-B - math.sqrt(disc)) / (2 * A)
                    lo_, hi_ = -min(b_, h_), min(a_, w_)
                    xeq = r1 if lo_ <= r1 <= hi_ else r2
                    xeq = max(lo_, min(hi_, xeq))
                alpha = 1 - exp(-p["k_shift"] * exp(-p["E_shift"] / RJ * (1 / Tg - 1 / 1000.0)))
                x = alpha * xeq
                gas[0] -= x
                gas[3] -= x
                gas[1] += x
                gas[2] += x
            # ---- gas temperature after reactions (enthalpy conserved inside gas)
            Tg = self._Tgas(gas, Hg, Tg)
            # ---- gas-solid heat exchange + wall loss (exact energy split)
            Cg = self._cpmix(gas, Tg)
            ntu = hv[j] / max(Cg, 1e-12)
            Qwj = Qw_cell[j]
            T_new = Tsj + (Tg - Tsj) * exp(-ntu) - qshare * Qwj / max(Cg, 1e-12)
            if T_new < 270.0:
                T_new = 270.0
            H_new = self._hmix(gas, T_new)
            Qgs = Hg - H_new - qshare * Qwj
            Hg = H_new
            Tg = T_new
            dHs[j] += Qgs - (1 - qshare) * Qwj
            ext["Qgs"][j] = Qgs
            ext["Qw"][j] = Qwj
            # ---- pressure drop (Ergun)
            ntot = gas[0] + gas[1] + gas[2] + gas[3] + gas[4] + gas[5]
            Mg = (gas[0] * 28.01 + gas[1] * 44.01 + gas[2] * 2.016 + gas[3] * 18.015 + gas[4] * 28.014) / max(ntot, 1e-12)
            rho = Pj * 1e5 * Mg / (8314.46 * Tg)
            Q = ntot * 8314.46 * Tg / (Pj * 1e5)          # m3/s
            u = Q / g.A_cross[j]
            mu = 1.663e-5 * (Tg / 273.0) ** 1.5 * (273.0 + 107.0) / (Tg + 107.0)
            a1, a2 = dp_cell[j]
            dP[j] = (a1 * mu * u + a2 * rho * u * u) * g.dz * p["dp_factor"] / 1e5   # bar
            Tg_out[j] = Tg
            ycomp[j] = (gas[0] / ntot, gas[1] / ntot, gas[2] / ntot, gas[3] / ntot, gas[4] / ntot, gas[5] / ntot)
        # pressure profile from top
        Ptop = ops["top_pressure"] + 1.01325
        Pn = np.zeros(N)
        acc = Ptop
        for j in range(N - 1, -1, -1):
            acc += dP[j]
            Pn[j] = acc - 0.5 * dP[j]
        self.P = Pn
        self.P_blast = acc + 0.08
        self.dP_cell = np.array(dP)
        self.T_g = np.array(Tg_out)
        self.y_g = np.array(ycomp)
        self.ext = {k: np.array(v) for k, v in ext.items()}
        self.top = {"gas": gas, "H": Hg, "T": Tg}
        return np.array(dS), np.array(dHs)

    def _hv_cells(self):
        """Volumetric gas-solid heat-transfer coefficient per cell (MW/K)."""
        g = self.geom
        p = self.p
        out = np.zeros(self.N)
        ore = self.inp["materials"]["burden"]
        d_ore = 0.016
        d_coke = self.inp["materials"]["coke"]["size_mm"] / 1000.0
        fo = self.V_ore / g.V_cell
        fc = self.V_coke / g.V_cell
        T = np.maximum(self.T_g, 300.0)
        kg = 0.0243 * (T / 273.0) ** 0.8
        # superficial velocity estimate
        u = 2.0
        rho = 1.0
        mu = 1.663e-5 * (T / 273.0) ** 1.5 * 380.0 / (T + 107.0)
        hv = np.zeros(self.N)
        for d, f, eps, ks in ((d_ore, fo, p["eps_ore"], 1.6), (d_coke, fc, p["eps_coke"], 4.0)):
            Re = rho * u * d / mu
            Nu = 2 + 0.6 * np.sqrt(Re) * 0.89
            h = Nu * kg / d
            heff = 1 / (1 / h + d / (10 * ks))
            a = 6 * (1 - eps) / d
            hv += heff * a * f
        drip = np.where(self.V_ore / g.V_cell < 0.02, p["hv_drip"], 1.0)
        # cohesive zone: softened ore layers are impermeable, gas flows through coke
        # slits and heats the ore layers mainly by conduction -> reduced exchange
        Tsoft, Tmelt = self.charge["T_soft"], self.charge["T_melt"]
        soft = np.clip((self.T_s - Tsoft) / max(Tmelt - Tsoft, 1.0), 0.0, 1.0)
        cz = 1.0 - (1.0 - p["hv_cz"]) * soft * (self.V_ore / g.V_cell > 0.02)
        return hv * g.V_cell * 1e-6 * p["hv_factor"] * drip * cz

    def _dp_coeffs(self):
        """Ergun coefficients (a1, a2) per cell: dP/dz = a1 mu u + a2 rho u^2."""
        g = self.geom
        p = self.p
        out = []
        d_ore = 0.015
        d_coke = self.inp["materials"]["coke"]["size_mm"] / 1000.0
        Ts = self.T_s
        Tsoft = self.charge["T_soft"]
        Tmelt = self.charge["T_melt"]
        for j in range(self.N):
            fo = self.V_ore[j] / g.V_cell[j]
            fc = 1 - fo
            res = []
            for d, eps, phi in ((d_ore, p["eps_ore"], 0.8), (d_coke, p["eps_coke"], 0.75)):
                dp = d * phi
                a1 = 150 * (1 - eps) ** 2 / (eps ** 3 * dp * dp)
                a2 = 1.75 * (1 - eps) / (eps ** 3 * dp)
                res.append((a1, a2))
            # cohesive zone: ore layers soften -> gas through coke slits
            soft = min(1.0, max(0.0, (Ts[j] - Tsoft) / max(Tmelt - Tsoft, 1.0))) if fo > 0.02 else 0.0
            if fo > 0.02:
                # series resistance of layers (fractions by volume), softening raises ore resistance
                ore_mult = 1 + 30 * soft ** 2
                a1 = fo * res[0][0] * ore_mult + fc * res[1][0]
                a2 = fo * res[0][1] * ore_mult + fc * res[1][1]
                if soft > 0:
                    # parallel bypass through coke slits limits the resistance
                    a1 = min(a1, res[1][0] * (1 + 3.0 * fo))
                    a2 = min(a2, res[1][1] * (1 + 3.0 * fo))
            else:
                # dripping zone: coke bed with liquid hold-up
                eps = p["eps_coke"] - (p["holdup"] if self.liq_present[j] else 0.0) if hasattr(self, "liq_present") else p["eps_coke"]
                dp = d_coke * 0.75
                a1 = 150 * (1 - eps) ** 2 / (eps ** 3 * dp * dp)
                a2 = 1.75 * (1 - eps) / (eps ** 3 * dp)
            out.append((a1, a2))
        return out

    # ------------------------------------------------------------------
    # time step
    # ------------------------------------------------------------------
    def step(self, dt: float, steady: bool = False):
        p = self.p
        g = self.geom
        N = self.N
        ops = self.ops
        if self.P_nom is None:
            self.P_nom = self._estimate_production()
        pci_kg_s = ops["pci"] * self.P_nom / 1000.0
        E0, H0 = self.inventory()
        self._dt_last_step = dt
        # ---------- raceway (coke from bottom cell)
        nb = self.n[0]
        ash_per_C = nb[IS["ASH"]] / max(nb[IS["C"]], 1e-9)
        S_per_C = nb[IS["S"]] / max(nb[IS["C"]], 1e-9)
        rw = self.raceway(ops, self.T_s[0], pci_kg_s, ash_per_C, S_per_C)
        self.rw = rw
        # ---------- liquid march (uses lagged gas temperature)
        if not hasattr(self, "liq_present"):
            self.liq_present = np.zeros(N, dtype=bool)
        dC_l = self._liquid_march(dt)
        # ---------- gas march
        dS, dHs = self._gas_march(dt, rw)
        dS[:, IS["C"]] -= dC_l
        dHs -= dC_l * TH.interp_tab(self.SOLH[[IS["C"]]], self.T_s)[0]
        dHs -= self.Q_ls
        # ---------- condensed update: reactions + heat
        self.n = self.n + dS * dt
        self.n = np.maximum(self.n, 0.0)
        self.H_s = self.H_s + dHs * dt
        self.T_s = self.T_cells(self.n, self.H_s, self.T_s)
        C_before = self.n[:, IS["C"]] - dS[:, IS["C"]] * dt
        # ---------- melting
        melt_vol = self._melting(dt)
        # ---------- volume freed by coke consumption
        dCcons = np.maximum(C_before - self.n[:, IS["C"]], 0.0)
        frac_c = np.where(C_before > 1e-12, dCcons / np.maximum(C_before, 1e-12), 0.0)
        dVc = self.V_coke * frac_c
        # ---------- bottom outflow: raceway + hearth coke demand
        C_dem = (rw["C_coke"] + self.hearth_coke_demand) * dt
        nC0 = self.n[0, IS["C"]]
        theta0 = min(0.95, C_dem / max(nC0, 1e-12))
        V0 = self.V_ore[0] + self.V_coke[0] - melt_vol[0] - dVc[0]
        F_bot0 = theta0 * max(V0, 1e-9)
        dV = melt_vol + dVc
        F = np.zeros(N + 1)          # face flows (m3 per step), F[j] = bottom face of cell j
        F[0] = F_bot0
        for j in range(N):
            F[j + 1] = F[j] + dV[j]
        self.V_ore = np.maximum(self.V_ore - melt_vol, 0.0)
        self.CRIf = self.CRIf * (1 - frac_c)
        self.V_coke = np.maximum(self.V_coke - dVc, 0.0)
        # ---------- advection (upwind)
        out_raceway = self._advect(F, dt)
        # ---------- hearth
        self._hearth_step(dt, rw, out_raceway, steady)
        # ---------- top gas & dust
        self._accumulate(dt, rw, pci_kg_s, E0, H0)
        if not steady:
            # coal-injection and cast-rate set-points follow the measured production
            # (PCI is dosed in kg/tHM on a 2-h filtered production signal, as in plant practice)
            Pk = float(self.liq_out["m"] @ TH.MW_METAL)
            if Pk > 0.2 * (self.P_nom or Pk):
                self.P_nom += min(1.0, dt / (2.0 * 3600.0)) * (Pk - self.P_nom)
        self.t += dt
        return rw

    def _melt_plan(self, j, delta):
        """Removal plan for melting a fraction `delta` of the ore-class material of cell j."""
        p = self.p
        Ts = self.T_s[j]
        sp = self.n[j, ORE_IDX] * delta
        hS = TH.interp_tab(self.SOLH[ORE_IDX], [Ts])[:, 0]
        H_removed = float(np.dot(sp, hS))
        spd = dict(zip(ORE_SPECIES, sp))
        nC_cell = self.n[j, IS["C"]]
        hC = TH.h_cond_species("C", Ts)
        take = min(spd["Fe2O3"] + spd["Fe3O4"], 0.5 * nC_cell)
        feo = spd["FeO"] + 2 * spd["Fe2O3"] + 3 * spd["Fe3O4"]
        mFe = spd["Fe"]
        c0 = min(p["c_initial"] / 100.0 * mFe * 55.845 / 12.011, 0.3 * nC_cell)
        H_removed += (take + c0) * hC
        co2 = spd["CaCO3"] + spd["MgCO3"]
        h2o = spd["H2Ol"]
        m = np.array([mFe, c0, 0.0, 0.0, 0.0, spd["Sb"]])
        s = np.zeros(len(TH.SLAG))
        s[TH.LI["SiO2"]] = spd["SiO2"]
        s[TH.LI["Al2O3"]] = spd["Al2O3"]
        s[TH.LI["CaO"]] = spd["CaO"] + spd["CaCO3"]
        s[TH.LI["MgO"]] = spd["MgO"] + spd["MgCO3"]
        s[TH.LI["MnO"]] = spd["MnO"]
        s[TH.LI["FeO"]] = feo
        s[TH.LI["TiO2"]] = spd["TiO2"]
        s[TH.LI["P2O5"]] = spd["P2O5"]
        Hgas = take * self._hg(iCO, Ts) + co2 * self._hg(iCO2, Ts) + h2o * self._hg(iH2O, Ts)
        Hl = TH.h_metal(list(m), Ts) + TH.h_slag(list(s), Ts)
        dn = np.zeros(NS)
        dn[ORE_IDX] = -sp
        dn[IS["C"]] -= take + c0
        return {"dn": dn, "dH": -(Hl + Hgas), "m": m, "s": s, "Hl": Hl,
                "gas": (take, co2, h2o), "Hgas": Hgas}

    def _melting(self, dt):
        """Implicit melting: the melted fraction is solved so that, after paying the
        latent heat, the residual unmelted fraction matches the melting curve at the
        new cell temperature (removes the explicit flip-flop)."""
        N = self.N
        p = self.p
        Tm = self.charge["T_melt"]
        band = p["melt_band"]
        Td = Tm - band
        melt_vol = np.zeros(N)

        def target(T):
            return min(1.0, max(0.0, 1.0 - (T - Td) / (2.0 * band)))

        for j in range(N):
            if self.V_ore[j] <= 1e-9 or self.T_s[j] <= Td:
                continue
            u = self.V_ore[j] / max(self.V_ore0[j], 1e-12)
            if u <= target(self.T_s[j]) + 1e-9:
                continue
            nj = self.n[j]
            Hj = self.H_s[j]
            Tj = self.T_s[j]

            def T_after(delta):
                pl = self._melt_plan(j, delta)
                n2 = np.maximum(nj + pl["dn"], 0.0)
                return float(self.T_cells(n2[None, :], np.array([Hj + pl["dH"]]), np.array([Tj]))[0]), pl

            lo, hi = 0.0, 1.0
            T1, _ = T_after(1.0)
            if u * 0.0 - target(T1) >= 0:      # everything melts
                delta = 1.0
            else:
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    Tm_, _ = T_after(mid)
                    if u * (1 - mid) - target(Tm_) > 0:
                        lo = mid
                    else:
                        hi = mid
                delta = 0.5 * (lo + hi)
            if delta <= 1e-9:
                continue
            Tn, pl = T_after(delta)
            self.n[j] = np.maximum(nj + pl["dn"], 0.0)
            self.H_s[j] = Hj + pl["dH"]
            self.T_s[j] = Tn
            take, co2, h2o = pl["gas"]
            self.pend_gas[j, iCO] += take
            self.pend_gas[j, iCO2] += co2
            self.pend_gas[j, iH2O] += h2o
            self.pend_H[j] += pl["Hgas"]
            buf = self.liq_buf[j]
            if buf is None:
                self.liq_buf[j] = {"m": pl["m"], "s": pl["s"], "H": pl["Hl"]}
            else:
                buf["m"] = buf["m"] + pl["m"]
                buf["s"] = buf["s"] + pl["s"]
                buf["H"] += pl["Hl"]
            melt_vol[j] = self.V_ore[j] * delta
            self.RIf[j] *= (1 - delta)
        return melt_vol

    def _advect(self, F, dt):
        """Upwind transport of condensed states with face flows F (m3/step)."""
        N = self.N
        Vcur = self.V_ore + self.V_coke
        theta = np.where(Vcur > 1e-9, np.minimum(F[:N] / np.maximum(Vcur, 1e-9), 0.98), 0.0)
        out_n = self.n * theta[:, None]
        out_H = self.H_s * theta
        out_Vo = self.V_ore * theta
        out_Vo0 = self.V_ore0 * theta
        out_Vc = self.V_coke * theta
        out_RI = self.RIf * theta
        out_CRI = self.CRIf * theta
        raceway_out = {"n": out_n[0].copy(), "H": float(out_H[0])}
        self.n -= out_n
        self.H_s -= out_H
        self.V_ore -= out_Vo
        self.V_ore0 -= out_Vo0
        self.V_coke -= out_Vc
        self.RIf -= out_RI
        self.CRIf -= out_CRI
        # inflow from the cell above
        self.n[:-1] += out_n[1:]
        self.H_s[:-1] += out_H[1:]
        self.V_ore[:-1] += out_Vo[1:]
        self.V_ore0[:-1] += out_Vo0[1:]
        self.V_coke[:-1] += out_Vc[1:]
        self.RIf[:-1] += out_RI[1:]
        self.CRIf[:-1] += out_CRI[1:]
        # charge into top cell
        Vch = F[N]
        ch = self.charge
        self.n[-1] += ch["n_per_m3"] * Vch
        self.H_s[-1] += ch["H_per_m3"] * Vch
        self.V_ore[-1] += ch["fore"] * Vch
        self.V_ore0[-1] += ch["fore"] * Vch
        self.V_coke[-1] += ch["fcoke"] * Vch
        self.RIf[-1] += ch["fore"] * Vch * ch["RI"] * self.ops.get("ore_reducibility", 1.0)
        self.CRIf[-1] += ch["fcoke"] * Vch * self.ops.get("coke_reactivity", 1.0)
        self.charged_m3 = Vch
        self.T_s = self.T_cells(self.n, self.H_s, self.T_s)
        return raceway_out

    # ------------------------------------------------------------------
    # hearth
    # ------------------------------------------------------------------
    def _hearth_step(self, dt, rw, out, steady):
        p = self.p
        lo = self.liq_out
        # liquids from dripping zone
        m_in = lo["m"] * dt
        s_in = lo["s"] * dt
        H_in = lo["H"] * dt
        # raceway condensed products: ash slag, Fe from ash, S
        for k, v in rw["slag_ox"].items():
            if k in TH.LI:
                s_in[TH.LI[k]] += v * dt
        m_in[0] += rw["nFe"] * dt
        m_in[5] += rw["S_liq"] * dt
        H_in += rw["H_cond"] * dt
        # bottom-cell out-flow: raceway took exactly (C_coke, ash, S_coke); residual -> hearth
        on = out["n"].copy()
        on[IS["C"]] -= rw["C_coke"] * dt
        on[IS["ASH"]] -= rw["ash"] * dt
        on[IS["S"]] -= rw["S_coke"] * dt
        H_res = out["H"] - rw["H_coke"] * dt
        Ts0 = self.T_h
        self.M_dm += on[IS["C"]]
        ash_r = on[IS["ASH"]]
        xFe2O3 = 0.0
        for k, w in self.ash_w.items():
            amt = ash_r * w / TH.MW[k]
            if k == "Fe2O3":
                s_in[TH.LI["FeO"]] += 2 * amt
                xFe2O3 += amt
            else:
                s_in[TH.LI[k]] += amt
        m_in[5] += on[IS["S"]] + on[IS["Sb"]]
        m_in[0] += on[IS["Fe"]]
        s_in[TH.LI["FeO"]] += on[IS["FeO"]] + 2 * on[IS["Fe2O3"]] + 3 * on[IS["Fe3O4"]]
        for k in ["SiO2", "Al2O3", "MnO", "TiO2", "P2O5"]:
            s_in[TH.LI[k]] += on[IS[k]]
        s_in[TH.LI["CaO"]] += on[IS["CaO"]] + on[IS["CaCO3"]]
        s_in[TH.LI["MgO"]] += on[IS["MgO"]] + on[IS["MgCO3"]]
        exO = on[IS["Fe2O3"]] + on[IS["Fe3O4"]] + xFe2O3
        co2 = on[IS["CaCO3"]] + on[IS["MgCO3"]]
        h2o = on[IS["H2Ol"]]
        self._raw_to_hearth = float(on[ORE_IDX].clip(min=0).sum())
        H_in += H_res
        self.hm += m_in
        self.sl += s_in
        self.H_h += H_in
        M0 = self.M_dm
        if exO != 0.0:
            self._co_from_hearth(exO, Ts0)
        if co2 > 0 or h2o > 0:
            Hg = co2 * self._hg(iCO2, Ts0) + h2o * self._hg(iH2O, Ts0)
            self.pend_gas[0, iCO2] += co2
            self.pend_gas[0, iH2O] += h2o
            self.pend_H[0] += Hg
            self.H_h -= Hg
        # ---- slag-metal reactions (kinetic approach to equilibrium)
        T = self._hearth_T(self.H_h, self.T_h)
        hm_mass = float(self.hm @ TH.MW_METAL)
        sl_mass = float(self.sl @ TH.MW_SLAG)
        demand = 0.0
        if hm_mass > 1.0 and sl_mass > 1.0:
            wm = self.hm * TH.MW_METAL / hm_mass * 100
            ws = {s: self.sl[i] * TH.MW_SLAG[i] / sl_mass * 100 for i, s in enumerate(TH.SLAG)}
            wsl = {k: ws[k] for k in ["SiO2", "Al2O3", "CaO", "MgO", "MnO", "FeO", "TiO2"]}
            # sulphur partition
            LS = SL.sulphur_partition(wsl, T, C=wm[1], Si=wm[2])
            S_tot = self.hm[5] + self.sl[TH.LI["CaS"]]
            S_m_eq = S_tot / (1 + LS * sl_mass / hm_mass)
            ks = p["k_S"] * dt
            x = (self.hm[5] - S_m_eq) * (1 - math.exp(-ks))
            # [S] + (CaO) + C -> (CaS) + CO : CO to pending gas
            x = min(x, self.hm[5], self.sl[TH.LI["CaO"]] * 0.5) if x > 0 else max(x, -self.sl[TH.LI["CaS"]])
            hTx = self._hearth_reaction_h(T)
            self.hm[5] -= x
            self.sl[TH.LI["CaS"]] += x
            self.sl[TH.LI["CaO"]] -= x
            self._co_from_hearth(x, T)
            # desiliconisation by slag FeO / re-oxidation
            kd = p["k_desi"] * math.exp(-p["E_desi"] / RJ * (1 / T - 1 / 1773.0)) * dt
            feo = max(ws["FeO"], 0.05)
            xs = self.hm[2] * (1 - math.exp(-kd * feo / 0.5))
            xs = min(xs, self.sl[TH.LI["FeO"]] / 2 + self.M_dm * 0.0 + self.hm[2])
            # [Si] + 2(FeO) -> (SiO2) + 2Fe, FeO limited -> rest by CO (C-CO oxygen)
            xf = min(xs, self.sl[TH.LI["FeO"]] / 2.0)
            self.hm[2] -= xs
            self.sl[TH.LI["SiO2"]] += xs
            self.sl[TH.LI["FeO"]] -= 2 * xf
            self.hm[0] += 2 * xf
            # remaining oxygen from CO dissociation: (xs-xf)*2 O -> 2CO consumed -> 2C deposited to deadman
            xo = xs - xf
            if xo > 0:
                self._co_from_hearth(-2 * xo, T)
            # Mn partition
            LMn = 10 ** (0.15 - 0.85 * (ws["CaO"] / max(ws["SiO2"], 1e-6) - 1.0) - 0.0025 * (T - 1773.0))
            Mn_tot = self.hm[3] + self.sl[TH.LI["MnO"]]
            Mn_m_eq = Mn_tot / (1 + LMn * (sl_mass / hm_mass) * TH.MW["Mn"] / TH.MW["MnO"])
            xm = (Mn_m_eq - self.hm[3]) * (1 - math.exp(-4e-4 * dt))
            xm = min(xm, self.sl[TH.LI["MnO"]]) if xm > 0 else max(xm, -self.hm[3])
            self.hm[3] += xm
            self.sl[TH.LI["MnO"]] -= xm
            self._co_from_hearth(xm, T)
            demand += max(xm, 0.0)
            # FeO reduction by deadman coke
            xf2 = self.sl[TH.LI["FeO"]] * (1 - math.exp(-p["k_hfeo"] * dt * math.exp(-160e3 / RJ * (1 / T - 1 / 1773.0))))
            self.sl[TH.LI["FeO"]] -= xf2
            self.hm[0] += xf2
            self._co_from_hearth(xf2, T)
            demand += xf2
            # carburisation (dissolution of deadman coke)
            Csat = SL.carbon_saturation(T - 273.15, Si=wm[2], Mn=wm[3], P=wm[4], S=wm[5])
            if Csat > wm[1]:
                dw = (Csat - wm[1]) * (1 - math.exp(-p["k_hcarb"] * dt))
                xc = dw / 100.0 * hm_mass / 12.011
                self.hm[1] += xc
                self.M_dm -= xc
                demand += xc
                self.H_h += 0.0
            self.LS_eq = LS
        self.hearth_coke_demand = max(0.0, M0 - self.M_dm) / dt
        # ---- heat loss
        Qh = self.wall.hearth_loss(T)
        self.H_h -= Qh * dt
        self.Q_hearth = Qh
        # ---- casting
        tapped = self._casting(dt, steady)
        self.T_h = self._hearth_T(self.H_h, T)
        self.tapped = tapped

    def _hearth_reaction_h(self, T):
        return 0.0

    def _co_from_hearth(self, x, T):
        """x kmol C removed from deadman to CO (x<0 means CO -> C deposition)."""
        # CO leaves via pending gas of cell 0 at hearth temperature; deadman C consumed
        hCO = self._hg(iCO, T)
        hC = TH.h_cond_species("C", T)
        self.M_dm -= x
        self.pend_gas[0, iCO] += x
        self.pend_H[0] += x * hCO
        self.H_h -= x * hCO
        # oxygen comes from the condensed phase composition change; carbon enthalpy is part of H_h

    def _casting(self, dt, steady):
        p = self.p
        out = {"m": np.zeros(6), "s": np.zeros(len(TH.SLAG)), "H": 0.0}
        if steady:
            # continuous tapping at the nominal inventory
            tgt_hm = getattr(self, "_hm_target", None)
            if tgt_hm is None:
                self._hm_target = self.hm.sum()
                self._sl_target = self.sl.sum()
                self._hm_target_mass = float(self.hm @ TH.MW_METAL)
                tgt_hm = self._hm_target
            fm = max(0.0, 1 - self._hm_target / max(self.hm.sum(), 1e-9))
            fs = max(0.0, 1 - self._sl_target / max(self.sl.sum(), 1e-9))
        else:
            c = self.cast
            t_min = self.t / 60.0
            if not c["open"] and t_min >= c["t_next"]:
                c["open"] = True
                c["t_close"] = t_min + p["cast_duration_min"]
                c["hole"] = (c["hole"] % self.geom.n_taphole) + 1
                c["count"] += 1
                c["sampled"] = False
                c["t_open"] = t_min
            fm = fs = 0.0
            if c["open"]:
                prod = max(self.P_nom, 1.0)  # kg/s
                rate_m = p["tap_rate_factor"] * prod                      # kg/s HM
                hm_mass = float(self.hm @ TH.MW_METAL)
                sl_mass = float(self.sl @ TH.MW_SLAG)
                # casts drain the hearth to ~80 % of the nominal working inventory (taphole
                # length / sump practice), keeping the cycle-average close to the steady state
                hm_min = max(self._hm_floor(), 0.80 * getattr(self, "_hm_target_mass", 0.0))
                fm = min(0.9, rate_m * dt / max(hm_mass, 1.0)) if hm_mass > hm_min else 0.0
                fs = min(0.9, rate_m * 0.32 * dt / max(sl_mass, 1.0)) if sl_mass > 0.25 * self._sl_floor() else 0.0
                if not c.get("sampled") and t_min >= c["t_open"] + 0.5 * p["cast_duration_min"]:
                    c["sampled"] = True
                    self._take_sample(c)
                if t_min >= c["t_close"] or hm_mass <= hm_min * 1.01:
                    c["open"] = False
                    c["t_next"] = t_min + p["cast_gap_min"]
        if fm > 0 or fs > 0:
            T = self.T_h
            Hm = TH.h_metal(list(self.hm * fm), T)
            Hs = TH.h_slag(list(self.sl * fs), T)
            out["m"] = self.hm * fm
            out["s"] = self.sl * fs
            out["H"] = Hm + Hs
            self.hm = self.hm * (1 - fm)
            self.sl = self.sl * (1 - fs)
            self.H_h -= Hm + Hs
        return out

    def _hm_floor(self):
        g = self.geom
        return g.A_hearth * self.p["sump_active"] * g.h_sump * self.p["deadman_void"] * self.p["rho_hm"]

    def _sl_floor(self):
        g = self.geom
        return g.A_hearth * 0.15 * g.h_hearth * self.p["deadman_void"] * self.p["rho_slag"]

    def _take_sample(self, c):
        hm_mass = float(self.hm @ TH.MW_METAL)
        sl_mass = float(self.sl @ TH.MW_SLAG)
        wm = self.hm * TH.MW_METAL / max(hm_mass, 1e-9) * 100
        ws = self.sl * TH.MW_SLAG / max(sl_mass, 1e-9) * 100
        rec = {"t_h": round(self.t / 3600.0, 4), "cast": c["count"], "taphole": c["hole"],
               "HMT_C": round(self.T_h - 273.15, 1), "Si": round(wm[2], 4), "S": round(wm[5], 4),
               "C": round(wm[1], 3), "Mn": round(wm[3], 3), "P": round(wm[4], 4),
               "slag_B2": round(ws[TH.LI["CaO"]] / max(ws[TH.LI["SiO2"]], 1e-9), 3),
               "slag_FeO": round(ws[TH.LI["FeO"]], 3), "slag_Al2O3": round(ws[TH.LI["Al2O3"]], 2),
               "t_avail_h": round(self.t / 3600.0 + self.p["lab_delay_min"] / 60.0, 4)}
        self.casts.append(rec)

    # ------------------------------------------------------------------
    def _estimate_production(self):
        """Production (kg HM/s) estimate from blast oxygen and the recipe."""
        ops = self.ops
        V = ops["blast_volume"] / 60.0 / TH.NM3_PER_KMOL
        Oat = 2 * V * (21 + ops["o2_enrich"]) / 100 + ops["moisture"] * ops["blast_volume"] / 60.0 / 1000 / 18.015
        # carbon gasified at tuyeres per tHM ~ fuel - solution loss - HM C
        fuel = self.recipe["coke"] * 0.86 + ops["pci"] * 0.78 * 0.95
        c_tuy = max(150.0, fuel - 46 - 95) / 12.011        # kmol/tHM
        return Oat / c_tuy * 1000.0

    def _accumulate(self, dt, rw, pci_kg_s, E0, H0):
        """Element/enthalpy accounting for this step + top gas/dust records."""
        acc = self.acc
        ops = self.ops
        g = self.geom
        Vch = self.charged_m3
        ch = self.charge
        # inputs
        E_in = (ch["n_per_m3"] + ch["dust_per_m3"]) @ self.ES * Vch
        H_in = (ch["H_per_m3"] + ch["Hdust_per_m3"]) * Vch
        bl = np.zeros(NE)
        bl[EI["O"]] = (2 * rw["nO2"] + rw["nH2Ob"]) * dt
        bl[EI["N"]] = 2 * rw["nN2"] * dt
        bl[EI["H"]] = 2 * rw["nH2Ob"] * dt
        E_in += bl
        H_in += rw["H_blast"] * dt
        dry = rw["pci_dry"]
        c = self.coal
        pc = np.zeros(NE)
        pc[EI["C"]] = dry * c["C"] / 12.011
        pc[EI["H"]] = dry * c["H"] / 1.008 + 2 * pci_kg_s * self.coal_moist / 18.015
        pc[EI["O"]] = dry * c["O"] / 15.999 + pci_kg_s * self.coal_moist / 18.015
        pc[EI["N"]] = dry * c["N"] / 14.007 + 2 * pci_kg_s * self.p["pci_carrier_nm3_per_kg"] / TH.NM3_PER_KMOL
        pc[EI["S"]] = dry * c["S"] / 32.06
        for k, w in self.pci_ash_w.items():
            pc += dry * c["ash"] * w / TH.MW[k] * _elem_vec(_FORMULA[k])
        E_in += pc * dt
        H_in += rw["H_coal"] * dt
        # outputs: top gas, dust, tapped liquids, losses
        gas = self.top["gas"]
        E_out = np.array(gas) @ self.EG * dt
        H_out = self.top["H"] * dt
        E_out += ch["dust_per_m3"] @ self.ES * Vch
        H_out += ch["Hdust_per_m3"] * Vch
        tp = self.tapped
        E_out += tp["m"] @ self.EM + tp["s"] @ self.EL
        H_out += tp["H"]
        Q_loss = (float(self.ext["Qw"].sum()) + self.Q_hearth + self.wall.tuyere_loss(rw["T_raft"])) * dt
        E1, H1 = self.inventory()
        resid_E = (E1 - E0) - (E_in - E_out)
        resid_H = (H1 - H0) - (H_in - H_out - Q_loss)
        acc["in_E"] += E_in
        acc["out_E"] += E_out
        acc["in_H"] += H_in
        acc["out_H"] += H_out
        acc["loss_H"] += Q_loss
        acc["time"] += dt
        acc["hm_kg"] += float(tp["m"] @ TH.MW_METAL)
        acc["slag_kg"] += float(tp["s"] @ TH.MW_SLAG)
        self.last = {"resid_E": resid_E, "resid_H": resid_H, "E_in": E_in, "H_in": H_in, "Q_loss": Q_loss,
                     "E_scale": np.maximum(np.abs(E_in), 1e-9)}


# ----------------------------------------------------------------------
# steady-state helpers
# ----------------------------------------------------------------------
def rescale_coke(bf: "BlastFurnace", c_new: float):
    """Change coke rate and re-proportion the lumpy/cohesive column (pseudo-transient
    acceleration used only by the steady-state solver)."""
    c_old = bf.recipe["coke"]
    f = c_new / c_old
    bf.ops["coke_rate"] = c_new
    bf.build_recipe(c_new)
    for j in range(bf.N):
        Vo, Vc = bf.V_ore[j], bf.V_coke[j]
        if Vo <= 1e-6:
            continue
        a = (Vo + Vc) / (Vo + f * Vc)
        bf.n[j, ORE_IDX] *= a
        bf.n[j, COKE_IDX] *= f * a
        bf.V_ore[j] *= a
        bf.V_ore0[j] *= a
        bf.V_coke[j] *= f * a
        bf.RIf[j] *= a
        bf.CRIf[j] *= f * a
    bf.H_s = bf.h_cells(bf.n, bf.T_s)


def courant_dt(bf: "BlastFurnace", target=0.6, dt_max=900.0, dt_min=60.0):
    """Time step so that the fastest cell turnover stays below `target`."""
    vol_rate = bf.charge["m3_per_tHM"] * (bf.P_nom or bf._estimate_production()) / 1000.0  # m3/s at top
    tau = bf.geom.V_cell / max(vol_rate, 1e-9)
    return float(np.clip(target * tau.min(), dt_min, dt_max))
