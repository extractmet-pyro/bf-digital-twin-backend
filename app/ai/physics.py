"""
Physics layer for the AI surrogates.

* Decision variables and their bounds (scaled to the furnace preset).
* Extraction of per-tHM stream flows from a converged first-principles state.
* Linear balance operators  r(x, y) = A(x) y + b(x)  for the carbon, oxygen,
  iron and enthalpy balances.  They are used as the physics-informed penalty
  in surrogate training and to report balance violations of surrogate
  predictions.
"""
from __future__ import annotations

import numpy as np

from ..core import thermo as TH
from ..core.furnace import IS, iCO, iCO2, iH2, iH2O, iN2

# decision variables of the NeuroMet abstract + coke rate (thermal closure variable)
DEC = ["blast_volume", "blast_temp", "o2_enrich", "pci", "moisture", "pellet_frac"]
X_NAMES = DEC + ["coke_rate"]
X_UNITS = ["Nm3/min", "C", "%-pt", "kg/tHM", "g/Nm3", "%", "kg/tHM"]

# outputs used by the balance operators (per tHM)
Y_BAL = ["ore_kg", "limestone_kg", "dolomite_kg", "quartzite_kg", "blast_nm3",
         "tg_CO", "tg_CO2", "tg_H2", "tg_H2O", "tg_N2",
         "dust_C", "dust_Fe", "dust_O", "dust_H",
         "hm_Fe", "hm_C", "hm_Si", "hm_Mn", "hm_P", "hm_S",
         "sl_SiO2", "sl_Al2O3", "sl_CaO", "sl_MgO", "sl_MnO", "sl_FeO", "sl_TiO2", "sl_P2O5", "sl_CaS",
         "tg_sens", "hm_sens", "sl_sens", "loss_MJ"]
# key performance outputs (not in balances)
Y_KPI = ["hmt", "ttop", "raft", "dP", "prod_tpd", "eta_co", "si_pct", "s_pct", "c_pct", "slag_rate", "slag_B2",
         "slag_Al2O3", "slag_MgO", "co2", "cost", "heat_loss_MW", "sol_loss", "burnout", "cz_top", "cz_bot",
         "tg_lhv", "o2_nm3", "tg_nm3", "fuel_rate", "trz_T"]
Y_NAMES = Y_BAL + Y_KPI
YB = {k: i for i, k in enumerate(Y_BAL)}


def bounds_for(preset: dict) -> dict:
    """Decision-variable bounds (6 vars) for a furnace preset."""
    Vb = preset["blast_volume"]
    return {
        "blast_volume": (0.87 * Vb, 1.12 * Vb), "blast_temp": (1050.0, 1250.0), "o2_enrich": (1.0, 8.0),
        "pci": (100.0, 230.0), "moisture": (5.0, 35.0), "pellet_frac": (0.0, 55.0),
    }


def per_tHM_flows(bf) -> dict:
    """Per-tHM flows from the last steady step (in = out at convergence)."""
    dt = bf._dt_last_step
    tp = bf.tapped
    P = float(tp["m"] @ TH.MW_METAL) / dt            # kg/s tapped hot metal
    if P <= 1e-6:
        P = float(bf.liq_out["m"] @ TH.MW_METAL)
    k = 1000.0 / P                                      # s/tHM
    Vdot = bf.charged_m3 / dt
    ch = bf.charge
    r = bf.recipe
    ore_recipe = r["sinter"] + r["pellet"] + r["lump"]
    ore_kg = ch["ore_kg_per_m3"] * Vdot * k
    scale = ore_kg / max(ore_recipe, 1e-9)
    rw = bf.rw
    gas = bf.top["gas"]
    out = {
        "ore_kg": ore_kg, "limestone_kg": r["limestone"] * scale, "dolomite_kg": r["dolomite"] * scale,
        "quartzite_kg": r["quartzite"] * scale,
        "blast_nm3": bf.ops["blast_volume"] / 60.0 * k,
        "tg_CO": gas[iCO] * k, "tg_CO2": gas[iCO2] * k, "tg_H2": gas[iH2] * k, "tg_H2O": gas[iH2O] * k,
        "tg_N2": gas[iN2] * k,
    }
    dust = ch["dust_per_m3"] * Vdot * k
    Ttop = bf.top["T"]
    out["dust_C"] = (dust[IS["C"]] + gas[6] * k) * 12.011
    fe = 2 * dust[IS["Fe2O3"]] + dust[IS["FeO"]] + 3 * dust[IS["Fe3O4"]] + dust[IS["Fe"]]
    ash = dust[IS["ASH"]]
    fe += ash * bf.ash_w["Fe2O3"] / TH.MW["Fe2O3"] * 2
    out["dust_Fe"] = fe * 55.845
    O = (3 * dust[IS["Fe2O3"]] + dust[IS["FeO"]] + 4 * dust[IS["Fe3O4"]] + 2 * dust[IS["SiO2"]] + 3 * dust[IS["Al2O3"]]
         + dust[IS["CaO"]] + dust[IS["MgO"]] + dust[IS["MnO"]] + 2 * dust[IS["TiO2"]] + 5 * dust[IS["P2O5"]]
         + 3 * dust[IS["CaCO3"]] + 3 * dust[IS["MgCO3"]] + dust[IS["H2Ol"]])
    O += ash * sum(w / TH.MW[s] * {"SiO2": 2, "Al2O3": 3, "Fe2O3": 3, "CaO": 1, "MgO": 1, "TiO2": 2, "P2O5": 5, "MnO": 1}[s]
                   for s, w in bf.ash_w.items())
    out["dust_O"] = O * 15.999
    Hd = float(dust @ TH.interp_tab(bf.SOLH, [bf.p["T_charge"]])[:, 0]) + gas[6] * k * TH.h_cond_species("C", Ttop)
    out["dust_H"] = Hd
    hm = tp["m"] / dt * k            # kmol/tHM
    for i, e in enumerate(["Fe", "C", "Si", "Mn", "P", "S"]):
        out["hm_" + e] = hm[i] * TH.MW_METAL[i]
    sl = tp["s"] / dt * k
    for i, s in enumerate(TH.SLAG):
        out["sl_" + s] = sl[i] * TH.MW_SLAG[i]
    T0 = TH.T0
    out["tg_sens"] = sum(gas[i] * (bf._hg(i, Ttop) - bf._hg(i, T0)) for i in range(6)) * k + gas[7] * k * 0.0
    Th = bf.T_h
    hf_m = np.array([0.0, 22.6, -81.3, 16.2, -50.0, -70.8])
    out["hm_sens"] = TH.h_metal(list(hm), Th) - float(hm @ hf_m)
    hf_s = np.array([TH.COND[s][0] for s in TH.SLAG])
    out["sl_sens"] = TH.h_slag(list(sl), Th) - float(sl @ hf_s)
    out["loss_MJ"] = (float(bf.ext["Qw"].sum()) + bf.Q_hearth + bf.wall.tuyere_loss(rw["T_raft"])) * k
    out["_Sg"] = gas[7] * k * 32.06
    out["_P"] = P
    return out


class BalanceOperator:
    """r = A(x) y + b(x) for [C, O, Fe, H] balances (kmol C, kmol O, kmol Fe, MJ per tHM)."""

    def __init__(self, inp: dict):
        self.inp = inp
        mat = inp["materials"]
        self.mat = mat
        ck = mat["coke"]
        cs = ck["C"] + ck["ash"] + ck["S"]
        self.coke_C = ck["C"] / cs / 12.011                    # kmol C per kg coke
        aw = ck["ash_comp"]
        at = sum(aw.values())
        self.coke_ash = ck["ash"] / cs
        self.coke_ash_w = {k: v / at for k, v in aw.items()}
        pc = mat["pci"]
        tot = sum(pc[k] for k in ["C", "H", "O", "N", "S", "ash"])
        self.pci = {k: pc[k] / tot for k in ["C", "H", "O", "N", "S", "ash"]}
        self.pci_moist = pc["moisture"] / 100.0
        paw = pc["ash_comp"]
        pt = sum(paw.values())
        self.pci_ash_w = {k: v / pt for k, v in paw.items()}
        self.ox_O = {"SiO2": 2, "Al2O3": 3, "Fe2O3": 3, "CaO": 1, "MgO": 1, "TiO2": 2, "P2O5": 5, "MnO": 1, "FeO": 1}
        # enthalpy of charge items at T_charge (MJ/kg)
        self.Tch = 303.15

    def _blend(self, pellet_frac, lump_frac=10.0):
        pf = pellet_frac / 100.0
        lf = lump_frac / 100.0
        sf = 1 - pf - lf
        bur = self.mat["burden"]
        comp = {}
        moist = 0.0
        for b, f in (("sinter", sf), ("pellet", pf), ("lump", lf)):
            c = bur[b]["comp"]
            t = sum(c.values())
            for k, v in c.items():
                comp[k] = comp.get(k, 0.0) + f * v / t
            moist += f * bur[b]["moisture"] / (100 - bur[b]["moisture"])
        return comp, moist

    def _h_kg(self, comp_frac: dict, T):
        h = 0.0
        for k, w in comp_frac.items():
            key = "S" if k == "S" else k
            h += w / TH.MW[key] * TH.h_cond_species(key, T)
        return h

    def build(self, x: np.ndarray):
        """x: (n,7) array [Vb, Tb, O2, PCI, moisture, pellet, coke]. Returns A (n,4,ny), b (n,4)."""
        n = x.shape[0]
        ny = len(Y_BAL)
        A = np.zeros((n, 4, ny))
        b = np.zeros((n, 4))
        mat = self.mat
        fl = mat["flux"]
        T0 = TH.T0
        # constant columns (outputs)
        for nm, (c, o, fe) in {
            "tg_CO": (1, 1, 0), "tg_CO2": (1, 2, 0), "tg_H2O": (0, 1, 0)}.items():
            A[:, 0, YB[nm]] -= c
            A[:, 1, YB[nm]] -= o
        A[:, 0, YB["dust_C"]] -= 1 / 12.011
        A[:, 0, YB["hm_C"]] -= 1 / 12.011
        A[:, 2, YB["hm_Fe"]] -= 1 / 55.845
        A[:, 2, YB["sl_FeO"]] -= 1 / 71.844
        A[:, 2, YB["dust_Fe"]] -= 1 / 55.845
        A[:, 1, YB["dust_O"]] -= 1 / 15.999
        for s in ["SiO2", "Al2O3", "CaO", "MgO", "MnO", "FeO", "TiO2", "P2O5"]:
            A[:, 1, YB["sl_" + s]] -= self.ox_O[s] / TH.MW[s]
        # enthalpy outputs: formation parts + sensible outputs
        hf = {g: TH.h_gas_exact(g, T0) for g in ["CO", "CO2", "H2", "H2O", "N2"]}
        for g in ["CO", "CO2", "H2", "H2O", "N2"]:
            A[:, 3, YB["tg_" + g]] -= hf[g]
        A[:, 3, YB["tg_sens"]] -= 1
        A[:, 3, YB["hm_sens"]] -= 1
        A[:, 3, YB["sl_sens"]] -= 1
        A[:, 3, YB["loss_MJ"]] -= 1
        A[:, 3, YB["dust_H"]] -= 1
        hfm = {"hm_C": 22.6 / 12.011, "hm_Si": -81.3 / 28.086, "hm_Mn": 16.2 / 54.938, "hm_P": -50.0 / 30.974,
               "hm_S": -70.8 / 32.06}
        for kk, v in hfm.items():
            A[:, 3, YB[kk]] -= v
        for s in TH.SLAG:
            A[:, 3, YB["sl_" + s]] -= TH.COND[s][0] / TH.MW[s]
        # flux columns (inputs scale with flux outputs)
        for f in ["limestone", "dolomite", "quartzite"]:
            c = fl[f]["comp"]
            C = c.get("CaCO3", 0) / 100 / 100.09 + c.get("MgCO3", 0) / 100 / 84.313
            O = sum(c.get(k, 0) / 100 / TH.MW[k] * {"CaCO3": 3, "MgCO3": 3, "SiO2": 2, "Al2O3": 3, "Fe2O3": 3, "CaO": 1, "MgO": 1}.get(k, 0) for k in c)
            O += fl[f]["moisture"] / 100 / 18.015
            Fe = c.get("Fe2O3", 0) / 100 / TH.MW["Fe2O3"] * 2
            H = sum(c.get(k, 0) / 100 / TH.MW[k] * TH.h_cond_species(k, self.Tch) for k in c) + \
                fl[f]["moisture"] / 100 / 18.015 * TH.h_cond_species("H2Ol", self.Tch)
            A[:, 0, YB[f + "_kg"]] += C
            A[:, 1, YB[f + "_kg"]] += O
            A[:, 2, YB[f + "_kg"]] += Fe
            A[:, 3, YB[f + "_kg"]] += H
        # ore (blend depends on pellet fraction)
        for i in range(n):
            comp, moist = self._blend(x[i, 5])
            Fe = comp.get("Fe2O3", 0) / TH.MW["Fe2O3"] * 2 + comp.get("FeO", 0) / TH.MW["FeO"]
            O = sum(comp.get(k, 0) / TH.MW[k] * self.ox_O.get(k, 0) for k in comp if k != "S") + moist / 18.015
            H = self._h_kg({k: v for k, v in comp.items()}, self.Tch) + moist / 18.015 * TH.h_cond_species("H2Ol", self.Tch)
            A[i, 1, YB["ore_kg"]] += O
            A[i, 2, YB["ore_kg"]] += Fe
            A[i, 3, YB["ore_kg"]] += H
            # blast per Nm3: O and enthalpy
            Tb = x[i, 1] + 273.15
            yO2 = (21.0 + x[i, 2]) / 100.0
            kmol = 1.0 / TH.NM3_PER_KMOL
            w = x[i, 4] / 1000.0 / 18.015
            A[i, 1, YB["blast_nm3"]] += 2 * yO2 * kmol + w
            A[i, 3, YB["blast_nm3"]] += (yO2 * kmol * TH.h_gas_exact("O2", Tb) + (1 - yO2) * kmol * TH.h_gas_exact("N2", Tb)
                                         + w * TH.h_gas_exact("H2O", Tb))
        # coke & PCI (inputs only -> b)
        coke = x[:, 6]
        pci = x[:, 3]
        dry = pci * (1 - self.pci_moist)
        ck = mat["coke"]
        b[:, 0] += coke * self.coke_C + dry * self.pci["C"] / 12.011
        ash_O = sum(w / TH.MW[k] * self.ox_O[k] for k, w in self.coke_ash_w.items())
        ash_Fe = self.coke_ash_w.get("Fe2O3", 0) / TH.MW["Fe2O3"] * 2
        pash_O = sum(w / TH.MW[k] * self.ox_O[k] for k, w in self.pci_ash_w.items())
        pash_Fe = self.pci_ash_w.get("Fe2O3", 0) / TH.MW["Fe2O3"] * 2
        cm = ck["moisture"] / (100 - ck["moisture"])
        b[:, 1] += coke * self.coke_ash * ash_O + coke * cm / 18.015 + dry * (self.pci["O"] / 15.999 + self.pci["ash"] * pash_O) \
            + pci * self.pci_moist / 18.015
        b[:, 2] += coke * self.coke_ash * ash_Fe + dry * self.pci["ash"] * pash_Fe
        h_ash = sum(w / TH.MW[k] * TH.h_cond_species(k, self.Tch) for k, w in self.coke_ash_w.items())
        hC = TH.h_cond_species("C", self.Tch) / 12.011
        hS = TH.h_cond_species("S", self.Tch) / 32.06
        cs = ck["C"] + ck["ash"] + ck["S"]
        b[:, 3] += coke * (ck["C"] / cs * hC * 12.011 / 12.011 + self.coke_ash * h_ash + ck["S"] / cs * hS) \
            + coke * cm / 18.015 * TH.h_cond_species("H2Ol", self.Tch)
        # PCI enthalpy (formation from GCV)
        from ..core.materials import pci_gcv
        gcv = pci_gcv(mat["pci"])
        c = self.pci
        h_prod = c["C"] / 12.011 * -393.52 + c["H"] / 2.016 * -285.83 + c["S"] / 32.06 * -296.81
        h_pash = sum(w / TH.MW[k] * TH.COND[k][0] for k, w in self.pci_ash_w.items())
        hf_coal = h_prod + gcv + c["ash"] * h_pash
        Tco = 333.15
        b[:, 3] += dry * (hf_coal + 1.30e-3 * (Tco - 298.15)) + pci * self.pci_moist / 18.015 * TH.h_cond_species("H2Ol", Tco) \
            + pci * 0.03 / TH.NM3_PER_KMOL * TH.h_gas_exact("N2", Tco)
        return A, b

    def residuals(self, x, Y):
        """Y: (n, ny) balance outputs. Returns (n,4) residuals and relative versions."""
        A, b = self.build(np.atleast_2d(x))
        r = np.einsum("nky,ny->nk", A, np.atleast_2d(Y)) + b
        return r
