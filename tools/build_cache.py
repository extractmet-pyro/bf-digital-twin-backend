"""
Pre-compute API caches so the Render free instance answers instantly:
  app/data/presets/<preset>.json      full steady report for every furnace size
  app/data/dynamic/<scenario>.json    24-h trajectories of the default furnace
  app/data/validation.json            validation report (windows, sensitivities, presets, balances)
Run after tools/train_all.py:  python tools/build_cache.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "app", "data")

from app.core.simulate import solve_steady, full_report, run_dynamic, SCENARIOS, kpi_block   # noqa: E402
from app.core.materials import PRESETS, DEFAULT_PRESET, default_inputs                       # noqa: E402
from app.main import _clean                                                                    # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dump(obj, *path):
    p = os.path.join(DATA, *path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(_clean(obj), fh, separators=(",", ":"))
    return p


# typical industrial windows (modern large BFs, open literature: Geerdes et al. "Modern Blast Furnace
# Ironmaking" 4th ed. 2020; Babich, Senk et al. "Ironmaking" 2016; Biswas "Principles of Blast Furnace
# Ironmaking" 1981; IEA/worldsteel benchmark data)
WINDOWS = [
    ("Coke rate", "coke_rate", "kg/tHM", 290, 400, "Geerdes 2020; worldsteel benchmarks"),
    ("PCI rate", "pci_rate", "kg/tHM", 100, 230, "Geerdes 2020"),
    ("Fuel rate (coke + PCI)", "fuel_rate", "kg/tHM", 470, 545, "Geerdes 2020; Babich 2016"),
    ("Productivity (working volume)", "productivity_work", "t/m3.d", 2.0, 2.9, "Geerdes 2020"),
    ("Blast volume", "blast_nm3_per_t", "Nm3/tHM", 850, 1150, "Babich 2016"),
    ("RAFT", "raft_C", "C", 2000, 2300, "Geerdes 2020"),
    ("Top-gas temperature", "top_gas_T_C", "C", 100, 220, "Geerdes 2020"),
    ("Top-gas CO utilisation", "eta_CO", "-", 0.45, 0.53, "Geerdes 2020; Biswas 1981"),
    ("Top-gas CO", "top_gas_CO", "vol %", 20, 26, "Babich 2016"),
    ("Top-gas CO2", "top_gas_CO2", "vol %", 19, 25, "Babich 2016"),
    ("Top-gas H2", "top_gas_H2", "vol %", 2.5, 7, "Babich 2016"),
    ("Hot-metal temperature", "hmt_C", "C", 1460, 1530, "Geerdes 2020"),
    ("Hot-metal Si", "hm_Si", "wt %", 0.25, 0.7, "Geerdes 2020"),
    ("Hot-metal S", "hm_S", "wt %", 0.015, 0.05, "Geerdes 2020"),
    ("Hot-metal C", "hm_C", "wt %", 4.3, 5.0, "Biswas 1981"),
    ("Slag rate", "slag_rate", "kg/tHM", 220, 380, "Geerdes 2020"),
    ("Slag basicity CaO/SiO2", "slag_B2", "-", 1.0, 1.2, "Geerdes 2020"),
    ("Slag Al2O3", "slag_Al2O3", "wt %", 10, 20, "Indian practice (high-alumina ores)"),
    ("Pressure drop (blast - top)", "pressure_drop_kPa", "kPa", 120, 190, "Geerdes 2020"),
    ("Bosh-gas index (bosh gas / blast)", "_bosh_ratio", "-", 1.25, 1.45, "Biswas 1981"),
    ("Wall + hearth heat loss", "heat_loss_GJ_t", "GJ/tHM", 0.15, 0.55, "Geerdes 2020"),
    ("Solution-loss carbon", "solution_loss_total_C", "kg C/tHM", 80, 130, "Rist diagram practice"),
    ("Solids residence time", "solids_residence_h", "h", 5, 9, "Geerdes 2020"),
]


def sensitivities(base_inp):
    out = []
    b0 = solve_steady(copy.deepcopy(base_inp))
    k0 = kpi_block(b0)

    def run(op):
        inp = copy.deepcopy(base_inp)
        inp.setdefault("operation", {}).update(op)
        return kpi_block(solve_steady(inp))

    o = base_inp["operation"]
    k = run({"pci": o["pci"] + 20})
    out.append({"label": "PCI replacement ratio (coke saved per kg coal)", "value": -(k["coke_rate"] - k0["coke_rate"]) / (k["pci_rate"] - k0["pci_rate"]),
                "lo": 0.75, "hi": 0.95, "unit": "kg/kg"})
    out.append({"label": "RAFT change per +10 kg/tHM PCI", "value": (k["raft_C"] - k0["raft_C"]) / 2.0, "lo": -32, "hi": -15, "unit": "C"})
    k = run({"blast_temp": o["blast_temp"] + 50})
    out.append({"label": "Coke change per +100 C blast temperature", "value": (k["coke_rate"] - k0["coke_rate"]) * 2, "lo": -25, "hi": -8, "unit": "kg/tHM"})
    out.append({"label": "RAFT change per +100 C blast temperature", "value": (k["raft_C"] - k0["raft_C"]) * 2, "lo": 55, "hi": 95, "unit": "C"})
    k = run({"moisture": o["moisture"] + 10})
    out.append({"label": "Coke change per +10 g/Nm3 blast moisture", "value": (k["coke_rate"] - k0["coke_rate"]), "lo": 4, "hi": 12, "unit": "kg/tHM"})
    out.append({"label": "RAFT change per +10 g/Nm3 blast moisture", "value": (k["raft_C"] - k0["raft_C"]), "lo": -75, "hi": -40, "unit": "C"})
    k = run({"o2_enrich": o["o2_enrich"] + 1.0})
    out.append({"label": "RAFT change per +1 %-pt O2 enrichment", "value": (k["raft_C"] - k0["raft_C"]), "lo": 35, "hi": 65, "unit": "C"})
    out.append({"label": "Productivity change per +1 %-pt O2 (constant blast)", "value": (k["production_tpd"] / k0["production_tpd"] - 1) * 100, "lo": 2, "hi": 5, "unit": "%"})
    # hot-metal temperature sensitivity to coke at fixed operation (thermal state gain)
    inp = copy.deepcopy(base_inp)
    inp.setdefault("operation", {})["coke_rate"] = b0.recipe["coke"] + 5
    k = kpi_block(solve_steady(inp, closure=False))
    out.append({"label": "HMT change per +1 kg/tHM coke (fixed operation)", "value": (k["hmt_C"] - k0["hmt_C"]) / 5.0, "lo": 1.5, "hi": 4.0, "unit": "C"})
    # RAFT sensitivities on the raceway sub-model at otherwise fixed operation (industry convention)
    from app.core.furnace import BlastFurnace
    bfr = BlastFurnace(copy.deepcopy(base_inp), ncell=36)
    P = k0["production_tph"] / 3.6
    Tc = b0.T_s[0] if hasattr(b0, "T_s") else 1923.15

    def raft(**d):
        ops = dict(bfr.ops); ops.update({kk: o[kk] for kk in ("pci", "moisture", "o2_enrich", "blast_temp", "blast_volume")}); ops.update(d)
        return bfr.raceway(ops, Tc, ops["pci"] * P / 1000.0)["T_raft"]
    r0 = raft()
    fixed = {"RAFT change per +10 kg/tHM PCI": (raft(pci=o["pci"] + 10) - r0),
             "RAFT change per +100 C blast temperature": (raft(blast_temp=o["blast_temp"] + 100) - r0),
             "RAFT change per +10 g/Nm3 blast moisture": (raft(moisture=o["moisture"] + 10) - r0),
             "RAFT change per +1 %-pt O2 enrichment": (raft(o2_enrich=o["o2_enrich"] + 1) - r0)}
    for s in out:
        if s["label"] in fixed:
            s["value"] = fixed[s["label"]]
            s["label"] += " (raceway, fixed operation)"
    for s in out:
        s["value"] = round(float(s["value"]), 3)
        s["ok"] = bool(s["lo"] <= s["value"] <= s["hi"])
    return out


def main(only=None):
    steps = set(only or ["presets", "dynamic", "validation"])
    t0 = time.time()
    preset_rows = []
    if "presets" in steps or "validation" in steps:
        for key in PRESETS:
            t = time.time()
            if key == DEFAULT_PRESET and os.path.exists(os.path.join(DATA, "base_report.json")):
                rep = json.load(open(os.path.join(DATA, "base_report.json")))
            else:
                rep = _clean(full_report(solve_steady({"preset": key})))
                dump(rep, "presets", f"{key}.json")
            kp = rep["kpi"]
            preset_rows.append({"key": key, "label": PRESETS[key]["label"], "V_work": PRESETS[key]["V_work"],
                                **{kk: kp.get(kk) for kk in ["production_tpd", "productivity_work", "coke_rate", "pci_rate", "fuel_rate", "eta_CO",
                                                             "raft_C", "hmt_C", "hm_Si", "top_gas_T_C", "pressure_drop_kPa", "heat_loss_GJ_t",
                                                             "co2_kg_t", "cost_inr_t"]}})
            log("preset", key, kp["coke_rate"], kp["hmt_C"], kp["production_tpd"], round(time.time() - t, 1), "s")
    if "dynamic" in steps:
        base = solve_steady(None)
        for sc in SCENARIOS:
            t = time.time()
            _, res = run_dynamic(bf=copy.deepcopy(base), scenario=sc, hours=24.0, record_every=2)
            res["hours"] = 24.0
            dump(res, "dynamic", f"{sc}.json")
            tr = res["trajectory"]
            log("dynamic", sc, "HMT", round(min(r["hmt_C"] for r in tr)), round(max(r["hmt_C"] for r in tr)), round(time.time() - t, 1), "s")
    if "validation" in steps:
        rep = json.load(open(os.path.join(DATA, "base_report.json")))
        kp = dict(rep["kpi"])
        kp["_bosh_ratio"] = kp["bosh_gas_nm3_t"] / kp["blast_nm3_per_t"]
        ranges = [{"label": l, "key": k, "unit": u, "value": round(float(kp[k]), 4), "lo": lo, "hi": hi, "source": src}
                  for (l, k, u, lo, hi, src) in WINDOWS if kp.get(k) is not None]
        dyn = json.load(open(os.path.join(DATA, "dynamic", "baseline.json"))) if os.path.exists(os.path.join(DATA, "dynamic", "baseline.json")) else None
        bal = {"max_rel": rep["balances"]["max_rel_element_residual"], "max_H_MJ": abs(rep["balances"]["enthalpy_step_residual_MJ"])}
        if dyn:
            bal["max_rel"] = max(bal["max_rel"], max(abs(r.get("bal_max_rel") or 0) for r in dyn["trajectory"]))
            bal["max_H_MJ"] = max(bal["max_H_MJ"], max(abs(r.get("bal_H_MJ") or 0) for r in dyn["trajectory"]))
        log("sensitivities…")
        sens = sensitivities(default_inputs(DEFAULT_PRESET))
        for s in sens:
            log("  ", s["label"], s["value"], s["ok"])
        st = json.load(open(os.path.join(DATA, "study_results.json"))) if os.path.exists(os.path.join(DATA, "study_results.json")) else {}
        lim = [
            "Plant data used for the soft-sensor and control studies are a synthetic 'Industry-X' historian produced by the same "
            "first-principles model with deliberately mismatched kinetic / heat-transfer parameters and unmeasured random-walk "
            "disturbances; site deployment requires calibration against the customer's historian before any closed-loop use.",
            "The thermal state of the 1-D model is more sensitive to fuel input than plants usually report (HMT change per kg/tHM coke "
            "is given in the sensitivity table); disturbance amplitudes in the studies were chosen accordingly.",
            "The transient solver uses an explicit, Courant-limited operator-split scheme; steady states carry a time-step-dependent "
            "numerical bias of the hot-metal temperature (the model is calibrated at its reference time step, which is used in all runs).",
            "Raceway flame-temperature sensitivities to PCI, blast moisture and O2 enrichment are about 60-90 % of the usual "
            "rules of thumb (the raceway is a lumped adiabatic balance with equilibrium gas at the RAFT); trends are correct but "
            "RAFT limits should be re-tuned against site flame-temperature practice.",
            "Radial gas distribution is represented by a reconstructed pseudo-2-D field for visualisation; burden distribution, "
            "deadman permeability and scaffold formation are not modelled mechanistically.",
            "The cohesive zone is resolved on the axial cell grid (about 0.75 m); very cold states (HMT below about 1420 C) with a "
            "collapsed cohesive zone are outside the surrogate's validity window.",
            "Prices, CO2 factors and refractory properties are editable defaults representative of Indian integrated plants and must "
            "be replaced with site values for commercial decisions.",
        ]
        val = {"generated": time.strftime("%Y-%m-%d %H:%M"), "ranges": ranges, "sensitivities": sens, "presets": preset_rows,
               "balance": bal, "limitations": lim,
               "softsensor": st.get("softsensor", {}).get("hmt"), "control": st.get("control", {}).get("reduction_vs_pi_pct")}
        dump(val, "validation.json")
        log("validation: inside windows", sum(1 for r in ranges if r["lo"] <= r["value"] <= r["hi"]), "/", len(ranges))
    log("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
