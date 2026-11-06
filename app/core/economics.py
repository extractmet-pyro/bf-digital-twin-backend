"""
Hot-metal cost (INR/tHM) and CO2 intensity (kg CO2/tHM) accounting.

CO2 boundary (default): BF gate-to-gate direct emissions (all carbon leaving in the
top gas is eventually burnt to CO2 in stoves/power plant) + scope-2 power for
oxygen and the blower + upstream burden/coke/flux factors (switchable).
"""
from __future__ import annotations

from . import thermo as TH


def stove_fuel_price(T_blast_C: float, u: dict) -> float:
    f_enr = min(1.0, max(0.0, (T_blast_C - 1100.0) / 250.0)) * 0.45
    return (1 - f_enr) * u["bfg_value"] + f_enr * u["enrich_fuel_price"]


def blast_sensible_GJ_per_nm3(T_blast_C: float, o2_pct: float, moisture_g: float) -> float:
    Tb = T_blast_C + 273.15
    T0 = TH.T0
    yO2 = o2_pct / 100.0
    dh_o2 = TH.h_gas_exact("O2", Tb) - TH.h_gas_exact("O2", T0)
    dh_n2 = TH.h_gas_exact("N2", Tb) - TH.h_gas_exact("N2", T0)
    dh_w = TH.h_gas_exact("H2O", Tb) - TH.h_gas_exact("H2O", T0)
    per_kmol = yO2 * dh_o2 + (1 - yO2) * dh_n2                       # MJ/kmol dry blast
    per_nm3 = per_kmol / TH.NM3_PER_KMOL + moisture_g / 18.015 / 1000.0 * dh_w
    return per_nm3 / 1000.0


def economics_from(k: dict, ops: dict, recipe: dict, mat: dict, u: dict, V_work: float,
                   include_upstream: bool = True, slag_kg: float | None = None,
                   topgas: dict | None = None) -> dict:
    """Core accounting from per-tHM quantities (also used with surrogate outputs)."""
    br = mat["burden"]
    fl = mat["flux"]
    cost = {}
    co2 = {}
    for b in ["sinter", "pellet", "lump"]:
        m = recipe.get(b, 0.0)
        cost[br[b]["label"]] = m / 1000.0 * br[b]["price"]
        if include_upstream:
            co2["Upstream: " + br[b]["label"]] = m / 1000.0 * br[b]["co2_upstream"]
    for f in ["limestone", "dolomite", "quartzite"]:
        m = recipe.get(f, 0.0)
        if m > 0:
            cost[fl[f]["label"]] = m / 1000.0 * fl[f]["price"]
            if include_upstream:
                co2["Upstream: " + fl[f]["label"]] = m / 1000.0 * fl[f]["co2_upstream"]
    coke = k["coke_rate"]
    pci = k["pci_rate"]
    cost["Coke"] = coke / 1000.0 * mat["coke"]["price"]
    cost["PCI coal"] = pci / 1000.0 * mat["pci"]["price"]
    if include_upstream:
        co2["Upstream: coke making"] = coke / 1000.0 * mat["coke"]["co2_upstream"]
        co2["Upstream: PCI grinding/drying"] = pci / 1000.0 * mat["pci"]["co2_upstream"]
    o2 = k["oxygen_nm3_per_t"]
    bl = k["blast_nm3_per_t"]
    cost["Oxygen"] = o2 * u["oxygen_price"]
    cost["Blower power"] = bl * u["blower_kwh_per_nm3"] * u["power_price"]
    co2["Scope 2: oxygen plant power"] = o2 * u["oxygen_kwh_per_nm3"] * u["grid_ef"]
    co2["Scope 2: blower power"] = bl * u["blower_kwh_per_nm3"] * u["grid_ef"]
    q_bl = blast_sensible_GJ_per_nm3(ops["blast_temp"], 21.0 + ops["o2_enrich"], ops["moisture"]) * bl
    stove_GJ = q_bl / u["stove_efficiency"]
    cost["Hot-stove fuel"] = stove_GJ * stove_fuel_price(ops["blast_temp"], u)
    # top gas energy credit (LHV) net of stove use
    tg_GJ = k["top_gas_LHV_MJ_nm3"] * k["top_gas_nm3_t_dry"] / 1000.0
    export = max(0.0, tg_GJ - stove_GJ)
    cost["BF-gas export credit"] = -export * u["bfg_value"]
    sk = slag_kg if slag_kg is not None else k.get("slag_rate", 300.0)
    cost["Slag credit"] = -sk / 1000.0 * u["slag_credit"]
    fixed = u["fixed_cost_per_day"] * (V_work / 2800.0) ** 0.8
    cost["Fixed cost (labour, maintenance, capital)"] = fixed / max(k["production_tpd"], 1.0)
    # direct CO2 from top-gas carbon
    if topgas is not None:
        c_top = topgas["C_kmol_t"]
    else:
        c_top = k["top_gas_nm3_t_dry"] * (k["top_gas_CO"] + k["top_gas_CO2"]) / 100.0 / TH.NM3_PER_KMOL
    co2["Direct: top-gas carbon (burnt downstream)"] = c_top * 44.01
    total_cost = sum(cost.values())
    total_co2 = sum(co2.values())
    return {
        "cost": {"items_inr_t": {a: round(b, 1) for a, b in cost.items()}, "total_inr_t": round(total_cost, 1),
                 "total_usd_t": round(total_cost / u["inr_per_usd"], 2),
                 "stove_heat_GJ_t": round(stove_GJ, 3), "topgas_GJ_t": round(tg_GJ, 3), "export_GJ_t": round(export, 3)},
        "co2": {"items_kg_t": {a: round(b, 1) for a, b in co2.items()}, "total_kg_t": round(total_co2, 1),
                "direct_kg_t": round(co2["Direct: top-gas carbon (burnt downstream)"], 1),
                "boundary": "BF direct + scope-2 (O2, blower)" + (" + upstream burden/coke" if include_upstream else "")},
    }


def economics(bf, k: dict, include_upstream: bool = True) -> dict:
    gas = bf.top["gas"]
    P = float(bf.liq_out["m"] @ TH.MW_METAL)
    c_top = (gas[0] + gas[1]) * 1000.0 / max(P, 1e-9)
    return economics_from(k, bf.ops, bf.recipe, bf.inp["materials"], bf.inp["utilities"], bf.geom.V_work,
                          include_upstream=include_upstream, topgas={"C_kmol_t": c_top})
