"""
Raw-material, fuel, flux, refractory and price defaults for BF-TWIN.

Compositions are representative of Indian integrated-plant practice
(high-alumina sinter and lump ore, imported-blend coke, low-volatile PCI
coal).  Every value is editable through the API/GUI; plant identities are
anonymised as Industry-X.
"""
from __future__ import annotations

import copy

# ---------------------------------------------------------------------------
# Ferrous burden (dry basis, wt%). Fe2O3/FeO given explicitly.
# ---------------------------------------------------------------------------
BURDEN = {
    "sinter": {
        "label": "Sinter (fluxed)",
        "comp": {"Fe2O3": 69.90, "FeO": 8.50, "SiO2": 5.40, "Al2O3": 2.90, "CaO": 10.80,
                 "MgO": 1.90, "MnO": 0.35, "TiO2": 0.15, "P2O5": 0.12, "S": 0.02},
        "moisture": 0.5, "bulk_density": 1750.0, "size_mm": 18.0, "voidage": 0.40,
        "T_soft": 1180.0, "T_melt": 1390.0, "reducibility": 1.00, "shape": 0.80,
        "price": 7200.0, "co2_upstream": 210.0, "fines_to_dust": 0.006,
    },
    "pellet": {
        "label": "Pellets (blast-furnace grade)",
        "comp": {"Fe2O3": 90.84, "FeO": 0.60, "SiO2": 3.60, "Al2O3": 2.60, "CaO": 1.60,
                 "MgO": 0.70, "MnO": 0.08, "TiO2": 0.10, "P2O5": 0.06, "S": 0.005},
        "moisture": 1.5, "bulk_density": 2100.0, "size_mm": 12.0, "voidage": 0.38,
        "T_soft": 1100.0, "T_melt": 1370.0, "reducibility": 0.85, "shape": 0.95,
        "price": 10500.0, "co2_upstream": 95.0, "fines_to_dust": 0.003,
    },
    "lump": {
        "label": "Calibrated lump ore",
        "comp": {"Fe2O3": 89.74, "FeO": 0.30, "SiO2": 2.80, "Al2O3": 2.60, "CaO": 0.10,
                 "MgO": 0.10, "MnO": 0.08, "TiO2": 0.15, "P2O5": 0.08, "S": 0.01},
        "moisture": 6.5, "bulk_density": 2250.0, "size_mm": 20.0, "voidage": 0.40,
        "T_soft": 1080.0, "T_melt": 1360.0, "reducibility": 0.70, "shape": 0.75,
        "price": 6000.0, "co2_upstream": 15.0, "fines_to_dust": 0.008,
    },
}

FLUX = {
    "limestone": {"label": "Limestone", "comp": {"CaCO3": 94.0, "MgCO3": 2.0, "SiO2": 2.5, "Al2O3": 0.8, "Fe2O3": 0.7},
                  "moisture": 1.0, "bulk_density": 1550.0, "size_mm": 30.0, "price": 1900.0, "co2_upstream": 5.0},
    "dolomite": {"label": "Dolomite", "comp": {"CaCO3": 54.5, "MgCO3": 42.0, "SiO2": 2.0, "Al2O3": 0.8, "Fe2O3": 0.7},
                 "moisture": 1.0, "bulk_density": 1600.0, "size_mm": 30.0, "price": 2100.0, "co2_upstream": 5.0},
    "quartzite": {"label": "Quartzite", "comp": {"SiO2": 95.0, "Al2O3": 2.5, "Fe2O3": 1.7, "CaO": 0.5, "MgO": 0.3},
                  "moisture": 1.0, "bulk_density": 1600.0, "size_mm": 30.0, "price": 1500.0, "co2_upstream": 5.0},
}

COKE = {
    "label": "Metallurgical coke (dry)",
    "C": 86.0, "ash": 13.2, "S": 0.60, "VM": 0.20,
    "ash_comp": {"SiO2": 52.0, "Al2O3": 28.0, "Fe2O3": 9.0, "CaO": 4.0, "MgO": 1.5,
                 "TiO2": 1.5, "P2O5": 0.8, "MnO": 0.2},
    "moisture": 4.5, "bulk_density": 500.0, "size_mm": 48.0, "voidage": 0.50,
    "CRI": 25.0, "CSR": 64.0, "price": 32000.0, "co2_upstream": 410.0,
    "fines_to_dust": 0.004,
}

PCI = {
    "label": "PCI coal (low-volatile blend)",
    "C": 78.5, "H": 4.3, "O": 5.6, "N": 1.7, "S": 0.5, "ash": 9.4, "VM": 22.0,
    "ash_comp": {"SiO2": 55.0, "Al2O3": 30.0, "Fe2O3": 6.0, "CaO": 3.0, "MgO": 1.2,
                 "TiO2": 1.5, "P2O5": 0.5},
    "moisture": 1.0, "size_d80_um": 75.0, "price": 15500.0, "co2_upstream": 25.0,
}


def pci_gcv(p=None):
    """Dulong gross calorific value (MJ/kg dry coal)."""
    p = p or PCI
    return 0.3383 * p["C"] + 1.443 * (p["H"] - p["O"] / 8.0) + 0.0942 * p["S"]


# ---------------------------------------------------------------------------
# Hot-blast, oxygen, utilities
# ---------------------------------------------------------------------------
UTIL = {
    "oxygen_purity": 0.995,
    "oxygen_price": 6.5,            # INR/Nm3 (incl. power)
    "oxygen_kwh_per_nm3": 0.45,
    "blower_kwh_per_nm3": 0.060,
    "power_price": 7.0,             # INR/kWh
    "grid_ef": 0.716,               # kg CO2/kWh (India grid, CEA)
    "stove_efficiency": 0.80,
    "bfg_value": 280.0,             # INR/GJ (opportunity value of BF gas)
    "enrich_fuel_price": 900.0,     # INR/GJ (COG/NG enrichment above ~1100 C)
    "slag_credit": 500.0,           # INR/t granulated slag
    "fixed_cost_per_day": 2.2e7,    # INR/day for a 2800 m3 class furnace (scaled)
    "inr_per_usd": 84.0,
}

# ---------------------------------------------------------------------------
# Refractory & cooling library (k in W/m K at service temperature)
# ---------------------------------------------------------------------------
REFRACTORY = {
    "conv_carbon": {"label": "Conventional carbon block", "k": 12.0, "rho": 1550, "tmax": 1600},
    "micropore_carbon": {"label": "Super-microporous carbon block", "k": 18.0, "rho": 1650, "tmax": 1700},
    "semigraphite": {"label": "Semi-graphite carbon block", "k": 40.0, "rho": 1650, "tmax": 1700},
    "graphite": {"label": "Graphite block", "k": 90.0, "rho": 1700, "tmax": 1800},
    "sic": {"label": "SiC (Si3N4-bonded) brick", "k": 18.0, "rho": 2650, "tmax": 1650},
    "high_alumina": {"label": "High-alumina brick (70% Al2O3)", "k": 2.0, "rho": 2600, "tmax": 1700},
    "ceramic_cup": {"label": "Corundum-mullite ceramic cup", "k": 3.2, "rho": 2900, "tmax": 1750},
    "fireclay": {"label": "Fireclay brick", "k": 1.3, "rho": 2200, "tmax": 1400},
    "alumina_carbon": {"label": "Alumina-carbon brick", "k": 8.0, "rho": 2900, "tmax": 1700},
    "castable": {"label": "Castable / ramming mass", "k": 1.6, "rho": 2400, "tmax": 1500},
    "skull": {"label": "Accretion / slag skull", "k": 1.5, "rho": 2800, "tmax": 1500},
    "copper_stave": {"label": "Copper stave", "k": 360.0, "rho": 8900, "tmax": 250},
    "ci_stave": {"label": "Cast-iron stave", "k": 35.0, "rho": 7100, "tmax": 700},
    "shell": {"label": "Steel shell", "k": 45.0, "rho": 7850, "tmax": 400},
}

# lining per zone: list of (material, thickness m) from hot face to shell,
# plus inner film coefficient (W/m2K) and cooling arrangement
LINING = {
    "throat": {"layers": [("castable", 0.10), ("high_alumina", 0.115), ("ci_stave", 0.16), ("shell", 0.04)],
               "h_in": 120.0, "cooling": "stave"},
    "upper_stack": {"layers": [("high_alumina", 0.23), ("ci_stave", 0.18), ("shell", 0.05)],
                    "h_in": 150.0, "cooling": "stave"},
    "lower_stack": {"layers": [("skull", 0.03), ("sic", 0.20), ("ci_stave", 0.18), ("shell", 0.05)],
                    "h_in": 250.0, "cooling": "stave"},
    "belly": {"layers": [("skull", 0.035), ("sic", 0.15), ("copper_stave", 0.125), ("shell", 0.05)],
              "h_in": 350.0, "cooling": "stave"},
    "bosh": {"layers": [("skull", 0.045), ("sic", 0.10), ("copper_stave", 0.125), ("shell", 0.05)],
             "h_in": 450.0, "cooling": "stave"},
    "tuyere_zone": {"layers": [("ceramic_cup", 0.35), ("micropore_carbon", 0.60), ("copper_stave", 0.125), ("shell", 0.06)],
                    "h_in": 600.0, "cooling": "stave"},
    "hearth_wall": {"layers": [("ceramic_cup", 0.35), ("micropore_carbon", 1.20), ("ci_stave", 0.16), ("shell", 0.06)],
                    "h_in": 900.0, "cooling": "stave"},
    "hearth_bottom": {"layers": [("ceramic_cup", 0.80), ("micropore_carbon", 1.20), ("graphite", 0.40), ("ci_stave", 0.10), ("shell", 0.06)],
                      "h_in": 900.0, "cooling": "underhearth"},
}

COOLING = {"h_water": 5000.0, "T_water_C": 35.0, "area_ratio": 0.55, "T_skull_C": 1200.0, "h_skull": 160.0,
           "h_air": 8.0, "emissivity": 0.80, "T_amb_C": 35.0,
           "tuyere_loss_kW": 140.0}

# ---------------------------------------------------------------------------
# Furnace presets: diameters (m), heights (m), tuyeres, tapholes, nominal ops
# Stack height is solved so that the working volume (tuyere -> stock line)
# matches the nominal value.
# ---------------------------------------------------------------------------
PRESETS = {
    "BF-350": {"label": "Mini BF 350 m3", "V_work": 350.0, "d_hearth": 5.4, "d_belly": 6.6, "d_throat": 4.6,
               "h_hearth": 2.4, "h_sump": 1.0, "h_bosh": 2.6, "h_belly": 1.4, "h_throat": 1.4,
               "n_tuyere": 12, "d_tuyere": 0.12, "n_taphole": 1,
               "blast_volume": 760.0, "blast_temp": 1050.0, "o2_enrich": 3.0, "pci": 100.0,
               "moisture": 12.0, "top_pressure": 0.8, "pellet_frac": 20.0, "coke_guess": 470.0},
    "BF-1050": {"label": "Medium BF 1050 m3", "V_work": 1050.0, "d_hearth": 8.2, "d_belly": 9.6, "d_throat": 6.6,
                "h_hearth": 3.0, "h_sump": 1.4, "h_bosh": 3.1, "h_belly": 1.8, "h_throat": 1.8,
                "n_tuyere": 18, "d_tuyere": 0.13, "n_taphole": 2,
                "blast_volume": 2050.0, "blast_temp": 1120.0, "o2_enrich": 3.5, "pci": 140.0,
                "moisture": 12.0, "top_pressure": 1.5, "pellet_frac": 25.0, "coke_guess": 380.0},
    "BF-1800": {"label": "Large BF 1800 m3", "V_work": 1800.0, "d_hearth": 9.8, "d_belly": 11.2, "d_throat": 7.6,
                "h_hearth": 3.4, "h_sump": 1.7, "h_bosh": 3.3, "h_belly": 1.9, "h_throat": 1.9,
                "n_tuyere": 24, "d_tuyere": 0.14, "n_taphole": 2,
                "blast_volume": 3300.0, "blast_temp": 1150.0, "o2_enrich": 4.0, "pci": 160.0,
                "moisture": 12.0, "top_pressure": 1.9, "pellet_frac": 25.0, "coke_guess": 350.0},
    "BF-2800": {"label": "Large BF 2800 m3 (Industry-X default)", "V_work": 2800.0, "d_hearth": 11.8, "d_belly": 13.4,
                "d_throat": 9.0, "h_hearth": 3.8, "h_sump": 2.0, "h_bosh": 3.6, "h_belly": 2.1, "h_throat": 2.1,
                "n_tuyere": 30, "d_tuyere": 0.14, "n_taphole": 3,
                "blast_volume": 4850.0, "blast_temp": 1170.0, "o2_enrich": 4.5, "pci": 170.0,
                "moisture": 12.0, "top_pressure": 2.2, "pellet_frac": 25.0, "coke_guess": 358.0},
    "BF-4000": {"label": "Very large BF 4000 m3", "V_work": 4000.0, "d_hearth": 13.8, "d_belly": 15.4, "d_throat": 10.4,
                "h_hearth": 4.3, "h_sump": 2.3, "h_bosh": 3.8, "h_belly": 2.2, "h_throat": 2.2,
                "n_tuyere": 36, "d_tuyere": 0.15, "n_taphole": 4,
                "blast_volume": 6700.0, "blast_temp": 1200.0, "o2_enrich": 5.0, "pci": 180.0,
                "moisture": 12.0, "top_pressure": 2.4, "pellet_frac": 25.0, "coke_guess": 330.0},
    "BF-5500": {"label": "Super-large BF 5500 m3", "V_work": 5500.0, "d_hearth": 15.6, "d_belly": 17.4,
                "d_throat": 11.6, "h_hearth": 4.8, "h_sump": 2.6, "h_bosh": 4.0, "h_belly": 2.4, "h_throat": 2.4,
                "n_tuyere": 42, "d_tuyere": 0.16, "n_taphole": 4,
                "blast_volume": 8900.0, "blast_temp": 1220.0, "o2_enrich": 5.5, "pci": 190.0,
                "moisture": 12.0, "top_pressure": 2.6, "pellet_frac": 25.0, "coke_guess": 320.0},
}
DEFAULT_PRESET = "BF-2800"


def default_inputs(preset: str = DEFAULT_PRESET) -> dict:
    """Full default input dictionary for a furnace preset."""
    p = PRESETS[preset]
    return {
        "preset": preset,
        "geometry": {k: p[k] for k in ["V_work", "d_hearth", "d_belly", "d_throat", "h_hearth", "h_sump",
                                        "h_bosh", "h_belly", "h_throat", "n_tuyere", "d_tuyere", "n_taphole"]},
        "operation": {
            "blast_volume": p["blast_volume"],     # Nm3/min, dry, incl. enrichment O2
            "blast_temp": p["blast_temp"],         # C
            "o2_enrich": p["o2_enrich"],           # %-points above 21 % O2
            "pci": p["pci"],                       # kg/tHM
            "moisture": p["moisture"],             # g/Nm3 dry blast
            "top_pressure": p["top_pressure"],     # bar(g)
            "pellet_frac": p["pellet_frac"],       # % of ferrous burden
            "lump_frac": 10.0,                     # % of ferrous burden
            "basicity_target": 1.08,               # CaO/SiO2 of slag
            "mgo_target": 8.5,                     # wt% MgO in slag
            "hmt_target": 1500.0,                  # C, thermal-state closure
            "coke_rate": p["coke_guess"],          # kg/tHM (dynamic mode / first guess)
            "coke_moisture": COKE["moisture"],
            "ore_reducibility": 1.0,               # multiplier
            "coke_reactivity": 1.0,                # multiplier on CRI kinetics
            "burden_distribution": 1.0,            # gas-utilisation factor (1 = nominal)
        },
        "materials": {"burden": copy.deepcopy(BURDEN), "flux": copy.deepcopy(FLUX),
                      "coke": copy.deepcopy(COKE), "pci": copy.deepcopy(PCI)},
        "lining": copy.deepcopy(LINING),
        "cooling": copy.deepcopy(COOLING),
        "utilities": copy.deepcopy(UTIL),
    }


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge user overrides into a default dict (returns new)."""
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def catalogue() -> dict:
    """Material/refractory/preset catalogue for the GUI."""
    return {
        "presets": {k: {"label": v["label"], "V_work": v["V_work"]} for k, v in PRESETS.items()},
        "default_preset": DEFAULT_PRESET,
        "burden": BURDEN, "flux": FLUX, "coke": COKE, "pci": PCI, "pci_gcv": round(pci_gcv(), 2),
        "refractory": REFRACTORY, "lining": LINING, "cooling": COOLING, "utilities": UTIL,
    }
