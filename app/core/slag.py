"""
Slag and hot-metal property models.

* basicities (B2, B3, B4) and optical basicity (Duffy & Ingram values)
* viscosity: Urbain model for CaO-MgO-Al2O3-SiO2(-FeO-MnO) slags
* free-running temperature (viscosity = 2.5 Pa s)
* sulphide capacity: Young et al. (1992) optical-basicity correlation
* sulphur partition L_S between slag and C-saturated iron
* carbon solubility in hot metal (Neumann-Schenck)
"""
from __future__ import annotations

import math

MW = {"SiO2": 60.084, "Al2O3": 101.96, "CaO": 56.077, "MgO": 40.304, "MnO": 70.937,
      "FeO": 71.844, "TiO2": 79.866, "P2O5": 141.94, "CaS": 72.14}
LAMBDA = {"CaO": 1.00, "MgO": 0.78, "SiO2": 0.48, "Al2O3": 0.61, "MnO": 0.59, "FeO": 0.51,
          "TiO2": 0.61, "P2O5": 0.40}
NOX = {"CaO": 1, "MgO": 1, "SiO2": 2, "Al2O3": 3, "MnO": 1, "FeO": 1, "TiO2": 2, "P2O5": 5}


def wt_to_mol(w: dict) -> dict:
    n = {k: w.get(k, 0.0) / MW[k] for k in MW if w.get(k, 0.0) > 0}
    s = sum(n.values()) or 1.0
    return {k: v / s for k, v in n.items()}


def basicity(w: dict) -> dict:
    c, s, m, a = w.get("CaO", 0.0), w.get("SiO2", 1e-9), w.get("MgO", 0.0), w.get("Al2O3", 0.0)
    return {"B2": c / s, "B3": (c + m) / s, "B4": (c + m) / (s + a)}


def optical_basicity(w: dict) -> float:
    x = wt_to_mol(w)
    num = sum(x[k] * NOX[k] * LAMBDA[k] for k in x if k in LAMBDA)
    den = sum(x[k] * NOX[k] for k in x if k in NOX)
    return num / den if den > 0 else 0.6


def urbain_viscosity(w: dict, T_K: float) -> float:
    """Urbain (1981) viscosity, Pa s."""
    x = wt_to_mol(w)
    XG = x.get("SiO2", 0.0) + x.get("P2O5", 0.0)
    XM = x.get("CaO", 0.0) + x.get("MgO", 0.0) + x.get("FeO", 0.0) + x.get("MnO", 0.0) + x.get("TiO2", 0.0)
    XA = x.get("Al2O3", 0.0)
    alpha = XM / max(XM + XA, 1e-9)
    B0 = 13.8 + 39.9355 * alpha - 44.049 * alpha ** 2
    B1 = 30.481 - 117.1505 * alpha + 129.9978 * alpha ** 2
    B2 = -40.9429 + 234.0486 * alpha - 300.04 * alpha ** 2
    B3 = 60.7619 - 153.9276 * alpha + 211.1616 * alpha ** 2
    B = B0 + B1 * XG + B2 * XG ** 2 + B3 * XG ** 3
    lnA = -(0.29 * B + 11.57)
    eta_poise = T_K * math.exp(lnA + 1000.0 * B / T_K)
    return eta_poise * 0.1


def free_running_T(w: dict, eta_crit: float = 2.5) -> float:
    """Temperature (C) where viscosity falls to eta_crit (Pa s)."""
    lo, hi = 1000.0, 1800.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if urbain_viscosity(w, mid + 273.15) > eta_crit:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sulphide_capacity(w: dict, T_K: float) -> float:
    """log10 C_S, Young et al. (1992), Lambda < 0.8."""
    L = optical_basicity(w)
    return (-13.913 + 42.84 * L - 23.82 * L * L - 11710.0 / T_K
            - 0.02223 * w.get("SiO2", 0.0) - 0.02275 * w.get("Al2O3", 0.0))


def sulphur_partition(w: dict, T_K: float, C=4.6, Si=0.45, p_co=2.5) -> float:
    """Equilibrium L_S = (%S)/[%S] for slag/carbon-saturated iron."""
    logCs = sulphide_capacity(w, T_K)
    logfS = 0.11 * C + 0.063 * Si
    logfC = 0.14 * C + 0.08 * Si
    hC = (10 ** logfC) * C
    K = 10 ** (1160.0 / T_K + 2.003)
    hO = p_co / (K * hC)
    return 10 ** (logCs + logfS + (-938.0 / T_K + 1.401) - math.log10(hO))


def carbon_saturation(T_C: float, Si=0.45, Mn=0.3, P=0.1, S=0.03) -> float:
    """Carbon solubility in hot metal (wt%), Neumann-Schenck."""
    return 1.30 + 2.57e-3 * T_C - 0.31 * Si - 0.33 * P + 0.027 * Mn - 0.40 * S


def slag_report(w: dict, T_K: float, hm: dict | None = None) -> dict:
    hm = hm or {}
    b = basicity(w)
    out = {
        "B2": round(b["B2"], 3), "B3": round(b["B3"], 3), "B4": round(b["B4"], 3),
        "optical_basicity": round(optical_basicity(w), 4),
        "viscosity_Pa_s": round(urbain_viscosity(w, T_K), 3),
        "viscosity_1450C_Pa_s": round(urbain_viscosity(w, 1723.15), 3),
        "free_running_T_C": round(free_running_T(w), 1),
        "log_Cs": round(sulphide_capacity(w, T_K), 3),
        "L_S_equilibrium": round(sulphur_partition(w, T_K, hm.get("C", 4.6), hm.get("Si", 0.45)), 2),
    }
    return out
