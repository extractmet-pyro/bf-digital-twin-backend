"""
Thermochemical data and functions for BF-TWIN.

All species enthalpies are ABSOLUTE (formation-based) enthalpies in MJ/kmol
(numerically identical to kJ/mol), referenced to the elements at 298.15 K.
Using absolute enthalpies means every heat of reaction is accounted for
automatically: an energy balance written as  sum(n_i * H_i(T))  conserves
energy exactly and the reaction heats appear as temperature changes.

Gas species use NIST Shomate polynomials (multi-range).
Condensed species use Maier-Kelley heat capacities integrated numerically
into lookup tables (with caps/plateaus where the polynomials diverge).

Sources of data: NIST-JANAF / NIST Chemistry WebBook (Shomate), Kubaschewski
& Alcock "Metallurgical Thermochemistry", Turkdogan "Physical Chemistry of
High Temperature Technology" (equilibrium constants), Sigworth & Elliott
(dissolution in liquid iron).
"""
from __future__ import annotations

import math
import numpy as np

R = 8.314462618e-3          # MJ/(kmol K)  (= kJ/(mol K))
R_J = 8.314462618           # J/(mol K)
T0 = 298.15
NM3_PER_KMOL = 22.414       # Nm3 per kmol (0 C, 1 atm)

# ---------------------------------------------------------------------------
# Molar masses (kg/kmol)
# ---------------------------------------------------------------------------
MW = {
    "CO": 28.010, "CO2": 44.009, "H2": 2.016, "H2O": 18.015, "N2": 28.014,
    "O2": 31.998, "SiO": 44.085, "Sg": 32.06,
    "Fe2O3": 159.69, "Fe3O4": 231.53, "FeO": 71.844, "Fe": 55.845,
    "C": 12.011, "SiO2": 60.084, "Al2O3": 101.96, "CaO": 56.077,
    "MgO": 40.304, "MnO": 70.937, "TiO2": 79.866, "P2O5": 141.94,
    "CaCO3": 100.09, "MgCO3": 84.313, "H2Ol": 18.015, "S": 32.06,
    "Si": 28.086, "Mn": 54.938, "P": 30.974, "CaS": 72.14, "Ti": 47.867,
    "Ar": 39.948,
}

GAS = ["CO", "CO2", "H2", "H2O", "N2", "SiO"]           # gas-march species
SOLID = ["Fe2O3", "Fe3O4", "FeO", "Fe", "C", "SiO2", "Al2O3", "CaO", "MgO",
         "MnO", "TiO2", "P2O5", "CaCO3", "MgCO3", "H2Ol", "S"]
SI = {s: i for i, s in enumerate(SOLID)}
GI = {g: i for i, g in enumerate(GAS)}
METAL = ["Fe", "C", "Si", "Mn", "P", "S"]                # dissolved in hot metal
SLAG = ["SiO2", "Al2O3", "CaO", "MgO", "MnO", "FeO", "TiO2", "P2O5", "CaS"]
MI = {s: i for i, s in enumerate(METAL)}
LI = {s: i for i, s in enumerate(SLAG)}

# ---------------------------------------------------------------------------
# Shomate coefficients (A,B,C,D,E,F) and range upper bounds (K)
# absolute H (kJ/mol) = A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t + F,  t=T/1000
# cp (J/mol K)        = A + B t + C t^2 + D t^3 + E/t^2
# ---------------------------------------------------------------------------
SHOMATE = {
    "CO": [(1300, (25.56759, 6.096130, 4.054656, -2.671301, 0.131021, -118.0089)),
           (6000, (35.15070, 1.300095, -0.205921, 0.013550, -3.282780, -127.8375))],
    "CO2": [(1200, (24.99735, 55.18696, -33.69137, 7.948387, -0.136638, -403.6075)),
            (6000, (58.16639, 2.720074, -0.492289, 0.038844, -6.447293, -425.9186))],
    "H2": [(1000, (33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797)),
           (2500, (18.563083, 12.257357, -2.859786, 0.268238, 1.977990, -1.147438)),
           (6000, (43.413560, -4.293079, 1.272428, -0.096876, -20.533862, -38.515158))],
    "H2O": [(1700, (30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810)),
            (6000, (41.96426, 8.622053, -1.499780, 0.098119, -11.15764, -272.1797))],
    "N2": [(500, (28.98641, 1.853978, -9.647459, 16.63537, 0.000117, -8.671914)),
           (2000, (19.50583, 19.88705, -8.598535, 1.369784, 0.527601, -4.935202)),
           (6000, (35.51872, 1.128728, -0.196103, 0.014662, -4.553760, -18.97091))],
    "O2": [(700, (31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471)),
           (2000, (30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468)),
           (6000, (20.91111, 10.72071, -2.020498, 0.146449, 9.245722, 5.337651))],
}
# SiO(g): simple fit (dHf = -100.4 kJ/mol), Sg lumped sulphur vapour (H2S/COS)
EXTRA_GAS_CP = {"SiO": (-100.4, 29.8, 0.0062, -4.0e5), "Sg": (-20.6, 34.0, 0.0050, -3.0e5)}


def _shomate_coeffs(sp, T):
    for tmax, c in SHOMATE[sp]:
        if T <= tmax:
            return c
    return SHOMATE[sp][-1][1]


def h_gas_exact(sp, T):
    """Absolute enthalpy of a gas species, MJ/kmol (kJ/mol)."""
    if sp in EXTRA_GAS_CP:
        hf, a, b, c = EXTRA_GAS_CP[sp]
        return hf + (a * (T - T0) + 0.5 * b * (T * T - T0 * T0) - c * (1.0 / T - 1.0 / T0)) * 1e-3
    A, B, C, D, E, F = _shomate_coeffs(sp, T)
    t = T / 1000.0
    return A * t + B * t * t / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + F


def cp_gas_exact(sp, T):
    """Heat capacity, MJ/(kmol K)."""
    if sp in EXTRA_GAS_CP:
        hf, a, b, c = EXTRA_GAS_CP[sp]
        return (a + b * T + c / (T * T)) * 1e-3
    A, B, C, D, E, F = _shomate_coeffs(sp, T)
    t = T / 1000.0
    return (A + B * t + C * t * t + D * t ** 3 + E / (t * t)) * 1e-3


# ---------------------------------------------------------------------------
# Condensed species: (dHf kJ/mol, list of (Tmax, a, b, c) Maier-Kelley pieces
# cp = a + b T + c/T^2  J/mol K), optional cap on cp.
# ---------------------------------------------------------------------------
COND = {
    "Fe2O3": (-824.2, [(950, 98.28, 0.07782, -1.4853e6), (4000, 150.6, 0.0, 0.0)]),
    "Fe3O4": (-1118.4, [(866, 91.55, 0.20167, 0.0), (4000, 200.8, 0.0, 0.0)]),
    "FeO": (-272.0, [(4000, 50.80, 0.008614, -3.309e5)]),
    "Fe": (0.0, [(4000, 23.5, 0.0134, 0.0)]),
    "C": (0.0, [(4000, 17.15, 0.00427, -8.79e5)]),
    "SiO2": (-910.7, [(848, 46.94, 0.03431, -1.130e6), (4000, 60.29, 0.00811, 0.0)]),
    "Al2O3": (-1675.7, [(4000, 114.77, 0.01280, -3.544e6)]),
    "CaO": (-635.1, [(4000, 49.62, 0.00452, -6.95e5)]),
    "MgO": (-601.6, [(4000, 42.59, 0.00728, -6.19e5)]),
    "MnO": (-385.2, [(4000, 46.48, 0.00812, -3.68e5)]),
    "TiO2": (-944.0, [(4000, 62.86, 0.01136, -9.96e5)]),
    "P2O5": (-1504.9, [(4000, 120.0, 0.0, 0.0)]),
    "CaCO3": (-1207.6, [(4000, 104.52, 0.02192, -2.594e6)]),
    "MgCO3": (-1095.8, [(4000, 77.91, 0.05774, -1.741e6)]),
    "H2Ol": (-285.83, [(4000, 75.3, 0.0, 0.0)]),
    "S": (0.0, [(4000, 22.6, 0.0, 0.0)]),
    "CaS": (-482.4, [(4000, 45.2, 0.0077, 0.0)]),
    # dissolved in liquid iron (formation from pure solid element at 298 K,
    # dissolution enthalpies after Sigworth & Elliott)
    "dC": (22.6, [(4000, 20.0, 0.0, 0.0)]),
    "dSi": (-81.3, [(4000, 27.0, 0.0, 0.0)]),
    "dMn": (16.2, [(4000, 46.0, 0.0, 0.0)]),
    "dP": (-50.0, [(4000, 26.0, 0.0, 0.0)]),
    "dS": (-70.8, [(4000, 30.0, 0.0, 0.0)]),
}

# latent heats (MJ/kmol) and mass-based melting enthalpies
L_FE_MELT = 13.0            # MJ/kmol Fe, carburised iron (approx. 233 kJ/kg)
L_SLAG_MELT = 0.200         # MJ/kg slag (net fusion enthalpy of BF slag)
HM_CP_LIQ_EXTRA = 0.0       # liquid Fe cp handled through table (approx)

# ---------------------------------------------------------------------------
# Lookup tables (uniform 1 K grid, 250..3600 K)
# ---------------------------------------------------------------------------
TGRID0 = 250.0
TGRID1 = 3600.0
NGRID = int(TGRID1 - TGRID0) + 1
TGRID = np.linspace(TGRID0, TGRID1, NGRID)


def _build_cond_table(sp):
    hf, pieces = COND[sp]
    cp = np.empty(NGRID)
    for k, T in enumerate(TGRID):
        Tc = max(T, 280.0)
        for tmax, a, b, c in pieces:
            if Tc <= tmax:
                cp[k] = a + b * Tc + c / (Tc * Tc)
                break
    # integrate cp from T0 (trapezoid on 1 K grid)
    i0 = int(round(T0 - TGRID0))
    H = np.zeros(NGRID)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (cp[1:] + cp[:-1]))])  # J/mol from TGRID0
    H = hf + (cum - cum[i0] - (T0 - TGRID[i0]) * cp[i0]) * 1e-3
    return H, cp * 1e-3


def _build_gas_table(sp):
    H = np.array([h_gas_exact(sp, T) for T in TGRID])
    cp = np.array([cp_gas_exact(sp, T) for T in TGRID])
    return H, cp


HTAB = {}
CPTAB = {}
for _sp in list(SHOMATE.keys()) + list(EXTRA_GAS_CP.keys()):
    HTAB["g_" + _sp], CPTAB["g_" + _sp] = _build_gas_table(_sp)
for _sp in COND:
    HTAB[_sp], CPTAB[_sp] = _build_cond_table(_sp)

# stacked tables for fast vector evaluation
GAS_H = np.stack([HTAB["g_" + g] for g in GAS])            # (ng, NGRID)
GAS_CP = np.stack([CPTAB["g_" + g] for g in GAS])
SOLID_KEYS = [("H2Ol" if s == "H2Ol" else s) for s in SOLID]
SOL_H = np.stack([HTAB[s] for s in SOLID_KEYS])           # (ns, NGRID)
SOL_CP = np.stack([CPTAB[s] for s in SOLID_KEYS])
MET_KEYS = ["Fe", "dC", "dSi", "dMn", "dP", "dS"]
MET_H = np.stack([HTAB[s] for s in MET_KEYS])
MET_CP = np.stack([CPTAB[s] for s in MET_KEYS])
SLAG_KEYS = ["SiO2", "Al2O3", "CaO", "MgO", "MnO", "FeO", "TiO2", "P2O5", "CaS"]
SLG_H = np.stack([HTAB[s] for s in SLAG_KEYS])
SLG_CP = np.stack([CPTAB[s] for s in SLAG_KEYS])
MW_SOLID = np.array([MW[s] for s in SOLID])
MW_GAS = np.array([MW[g] for g in GAS])
MW_METAL = np.array([MW["Fe"], MW["C"], MW["Si"], MW["Mn"], MW["P"], MW["S"]])
MW_SLAG = np.array([MW[s] for s in SLAG])

# Python-list copies for scalar inner loops (faster than numpy indexing)
_GH = [list(row) for row in GAS_H]
_GCP = [list(row) for row in GAS_CP]


def _idx(T):
    x = T - TGRID0
    if x < 0.0:
        x = 0.0
    elif x > NGRID - 1.001:
        x = NGRID - 1.001
    i = int(x)
    return i, x - i


def hg(k, T):
    """Scalar absolute enthalpy of gas species index k at T (MJ/kmol)."""
    i, f = _idx(T)
    row = _GH[k]
    return row[i] + f * (row[i + 1] - row[i])


def cpg(k, T):
    i, f = _idx(T)
    row = _GCP[k]
    return row[i] + f * (row[i + 1] - row[i])


def h_gas_mix(n, T):
    """Enthalpy flow of gas mixture (list of kmol or kmol/s) at T."""
    i, f = _idx(T)
    s = 0.0
    for k in range(len(n)):
        nk = n[k]
        if nk != 0.0:
            row = _GH[k]
            s += nk * (row[i] + f * (row[i + 1] - row[i]))
    return s


def cp_gas_mix(n, T):
    i, f = _idx(T)
    s = 0.0
    for k in range(len(n)):
        nk = n[k]
        if nk != 0.0:
            row = _GCP[k]
            s += nk * (row[i] + f * (row[i + 1] - row[i]))
    return s


def T_from_h_gas(n, H, Tguess=1000.0):
    """Invert gas mixture enthalpy (scalar Newton)."""
    T = Tguess
    for _ in range(30):
        h = h_gas_mix(n, T)
        c = cp_gas_mix(n, T)
        if c <= 0:
            return T
        dT = (H - h) / c
        if dT > 400.0:
            dT = 400.0
        elif dT < -400.0:
            dT = -400.0
        T += dT
        if T < 260.0:
            T = 260.0
        if T > 3550.0:
            T = 3550.0
        if abs(dT) < 1e-4:
            break
    return T


def interp_tab(tab, T):
    """Vectorised table interpolation. tab: (ns, NGRID); T: array (m,) -> (ns, m)."""
    x = np.clip(np.asarray(T, dtype=float) - TGRID0, 0.0, NGRID - 1.001)
    i = x.astype(int)
    f = x - i
    return tab[:, i] * (1.0 - f) + tab[:, i + 1] * f


def h_solid_cells(n, T):
    """n: (ncell, ns) kmol; T: (ncell,) -> enthalpy (ncell,) MJ."""
    Hs = interp_tab(SOL_H, T)        # (ns, ncell)
    return np.einsum("cs,sc->c", n, Hs)


def cp_solid_cells(n, T):
    Cs = interp_tab(SOL_CP, T)
    return np.einsum("cs,sc->c", n, Cs)


def T_from_h_solid_cells(n, H, Tguess):
    """Vectorised Newton inversion for condensed cells."""
    T = np.array(Tguess, dtype=float).copy()
    for _ in range(25):
        h = h_solid_cells(n, T)
        c = cp_solid_cells(n, T)
        c = np.where(c > 1e-9, c, 1e-9)
        dT = np.clip((H - h) / c, -300.0, 300.0)
        T = np.clip(T + dT, 260.0, 3500.0)
        if np.max(np.abs(dT)) < 1e-3:
            break
    return T


def h_metal(nm, T):
    """Hot-metal enthalpy (MJ) for composition nm (kmol Fe,C,Si,Mn,P,S), liquid."""
    i, f = _idx(T)
    s = 0.0
    for k in range(6):
        if nm[k] != 0.0:
            row = MET_H[k]
            s += nm[k] * (row[i] + f * (row[i + 1] - row[i]))
    return s + nm[0] * L_FE_MELT


def cp_metal(nm, T):
    i, f = _idx(T)
    s = 0.0
    for k in range(6):
        if nm[k] != 0.0:
            row = MET_CP[k]
            s += nm[k] * (row[i] + f * (row[i + 1] - row[i]))
    return s


def h_slag(nl, T):
    """Liquid slag enthalpy (MJ) for composition nl (kmol of SLAG species)."""
    i, f = _idx(T)
    s = 0.0
    mass = 0.0
    for k in range(len(SLAG)):
        if nl[k] != 0.0:
            row = SLG_H[k]
            s += nl[k] * (row[i] + f * (row[i + 1] - row[i]))
            mass += nl[k] * MW_SLAG[k]
    return s + mass * L_SLAG_MELT


def cp_slag(nl, T):
    i, f = _idx(T)
    s = 0.0
    for k in range(len(SLAG)):
        if nl[k] != 0.0:
            row = SLG_CP[k]
            s += nl[k] * (row[i] + f * (row[i + 1] - row[i]))
    return s


def T_from_h_liquids(nm, nl, H, extra_cp=0.0, extra_h_fn=None, Tguess=1700.0):
    """Invert enthalpy of a metal+slag (+ optional extra solid) system."""
    T = Tguess
    for _ in range(40):
        h = h_metal(nm, T) + h_slag(nl, T)
        c = cp_metal(nm, T) + cp_slag(nl, T)
        if extra_h_fn is not None:
            he, ce = extra_h_fn(T)
            h += he
            c += ce
        c += extra_cp
        if c <= 1e-12:
            return T
        dT = max(-300.0, min(300.0, (H - h) / c))
        T = max(500.0, min(3200.0, T + dT))
        if abs(dT) < 1e-4:
            break
    return T


def h_cond_species(sp, T):
    """Scalar absolute enthalpy of a condensed species (MJ/kmol)."""
    i, f = _idx(T)
    row = HTAB[sp]
    return row[i] + f * (row[i + 1] - row[i])


def cp_cond_species(sp, T):
    i, f = _idx(T)
    row = CPTAB[sp]
    return row[i] + f * (row[i + 1] - row[i])


# ---------------------------------------------------------------------------
# Standard Gibbs energies of reaction (J/mol) -> equilibrium constants
# Consistent set: Fe-O-C from Turkdogan; Fe-O-H derived via the WGS reaction.
# ---------------------------------------------------------------------------
def K_wustite_CO(T):     # FeO + CO = Fe + CO2
    return math.exp(-(-17883.0 + 21.08 * T) / (R_J * T))


def K_magnetite_CO(T):   # Fe3O4 + CO = 3FeO + CO2
    return math.exp(-(35380.0 - 40.16 * T) / (R_J * T))


def K_shift(T):          # CO + H2O = CO2 + H2
    return math.exp(-(-36185.0 + 32.01 * T) / (R_J * T))


def K_wustite_H2(T):     # FeO + H2 = Fe + H2O   (= K_w,CO / K_shift)
    return K_wustite_CO(T) / K_shift(T)


def K_magnetite_H2(T):   # Fe3O4 + H2 = 3FeO + H2O
    return K_magnetite_CO(T) / K_shift(T)


def K_boudouard(T):      # C + CO2 = 2CO   (pressures in bar)
    return math.exp(-(170700.0 - 174.5 * T) / (R_J * T))


def K_watergas(T):       # C + H2O = CO + H2
    return K_boudouard(T) / K_shift(T)


def p_eq_CaCO3(T):       # CaCO3 = CaO + CO2, bar
    return math.exp(-(168400.0 - 143.9 * T) / (R_J * T))


def p_eq_MgCO3(T):       # MgCO3 = MgO + CO2, bar
    return math.exp(-(117600.0 - 170.0 * T) / (R_J * T))


def co_fraction_wustite(T):
    """Equilibrium CO/(CO+CO2) for Fe/FeO (Baur-Glaessner)."""
    return 1.0 / (1.0 + K_wustite_CO(T))


def co_fraction_magnetite(T):
    return 1.0 / (1.0 + K_magnetite_CO(T))


def co_fraction_boudouard(T, P=1.0):
    """CO/(CO+CO2) at Boudouard equilibrium for total P (bar)."""
    K = K_boudouard(T)
    # x^2 P / (1-x) = K -> P x^2 + K x - K = 0
    a, b, c = P, K, -K
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


def h2_fraction_wustite(T):
    return 1.0 / (1.0 + K_wustite_H2(T))


def h2_fraction_magnetite(T):
    return 1.0 / (1.0 + K_magnetite_H2(T))


# reaction enthalpies at 298 K for reporting (MJ/kmol of reaction)
def dH298(reactants: dict, products: dict):
    def h(sp):
        if sp in SHOMATE or sp in EXTRA_GAS_CP:
            return h_gas_exact(sp, T0)
        return COND[sp][0]
    return sum(v * h(k) for k, v in products.items()) - sum(v * h(k) for k, v in reactants.items())


REACTIONS = [
    # (label, reactants, products, zone/notes)
    ("3Fe2O3 + CO = 2Fe3O4 + CO2", {"Fe2O3": 3, "CO": 1}, {"Fe3O4": 2, "CO2": 1}, "Upper stack, 400-650 C, irreversible"),
    ("Fe3O4 + CO = 3FeO + CO2", {"Fe3O4": 1, "CO": 1}, {"FeO": 3, "CO2": 1}, "Stack, 600-900 C"),
    ("FeO + CO = Fe + CO2", {"FeO": 1, "CO": 1}, {"Fe": 1, "CO2": 1}, "Thermal reserve zone, W-point"),
    ("3Fe2O3 + H2 = 2Fe3O4 + H2O", {"Fe2O3": 3, "H2": 1}, {"Fe3O4": 2, "H2O": 1}, "Upper stack"),
    ("Fe3O4 + H2 = 3FeO + H2O", {"Fe3O4": 1, "H2": 1}, {"FeO": 3, "H2O": 1}, "Stack"),
    ("FeO + H2 = Fe + H2O", {"FeO": 1, "H2": 1}, {"Fe": 1, "H2O": 1}, "Stack / TRZ"),
    ("C + CO2 = 2CO (solution loss)", {"C": 1, "CO2": 1}, {"CO": 2}, "> 950 C, coke reactivity (CRI)"),
    ("C + H2O = CO + H2 (water-gas)", {"C": 1, "H2O": 1}, {"CO": 1, "H2": 1}, "> 900 C"),
    ("CO + H2O = CO2 + H2 (shift)", {"CO": 1, "H2O": 1}, {"CO2": 1, "H2": 1}, "Gas phase, Fe-catalysed"),
    ("FeO(l) + C = Fe(l) + CO (direct)", {"FeO": 1, "C": 1}, {"Fe": 1, "CO": 1}, "Cohesive / dripping zone, hearth"),
    ("2C + O2 = 2CO (raceway)", {"C": 2, "O2": 1}, {"CO": 2}, "Raceway"),
    ("CaCO3 = CaO + CO2", {"CaCO3": 1}, {"CaO": 1, "CO2": 1}, "800-1000 C (raw flux)"),
    ("MgCO3 = MgO + CO2", {"MgCO3": 1}, {"MgO": 1, "CO2": 1}, "500-750 C"),
    ("SiO2 + C = SiO(g) + CO", {"SiO2": 1, "C": 1}, {"SiO": 1, "CO": 1}, "Raceway (> 1800 C)"),
    ("SiO(g) + C = [Si] + CO", {"SiO": 1, "C": 1}, {"dSi": 1, "CO": 1}, "Dripping zone"),
    ("MnO + C = [Mn] + CO", {"MnO": 1, "C": 1}, {"dMn": 1, "CO": 1}, "Dripping zone / hearth"),
    ("P2O5 + 5C = 2[P] + 5CO", {"P2O5": 1, "C": 5}, {"dP": 2, "CO": 5}, "Dripping zone"),
    ("C(coke) = [C] (carburisation)", {"C": 1}, {"dC": 1}, "Dissolution into iron"),
]


def reaction_table():
    out = []
    for lab, r, p, note in REACTIONS:
        out.append({"reaction": lab, "dH298_kJ_per_mol": round(dH298(r, p), 1), "zone": note})
    return out


def baur_glaessner(Tmin=673.0, Tmax=1573.0, n=91):
    """Equilibrium curves for the Fe-O-C and Fe-O-H systems (fractions)."""
    Ts = np.linspace(Tmin, Tmax, n)
    out = {"T_C": (Ts - 273.15).round(1).tolist(), "FeO_Fe_CO": [], "Fe3O4_FeO_CO": [],
           "FeO_Fe_H2": [], "Fe3O4_FeO_H2": [], "Boudouard_1bar": [], "Boudouard_3bar": []}
    for T in Ts:
        wus = T > 843.0
        out["FeO_Fe_CO"].append(round(co_fraction_wustite(T) * 100, 2) if wus else None)
        out["Fe3O4_FeO_CO"].append(round(co_fraction_magnetite(T) * 100, 2) if wus else None)
        out["FeO_Fe_H2"].append(round(h2_fraction_wustite(T) * 100, 2) if wus else None)
        out["Fe3O4_FeO_H2"].append(round(h2_fraction_magnetite(T) * 100, 2) if wus else None)
        out["Boudouard_1bar"].append(round(co_fraction_boudouard(T, 1.0) * 100, 2))
        out["Boudouard_3bar"].append(round(co_fraction_boudouard(T, 3.0) * 100, 2))
    return out


def ellingham(Tmin=773.0, Tmax=2273.0, n=61):
    """Standard Gibbs energies (kJ/mol O2) for oxide formation lines."""
    Ts = np.linspace(Tmin, Tmax, n)
    lines = {
        "2C+O2=2CO": (-223426.0, -175.3),
        "C+O2=CO2": (-394133.0, -0.84),
        "2Fe+O2=2FeO": (-528000.0, 129.2),
        "6FeO+O2=2Fe3O4": (-636130.0, 255.9),
        "4/3Si... Si+O2=SiO2": (-907100.0, 175.7),
        "2Mn+O2=2MnO": (-769400.0, 145.6),
        "2H2+O2=2H2O": (-492900.0, 109.6),
        "2Ca+O2=2CaO": (-1267600.0, 201.2),
        "4/3Al+O2=2/3Al2O3": (-1120500.0, 214.2),
    }
    out = {"T_C": (Ts - 273.15).round(0).tolist()}
    for k, (a, b) in lines.items():
        out[k] = [round((a + b * T) / 1000.0, 1) for T in Ts]
    return out


if __name__ == "__main__":  # quick self-check
    for sp in ["CO", "CO2", "H2", "H2O", "N2", "O2"]:
        print(sp, round(h_gas_exact(sp, T0), 3), round(hg(GI[sp], 1500.0) if sp in GI else 0, 3))
    print("FeO/Fe CO eq at 900C", co_fraction_wustite(1173.15))
    print(reaction_table()[:4])
