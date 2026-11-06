"""
Wall heat-loss model: coupled conduction, convection and radiation through
the multi-layer furnace lining of every zone.

Heat path (stave-cooled zone):

  process (gas/burden/liquid) --[convection h_in + radiation eps*sigma]-->
  hot face --[conduction through refractory/skull layers]-->
  stave --[conduction to embedded pipes]--> cooling water film (h_w)
  in parallel: stave back --> steel shell --[external convection + radiation]--> ambient

Air-cooled (no stave) zones send the whole flux through the shell to ambient.
All resistances are per m2 of hot face (planar approximation; wall thickness
is small relative to radius except in the hearth, where a cylindrical
correction factor is applied).
"""
from __future__ import annotations

import math
import numpy as np

from .materials import REFRACTORY

SIGMA = 5.670374e-8


class WallZone:
    def __init__(self, name: str, spec: dict, cooling: dict, radius: float | None = None):
        self.name = name
        self.layers = [(m, float(t)) for m, t in spec["layers"]]
        self.h_in = float(spec.get("h_in", 200.0))
        self.mode = spec.get("cooling", "stave")
        self.cool = cooling
        self.radius = radius
        self.eps_in = 0.55            # effective emissivity bed->wall
        self.T_skull = cooling.get("T_skull_C", 1250.0) + 273.15
        self.h_skull = cooling.get("h_skull", 160.0)
        self.Tw = cooling["T_water_C"] + 273.15
        self.Ta = cooling["T_amb_C"] + 273.15
        self.hw = cooling["h_water"] * cooling["area_ratio"]
        self.ha = cooling["h_air"]
        self.eps_sh = cooling["emissivity"]
        # split layers into refractory part, cooling element, shell
        self.ref_layers, self.stave, self.shell = [], None, None
        for m, t in self.layers:
            if m in ("copper_stave", "ci_stave"):
                self.stave = (m, t)
            elif m == "shell":
                self.shell = (m, t)
            else:
                self.ref_layers.append((m, t))
        self._table = None

    # cylindrical correction: R = ln(r2/r1) r1 / k  vs planar t/k
    def _layer_R(self, m, t, r_in):
        k = REFRACTORY[m]["k"]
        if self.radius and self.name.startswith("hearth_wall"):
            r2 = r_in + t
            return r_in * math.log(r2 / r_in) / k, r2
        return t / k, r_in + t

    def resistances(self):
        r = self.radius or 5.0
        out = []
        for m, t in self.ref_layers:
            R, r = self._layer_R(m, t, r)
            out.append((m, t, R))
        return out

    def _shell_back(self, Tnode, R_to_shell):
        """Solve the external shell temperature: (Tnode-Tsh)/R = ha(Tsh-Ta) + eps sigma (Tsh^4-Ta^4)."""
        lo, hi = self.Ta, max(Tnode, self.Ta + 1e-6)
        for _ in range(60):
            Tsh = 0.5 * (lo + hi)
            q_c = (Tnode - Tsh) / R_to_shell
            q_o = self.ha * (Tsh - self.Ta) + self.eps_sh * SIGMA * (Tsh ** 4 - self.Ta ** 4)
            if q_c > q_o:
                lo = Tsh
            else:
                hi = Tsh
        Tsh = 0.5 * (lo + hi)
        q_conv = self.ha * (Tsh - self.Ta)
        q_rad = self.eps_sh * SIGMA * (Tsh ** 4 - self.Ta ** 4)
        return Tsh, q_conv, q_rad

    def solve(self, Tp: float) -> dict:
        """Full solution with dynamic skull (accretion) equilibrium.

        If the zone carries a skull layer, its thickness adjusts so that the skull hot
        face sits at the slag/metal solidification temperature (T_skull) whenever the
        process is hotter than that: the skull melts back when the furnace runs hot and
        grows (up to the design thickness) when it runs cold.  This makes the stave heat
        load rise steeply with the lower-zone thermal state, as observed in practice."""
        skull_i = [i for i, (m, t) in enumerate(self.ref_layers) if m == "skull"]
        if not skull_i:
            return self._solve_fixed(Tp)
        i = skull_i[0]
        base = self._solve_fixed(Tp)
        if Tp <= self.T_skull + 1.0:
            base["skull_thickness_m"] = self.ref_layers[i][1]
            return base
        q_eq = self.h_skull * (Tp - self.T_skull)
        q_fix = base["q_total_kW_m2"] * 1000.0
        if q_eq <= q_fix:
            base["skull_thickness_m"] = self.ref_layers[i][1]
            return base
        # thinner skull: find thickness t so that conduction carries q_eq
        t_max = self.ref_layers[i][1]
        lo, hi = 0.0, t_max
        saved = list(self.ref_layers)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            self.ref_layers[i] = ("skull", max(mid, 1e-6))
            q = self._solve_fixed(Tp)["q_total_kW_m2"] * 1000.0
            if q > q_eq:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        self.ref_layers[i] = ("skull", max(t, 1e-6))
        res = self._solve_fixed(Tp)
        self.ref_layers[:] = saved
        res["skull_thickness_m"] = round(t, 4)
        return res

    def _solve_fixed(self, Tp: float) -> dict:
        Rref = sum(R for _, _, R in self.resistances())
        sh_R = (self.shell[1] / REFRACTORY["shell"]["k"]) if self.shell else 0.0
        if self.stave is not None:
            st_k = REFRACTORY[self.stave[0]]["k"]
            R_st_half = 0.5 * self.stave[1] / st_k
            R_cold = R_st_half + 1.0 / self.hw          # hot face of stave -> water
        else:
            R_st_half = 0.0
            R_cold = None

        def q_cond(Thf):
            if R_cold is not None:
                return (Thf - self.Tw) / (Rref + R_cold)
            # air-cooled: nested solve with shell
            Tsh, qc, qr = self._shell_back(Thf, Rref + sh_R)
            return qc + qr

        def q_in(Thf):
            return self.h_in * (Tp - Thf) + self.eps_in * SIGMA * (Tp ** 4 - Thf ** 4)

        lo, hi = min(self.Tw, Tp), Tp
        if Tp <= self.Tw + 1e-6:
            Thf = Tp
        else:
            for _ in range(70):
                Thf = 0.5 * (lo + hi)
                if q_in(Thf) > q_cond(Thf):
                    lo = Thf
                else:
                    hi = Thf
            Thf = 0.5 * (lo + hi)
        q = max(0.0, q_cond(Thf))
        q_conv_in = self.h_in * (Tp - Thf)
        q_rad_in = self.eps_in * SIGMA * (Tp ** 4 - Thf ** 4)
        # temperature profile through layers
        prof = [{"x_m": 0.0, "T_C": round(Tp - 273.15, 1), "label": "process"},
                {"x_m": 0.0, "T_C": round(Thf - 273.15, 1), "label": "hot face"}]
        T = Thf
        x = 0.0
        for m, t, R in self.resistances():
            T = T - q * R
            x += t
            prof.append({"x_m": round(x, 3), "T_C": round(T - 273.15, 1), "label": REFRACTORY[m]["label"]})
        q_back = 0.0
        Tsh = None
        q_ext_conv = q_ext_rad = 0.0
        if self.stave is not None:
            T_pipe = self.Tw + q / self.hw
            x += self.stave[1]
            prof.append({"x_m": round(x - 0.5 * self.stave[1], 3), "T_C": round(T_pipe - 273.15, 1),
                         "label": REFRACTORY[self.stave[0]]["label"] + " (pipe wall)"})
            if self.shell:
                Tsh, q_ext_conv, q_ext_rad = self._shell_back(T_pipe, R_st_half + sh_R)
                q_back = q_ext_conv + q_ext_rad
                x += self.shell[1]
                prof.append({"x_m": round(x, 3), "T_C": round(Tsh - 273.15, 1), "label": "steel shell"})
            q_water = q
        else:
            if self.shell:
                Tsh, q_ext_conv, q_ext_rad = self._shell_back(T, sh_R if sh_R > 0 else 1e-4)
                x += self.shell[1]
                prof.append({"x_m": round(x, 3), "T_C": round(Tsh - 273.15, 1), "label": "steel shell"})
            q_water = 0.0
        return {
            "zone": self.name, "T_process_C": round(Tp - 273.15, 1), "T_hotface_C": round(Thf - 273.15, 1),
            "q_total_kW_m2": round((q + q_back) / 1000.0, 3), "q_water_kW_m2": round(q_water / 1000.0, 3),
            "q_in_convection_kW_m2": round(q_conv_in / 1000.0, 3), "q_in_radiation_kW_m2": round(q_rad_in / 1000.0, 3),
            "q_shell_convection_W_m2": round(q_ext_conv, 2), "q_shell_radiation_W_m2": round(q_ext_rad, 2),
            "T_shell_C": None if Tsh is None else round(Tsh - 273.15, 1),
            "R_refractory_m2K_W": round(Rref, 5), "profile": prof,
            "layers": [{"material": REFRACTORY[m]["label"], "key": m, "thickness_m": t,
                        "k_W_mK": REFRACTORY[m]["k"]} for m, t in self.layers],
        }

    def flux(self, Tp: float) -> float:
        """Total loss flux W/m2 (fast, from table)."""
        if self._table is None:
            Tg = np.arange(300.0, 2601.0, 20.0)
            qs = np.array([self.solve(T)["q_total_kW_m2"] * 1000.0 for T in Tg])
            self._table = (Tg, qs)
        Tg, qs = self._table
        return float(np.interp(Tp, Tg, qs))

    def flux_table(self):
        self.flux(1000.0)
        return self._table


class WallModel:
    """Collection of wall zones for a furnace."""

    def __init__(self, lining: dict, cooling: dict, geom):
        self.zones = {}
        for name, spec in lining.items():
            r = geom.r_h if name.startswith("hearth") or name == "tuyere_zone" else None
            self.zones[name] = WallZone(name, spec, cooling, radius=r)
        self.cooling = cooling
        self.geom = geom
        # cell zone tables (vectorised interpolation for the shaft)
        self.cell_tables = []
        for z in geom.zones:
            Tg, qs = self.zones[z].flux_table()
            self.cell_tables.append((Tg, qs))
        Tg0 = self.cell_tables[0][0]
        self.Tgrid = Tg0
        self.Q = np.stack([t[1] for t in self.cell_tables])        # (ncell, nT) W/m2

    def cell_losses(self, Tp_cells):
        """Wall heat loss per cell (MW) for process temperatures (K)."""
        x = np.clip((np.asarray(Tp_cells) - self.Tgrid[0]) / (self.Tgrid[1] - self.Tgrid[0]), 0, len(self.Tgrid) - 1.001)
        i = x.astype(int)
        f = x - i
        rows = np.arange(len(i))
        q = self.Q[rows, i] * (1 - f) + self.Q[rows, i + 1] * f
        return q * self.geom.A_wall * 1e-6

    def hearth_loss(self, T_hearth):
        """Hearth wall + bottom + tuyere-zone wall losses (MW)."""
        g = self.geom
        qw = self.zones["hearth_wall"].flux(T_hearth)
        qb = self.zones["hearth_bottom"].flux(T_hearth)
        qt = self.zones["tuyere_zone"].flux(T_hearth + 30.0)
        A_tz = 2 * math.pi * g.r_h * 0.6
        return (qw * g.A_hearth_wall + qb * g.A_hearth_bottom + qt * A_tz) * 1e-6

    def tuyere_loss(self, raft_K):
        return self.geom.n_tuyere * self.cooling["tuyere_loss_kW"] * 1e-3 * (raft_K / 2400.0) ** 1.5

    def report(self, T_by_zone: dict) -> list:
        out = []
        g = self.geom
        areas = {}
        for z, a in zip(g.zones, g.A_wall):
            areas[z] = areas.get(z, 0.0) + a
        areas["hearth_wall"] = g.A_hearth_wall
        areas["hearth_bottom"] = g.A_hearth_bottom
        areas["tuyere_zone"] = 2 * math.pi * g.r_h * 0.6
        for name, wz in self.zones.items():
            Tp = T_by_zone.get(name)
            if Tp is None:
                continue
            res = wz.solve(Tp)
            res["area_m2"] = round(areas.get(name, 0.0), 1)
            res["loss_MW"] = round(res["q_total_kW_m2"] * res["area_m2"] / 1000.0, 3)
            out.append(res)
        return out
