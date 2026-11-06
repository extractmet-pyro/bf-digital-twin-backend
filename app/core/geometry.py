"""
Blast-furnace geometry: inner profile, working volume, discretisation of the
shaft into axial cells, zone labels, hearth dimensions and drawing data for
the 2-D schematic and the 3-D model.

Vertical coordinate z is measured upward from the tuyere level (z = 0) to the
stock line (z = H).  The hearth lies below z = 0.
"""
from __future__ import annotations

import math
import numpy as np


class Geometry:
    def __init__(self, g: dict, ncell: int = 40):
        self.g = dict(g)
        self.r_h = g["d_hearth"] / 2.0
        self.r_b = g["d_belly"] / 2.0
        self.r_t = g["d_throat"] / 2.0
        self.h_bosh = g["h_bosh"]
        self.h_belly = g["h_belly"]
        self.h_throat = g["h_throat"]
        self.h_hearth = g["h_hearth"]          # taphole -> tuyere
        self.h_sump = g["h_sump"]              # hearth bottom -> taphole
        if g.get("h_stack"):
            self.h_stack = float(g["h_stack"])
        else:
            self.h_stack = self._solve_stack_height(g["V_work"])
        self.H = self.h_bosh + self.h_belly + self.h_stack + self.h_throat
        self.z_bosh_top = self.h_bosh
        self.z_belly_top = self.h_bosh + self.h_belly
        self.z_stack_top = self.z_belly_top + self.h_stack
        self.n_tuyere = int(g["n_tuyere"])
        self.d_tuyere = float(g["d_tuyere"])
        self.n_taphole = int(g["n_taphole"])
        self.ncell = ncell
        self._discretise(ncell)

    # ------------------------------------------------------------------
    def _frustum(self, r1, r2, h):
        return math.pi * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2)

    def _solve_stack_height(self, V):
        v_fixed = (self._frustum(self.r_h, self.r_b, self.h_bosh)
                   + math.pi * self.r_b ** 2 * self.h_belly
                   + math.pi * self.r_t ** 2 * self.h_throat)
        per_m = math.pi / 3.0 * (self.r_b ** 2 + self.r_b * self.r_t + self.r_t ** 2)
        return max(4.0, (V - v_fixed) / per_m)

    def radius(self, z):
        """Inner (hot-face) radius at height z above tuyere level."""
        z = np.asarray(z, dtype=float)
        r = np.where(z <= self.z_bosh_top, self.r_h + (self.r_b - self.r_h) * np.clip(z, 0, None) / self.h_bosh,
            np.where(z <= self.z_belly_top, self.r_b,
            np.where(z <= self.z_stack_top,
                     self.r_b - (self.r_b - self.r_t) * (z - self.z_belly_top) / self.h_stack, self.r_t)))
        return np.where(z < 0, self.r_h, r)

    def zone_of(self, z):
        if z < 0:
            return "hearth_wall"
        if z <= self.z_bosh_top:
            return "bosh"
        if z <= self.z_belly_top:
            return "belly"
        if z <= self.z_belly_top + 0.5 * self.h_stack:
            return "lower_stack"
        if z <= self.z_stack_top:
            return "upper_stack"
        return "throat"

    def _discretise(self, n):
        self.dz = self.H / n
        self.z_face = np.linspace(0.0, self.H, n + 1)
        self.z_mid = 0.5 * (self.z_face[1:] + self.z_face[:-1])
        V = np.zeros(n)
        Aw = np.zeros(n)
        for j in range(n):
            zs = np.linspace(self.z_face[j], self.z_face[j + 1], 41)
            r = self.radius(zs)
            V[j] = np.trapezoid(math.pi * r * r, zs)
            drdz = np.gradient(r, zs)
            Aw[j] = np.trapezoid(2 * math.pi * r * np.sqrt(1 + drdz ** 2), zs)
        self.V_cell = V
        self.A_wall = Aw
        self.A_cross = math.pi * self.radius(self.z_mid) ** 2
        self.zones = [self.zone_of(z) for z in self.z_mid]
        self.V_work = float(V.sum())
        # hearth
        self.A_hearth = math.pi * self.r_h ** 2
        self.V_hearth = self.A_hearth * (self.h_hearth + self.h_sump)
        self.A_hearth_wall = 2 * math.pi * self.r_h * (self.h_hearth + self.h_sump)
        self.A_hearth_bottom = self.A_hearth
        self.V_inner = self.V_work + self.A_hearth * self.h_hearth

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "working_volume_m3": round(self.V_work, 1),
            "inner_volume_m3": round(self.V_inner, 1),
            "working_height_m": round(self.H, 2),
            "total_height_m": round(self.H + self.h_hearth + self.h_sump, 2),
            "h_stack_m": round(self.h_stack, 2),
            "hearth_area_m2": round(self.A_hearth, 2),
            "hearth_volume_m3": round(self.V_hearth, 1),
            "belly_area_m2": round(math.pi * self.r_b ** 2, 2),
            "throat_area_m2": round(math.pi * self.r_t ** 2, 2),
            "bosh_angle_deg": round(math.degrees(math.atan2(self.h_bosh, self.r_b - self.r_h)), 1),
            "stack_angle_deg": round(math.degrees(math.atan2(self.h_stack, self.r_b - self.r_t)), 1),
            "wall_area_m2": round(float(self.A_wall.sum()) + self.A_hearth_wall, 1),
            "n_tuyere": self.n_tuyere, "d_tuyere_m": self.d_tuyere, "n_taphole": self.n_taphole,
            "d_hearth_m": round(2 * self.r_h, 2), "d_belly_m": round(2 * self.r_b, 2),
            "d_throat_m": round(2 * self.r_t, 2),
            "h_bosh_m": self.h_bosh, "h_belly_m": self.h_belly, "h_throat_m": self.h_throat,
            "h_hearth_m": self.h_hearth, "h_sump_m": self.h_sump,
        }

    def drawing(self, lining: dict | None = None) -> dict:
        """Coordinates for the schematic: inner profile, lining thickness per zone."""
        pts = [(-self.h_hearth - self.h_sump, self.r_h), (0.0, self.r_h), (self.z_bosh_top, self.r_b),
               (self.z_belly_top, self.r_b), (self.z_stack_top, self.r_t), (self.H, self.r_t)]
        thick = {}
        if lining:
            for zname, spec in lining.items():
                thick[zname] = round(sum(t for _, t in spec["layers"]), 3)
        return {
            "profile": [{"z": round(z, 3), "r": round(r, 3)} for z, r in pts],
            "levels": {
                "hearth_bottom": round(-self.h_hearth - self.h_sump, 3), "taphole": round(-self.h_hearth, 3),
                "tuyere": 0.0, "bosh_top": round(self.z_bosh_top, 3), "belly_top": round(self.z_belly_top, 3),
                "stack_top": round(self.z_stack_top, 3), "stock_line": round(self.H, 3),
            },
            "lining_thickness": thick,
            "cells": {"z_mid": self.z_mid.round(3).tolist(), "r_mid": self.radius(self.z_mid).round(3).tolist(),
                      "zone": self.zones},
            "summary": self.summary(),
        }
