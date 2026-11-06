"""
Surrogate-based supervisory optimiser.

* PI-NN surrogate y = f(x) of the first-principles steady state
  (x = 6 decision variables + coke rate).
* Thermal closure on the surrogate: for every candidate the coke rate is solved
  so that the predicted hot-metal temperature equals the target (vectorised
  bisection), i.e. coke rate "follows from" the thermal state.
* NSGA-II problem definition: minimise CO2 intensity and hot-metal cost,
  maximise productivity, subject to Si, HMT, RAFT, top-gas temperature,
  pressure-drop and slag-chemistry limits.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .nn import MLP
from .physics import DEC, X_NAMES, Y_NAMES, Y_BAL, Y_KPI, BalanceOperator
from .nsga2 import nsga2, knee_point

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DEFAULT_LIMITS = {
    "si_min": 0.30, "si_max": 0.70, "hmt_min": 1480.0, "hmt_max": 1520.0, "raft_min": 2000.0, "raft_max": 2300.0,
    "ttop_min": 100.0, "ttop_max": 250.0, "dP_max": 175.0, "b2_min": 1.0, "b2_max": 1.2, "al2o3_max": 20.0,
    "mgo_min": 7.0, "mgo_max": 10.0, "coke_min": 280.0,
}


class Surrogate:
    def __init__(self, path=None):
        path = path or os.path.join(DATA, "surrogate_pinn.npz")
        self.model = MLP.load_npz(path)
        with open(os.path.join(DATA, "surrogate_meta.json")) as fh:
            self.meta = json.load(fh)
        self.coef = np.array(self.meta["coke_est_coef"])
        self.xlo = np.array(self.meta["x_lo"])
        self.xhi = np.array(self.meta["x_hi"])
        self.yidx = {k: i for i, k in enumerate(self.meta["y_names"])}

    def predict(self, X):
        return self.model.predict(np.atleast_2d(X))

    def c_est(self, X6):
        X6 = np.atleast_2d(X6)
        return np.hstack([np.ones((len(X6), 1)), X6]) @ self.coef

    def closure(self, X6, target=1500.0, iters=28):
        """Solve coke rate (per row) so that predicted HMT = target."""
        X6 = np.atleast_2d(X6).astype(float)
        c0 = self.c_est(X6)
        lo = np.maximum(c0 - 32.0, self.xlo[6])
        hi = np.minimum(c0 + 32.0, self.xhi[6])
        ih = self.yidx["hmt"]
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            y = self.predict(np.hstack([X6, mid[:, None]]))[:, ih]
            up = y < target
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        c = 0.5 * (lo + hi)
        Y = self.predict(np.hstack([X6, c[:, None]]))
        return c, Y

    def evaluate(self, X6, V_work, limits=None, target=1500.0):
        L = dict(DEFAULT_LIMITS)
        L.update(limits or {})
        c, Y = self.closure(X6, target)
        g = lambda k: Y[:, self.yidx[k]]
        prod = g("prod_tpd") / V_work
        F = np.column_stack([g("co2"), g("cost"), -prod])
        G = np.column_stack([
            L["si_min"] - g("si_pct"), g("si_pct") - L["si_max"],
            (L["hmt_min"] - g("hmt")) / 100, (g("hmt") - L["hmt_max"]) / 100,
            (L["raft_min"] - g("raft")) / 100, (g("raft") - L["raft_max"]) / 100,
            (L["ttop_min"] - g("ttop")) / 100, (g("ttop") - L["ttop_max"]) / 100,
            (g("dP") - L["dP_max"]) / 10,
            L["b2_min"] - g("slag_B2"), g("slag_B2") - L["b2_max"],
            g("slag_Al2O3") - L["al2o3_max"],
            (L["coke_min"] - c) / 10,
        ])
        return F, G, c, Y

    def optimise(self, V_work, bounds=None, limits=None, pop=100, gens=120, seed=3, target=1500.0, base_x=None):
        lo = self.xlo[:6].copy()
        hi = self.xhi[:6].copy()
        if bounds:
            for i, k in enumerate(DEC):
                if k in bounds:
                    lo[i], hi[i] = bounds[k]
        t0 = time.time()
        res = nsga2(lambda X: self.evaluate(X, V_work, limits, target)[:2], lo, hi, pop=pop, gens=gens, seed=seed,
                    init=None if base_x is None else np.atleast_2d(base_x))
        X = res["X"][res["front"]]
        F, G, c, Y = self.evaluate(X, V_work, limits, target)
        order = np.argsort(F[:, 0])
        X, F, G, c, Y = X[order], F[order], G[order], c[order], Y[order]
        k = knee_point(F)
        return {"X": X, "F": F, "coke": c, "Y": Y, "knee": k, "history": res["history"],
                "cpu_s": time.time() - t0, "lo": lo, "hi": hi}


def balance_violation(sur: Surrogate, X, Y, inputs):
    """Balance residuals (C, O, Fe, H) of surrogate predictions."""
    op = BalanceOperator(inputs)
    nb = len(Y_BAL)
    return op.residuals(np.atleast_2d(X), np.atleast_2d(Y)[:, :nb])
