"""
Generate the Latin-hypercube steady-state data set from the first-principles model.

Pass 1: thermal-closure solves at a small LHS design -> linear model of the closure
        coke rate c_est(x).
Pass 2: fixed-coke steady states at a large LHS design with coke = c_est(x) + U(-d, d).
Outputs app/data/lhs_dataset.npz (+ json meta).
"""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from app.core.simulate import solve_steady, converge, kpi_block            # noqa: E402
from app.core.economics import economics                                    # noqa: E402
from app.core.furnace import rescale_coke                                   # noqa: E402
from app.core.materials import PRESETS, DEFAULT_PRESET                     # noqa: E402
from app.ai.physics import DEC, X_NAMES, Y_BAL, Y_KPI, bounds_for, per_tHM_flows   # noqa: E402

BASE = None


def lhs(n, d, rng):
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return u


def init_worker():
    global BASE
    BASE = solve_steady(None)


def apply_x(b, x):
    for i, k in enumerate(DEC):
        b.ops[k] = float(x[i])
    b.P_nom = b._estimate_production()


def collect(b):
    f = per_tHM_flows(b)
    k = kpi_block(b)
    e = economics(b, k)
    out = {kk: float(f[kk]) for kk in Y_BAL}
    out.update({
        "hmt": k["hmt_C"], "ttop": k["top_gas_T_C"], "raft": k["raft_C"], "dP": k["pressure_drop_kPa"],
        "prod_tpd": k["production_tpd"], "eta_co": k["eta_CO"], "si_pct": k["hm_Si"], "s_pct": k["hm_S"],
        "c_pct": k["hm_C"], "slag_rate": k["slag_rate"], "slag_B2": k["slag_B2"], "slag_Al2O3": k["slag_Al2O3"],
        "slag_MgO": k["slag_MgO"], "co2": e["co2"]["total_kg_t"], "cost": e["cost"]["total_inr_t"],
        "heat_loss_MW": k["heat_loss_MW"], "sol_loss": k["solution_loss_C"], "burnout": k["pci_burnout_pct"],
        "cz_top": k["cz_top_m"], "cz_bot": k["cz_bottom_m"], "tg_lhv": k["top_gas_LHV_MJ_nm3"],
        "o2_nm3": k["oxygen_nm3_per_t"], "tg_nm3": k["top_gas_nm3_t_dry"], "fuel_rate": k["fuel_rate"],
        "trz_T": k["trz_T_C"],
    })
    x_act = [b.ops[kk] for kk in DEC] + [k["coke_rate"]]
    x_act[3] = k["pci_rate"]
    return x_act, out


def run_closure(x):
    b = copy.deepcopy(BASE)
    apply_x(b, x)
    t0 = time.time()
    b = solve_steady(bf=b, closure=True)
    xa, out = collect(b)
    return {"x": xa, "y": out, "coke_set": b.recipe["coke"], "cpu": time.time() - t0}


def run_fixed(args):
    x, coke = args
    b = copy.deepcopy(BASE)
    apply_x(b, x)
    t0 = time.time()
    rescale_coke(b, coke)
    steps = converge(b, max_steps=600)
    xa, out = collect(b)
    return {"x": xa, "y": out, "coke_set": coke, "cpu": time.time() - t0, "steps": steps}


def main(n1=20, n2=900, seed=7, procs=2):
    rng = np.random.default_rng(seed)
    pre = PRESETS[DEFAULT_PRESET]
    bnd = bounds_for(pre)
    lo = np.array([bnd[k][0] for k in DEC])
    hi = np.array([bnd[k][1] for k in DEC])
    t0 = time.time()
    X1 = lo + lhs(n1, len(DEC), rng) * (hi - lo)
    with mp.Pool(procs, initializer=init_worker) as pool:
        res1 = pool.map(run_closure, list(X1))
        print(f"pass1 closures done in {time.time() - t0:.0f}s", flush=True)
        # linear fit of closure coke setpoint vs x
        Xc = np.array([r["x"][:6] for r in res1])
        cs = np.array([r["coke_set"] for r in res1])
        hm = np.array([r["y"]["hmt"] for r in res1])
        M = np.hstack([np.ones((len(Xc), 1)), Xc])
        coef, *_ = np.linalg.lstsq(M, cs, rcond=None)
        resid = cs - M @ coef
        print("c_est coef", coef.round(4).tolist(), "rmse", float(np.sqrt(np.mean(resid ** 2))), "hmt", hm.round(0).tolist(), flush=True)
        X2 = lo + lhs(n2, len(DEC), rng) * (hi - lo)
        c2 = np.hstack([np.ones((n2, 1)), X2]) @ coef + rng.uniform(-16, 16, n2)
        t1 = time.time()
        res2 = []
        for i, r in enumerate(pool.imap(run_fixed, list(zip(list(X2), list(c2))), chunksize=4)):
            res2.append(r)
            if (i + 1) % 50 == 0:
                print(f"  pass2 {i + 1}/{n2}  {time.time() - t1:.0f}s", flush=True)
    allres = res1 + res2
    X = np.array([r["x"] for r in allres])
    Y = np.array([[r["y"][k] for k in Y_BAL + Y_KPI] for r in allres])
    cpu = np.array([r["cpu"] for r in allres])
    out = os.path.join(os.path.dirname(HERE), "app", "data")
    np.savez_compressed(os.path.join(out, "lhs_dataset.npz"), X=X, Y=Y, cpu=cpu, closure=np.array([1] * n1 + [0] * n2))
    meta = {"x_names": X_NAMES, "y_bal": Y_BAL, "y_kpi": Y_KPI, "n_closure": n1, "n_fixed": n2,
            "coke_est_coef": coef.tolist(), "bounds": {k: list(v) for k, v in bnd.items()},
            "fp_cpu_mean_s": float(cpu[n1:].mean()), "fp_closure_cpu_mean_s": float(cpu[:n1].mean()),
            "generated_s": time.time() - t0, "seed": seed}
    with open(os.path.join(out, "lhs_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("saved", X.shape, Y.shape, "total", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    n2 = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    main(n2=n2)
