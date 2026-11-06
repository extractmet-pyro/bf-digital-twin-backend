"""
Train the AI layer and run the studies reported in the NeuroMet 2027 abstract.

Steps
 1. physics-informed (PINN) and plain MLP surrogates on the LHS data set
 2. NSGA-II on the PINN surrogate (CO2, cost, productivity) + knee point,
    equal-productivity selection and first-principles verification
 3. GRU thermal-state model on random-excitation trajectories
 4. synthetic Industry-X historian -> gradient-boosted mismatch correction /
    soft sensors for HMT and Si
 5. closed-loop study: open loop vs PI vs NMPC under coke-moisture and
    ore-reducibility disturbances
Artifacts go to app/data/.
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

from app.ai.nn import MLP, r2_scores                                        # noqa: E402
from app.ai.physics import X_NAMES, Y_BAL, Y_KPI, BalanceOperator, DEC       # noqa: E402
from app.ai.gbm import GBM                                                  # noqa: E402
from app.ai.gru import GRUModel                                             # noqa: E402
from app.ai import control as CT                                            # noqa: E402
from app.core.simulate import solve_steady, full_report, kpi_block          # noqa: E402
from app.core.materials import default_inputs, PRESETS, DEFAULT_PRESET      # noqa: E402

R_SCALE = np.array([1.0, 1.5, 0.4, 60.0])        # kmol C, kmol O, kmol Fe, MJ per tHM
LAM_PINN = 0.05


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def train_surrogates(results):
    d = np.load(os.path.join(DATA, "lhs_dataset.npz"))
    X, Y = d["X"], d["Y"]
    cpu = d["cpu"]
    y_names = Y_BAL + Y_KPI
    ih = y_names.index("hmt")
    # operating window: HMT 1420-1660 C (cold, chilled-hearth states with a collapsed cohesive
    # zone are outside the supervisory-control envelope) and no failed / non-producing runs
    ok = (np.all(np.isfinite(Y), axis=1) & (Y[:, ih] > 1420) & (Y[:, ih] < 1660)
          & (Y[:, y_names.index("prod_tpd")] > 3000))
    X, Y = X[ok], Y[ok]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    n = len(X)
    ntr, nva = int(0.8 * n), int(0.1 * n)
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]
    inp = default_inputs(DEFAULT_PRESET)
    op = BalanceOperator(inp)
    nb = len(Y_BAL)
    phys = lambda Xb: op.build(Xb)
    models = {}
    for tag, lam in (("pinn", LAM_PINN), ("plain", 0.0)):
        t0 = time.time()
        m = MLP(X.shape[1], Y.shape[1], hidden=(128, 128, 96), seed=1)
        m.fit(X[tr], Y[tr], X[va], Y[va], epochs=1400, lr=2.5e-3, batch=64, phys=phys if lam > 0 else None,
              lam=lam, bal_idx=list(range(nb)), r_scale=R_SCALE, patience=300, verbose=True, log_every=200)
        P = m.predict(X[te])
        r2 = r2_scores(Y[te], P)
        rb = op.residuals(X[te], P[:, :nb])
        rb_fp = op.residuals(X[te], Y[te][:, :nb])
        models[tag] = m
        results["surrogate_" + tag] = {
            "lambda": lam, "train_s": round(time.time() - t0, 1),
            "r2": {k: round(float(v), 5) for k, v in zip(y_names, r2)},
            "r2_kpi_min": round(float(min(r2[y_names.index(k)] for k in ["hmt", "si_pct", "co2", "cost", "prod_tpd", "raft", "ttop", "dP"])), 5),
            "r2_median": round(float(np.median(r2)), 5),
            "rmse": {k: round(float(np.sqrt(np.mean((Y[te][:, i] - P[:, i]) ** 2))), 4) for i, k in enumerate(y_names)},
            "balance_resid_mean_abs": {b: round(float(np.mean(np.abs(rb[:, i]))), 5) for i, b in enumerate(["C_kmol", "O_kmol", "Fe_kmol", "H_MJ"])},
            "balance_resid_fp_mean_abs": {b: round(float(np.mean(np.abs(rb_fp[:, i]))), 5) for i, b in enumerate(["C_kmol", "O_kmol", "Fe_kmol", "H_MJ"])},
            "history": m.history[::5],
            "parity": {k: {"fp": Y[te][:, y_names.index(k)].round(4).tolist(), "nn": P[:, y_names.index(k)].round(4).tolist()}
                       for k in ["hmt", "si_pct", "co2", "cost", "prod_tpd", "raft", "ttop", "dP", "eta_co"]},
        }
        log(tag, "R2 hmt/si/co2/cost/prod", [round(float(r2[y_names.index(k)]), 4) for k in ["hmt", "si_pct", "co2", "cost", "prod_tpd"]],
            "balance resid", results["surrogate_" + tag]["balance_resid_mean_abs"])
    models["pinn"].save_npz(os.path.join(DATA, "surrogate_pinn.npz"))
    models["plain"].save_npz(os.path.join(DATA, "surrogate_plain.npz"))
    meta = json.load(open(os.path.join(DATA, "lhs_meta.json")))
    t_eval = time.time()
    for _ in range(200):
        models["pinn"].predict(X[:1])
    t_eval = (time.time() - t_eval) / 200
    smeta = {"x_names": X_NAMES, "y_names": y_names, "x_lo": X.min(0).tolist(), "x_hi": X.max(0).tolist(),
             "coke_est_coef": meta["coke_est_coef"], "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
             "fp_cpu_s": meta["fp_cpu_mean_s"], "fp_closure_cpu_s": meta["fp_closure_cpu_mean_s"],
             "nn_eval_s": t_eval, "speedup": meta["fp_cpu_mean_s"] / t_eval, "speedup_closure": meta["fp_closure_cpu_mean_s"] / t_eval}
    json.dump(smeta, open(os.path.join(DATA, "surrogate_meta.json"), "w"), indent=1)
    results["surrogate_meta"] = smeta
    results["dataset"] = {"n_cases": int(n), "n_dropped": int((~ok).sum()), "x_names": X_NAMES,
                          "x_min": X.min(0).round(2).tolist(), "x_max": X.max(0).round(2).tolist()}
    return models


def optimisation_study(results, base_bf):
    from app.ai.surrogate import Surrogate
    sur = Surrogate()
    g = base_bf.geom
    ops = base_bf.ops
    base_x = np.array([ops[k] for k in DEC])
    kb = kpi_block(base_bf)
    from app.core.economics import economics
    eb = economics(base_bf, kb)
    base = {"x": base_x.tolist(), "coke": kb["coke_rate"], "co2": eb["co2"]["total_kg_t"], "cost": eb["cost"]["total_inr_t"],
            "prod": kb["production_tpd"] / g.V_work, "hmt": kb["hmt_C"], "si": kb["hm_Si"], "raft": kb["raft_C"],
            "ttop": kb["top_gas_T_C"], "dP": kb["pressure_drop_kPa"]}
    Fb, Gb, cb, Yb = sur.evaluate(base_x[None], g.V_work)
    base["surrogate"] = {"co2": float(Fb[0, 0]), "cost": float(Fb[0, 1]), "prod": float(-Fb[0, 2]), "coke": float(cb[0])}
    log("base", base)
    res = sur.optimise(g.V_work, pop=100, gens=120, seed=3, base_x=base_x)
    F = res["F"]
    k = res["knee"]
    prod = -F[:, 2]
    # equal productivity: points with productivity >= base (surrogate-consistent)
    pb = base["surrogate"]["prod"]
    eq = np.nonzero(prod >= pb * 0.997)[0]
    k_eq = None
    if len(eq):
        from app.ai.nsga2 import knee_point
        sub = F[eq][:, :2]
        k_eq = int(eq[knee_point(sub)])
    log("front size", len(F), "knee", k, "k_eq", k_eq, "cpu", round(res["cpu_s"], 1))
    out = {"front": [{"x": res["X"][i].round(3).tolist(), "coke": round(float(res["coke"][i]), 2),
                      "co2": round(float(F[i, 0]), 2), "cost": round(float(F[i, 1]), 1), "prod": round(float(prod[i]), 4),
                      "si": round(float(res["Y"][i, sur.yidx["si_pct"]]), 4), "raft": round(float(res["Y"][i, sur.yidx["raft"]]), 1),
                      "ttop": round(float(res["Y"][i, sur.yidx["ttop"]]), 1), "dP": round(float(res["Y"][i, sur.yidx["dP"]]), 1),
                      "hmt": round(float(res["Y"][i, sur.yidx["hmt"]]), 1)} for i in range(len(F))],
           "knee": int(k), "knee_eq_prod": None if k_eq is None else int(k_eq), "nsga_cpu_s": round(res["cpu_s"], 2),
           "history": res["history"][::4], "base": base, "bounds_lo": res["lo"].tolist(), "bounds_hi": res["hi"].tolist(),
           "dec_names": DEC}
    # first-principles verification of the knee and equal-productivity point
    ver = {}
    for tag, idx in (("knee", k), ("eq_prod", k_eq)):
        if idx is None:
            continue
        xin = {"operation": {kk: float(v) for kk, v in zip(DEC, res["X"][idx])}}
        xin["operation"]["coke_rate"] = float(res["coke"][idx])
        t0 = time.time()
        bfv = solve_steady(xin)
        kv = kpi_block(bfv)
        ev = economics(bfv, kv)
        ver[tag] = {"coke": kv["coke_rate"], "co2": ev["co2"]["total_kg_t"], "cost": ev["cost"]["total_inr_t"],
                    "prod": kv["production_tpd"] / g.V_work, "hmt": kv["hmt_C"], "si": kv["hm_Si"], "raft": kv["raft_C"],
                    "ttop": kv["top_gas_T_C"], "dP": kv["pressure_drop_kPa"], "cpu_s": round(time.time() - t0, 1),
                    "dCO2_kg_t": round(ev["co2"]["total_kg_t"] - base["co2"], 1),
                    "dCost_pct": round((ev["cost"]["total_inr_t"] - base["cost"]) / base["cost"] * 100, 2),
                    "dProd_pct": round((kv["production_tpd"] / g.V_work - base["prod"]) / base["prod"] * 100, 2),
                    "surrogate_pred": {"co2": float(F[idx, 0]), "cost": float(F[idx, 1]), "prod": float(prod[idx])}}
        log("verify", tag, ver[tag])
    out["fp_verification"] = ver
    results["optimisation"] = out


def gru_study(results, base_bf):
    t0 = time.time()
    U, Y = CT.gru_runs(base_bf, n_runs=26, hours=48.0, seed=11)
    log("gru data", len(U), "runs", round(time.time() - t0), "s")
    Ut, Yt, Uv, Yv = U[:22], Y[:22], U[22:], Y[22:]
    gm = GRUModel(len(CT.GRU_IN), 2, n_h=32, seed=2)
    t1 = time.time()
    gm.fit(Ut, Yt, Uv, Yv, washout=24, win=96, stride=12, epochs=120, lr=4e-3, batch=16, verbose=True)
    gm.to_npz(os.path.join(DATA, "gru_thermal.npz"))
    # validation metrics on held-out runs (free-run simulation)
    yv_all, yp_all = [], []
    for u, y in zip(Uv, Yv):
        yp = gm.simulate(u)
        yv_all.append(y[24:])
        yp_all.append(yp[24:])
    yv_all = np.concatenate(yv_all)
    yp_all = np.concatenate(yp_all)
    r2 = r2_scores(yv_all, yp_all)
    rmse = np.sqrt(np.mean((yv_all - yp_all) ** 2, axis=0))
    results["gru"] = {"train_s": round(time.time() - t1, 1), "r2_hmt": round(float(r2[0]), 4), "r2_si": round(float(r2[1]), 4),
                      "rmse_hmt": round(float(rmse[0]), 2), "rmse_si": round(float(rmse[1]), 4), "inputs": CT.GRU_IN,
                      "history": gm.history[::4],
                      "val_example": {"u": Uv[0].round(3).tolist(), "y": Yv[0].round(4).tolist(), "yp": gm.simulate(Uv[0]).round(4).tolist()}}
    log("gru", results["gru"]["r2_hmt"], results["gru"]["r2_si"], results["gru"]["rmse_hmt"])
    return gm, U


def softsensor_study(results, base_twin):
    t0 = time.time()
    plant0 = solve_steady(None, params=CT.PLANT_MISMATCH)
    log("plant steady", kpi_block(plant0)["hmt_C"], kpi_block(plant0)["coke_rate"])
    rows, casts = CT.historian(plant0, base_twin, days=18.0, seed=5)
    log("historian rows", len(rows), "casts", len(casts), round(time.time() - t0), "s")
    t_rows = np.array([r["t_h"] for r in rows])
    ci = [int(np.argmin(np.abs(t_rows - c["t_h"]))) for c in casts]
    Xc = CT.build_features(rows, casts, ci)
    yh = np.array([c["HMT_lab"] for c in casts]) - Xc[:, 0]
    ys = np.array([c["Si_lab"] for c in casts]) - Xc[:, 1]
    ncut = int(0.7 * len(casts))
    nval = int(0.5 * len(casts))
    kw = dict(n_estimators=220, learning_rate=0.04, max_depth=2, min_leaf=8, subsample=0.85)
    # model selection on a validation block inside the training period
    gv = GBM(seed=1, **kw).fit(Xc[:nval], yh[:nval])
    use_gbm_h = float(np.sqrt(np.mean((yh[nval:ncut] - gv.predict(Xc[nval:ncut])) ** 2))) < float(np.sqrt(np.mean(yh[nval:ncut] ** 2)))
    gvs = GBM(seed=2, **kw).fit(Xc[:nval], ys[:nval])
    use_gbm_s = float(np.sqrt(np.mean((ys[nval:ncut] - gvs.predict(Xc[nval:ncut])) ** 2))) < float(np.sqrt(np.mean(ys[nval:ncut] ** 2)))
    gh = GBM(seed=1, **kw).fit(Xc[:ncut], yh[:ncut], Xc[ncut:], yh[ncut:])
    gs = GBM(seed=2, **kw).fit(Xc[:ncut], ys[:ncut], Xc[ncut:], ys[ncut:])
    json.dump({"hmt": gh.to_dict(), "si": gs.to_dict(), "features": CT.FEATURES, "use_gbm_hmt": bool(use_gbm_h),
               "use_gbm_si": bool(use_gbm_s)}, open(os.path.join(DATA, "softsensor_gbm.json"), "w"))
    t_cut = casts[ncut]["t_h"]
    idx = [i for i, r in enumerate(rows) if r["t_h"] >= t_cut]
    Xa = CT.build_features(rows, casts, idx)
    gbm_h = Xa[:, 0] + gh.predict(Xa)
    gbm_s = Xa[:, 1] + gs.predict(Xa)
    tru_h = np.array([rows[i]["plant"]["hmt"] for i in idx])
    tru_s = np.array([rows[i]["plant"]["si"] for i in idx])
    open_h = np.array([rows[i]["twin_open"]["hmt"] for i in idx])
    open_s = np.array([rows[i]["twin_open"]["si"] for i in idx])
    twin_h, twin_s = Xa[:, 2], Xa[:, 3]
    bias_h, bias_s = Xa[:, 0], Xa[:, 1]
    last_h = Xa[:, CT.FEATURES.index("last_hmt")]
    last_s = Xa[:, CT.FEATURES.index("last_si")]
    est_h = gbm_h if use_gbm_h else bias_h
    est_s = gbm_s if use_gbm_s else bias_s

    def m(a, b):
        return {"rmse": round(float(np.sqrt(np.mean((a - b) ** 2))), 4), "mae": round(float(np.mean(np.abs(a - b))), 4),
                "r2": round(float(1 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2)), 4)}
    allp = np.array([r["plant"]["hmt"] for r in rows])
    results["softsensor"] = {
        "n_casts": len(casts), "n_train_casts": ncut, "days": 18.0, "mismatch_params": CT.PLANT_MISMATCH,
        "selected": {"hmt": "twin_assim_bias_gbm" if use_gbm_h else "twin_assim_bias", "si": "twin_assim_bias_gbm" if use_gbm_s else "twin_assim_bias"},
        "plant_hmt_stats": {"mean": round(float(allp.mean()), 1), "std": round(float(allp.std()), 1),
                            "min": round(float(allp.min()), 1), "max": round(float(allp.max()), 1)},
        "hmt": {"open_twin": m(open_h, tru_h), "last_cast_hold": m(last_h, tru_h), "twin_assimilated": m(twin_h, tru_h),
                "twin_assim_bias": m(bias_h, tru_h), "twin_assim_bias_gbm": m(gbm_h, tru_h), "soft_sensor": m(est_h, tru_h)},
        "si": {"open_twin": m(open_s, tru_s), "last_cast_hold": m(last_s, tru_s), "twin_assimilated": m(twin_s, tru_s),
               "twin_assim_bias": m(bias_s, tru_s), "twin_assim_bias_gbm": m(gbm_s, tru_s), "soft_sensor": m(est_s, tru_s)},
        "importance_hmt": dict(zip(CT.FEATURES, np.round(gh.importance, 4).tolist())),
        "importance_si": dict(zip(CT.FEATURES, np.round(gs.importance, 4).tolist())),
        "curve_hmt": gh.curve, "curve_si": gs.curve,
        "trace": {"t_h": [round(rows[i]["t_h"], 3) for i in idx][::2], "true_hmt": tru_h[::2].round(2).tolist(),
                  "est_hmt": est_h[::2].round(2).tolist(), "twin_hmt": twin_h[::2].round(2).tolist(),
                  "open_hmt": open_h[::2].round(2).tolist(), "last_hmt": last_h[::2].round(2).tolist(),
                  "true_si": tru_s[::2].round(4).tolist(), "est_si": est_s[::2].round(4).tolist(),
                  "twin_si": twin_s[::2].round(4).tolist()},
        "assimilation": {"t_h": [round(r["t_h"], 2) for r in rows[::4]], "twin_cm": [round(r["twin_cm"], 3) for r in rows[::4]],
                         "plant_cm": [round(r["plant_cm"], 3) for r in rows[::4]], "plant_ri": [round(r["plant_ri"], 4) for r in rows[::4]]},
        "casts": [{"t_h": round(c["t_h"], 3), "HMT_lab": round(c["HMT_lab"], 1), "Si_lab": round(c["Si_lab"], 4)} for c in casts],
        "historian_sample": [{"t_h": round(r["t_h"], 2), **{k: round(v, 4) for k, v in r["meas"].items()},
                              **{"op_" + k: round(v, 3) for k, v in r["ops"].items()}} for r in rows[::8]],
    }
    log("softsensor HMT", results["softsensor"]["hmt"], "Si", results["softsensor"]["si"], "plant", results["softsensor"]["plant_hmt_stats"])
    return plant0


def control_study(results, plant0, gm, base_twin):
    raft_fn, coef = CT.raft_model(base_twin)
    events = [{"t": 3.0, "var": "coke_moisture", "delta": 1.5}, {"t": 14.0, "var": "ore_reducibility", "value": 0.92}]
    hours = 36.0
    sh = results["softsensor"]["hmt"]["soft_sensor"]["rmse"]
    out = {"events": events, "hours": hours, "raft_coef": coef.tolist(),
           "feedback_noise": {"hmt_sd": sh, "si_sd": results["softsensor"]["si"]["soft_sensor"]["rmse"]}}
    for mode in ("open", "pi", "mpc"):
        t0 = time.time()
        traj, cpu = CT.closed_loop(plant0, events, hours, mode, gru=gm, raft_fn=raft_fn, sp=(1500.0, 0.45),
                                   noise=(sh, results["softsensor"]["si"]["soft_sensor"]["rmse"]))
        met = CT.spread_metrics(traj)
        out[mode] = {"metrics": met, "trajectory": traj, "cpu_mean_s": round(float(np.mean(cpu)), 3) if cpu else None,
                     "cpu_max_s": round(float(np.max(cpu)), 3) if cpu else None, "wall_s": round(time.time() - t0, 1)}
        log(mode, met, "cpu", out[mode]["cpu_mean_s"])
    m_pi, m_mpc = out["pi"]["metrics"], out["mpc"]["metrics"]
    out["reduction_vs_pi_pct"] = {"hmt_std": round((1 - m_mpc["hmt_std"] / m_pi["hmt_std"]) * 100, 1),
                                  "si_std": round((1 - m_mpc["si_std"] / m_pi["si_std"]) * 100, 1)}
    results["control"] = out


def main(only=None):
    """only: subset of {"surrogate", "opt", "gru", "soft", "control"}; other results are kept from
    the existing study_results.json."""
    path = os.path.join(DATA, "study_results.json")
    steps = set(only or ["surrogate", "opt", "gru", "soft", "control"])
    results = json.load(open(path)) if (only and os.path.exists(path)) else {}
    results.update({"generated": time.strftime("%Y-%m-%d %H:%M"), "preset": DEFAULT_PRESET})
    log("base steady state")
    base = solve_steady(None)
    rep = full_report(base)
    results["base_kpi"] = rep["kpi"]
    json.dump(rep, open(os.path.join(DATA, "base_report.json"), "w"), default=float)
    save = lambda: json.dump(results, open(path, "w"), default=float)
    if "surrogate" in steps:
        train_surrogates(results)
        save()
    if "opt" in steps:
        optimisation_study(results, base)
        save()
    if "gru" in steps:
        gm, _ = gru_study(results, base)
        save()
    else:
        gm = GRUModel.from_npz(os.path.join(DATA, "gru_thermal.npz"))
    plant0 = None
    if "soft" in steps:
        plant0 = softsensor_study(results, base)
        save()
    if "control" in steps:
        if plant0 is None:
            plant0 = solve_steady(None, params=CT.PLANT_MISMATCH)
        control_study(results, plant0, gm, base)
    t_cycle = results["optimisation"]["nsga_cpu_s"] + (results["control"]["mpc"]["cpu_max_s"] or 0)
    results["timing"] = {"nsga2_s": results["optimisation"]["nsga_cpu_s"],
                         "mpc_solve_mean_s": results["control"]["mpc"]["cpu_mean_s"],
                         "mpc_solve_max_s": results["control"]["mpc"]["cpu_max_s"],
                         "full_cycle_s": round(t_cycle, 2), "control_interval_s": 1800,
                         "fp_steady_s": results["surrogate_meta"]["fp_cpu_s"],
                         "fp_closure_s": results["surrogate_meta"]["fp_closure_cpu_s"],
                         "surrogate_eval_s": results["surrogate_meta"]["nn_eval_s"]}
    save()
    log("done")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
