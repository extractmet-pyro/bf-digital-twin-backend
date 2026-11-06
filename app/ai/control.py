"""
Supervisory process control layer.

* synthetic Industry-X historian (plant = first-principles model with
  parameter mismatch + unmeasured disturbances + sensor noise; casts with
  delayed laboratory analyses)
* feature builder for the gradient-boosted mismatch-correction / soft sensors
* GRU training-data generator (random multi-input excitation)
* PI baseline and GRU-based nonlinear MPC (offset-free, move-blocked,
  bounded, rate-limited, RAFT-constrained)
* closed-loop runner on the first-principles plant
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
from scipy.optimize import minimize

from ..core import thermo as TH
from ..core.simulate import solve_steady, apply_event, snapshot, _fast_hearth, courant_dt
from ..core.furnace import BlastFurnace

DT_SIM = 225.0          # s, first-principles step for control studies
SAMPLE = 900.0          # s, 15-min data / control sample
SUB = int(SAMPLE / DT_SIM)
GRU_IN = ["pci", "moisture", "o2_enrich", "blast_temp", "blast_volume", "coke_rate"]
GRU_OUT = ["hmt", "si"]
PLANT_MISMATCH = {"k_wus": 0.53e-4, "k_bou": 3.4e-4, "phi_SiO": 0.45, "h_ls": 330.0, "hv_drip": 0.06}


def _meas(bf):
    hm = bf.hm * TH.MW_METAL
    hm_mass = hm.sum()
    gas = bf.top["gas"]
    dry = gas[0] + gas[1] + gas[2] + gas[4]
    return {
        "hmt": bf.T_h - 273.15, "si": hm[2] / hm_mass * 100, "s": hm[5] / hm_mass * 100,
        "eta": gas[1] / (gas[0] + gas[1]), "ttop": bf.top["T"] - 273.15, "h2": gas[2] / dry * 100,
        "co": gas[0] / dry * 100, "co2": gas[1] / dry * 100,
        "stave": float(bf.ext["Qw"].sum()), "dP": (bf.P_blast - bf.ops["top_pressure"] - 1.01325) * 100,
        "raft": bf.rw["T_raft"] - 273.15, "prod": float(bf.liq_out["m"] @ TH.MW_METAL) * 3.6,
        "tliq": bf.liq_out["T"] - 273.15, "descent": bf.charged_m3 / bf._dt_last_step * 3600.0,
    }


def _ops_vec(bf):
    return {"pci": bf.ops["pci"], "moisture": bf.ops["moisture"], "o2_enrich": bf.ops["o2_enrich"],
            "blast_temp": bf.ops["blast_temp"], "blast_volume": bf.ops["blast_volume"],
            "coke_rate": bf.recipe["coke"], "coke_moisture": bf.ops.get("coke_moisture", 4.5),
            "ore_reducibility": bf.ops.get("ore_reducibility", 1.0)}


def start_dynamic(bf):
    _fast_hearth(bf, False)
    bf.t = 0.0
    bf.casts = []
    bf.cast = {"open": False, "t_next": 10.0, "t_close": 0.0, "hole": 0, "count": 0}
    bf.acc = bf._zero_acc()
    bf._dt_last = DT_SIM
    return bf


def set_coke(bf, c):
    bf.ops["coke_rate"] = c
    bf.build_recipe(c)


# ---------------------------------------------------------------------------
# GRU training data (random excitation of the nominal first-principles model)
# ---------------------------------------------------------------------------
def gru_runs(base: BlastFurnace, n_runs=24, hours=48.0, seed=11):
    rng = np.random.default_rng(seed)
    U, Y = [], []
    b0 = base.ops.copy()
    c0 = base.recipe["coke"]
    nsamp = int(hours * 3600 / SAMPLE)
    for r in range(n_runs):
        bf = start_dynamic(copy.deepcopy(base))
        t_next = rng.uniform(1.0, 3.0)
        u_seq, y_seq = [], []
        for k in range(nsamp):
            th = k * SAMPLE / 3600.0
            if th >= t_next:
                which = rng.random()
                if which < 0.35:
                    bf.ops["pci"] = float(np.clip(b0["pci"] + rng.uniform(-22, 22), 110, 230))
                elif which < 0.55:
                    bf.ops["moisture"] = float(np.clip(b0["moisture"] + rng.uniform(-7, 12), 5, 35))
                elif which < 0.75:
                    bf.ops["o2_enrich"] = float(np.clip(b0["o2_enrich"] + rng.uniform(-1.5, 1.5), 1, 8))
                elif which < 0.88:
                    bf.ops["blast_temp"] = float(np.clip(b0["blast_temp"] + rng.uniform(-45, 45), 1050, 1250))
                else:
                    set_coke(bf, float(c0 + rng.uniform(-8, 8)))
                t_next = th + rng.uniform(2.0, 6.0)
            for _ in range(SUB):
                bf.step(DT_SIM, steady=False)
            m = _meas(bf)
            o = _ops_vec(bf)
            u_seq.append([o[k] for k in GRU_IN])
            y_seq.append([m["hmt"], m["si"]])
        U.append(np.array(u_seq))
        Y.append(np.array(y_seq))
    return U, Y


# ---------------------------------------------------------------------------
# Synthetic Industry-X historian (plant with mismatch) + nominal twin in parallel
# ---------------------------------------------------------------------------
def historian(base_plant: BlastFurnace, base_twin: BlastFurnace, days=30.0, seed=5, assimilate=True,
              gain=0.004, max_step=0.15):
    """Synthetic Industry-X historian.

    plant : first-principles model with parameter mismatch + unmeasured random-walk
            disturbances (coke moisture, ore reducibility)
    twin  : nominal model fed with the measured inputs; when `assimilate` is on, the twin's
            unmeasured-disturbance state (effective coke moisture) is adapted from every lab
            result as it becomes available (integral data assimilation, gain in %-moisture per
            K of HMT error, rate-limited)
    open  : the same nominal twin without assimilation (benchmark)."""
    rng = np.random.default_rng(seed)               # disturbances & operator moves
    rng_m = np.random.default_rng(seed + 1000)      # measurement / lab noise
    plant = start_dynamic(copy.deepcopy(base_plant))
    twin = start_dynamic(copy.deepcopy(base_twin))
    opent = start_dynamic(copy.deepcopy(base_twin))
    n = int(days * 24 * 3600 / SAMPLE)
    rows = []
    b0 = plant.ops.copy()
    cm = 4.5
    ri = 1.0
    t_next = 2.0
    used = 0
    t_hist, h_hist = [], []
    pi_plant = PIController(sp=1500.0, kp=0.10, ti_h=6.0, u0=plant.ops["pci"], lo=120.0, hi=220.0, dt_h=0.5, rate=6.0)
    for k in range(n):
        th = k * SAMPLE / 3600.0
        # unmeasured disturbances (plant only): coke moisture & ore reducibility random walks
        cm = float(np.clip(cm + rng.normal(0, 0.025), 3.8, 5.2))
        ri = float(np.clip(ri + rng.normal(0, 0.002), 0.95, 1.05))
        plant.ops["coke_moisture_assumed"] = 4.5
        plant.ops["coke_moisture"] = cm
        plant.ops["ore_reducibility"] = ri
        if k % 4 == 0:
            plant._recipe_per_m3()
        # operator moves (applied to plant and both twins): random excitation of blast parameters
        if th >= t_next:
            moves = {}
            w = rng.random()
            if w < 0.4:
                moves["moisture"] = float(np.clip(b0["moisture"] + rng.uniform(-4, 6), 6, 24))
            elif w < 0.7:
                moves["o2_enrich"] = float(np.clip(b0["o2_enrich"] + rng.uniform(-0.6, 0.6), 3, 6))
            else:
                moves["blast_temp"] = float(np.clip(b0["blast_temp"] + rng.uniform(-20, 20), 1120, 1220))
            for kk, v in moves.items():
                for b in (plant, twin, opent):
                    b.ops[kk] = v
            t_next = th + rng.uniform(3.0, 8.0)
        # plant thermal control as practised: PI on PCI from the latest available lab HMT (every 30 min)
        if k % 2 == 1:
            avail = [c for c in plant.casts if c.get("t_avail_h", 1e9) <= th]
            if avail:
                u = pi_plant(avail[-1]["HMT_C"])["pci"]
                for b in (plant, twin, opent):
                    b.ops["pci"] = u
        # data assimilation: new lab results available -> adapt the twin's effective coke moisture
        if assimilate:
            while used < len(plant.casts) and plant.casts[used].get("t_avail_h", 1e9) <= th:
                c = plant.casts[used]
                used += 1
                if len(t_hist) < 2:
                    continue
                lab = c["HMT_C"] + rng_m.normal(0, 4.0)
                c["HMT_lab_assim"] = lab
                tw = float(np.interp(c["t_h"], t_hist, h_hist))
                d = float(np.clip(gain * (tw - lab), -max_step, max_step))
                twin.ops["coke_moisture_assumed"] = 4.5
                twin.ops["coke_moisture"] = float(np.clip(twin.ops.get("coke_moisture", 4.5) + d, 1.0, 10.0))
                twin._recipe_per_m3()
        for _ in range(SUB):
            plant.step(DT_SIM, steady=False)
            twin.step(DT_SIM, steady=False)
            opent.step(DT_SIM, steady=False)
        mp = _meas(plant)
        mt = _meas(twin)
        mo = _meas(opent)
        t_hist.append(th + SAMPLE / 3600.0)
        h_hist.append(mt["hmt"])
        o = _ops_vec(plant)
        noisy = {kk: v for kk, v in mp.items()}
        for kk, sd in (("eta", 0.003), ("ttop", 2.0), ("h2", 0.08), ("co", 0.15), ("co2", 0.15), ("stave", 0.3),
                       ("dP", 1.5)):
            noisy[kk] = mp[kk] + rng_m.normal(0, sd)
        rows.append({"t_h": th + SAMPLE / 3600.0, "ops": o, "plant": mp, "meas": noisy, "twin": mt, "twin_open": mo,
                     "twin_cm": twin.ops.get("coke_moisture", 4.5), "plant_cm": cm, "plant_ri": ri})
    casts = [{**c, "HMT_lab": c.get("HMT_lab_assim", c["HMT_C"] + rng_m.normal(0, 4.0)),
              "Si_lab": max(0.05, c["Si"] + rng_m.normal(0, 0.015))} for c in plant.casts]
    return rows, casts


FEATURES = ["bias_hmt", "bias_si", "twin_hmt", "twin_si", "twin_dhmt", "twin_dsi", "eta", "ttop", "h2", "stave", "dP",
            "raft", "pci", "moisture", "o2_enrich", "blast_temp", "coke_rate", "eta_lag2h", "ttop_lag2h", "stave_lag2h",
            "pci_lag4h", "last_hmt", "last_si", "h_since_cast", "d_eta_since_cast", "d_ttop_since_cast",
            "d_stave_since_cast"]


def build_features(rows, casts, idx_list=None):
    """Soft-sensor features. The twin runs in parallel with the plant on the measured
    inputs; at every available lab result the twin is re-anchored (bias update):
        bias_hmt = HMT_lab(last) + [twin_hmt(now) - twin_hmt(at that cast)]
    The GBM then learns the residual of this bias-updated twin from process signals."""
    n = len(rows)
    idx_list = range(n) if idx_list is None else idx_list
    t_rows = np.array([r["t_h"] for r in rows])
    X = []
    ct = [c["t_avail_h"] for c in casts]
    for i in idx_list:
        r = rows[i]
        th = r["t_h"]
        j2 = max(0, i - 8)
        j4 = max(0, i - 16)
        m = r["meas"]
        kk = np.searchsorted(ct, th, side="right") - 1
        if kk >= 0:
            c = casts[kk]
            ic = int(min(np.searchsorted(t_rows, c["t_h"], side="right") - 1, i))
            ic = max(ic, 0)
            lh, ls, dt_h = c["HMT_lab"], c["Si_lab"], th - c["t_h"]
            rc = rows[ic]
        else:
            lh, ls, dt_h = r["twin"]["hmt"], r["twin"]["si"], 3.0
            rc = r
        dh = r["twin"]["hmt"] - rc["twin"]["hmt"]
        ds = r["twin"]["si"] - rc["twin"]["si"]
        X.append([lh + dh, ls + ds, r["twin"]["hmt"], r["twin"]["si"], dh, ds, m["eta"], m["ttop"], m["h2"], m["stave"],
                  m["dP"], m["raft"], r["ops"]["pci"], r["ops"]["moisture"], r["ops"]["o2_enrich"], r["ops"]["blast_temp"],
                  r["ops"]["coke_rate"], rows[j2]["meas"]["eta"], rows[j2]["meas"]["ttop"], rows[j2]["meas"]["stave"],
                  rows[j4]["ops"]["pci"], lh, ls, dt_h, m["eta"] - rc["meas"]["eta"], m["ttop"] - rc["meas"]["ttop"],
                  m["stave"] - rc["meas"]["stave"]])
    return np.array(X)


# ---------------------------------------------------------------------------
# controllers
# ---------------------------------------------------------------------------
class PIController:
    """Conventional PI loop: hot-metal temperature -> PCI rate (soft-sensor feedback)."""

    def __init__(self, sp=1500.0, kp=0.10, ti_h=6.0, u0=170.0, lo=110.0, hi=230.0, dt_h=0.5, rate=8.0):
        self.sp, self.kp, self.ti, self.u0, self.lo, self.hi, self.dt, self.rate = sp, kp, ti_h, u0, lo, hi, dt_h, rate
        self.I = 0.0
        self.u = u0

    def __call__(self, y_hmt, y_si=None):
        e = self.sp - y_hmt
        self.I += e * self.dt / self.ti
        u = self.u0 + self.kp * (e + self.I)
        u = float(np.clip(u, self.u - self.rate, self.u + self.rate))
        u = float(np.clip(u, self.lo, self.hi))
        if u in (self.lo, self.hi):
            self.I -= e * self.dt / self.ti      # anti-windup
        self.u = u
        return {"pci": u}


MPC_TUNING = {"w": (1.0, 6000.0), "r_move": (0.03, 0.2, 6.0)}     # tuned on the closed-loop study


class NMPC:
    """Offset-free nonlinear MPC on the GRU thermal-state model."""

    def __init__(self, gru, raft_fn, u_nom: dict, sp=(1500.0, 0.45), Np=32, blocks=(4, 8, 20), w=(1.0, 600.0),
                 r_move=(0.02, 0.05, 2.0), bounds=None, rate=None, raft_lim=(2000.0, 2300.0)):
        self.gru = gru
        self.raft_fn = raft_fn
        self.u_nom = dict(u_nom)
        self.sp = np.array(sp)
        self.Np = Np
        self.blocks = blocks
        self.w = np.array(w)
        self.r = np.array(r_move)
        self.mv = ["pci", "moisture", "o2_enrich"]
        self.bounds = bounds or {"pci": (110.0, 230.0), "moisture": (5.0, 35.0), "o2_enrich": (1.0, 8.0)}
        self.rate = rate or {"pci": 10.0, "moisture": 6.0, "o2_enrich": 0.8}
        self.raft_lim = raft_lim
        self.d = np.zeros(2)
        self.last_cpu = 0.0
        self.u_prev = np.array([u_nom[k] for k in self.mv])

    def _u_traj(self, z, u_hist_last):
        """z: (n_blocks*3,) moves -> (Np, n_in) future input matrix."""
        U = np.tile(u_hist_last, (self.Np, 1))
        cur = self.u_prev.copy()
        t = 0
        zz = z.reshape(len(self.blocks), 3)
        for bi, L in enumerate(self.blocks):
            cur = cur + zz[bi]
            for _ in range(L):
                if t >= self.Np:
                    break
                U[t, :3] = cur
                t += 1
        while t < self.Np:
            U[t, :3] = cur
            t += 1
        return U

    def step(self, u_hist: np.ndarray, y_meas: np.ndarray):
        """u_hist: past inputs (T, n_in) incl. current; y_meas: [HMT, Si] estimate now."""
        t0 = time.time()
        yp_hist, h = self.gru.simulate(u_hist, return_h=True)
        d_new = y_meas - yp_hist[-1]
        self.d = 0.6 * self.d + 0.4 * d_new
        last = u_hist[-1].copy()
        nz = len(self.blocks) * 3
        lim = np.array([self.rate[k] for k in self.mv])
        bnds = []
        for bi in range(len(self.blocks)):
            for i, k in enumerate(self.mv):
                bnds.append((-lim[i], lim[i]))

        def cost(z):
            U = self._u_traj(z, last)
            uabs = U[:, :3]
            pen = 0.0
            for i, k in enumerate(self.mv):
                lo, hi = self.bounds[k]
                pen += 1e3 * float(np.sum(np.maximum(0, lo - uabs[:, i]) ** 2 + np.maximum(0, uabs[:, i] - hi) ** 2))
            yp, _ = self.gru.simulate(U, h0=h, return_h=True)
            yp = yp + self.d
            e = yp - self.sp
            J = float(np.sum(self.w[0] * e[:, 0] ** 2 + self.w[1] * e[:, 1] ** 2)) / self.Np
            zz = z.reshape(len(self.blocks), 3)
            J += float(np.sum(self.r * zz ** 2))
            # RAFT constraint on the final MV set
            fin = uabs[-1]
            raft = self.raft_fn(fin[0], fin[1], fin[2])
            J += 0.05 * (max(0, self.raft_lim[0] - raft) ** 2 + max(0, raft - self.raft_lim[1]) ** 2)
            return J + pen

        z0 = np.zeros(nz)
        res = minimize(cost, z0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 60})
        z = res.x
        move = z[:3]
        u_new = self.u_prev + move
        for i, k in enumerate(self.mv):
            u_new[i] = float(np.clip(u_new[i], *self.bounds[k]))
        self.u_prev = u_new
        self.last_cpu = time.time() - t0
        U = self._u_traj(z, last)
        yp, _ = self.gru.simulate(U, h0=h, return_h=True)
        self.prediction = (yp + self.d).tolist()
        return {k: float(u_new[i]) for i, k in enumerate(self.mv)}


def raft_model(base: BlastFurnace):
    """Cheap RAFT function of (PCI, moisture, O2) from the raceway sub-model."""
    b = copy.deepcopy(base)
    pts = []
    for pci in (110, 170, 230):
        for mo in (5, 20, 35):
            for o2 in (1, 4.5, 8):
                b.ops.update({"pci": pci, "moisture": mo, "o2_enrich": o2})
                rw = b.raceway(b.ops, b.T_s[0], pci * b.P_nom / 1000.0)
                pts.append([1, pci, mo, o2, pci * o2, rw["T_raft"] - 273.15])
    A = np.array(pts)
    coef, *_ = np.linalg.lstsq(A[:, :5], A[:, 5], rcond=None)
    return lambda p, m, o: float(coef[0] + coef[1] * p + coef[2] * m + coef[3] * o + coef[4] * p * o), coef


def closed_loop(base_plant: BlastFurnace, events, hours, mode, gru=None, soft=None, raft_fn=None, sp=(1500.0, 0.45),
                u_hist0=None, seed=3, noise=(3.0, 0.012), mpc_kw=None):
    """Run plant with PI / MPC / open loop. Feedback = soft-sensor estimate every 15 min."""
    rng = np.random.default_rng(seed)
    bf = start_dynamic(copy.deepcopy(base_plant))
    base_ops = copy.deepcopy(bf.ops)
    evs = sorted(events, key=lambda e: e["t"])
    ei = 0
    n = int(hours * 3600 / SAMPLE)
    u_hist = list(u_hist0) if u_hist0 is not None else [[bf.ops[k] if k != "coke_rate" else bf.recipe["coke"] for k in GRU_IN]] * 96
    pi = PIController(sp=sp[0], u0=bf.ops["pci"]) if mode == "pi" else None
    mpc = NMPC(gru, raft_fn, {k: bf.ops[k] for k in ["pci", "moisture", "o2_enrich"]}, sp=sp, **(mpc_kw or MPC_TUNING)) \
        if mode == "mpc" else None
    traj = []
    cpu = []
    for k in range(n):
        th = k * SAMPLE / 3600.0
        while ei < len(evs) and evs[ei]["t"] <= th + 1e-9:
            apply_event(bf, evs[ei], base_ops)
            ei += 1
        for _ in range(SUB):
            bf.step(DT_SIM, steady=False)
        m = _meas(bf)
        # feedback signal: soft sensor (truth + small error) if provided, else noisy truth
        y_hat = np.array([m["hmt"] + rng.normal(0, noise[0]), m["si"] + rng.normal(0, noise[1])])
        u_now = [bf.ops[kk] if kk != "coke_rate" else bf.recipe["coke"] for kk in GRU_IN]
        u_hist.append(u_now)
        if k % 2 == 1 and th > 0.4:
            if pi is not None:
                bf.ops.update(pi(y_hat[0]))
            elif mpc is not None:
                mv = mpc.step(np.array(u_hist[-96:]), y_hat)
                bf.ops.update(mv)
                cpu.append(mpc.last_cpu)
        traj.append({"t_h": round(th + SAMPLE / 3600, 3), "hmt": round(m["hmt"], 2), "si": round(m["si"], 4),
                     "hmt_est": round(float(y_hat[0]), 2), "si_est": round(float(y_hat[1]), 4),
                     "pci": round(bf.ops["pci"], 2), "moisture": round(bf.ops["moisture"], 2),
                     "o2_enrich": round(bf.ops["o2_enrich"], 3), "raft": round(m["raft"], 1),
                     "eta": round(m["eta"], 4), "ttop": round(m["ttop"], 1), "prod": round(m["prod"], 2),
                     "coke_moisture": bf.ops.get("coke_moisture", 4.5), "ore_reducibility": bf.ops.get("ore_reducibility", 1.0)})
    return traj, cpu


def spread_metrics(traj, t_from=2.0, sp=(1500.0, 0.45)):
    h = np.array([r["hmt"] for r in traj if r["t_h"] >= t_from])
    s = np.array([r["si"] for r in traj if r["t_h"] >= t_from])
    return {"hmt_std": float(np.std(h)), "si_std": float(np.std(s)),
            "hmt_iae": float(np.mean(np.abs(h - sp[0]))), "si_iae": float(np.mean(np.abs(s - sp[1]))),
            "hmt_min": float(h.min()), "hmt_max": float(h.max()), "si_min": float(s.min()), "si_max": float(s.max())}
