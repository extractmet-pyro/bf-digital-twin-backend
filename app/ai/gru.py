"""
Recurrent neural model of the blast-furnace thermal state (NumPy GRU).

Simulation (output-error) structure: the GRU is driven by the manipulated and
measured inputs only; outputs are hot-metal temperature and silicon.  Trained
with truncated back-propagation through time and Adam on step-response /
random-excitation trajectories of the first-principles dynamic model.
"""
from __future__ import annotations

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class GRUModel:
    def __init__(self, n_in, n_out, n_h=32, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(n_h)
        self.P = {
            "Wz": rng.uniform(-s, s, (n_in, n_h)), "Uz": rng.uniform(-s, s, (n_h, n_h)), "bz": np.zeros(n_h),
            "Wr": rng.uniform(-s, s, (n_in, n_h)), "Ur": rng.uniform(-s, s, (n_h, n_h)), "br": np.zeros(n_h),
            "Wn": rng.uniform(-s, s, (n_in, n_h)), "Un": rng.uniform(-s, s, (n_h, n_h)), "bn": np.zeros(n_h),
            "bhn": np.zeros(n_h), "Wo": rng.uniform(-s, s, (n_h, n_out)), "bo": np.zeros(n_out),
        }
        self.n_h = n_h
        self.um = self.us = self.ym = self.ys = None

    # ---------------------------------------------------------------
    def _forward(self, X, h0=None, keep=False):
        P = self.P
        B, T, _ = X.shape
        h = np.zeros((B, self.n_h)) if h0 is None else h0
        Y = np.zeros((B, T, P["Wo"].shape[1]))
        cache = [] if keep else None
        for t in range(T):
            x = X[:, t]
            z = _sig(x @ P["Wz"] + h @ P["Uz"] + P["bz"])
            r = _sig(x @ P["Wr"] + h @ P["Ur"] + P["br"])
            hn = h @ P["Un"] + P["bhn"]
            n = np.tanh(x @ P["Wn"] + P["bn"] + r * hn)
            hnew = (1 - z) * n + z * h
            if keep:
                cache.append((x, h, z, r, n, hn))
            h = hnew
            Y[:, t] = h @ P["Wo"] + P["bo"]
        return Y, h, cache

    def _backward(self, cache, dY, H):
        P = self.P
        g = {k: np.zeros_like(v) for k, v in P.items()}
        dh_next = np.zeros_like(cache[0][1])
        T = len(cache)
        for t in range(T - 1, -1, -1):
            x, hp, z, r, n, hn = cache[t]
            ht = H[t]
            dy = dY[:, t]
            g["Wo"] += ht.T @ dy
            g["bo"] += dy.sum(0)
            dh = dy @ P["Wo"].T + dh_next
            dz = dh * (hp - n)
            dn = dh * (1 - z)
            dhp = dh * z
            dan = dn * (1 - n * n)
            g["Wn"] += x.T @ dan
            g["bn"] += dan.sum(0)
            dr = dan * hn
            dhn = dan * r
            g["Un"] += hp.T @ dhn
            g["bhn"] += dhn.sum(0)
            dhp += dhn @ P["Un"].T
            dar = dr * r * (1 - r)
            g["Wr"] += x.T @ dar
            g["Ur"] += hp.T @ dar
            g["br"] += dar.sum(0)
            dhp += dar @ P["Ur"].T
            daz = dz * z * (1 - z)
            g["Wz"] += x.T @ daz
            g["Uz"] += hp.T @ daz
            g["bz"] += daz.sum(0)
            dhp += daz @ P["Uz"].T
            dh_next = dhp
        return g

    # ---------------------------------------------------------------
    def fit(self, U, Y, Uv=None, Yv=None, washout=24, win=96, stride=24, epochs=150, lr=3e-3, batch=16,
            clip=1.0, seed=0, verbose=False):
        """U: list of (T_i, n_in) sequences; Y: list of (T_i, n_out)."""
        rng = np.random.default_rng(seed)
        Ucat = np.concatenate(U)
        Ycat = np.concatenate(Y)
        self.um, self.us = Ucat.mean(0), Ucat.std(0) + 1e-9
        self.ym, self.ys = Ycat.mean(0), Ycat.std(0) + 1e-9
        Xw, Yw = [], []
        for u, y in zip(U, Y):
            us = (u - self.um) / self.us
            ys = (y - self.ym) / self.ys
            for s in range(0, len(u) - win + 1, stride):
                Xw.append(us[s:s + win])
                Yw.append(ys[s:s + win])
        Xw = np.array(Xw)
        Yw = np.array(Yw)
        m = {k: np.zeros_like(v) for k, v in self.P.items()}
        v = {k: np.zeros_like(v) for k, v in self.P.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        best = (np.inf, None)
        self.history = []
        for ep in range(epochs):
            perm = rng.permutation(len(Xw))
            lr_ep = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs))) + 1e-5
            tot = 0.0
            for s in range(0, len(Xw), batch):
                idx = perm[s:s + batch]
                Xb, Yb = Xw[idx], Yw[idx]
                Yp, _, cache = self._forward(Xb, keep=True)
                H = []
                h = np.zeros((len(idx), self.n_h))
                for c in cache:
                    x, hp, z, r, n, hn = c
                    h = (1 - z) * n + z * hp
                    H.append(h)
                err = Yp - Yb
                err[:, :washout] = 0.0
                cnt = err[:, washout:].size
                dY = 2 * err / cnt
                tot += float((err[:, washout:] ** 2).sum()) / cnt * len(idx)
                g = self._backward(cache, dY, H)
                gn = np.sqrt(sum(float((gg ** 2).sum()) for gg in g.values()))
                if gn > clip:
                    for k in g:
                        g[k] *= clip / gn
                t += 1
                for k in self.P:
                    m[k] = b1 * m[k] + (1 - b1) * g[k]
                    v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k]
                    self.P[k] -= lr_ep * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
            rec = {"epoch": ep, "train_mse": tot / len(Xw)}
            if Uv is not None:
                vm = self.evaluate(Uv, Yv, washout)
                rec["val_mse"] = vm
                if vm < best[0]:
                    best = (vm, {k: vv.copy() for k, vv in self.P.items()})
            self.history.append(rec)
            if verbose and ep % 10 == 0:
                print("   gru ep", ep, rec, flush=True)
        if best[1] is not None:
            self.P = best[1]
        return self

    def evaluate(self, U, Y, washout=24):
        errs = []
        for u, y in zip(U, Y):
            yp = self.simulate(u)
            errs.append(((yp[washout:] - y[washout:]) / self.ys) ** 2)
        return float(np.mean(np.concatenate(errs)))

    def simulate(self, u, h0=None, return_h=False):
        X = ((np.asarray(u) - self.um) / self.us)[None]
        Yp, h, _ = self._forward(X, h0=h0)
        y = Yp[0] * self.ys + self.ym
        return (y, h) if return_h else y

    def state(self, u_hist):
        _, h = self.simulate(u_hist, return_h=True)
        return h

    def to_npz(self, path):
        np.savez_compressed(path, **{k: v for k, v in self.P.items()}, um=self.um, us=self.us, ym=self.ym, ys=self.ys,
                            n_h=np.array(self.n_h))

    @classmethod
    def from_npz(cls, path):
        d = np.load(path)
        m = cls.__new__(cls)
        m.P = {k: d[k] for k in ["Wz", "Uz", "bz", "Wr", "Ur", "br", "Wn", "Un", "bn", "bhn", "Wo", "bo"]}
        m.um, m.us, m.ym, m.ys = d["um"], d["us"], d["ym"], d["ys"]
        m.n_h = int(d["n_h"])
        return m
