"""
Synthetic data generator for the AGD-Sentinel deep surrogate (v2.0).

Physical oracle = v1.0 infinite-slope engine extended with a hydro-climatic forcing,
a damage clock and a post-failure recovery law (AGD regime):
  E1  FS = [c + (g z_e cos^2 b (1-r_u) - kh g z_e sin b cos b) tan phi]
           / [g z_e sin b cos b + kh g z_e cos^2 b]          (r_u=0, z_e=z -> v1.0)
  E2  synthetic monthly rainfall p_k = P_k / P_bar (seasonal climatology x interannual
      factor x Gamma noise + extreme events), P_bar = 250 mm/month
  E3  antecedent water W_k = lam W_{k-1} + (1-lam) p_k, r_u = clip(r1 (W_k - 1), 0, 0.5)
  E4  damage clock D_k = D_{k-1} + dt [(1-eta) + eta p_{k-1}];  c = max(c0 - a D, 1),
      phi = max(phi0 - d D, 5)   (p = 1 -> D = t -> v1.0 linear law)
  E5  recovery after failure (t >= t_f, damage frozen):
      X(t) = X_inf - (X_inf - X_f) exp(-kappa (t - t_f) / tau_X), z_e = zeta z
  E6  t_crit = first downward crossing of FS = 1.1 (linear interpolation, as v1.0 cross())
With the forcing switched off (p = 1, no recovery) the oracle reproduces v1.0 to
machine precision (checked against a literal copy of the v1.0 engine, see --check).

Calibration constants (rho_c, tau_c, tau_phi, zeta, r1) are fitted once on the
"Cortinas historical" trajectory by calibrate() and frozen in dl_cortinas_ref.json.

Usage:
  python dl_data_generator.py            calibrate + build data/dl_dataset.npz (seed 42)
  python dl_data_generator.py --check    parity and calibration acceptance checks only
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
REF_FILE = HERE / "dl_cortinas_ref.json"
DATA_FILE = HERE / "data" / "dl_dataset.npz"
SEED = 42

# ---------------- time grid (identical to master_run: t = arange(0, 12, 1/12)) ----------------
TH, DT = 12.0, 1 / 12
t = np.arange(0, TH, DT)
T = len(t)                                   # 144 monthly steps, 2017.0 -> 2028.92
YEAR0 = 2017
MONTH = np.arange(T) % 12 + 1                # calendar month of each step (1..12)
FS_CRIT = 1.1

# ---------------- v1.0 nominal parameters (cortinas_config.json + master_run) ----------------
C0, PHI0, ALPHA, DELTA = 18.5, 30.8, 1.60, 2.375
GAMMA, Z, BETA, KH = 19.68, 13.0, 30.0, 0.0
SC, SP, RHO = 3.0, 2.5, -0.62

# ---------------- forcing constants (brief E2-E4) ----------------
P_BAR = 250.0                                # mm/month
TAU_D = 0.25                                 # antecedent-water memory (years)
LAM = np.exp(-DT / TAU_D)
ETA = 0.6                                    # rainfall weight of the damage clock
RU_MAX = 0.5
HIST_EVENTS = [(2021, 7, 2790.5), (2024, 5, 2803.9)]   # documented rainfall events (mm, 2 months)
T_F_HIST = 4.0                               # v1.0 annual dating: failure campaign 2021 -> t = 4

FEATURES = ["tau", "log1p_p", "rec", "z_e", "c0", "phi0", "gamma", "beta",
            "alpha", "delta", "kh", "kappa_r"]
SCALED_IDX = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]   # tau and rec are left unscaled


# ============================== physics ==============================
def fs_infinite_slope(c, phi, gamma, z, beta, kh=0.0, ru=0.0):
    """E1: pseudo-static infinite-slope FS with pore-pressure ratio r_u."""
    b = np.radians(beta); p = np.radians(phi)
    cb, sb = np.cos(b), np.sin(b)
    return (c + (gamma * z * cb**2 * (1 - ru) - kh * gamma * z * sb * cb) * np.tan(p)) / \
           (gamma * z * sb * cb + kh * gamma * z * cb**2)


def seasonal_factor(month):
    """Normalised seasonal rainfall shape s(m), annual mean = 1."""
    m = np.asarray(month, float)
    return 1 + 0.35 * np.cos(2 * np.pi * (m - 6) / 12) + 0.25 * np.cos(4 * np.pi * (m - 5) / 12)


S_CLIM = seasonal_factor(MONTH)


def step_index(year, month):
    return int((year - YEAR0) * 12 + (month - 1))


def climatology_rain(events=(), scale=1.0):
    """Noise-free rainfall ratio p_k: scale * s(m_k) + documented events split 1/3-2/3."""
    p = scale * S_CLIM.copy()
    for year, month, total in events:
        k = step_index(year, month)
        if 0 <= k < T: p[k] += total / 3 / P_BAR
        if 0 <= k + 1 < T: p[k + 1] += 2 * total / 3 / P_BAR
    return p


def sample_rainfall(rng, n):
    """E2: stochastic rainfall ratio, shape (n, T); 10 % of the series with forcing off."""
    a_y = rng.lognormal(-0.02, 0.20, (n, T // 12))
    eps = rng.gamma(4.0, 0.25, (n, T))
    p = S_CLIM[None, :] * np.repeat(a_y, 12, axis=1) * eps
    n_ev = rng.poisson(3.0, n)
    idx = np.repeat(np.arange(n), n_ev)
    k0 = rng.integers(0, T - 1, idx.size)
    M = rng.uniform(1500.0, 3200.0, idx.size)
    np.add.at(p, (idx, k0), M / 3 / P_BAR)
    np.add.at(p, (idx, k0 + 1), 2 * M / 3 / P_BAR)
    off = rng.random(n) < 0.10
    p[off] = 1.0
    return p, off


def antecedent_water(p):
    """E3: exponentially weighted rainfall index, W_{-1} = 1."""
    p = np.atleast_2d(p)
    W = np.empty_like(p, dtype=float)
    w = np.ones(p.shape[0])
    for k in range(p.shape[1]):
        w = LAM * w + (1 - LAM) * p[:, k]
        W[:, k] = w
    return W


def pore_pressure_ratio(W, r1):
    return np.clip(r1 * (W - 1.0), 0.0, RU_MAX)


def damage_clock(p):
    """E4: D_0 = 0, D_k = D_{k-1} + dt [(1-eta) + eta p_{k-1}]."""
    p = np.atleast_2d(p)
    inc = DT * ((1 - ETA) + ETA * p[:, :-1])
    return np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)


def run_oracle(c0, phi0, gamma, z, beta, alpha, delta, kh, p, k_f=None, zeta=1.0,
               kappa=1.0, const=None):
    """Vectorised oracle E1-E5. Scalars or (n,) arrays; p is (T,) or (n, T).
    k_f: failure step (>= 0 activates the recovery regime from that step, -1 = none)."""
    const = const or load_constants()
    p = np.atleast_2d(np.asarray(p, float))
    n = max(p.shape[0], np.size(c0))
    if p.shape[0] == 1 and n > 1: p = np.repeat(p, n, axis=0)
    col = lambda v: np.broadcast_to(np.asarray(v, float), (n,))[:, None]
    c0, phi0, gamma, z, beta, alpha, delta, kh, zeta, kappa = map(
        col, (c0, phi0, gamma, z, beta, alpha, delta, kh, zeta, kappa))
    kf = np.full(n, -1, int) if k_f is None else np.broadcast_to(np.asarray(k_f, int), (n,)).copy()

    W = antecedent_water(p)
    ru = pore_pressure_ratio(W, const["r1"])
    D = damage_clock(p)
    k = np.arange(T)[None, :]
    rec = (kf[:, None] >= 0) & (k >= kf[:, None])
    rows = np.arange(n)
    Df = D[rows, np.clip(kf, 0, T - 1)][:, None]
    D = np.where(rec, Df, D)
    c_dmg = np.maximum(c0 - alpha * D, 1.0)
    phi_dmg = np.maximum(phi0 - delta * D, 5.0)
    c_f = np.maximum(c0 - alpha * Df, 1.0)
    phi_f = np.maximum(phi0 - delta * Df, 5.0)
    c_inf = np.maximum(const["rho_c"] * c0, c_f)
    phi_inf = phi0
    trec = np.where(rec, t[None, :] - np.clip(kf, 0, None)[:, None] * DT, 0.0)
    c = np.where(rec, c_inf - (c_inf - c_f) * np.exp(-kappa * trec / const["tau_c"]), c_dmg)
    phi = np.where(rec, phi_inf - (phi_inf - phi_f) * np.exp(-kappa * trec / const["tau_phi"]), phi_dmg)
    ze = np.where(rec, zeta * z, z) * np.ones((1, T))
    fs = fs_infinite_slope(c, phi, gamma, ze, beta, kh, ru)
    return {"fs": fs, "c": c, "phi": phi, "ru": ru, "W": W, "D": D,
            "rec": rec.astype(np.float64), "z_e": ze, "p": p}


def crossing_time(fs, thr=FS_CRIT):
    """E6: first index with FS <= thr + linear interpolation (v1.0 cross()).
    Returns (t_crit [years], cross flag); FS_0 <= thr -> 0, no crossing -> TH, flag 0."""
    fs = np.atleast_2d(fs)
    below = fs <= thr
    has = below.any(axis=1)
    i = np.argmax(below, axis=1)
    r = np.arange(fs.shape[0])
    i0 = np.maximum(i - 1, 0)
    f0, f1 = fs[r, i0], fs[r, i]
    t0, t1 = t[i0], t[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = np.where(f1 != f0, t0 + (thr - f0) * (t1 - t0) / (f1 - f0), t1)
    tc = np.where(i == 0, 0.0, tc)
    tc = np.where(has, tc, TH)
    return tc, has.astype(np.int8)


# ============================== features ==============================
def build_features(p, rec, ze, c0, phi0, gamma, beta, alpha, delta, kh, kappa):
    """Per-step feature tensor (n, T, 12) in the order of FEATURES (unscaled)."""
    p = np.atleast_2d(p); n = p.shape[0]
    X = np.empty((n, T, len(FEATURES)), np.float32)
    X[:, :, 0] = t / TH
    X[:, :, 1] = np.log1p(p)
    X[:, :, 2] = np.broadcast_to(rec, (n, T))
    X[:, :, 3] = np.broadcast_to(ze, (n, T))
    for j, v in enumerate((c0, phi0, gamma, beta, alpha, delta, kh, kappa), start=4):
        X[:, :, j] = np.broadcast_to(np.asarray(v, float).reshape(-1, 1), (n, T))
    return X


# ============================== sampling ==============================
def rosenblatt(U):
    """Identical to resubmission_analyses.model: U (n,7) uniforms -> c0, phi0|c0, g, z, b, a, d."""
    c0 = C0 + SC * norm.ppf(np.clip(U[:, 0], 1e-6, 1 - 1e-6))
    zc = (c0 - C0) / SC
    phi0 = PHI0 + RHO * SP * zc + SP * np.sqrt(1 - RHO**2) * norm.ppf(np.clip(U[:, 1], 1e-6, 1 - 1e-6))
    c0 = np.clip(c0, 10, 30); phi0 = np.clip(phi0, 20, 40)
    gamma = 18.5 + U[:, 2] * (21 - 18.5)
    z = 8 + U[:, 3] * (16 - 8)
    beta = 26 + U[:, 4] * (34 - 26)
    alpha = ALPHA * (0.5 + U[:, 5])
    delta = DELTA * (0.5 + U[:, 6])
    return c0, phi0, gamma, z, beta, alpha, delta


def build_dataset(n=20000, seed=SEED, const=None):
    const = const or load_constants()
    rng = np.random.default_rng(seed)
    c0, phi0, gamma, z, beta, alpha, delta = rosenblatt(rng.random((n, 7)))
    kh = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.0, 0.30, n))
    kappa = rng.uniform(0.5, 1.5, n)
    p, forcing_off = sample_rainfall(rng, n)
    regime_b = rng.random(n) < 0.5
    k_f = np.rint(rng.uniform(2.5, 7.0, n) / DT).astype(int)
    zeta = rng.uniform(0.30, 0.55, n)
    split = np.zeros(n, np.int8)                       # 0 train / 1 val / 2 test
    perm = rng.permutation(n)
    split[perm[int(0.70 * n):int(0.85 * n)]] = 1
    split[perm[int(0.85 * n):]] = 2

    pars = (c0, phi0, gamma, z, beta, alpha, delta, kh)
    fs_a = run_oracle(*pars, p, None, zeta, kappa, const)["fs"]
    keep_b = regime_b & (fs_a[np.arange(n), k_f] < 1.0)   # FS(t_f-) >= 1 -> back to regime A
    k_f = np.where(keep_b, k_f, -1)
    out = run_oracle(*pars, p, k_f, zeta, kappa, const)
    tc, cr = crossing_time(out["fs"])
    X = build_features(p, out["rec"], out["z_e"], c0, phi0, gamma, beta, alpha, delta, kh, kappa)
    return {"X": X, "Y": out["fs"].astype(np.float32), "t_crit": tc.astype(np.float32),
            "cross": cr, "split": split, "c": out["c"].astype(np.float32),
            "phi": out["phi"].astype(np.float32), "ru": out["ru"].astype(np.float32),
            "D": out["D"].astype(np.float32), "regime": keep_b.astype(np.int8),
            "k_f": k_f.astype(np.int16), "zeta": zeta.astype(np.float32),
            "forcing_off": forcing_off.astype(np.int8)}


# ============================== calibration ==============================
def campaign_data():
    rows = list(csv.DictReader(open(HERE / "cortinas_campaign_data.csv", newline="")))
    return {int(r["year"]): (float(r["cohesion_kPa"]), float(r["friction_deg"])) for r in rows}


SPENCER = {2017: 1.216, 2021: 0.906, 2023: 1.155, 2025: 1.198}   # published static LEM (master_metrics)


def cortinas_history(const):
    """Cortinas historical trajectory: v1.0 nominal, kh=0, climatology + documented events,
    failure at t_f = 4.0 (k_f = 48), recovery with kappa = 1."""
    p = climatology_rain(HIST_EVENTS)
    k_f = int(round(T_F_HIST / DT))
    return run_oracle(C0, PHI0, GAMMA, Z, BETA, ALPHA, DELTA, KH, p, k_f,
                      const["zeta"], 1.0, const)


def nominal_v1(const):
    return run_oracle(C0, PHI0, GAMMA, Z, BETA, ALPHA, DELTA, KH, np.ones(T), None, 1.0, 1.0, const)


def calibrate(verbose=True):
    """Fit (rho_c, tau_c) exactly, tau_phi and zeta by least squares, r1 by root finding."""
    camp = campaign_data()
    kf = int(round(T_F_HIST / DT))
    d_f = damage_clock(climatology_rain(HIST_EVENTS))[0, kf]
    c_f, phi_f = max(C0 - ALPHA * d_f, 1.0), max(PHI0 - DELTA * d_f, 5.0)
    (c6, p6), (c8, p8) = camp[2023], camp[2025]
    s6, s8 = 2023 - YEAR0 - T_F_HIST, 2025 - YEAR0 - T_F_HIST

    def c_inf_of(tau):
        e1 = np.exp(-s6 / tau)
        return (c6 - c_f * e1) / (1 - e1)

    def res_c8(tau):
        ci = c_inf_of(tau)
        return ci - (ci - c_f) * np.exp(-s8 / tau) - c8
    tau_c = brentq(res_c8, 0.05, 50.0)
    rho_c = c_inf_of(tau_c) / C0

    def ss_phi(tau):
        return sum((PHI0 - (PHI0 - phi_f) * np.exp(-s / tau) - obs) ** 2 for s, obs in ((s6, p6), (s8, p8)))
    tau_phi = minimize_scalar(ss_phi, bounds=(0.5, 60.0), method="bounded", options={"xatol": 1e-10}).x

    const = {"rho_c": rho_c, "tau_c": tau_c, "tau_phi": tau_phi, "zeta": 0.4, "r1": 0.05}
    k6, k8 = step_index(2023, 1), step_index(2025, 1)
    k21 = [step_index(2021, m) for m in (7, 8, 9)]

    def ss_zeta(zeta):
        fs = cortinas_history({**const, "zeta": zeta})["fs"][0]
        return (fs[k6] - SPENCER[2023]) ** 2 + (fs[k8] - SPENCER[2025]) ** 2

    def res_r1(r1):
        return cortinas_history({**const, "r1": r1})["fs"][0][k21].min() - SPENCER[2021]

    for _ in range(50):
        z_old, r_old = const["zeta"], const["r1"]
        const["zeta"] = minimize_scalar(ss_zeta, bounds=(0.15, 1.0), method="bounded",
                                        options={"xatol": 1e-12}).x
        const["r1"] = brentq(res_r1, 0.0, 1.0, xtol=1e-14)
        if abs(const["zeta"] - z_old) < 1e-12 and abs(const["r1"] - r_old) < 1e-12: break
    const = {k: float(v) for k, v in const.items()}
    ref = reference_record(const)
    if verbose:
        print("Calibration: rho_c=%.4f tau_c=%.3f a tau_phi=%.3f a zeta=%.4f (z_e=%.2f m) r1=%.4f"
              % (const["rho_c"], const["tau_c"], const["tau_phi"], const["zeta"], const["zeta"] * Z, const["r1"]))
        for row in ref["residuals"]:
            print("  %-34s model=%8.4f  target=%8.4f  resid=%+.4f  [%s]" %
                  (row["quantity"], row["model"], row["target"], row["model"] - row["target"], row["use"]))
    return const, ref


def reference_record(const):
    camp = campaign_data()
    h = cortinas_history(const)
    nom = nominal_v1(const)
    fs = h["fs"][0]
    k = lambda y, m=1: step_index(y, m)
    k21 = [k(2021, m) for m in (7, 8, 9)]
    k24 = [k(2024, m) for m in (5, 6, 7)]
    rows = [
        ("c'(2023) kPa", h["c"][0, k(2023)], camp[2023][0], "calibrated (exact)"),
        ("c'(2025) kPa", h["c"][0, k(2025)], camp[2025][0], "calibrated (exact)"),
        ("phi'(2023) deg", h["phi"][0, k(2023)], camp[2023][1], "calibrated (LS)"),
        ("phi'(2025) deg", h["phi"][0, k(2025)], camp[2025][1], "calibrated (LS)"),
        ("FS(2023) vs Spencer", fs[k(2023)], SPENCER[2023], "calibrated (LS)"),
        ("FS(2025) vs Spencer", fs[k(2025)], SPENCER[2025], "calibrated (LS)"),
        ("min FS jul-sep 2021 vs Spencer", fs[k21].min(), SPENCER[2021], "calibrated (root)"),
        ("FS(2017) vs Spencer", fs[0], SPENCER[2017], "verification"),
        ("FS(2021.0+) post-failure", fs[k(2021)], 0.95, "verification (approx.)"),
        ("min FS may-jul 2024 (> 1.0)", fs[k24].min(), 1.06, "verification (approx.)"),
    ]
    tc11, _ = crossing_time(nom["fs"], 1.1)
    tc10, _ = crossing_time(nom["fs"], 1.0)
    tch, crh = crossing_time(h["fs"], 1.1)
    return {
        "constants": const,
        "fixed": {"P_bar_mm": P_BAR, "tau_d_yr": TAU_D, "eta": ETA, "ru_max": RU_MAX,
                  "t_f_hist_yr": T_F_HIST, "kappa_hist": 1.0, "events_hist": HIST_EVENTS,
                  "nominal": {"c0": C0, "phi0": PHI0, "gamma": GAMMA, "z": Z, "beta": BETA,
                              "alpha": ALPHA, "delta": DELTA, "kh": KH}},
        "residuals": [{"quantity": q, "model": float(m), "target": float(tg), "use": u} for q, m, tg, u in rows],
        "nominal_v1": {"fs": nom["fs"][0].round(6).tolist(),
                       "t_crit_1p1_year": float(YEAR0 + tc11[0]), "t_crit_1p0_year": float(YEAR0 + tc10[0])},
        "cortinas_history": {key: h[key][0].round(6).tolist() for key in ("p", "W", "ru", "D", "c", "phi", "z_e", "rec", "fs")}
                            | {"t_crit_1p1_year": float(YEAR0 + tch[0]), "cross": int(crh[0])},
        "campaigns": {str(y): {"c": v[0], "phi": v[1], "fs_spencer": SPENCER[y]} for y, v in camp.items()},
    }


def load_constants():
    if REF_FILE.exists():
        return json.load(open(REF_FILE))["constants"]
    const, ref = calibrate(verbose=False)
    return const


# ============================== acceptance checks ==============================
def _fs_v1(c, phi, gamma, z, beta, kh):          # literal copy of the v1.0 engine (master_run.py)
    b = np.radians(beta); p = np.radians(phi)
    cb, sb = np.cos(b), np.sin(b)
    return (c + (gamma * z * cb**2 - kh * gamma * z * sb * cb) * np.tan(p)) / (gamma * z * sb * cb + kh * gamma * z * cb**2)


def _traj_v1(c0, p0, g, z, b, a, d, kh=0.0):
    return _fs_v1(np.clip(c0 - a * t, 1, None), np.clip(p0 - d * t, 5, None), g, z, b, kh)


def check(const=None, ref=None):
    if const is None:
        if REF_FILE.exists():
            ref = json.load(open(REF_FILE)); const = ref["constants"]
        else:
            const, ref = calibrate(verbose=False)
    rng = np.random.default_rng(SEED)
    c0, phi0, gamma, z, beta, alpha, delta = rosenblatt(rng.random((500, 7)))
    kh = np.where(rng.random(500) < 0.5, 0.0, rng.uniform(0, 0.3, 500))
    fs_or = run_oracle(c0, phi0, gamma, z, beta, alpha, delta, kh, np.ones(T), None, 1.0, 1.0, const)["fs"]
    fs_v1 = np.array([_traj_v1(*v) for v in zip(c0, phi0, gamma, z, beta, alpha, delta, kh)])
    nom = nominal_v1(const)["fs"][0]
    parity = float(max(np.abs(fs_or - fs_v1).max(), np.abs(nom - _traj_v1(C0, PHI0, GAMMA, Z, BETA, ALPHA, DELTA)).max()))
    res = {r["quantity"]: r for r in ref["residuals"]}
    d = lambda q: abs(res[q]["model"] - res[q]["target"])
    t11, t10 = ref["nominal_v1"]["t_crit_1p1_year"], ref["nominal_v1"]["t_crit_1p0_year"]
    checks = [
        ("C1 parity |FS_oracle - FS_v1.0| < 1e-9", parity, parity < 1e-9),
        ("C1 nominal crossing FS=1.1 ~ 2017.9 (+-0.05)", t11, abs(t11 - 2017.9) <= 0.05),
        ("C1 nominal crossing FS=1.0 ~ 2018.9 (+-0.05)", t10, abs(t10 - 2018.9) <= 0.05),
        ("C2 |FS(2017) - Spencer| <= 0.05", d("FS(2017) vs Spencer"), d("FS(2017) vs Spencer") <= 0.05),
        ("C2 |FS(2023) - Spencer| <= 0.05", d("FS(2023) vs Spencer"), d("FS(2023) vs Spencer") <= 0.05),
        ("C2 |FS(2025) - Spencer| <= 0.05", d("FS(2025) vs Spencer"), d("FS(2025) vs Spencer") <= 0.05),
        ("C2 min FS 2021 = 0.906 +- 0.01", res["min FS jul-sep 2021 vs Spencer"]["model"],
         d("min FS jul-sep 2021 vs Spencer") <= 0.01),
        ("C2 min FS 2024 > 1.0", res["min FS may-jul 2024 (> 1.0)"]["model"],
         res["min FS may-jul 2024 (> 1.0)"]["model"] > 1.0),
    ]
    for name, val, ok in checks:
        print("  [%s] %-48s %.6g" % ("PASS" if ok else "FAIL", name, val))
    return {"parity_max_abs": parity, "nominal_t_crit_1p1_year": t11, "nominal_t_crit_1p0_year": t10,
            "all_pass": bool(all(c[2] for c in checks)),
            "checks": {c[0]: {"value": float(c[1]), "pass": bool(c[2])} for c in checks}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true", help="run acceptance checks only")
    ap.add_argument("--n", type=int, default=20000, help="number of synthetic series")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    if a.check:
        print("Acceptance checks (criteria 1-2):")
        sys.exit(0 if check()["all_pass"] else 1)

    t0 = time.perf_counter()
    const, ref = calibrate()
    json.dump(ref, open(REF_FILE, "w"), indent=1)
    print("Frozen constants ->", REF_FILE.name)
    print("Acceptance checks (criteria 1-2):")
    check(const, ref)
    ds = build_dataset(a.n, a.seed, const)
    DATA_FILE.parent.mkdir(exist_ok=True)
    np.savez(DATA_FILE, **ds)
    Y, sp = ds["Y"], ds["split"]
    print("Dataset: N=%d T=%d D=%d | train/val/test=%d/%d/%d | regime B=%.1f%% | forcing off=%.1f%% | "
          "cross(1.1)=%.1f%% | FS range [%.3f, %.3f] | %.1f s -> %s"
          % (len(Y), T, ds["X"].shape[2], (sp == 0).sum(), (sp == 1).sum(), (sp == 2).sum(),
             100 * ds["regime"].mean(), 100 * ds["forcing_off"].mean(), 100 * ds["cross"].mean(),
             Y.min(), Y.max(), time.perf_counter() - t0, DATA_FILE.relative_to(HERE)))


if __name__ == "__main__":
    main()
