"""
Training and evaluation of the AGD-Sentinel deep surrogates (v2.0).

Models (causal, sequence-to-sequence, ~3.5-5.6e4 parameters each):
  lstm  LSTM(12->64)-LN-Dropout-LSTM(64->64)-LN-Dropout-MLP head-softplus + aux t_crit head
  gru   same topology with GRU cells (baseline)
  cnn   causal 1-D CNN, 4 layers k=3, 64 channels, dilations 1/2/4/8 (31-month receptive field)
  mlp   static per-step MLP 12->128->128->128->1 (no temporal memory)
Loss: MSE + 2.0 mean(g e^2) + 0.1 mean(m relu(dy)) [+ 0.05 SmoothL1(t_crit/T_h), LSTM only]
      g = exp(-(y-1.1)^2 / (2 0.08^2)) (critical-zone weight), m = monotonic mask (no recovery, y non-increasing)
Optimiser: AdamW(2e-3, wd 1e-4), batch 256, grad-clip 1.0, ReduceLROnPlateau(0.5, 3), early stopping
(patience 8, best weights restored). Seed 42.

Outputs: models/dl_lstm.npz (NumPy runtime weights), models/dl_{lstm,gru,cnn,mlp}.pt, models/dl_scaler.json,
dl_metrics.json, dl_training_log.csv.
Usage: python dl_train_evaluate.py [--models lstm,gru,cnn,mlp] [--eval-only] [--quick] [--batch 256 --lr 2e-3]
"""
import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dl_data_generator as gen
from dl_inference import LSTMSurrogate

HERE = Path(__file__).resolve().parent
SEED = 42
TH, FS_CRIT = gen.TH, gen.FS_CRIT
HIDDEN, DROPOUT = 64, 0.10
MAX_EPOCHS = {"lstm": 40, "gru": 40, "cnn": 30, "mlp": 30}
W_ZONE, W_MONO, W_AUX, SIG_ZONE = 2.0, 0.1, 0.05, 0.08


# ============================== models ==============================
class RecurrentSurrogate(nn.Module):
    def __init__(self, cell="lstm", d_in=12, h=HIDDEN, p=DROPOUT, aux=True):
        super().__init__()
        R = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn1, self.ln1, self.do1 = R(d_in, h, batch_first=True), nn.LayerNorm(h), nn.Dropout(p)
        self.rnn2, self.ln2, self.do2 = R(h, h, batch_first=True), nn.LayerNorm(h), nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))
        self.aux = nn.Linear(h, 1) if aux else None

    def forward(self, x):
        a1 = self.do1(self.ln1(self.rnn1(x)[0]))
        a2 = self.do2(self.ln2(self.rnn2(a1)[0]))
        fs = F.softplus(self.head(a2)).squeeze(-1)
        tc = torch.sigmoid(self.aux(a2[:, -1])).squeeze(-1) if self.aux is not None else None
        return fs, tc


class CausalCNN(nn.Module):
    def __init__(self, d_in=12, ch=64, k=3, dilations=(1, 2, 4, 8), p=DROPOUT):
        super().__init__()
        self.pads = [(k - 1) * d for d in dilations]
        self.convs = nn.ModuleList(nn.Conv1d(d_in if j == 0 else ch, ch, k, dilation=d)
                                   for j, d in enumerate(dilations))
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(ch, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = x.transpose(1, 2)
        for pad, conv in zip(self.pads, self.convs):
            z = self.drop(F.relu(conv(F.pad(z, (pad, 0)))))       # left padding only -> causal
        return F.softplus(self.head(z.transpose(1, 2))).squeeze(-1), None


class StaticMLP(nn.Module):
    def __init__(self, d_in=12, h=128, p=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x):
        return F.softplus(self.net(x)).squeeze(-1), None


def make_model(name):
    torch.manual_seed(SEED)
    return {"lstm": lambda: RecurrentSurrogate("lstm", aux=True),
            "gru": lambda: RecurrentSurrogate("gru", aux=False),
            "cnn": CausalCNN, "mlp": StaticMLP}[name]()


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# ============================== loss ==============================
def composite_loss(yhat, y, rec, tc_hat=None, tc=None):
    e = yhat - y
    mse = (e ** 2).mean()
    g = torch.exp(-(y - FS_CRIT) ** 2 / (2 * SIG_ZONE ** 2))
    zone = (g * e ** 2).mean()
    m = ((rec[:, :-1] == 0) & (rec[:, 1:] == 0) & (y[:, 1:] <= y[:, :-1])).float()
    mono = (m * F.relu(yhat[:, 1:] - yhat[:, :-1])).mean()
    loss = mse + W_ZONE * zone + W_MONO * mono
    if tc_hat is not None:
        loss = loss + W_AUX * F.smooth_l1_loss(tc_hat, tc / TH)
    return loss


# ============================== data ==============================
def load_data():
    if not gen.DATA_FILE.exists():
        raise SystemExit("Dataset not found: run `python dl_data_generator.py` first.")
    d = np.load(gen.DATA_FILE)
    return {k: d[k] for k in d.files}


def fit_scaler(X_train):
    idx = gen.SCALED_IDX
    flat = X_train[:, :, idx].reshape(-1, len(idx)).astype(np.float64)
    std = flat.std(0)
    std[std < 1e-8] = 1.0
    return {"features": gen.FEATURES, "scaled_idx": idx, "mean": flat.mean(0).tolist(),
            "std": std.tolist(), "T_h": TH, "dt": gen.DT, "T": gen.T, "fs_crit": FS_CRIT,
            "dropout": DROPOUT, "year0": gen.YEAR0}


def apply_scaler(X, sc):
    Xs = X.astype(np.float32, copy=True)
    idx = sc["scaled_idx"]
    Xs[:, :, idx] = (Xs[:, :, idx] - np.float32(sc["mean"])) / np.float32(sc["std"])
    return Xs


@torch.no_grad()
def predict(model, Xs, batch=1000):
    model.eval()
    out = [model(torch.from_numpy(Xs[i:i + batch]))[0].numpy() for i in range(0, len(Xs), batch)]
    return np.concatenate(out)


# ============================== training ==============================
def train_model(name, model, data, log_rows, max_epochs, batch=256, lr=2e-3):
    Xtr, Ytr, Rtr, TCtr = (torch.from_numpy(data[k]) for k in ("Xtr", "Ytr", "Rtr", "TCtr"))
    use_aux = getattr(model, "aux", None) is not None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-5)
    gen_t = torch.Generator().manual_seed(SEED)
    best, best_state, best_ep, wait = np.inf, None, 0, 0
    t0 = time.perf_counter()
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen_t)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            fs, tc = model(Xtr[j])
            loss = composite_loss(fs, Ytr[j], Rtr[j], tc if use_aux else None, TCtr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        val_rmse = float(np.sqrt(np.mean((predict(model, data["Xva"]) - data["Yva"]) ** 2)))
        sched.step(val_rmse)
        lr_now = opt.param_groups[0]["lr"]
        el = time.perf_counter() - t0
        log_rows.append({"model": name, "epoch": ep, "train_loss": tot / nb, "val_rmse": val_rmse,
                         "lr": lr_now, "elapsed_s": round(el, 1)})
        print(f"  {name} ep{ep:02d} loss={tot / nb:.5f} val_rmse={val_rmse:.5f} lr={lr_now:.1e} t={el:.0f}s", flush=True)
        if val_rmse < best - 1e-6:
            best, best_ep, wait = val_rmse, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                print(f"  {name}: early stopping at epoch {ep} (best {best_ep})", flush=True)
                break
    model.load_state_dict(best_state)
    return {"epochs_run": ep, "best_epoch": best_ep, "best_val_rmse": best,
            "train_time_s": round(time.perf_counter() - t0, 1)}


# ============================== evaluation ==============================
def regression_metrics(yhat, y, tc_true, cr_true, rec):
    e = yhat - y
    zone = np.abs(y - FS_CRIT) < 0.1
    tc_pred, cr_pred = gen.crossing_time(yhat)
    interior = (cr_true == 1) & (tc_true > 0) & (tc_true < TH)
    dt = np.abs(tc_pred - tc_true)[interior]
    m = (rec[:, :-1] == 0) & (rec[:, 1:] == 0) & (y[:, 1:] <= y[:, :-1])
    viol = (np.diff(yhat, axis=1) > 1e-3) & m
    return {"R2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
            "RMSE_zone_1p1": float(np.sqrt((e[zone] ** 2).mean())), "n_zone_points": int(zone.sum()),
            "dt_crit_median_yr": float(np.median(dt)), "dt_crit_MAE_yr": float(dt.mean()),
            "dt_crit_p90_yr": float(np.percentile(dt, 90)), "n_interior_crossings": int(interior.sum()),
            "cross_accuracy": float((cr_pred == cr_true).mean()),
            "monotonic_violation_pct": float(100 * viol.sum() / max(m.sum(), 1))}


def evaluate(yhat, d):
    te = d["split"] == 2
    y, tc, cr, rec, reg = d["Y"][te], d["t_crit"][te], d["cross"][te], d["X"][te][:, :, 2], d["regime"][te]
    out = regression_metrics(yhat, y, tc, cr, rec)
    out["regime_A"] = regression_metrics(yhat[reg == 0], y[reg == 0], tc[reg == 0], cr[reg == 0], rec[reg == 0])
    out["regime_B"] = regression_metrics(yhat[reg == 1], y[reg == 1], tc[reg == 1], cr[reg == 1], rec[reg == 1])
    out["regime_A"]["n_series"], out["regime_B"]["n_series"] = int((reg == 0).sum()), int((reg == 1).sum())
    return out


@torch.no_grad()
def mc_dropout_torch(model, Xs, K=50, chunk=20):
    model.train()                                   # dropout active; LayerNorm is deterministic
    mus, sds, p5s, p95s = [], [], [], []
    for i in range(0, len(Xs), chunk):
        xb = torch.from_numpy(np.repeat(Xs[i:i + chunk], K, axis=0))
        S = model(xb)[0].numpy().reshape(-1, K, Xs.shape[1])
        mus.append(S.mean(1)); sds.append(S.std(1))
        p5, p95 = np.percentile(S, [5, 95], axis=1); p5s.append(p5); p95s.append(p95)
    model.eval()
    return tuple(np.concatenate(a) for a in (mus, sds, p5s, p95s))


def export_lstm_npz(model, path):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    w = {}
    for j in (1, 2):
        w[f"l{j}_w_ih"] = sd[f"rnn{j}.weight_ih_l0"]; w[f"l{j}_w_hh"] = sd[f"rnn{j}.weight_hh_l0"]
        w[f"l{j}_b"] = sd[f"rnn{j}.bias_ih_l0"] + sd[f"rnn{j}.bias_hh_l0"]
        w[f"ln{j}_w"] = sd[f"ln{j}.weight"]; w[f"ln{j}_b"] = sd[f"ln{j}.bias"]
    w["fc1_w"], w["fc1_b"] = sd["head.0.weight"], sd["head.0.bias"]
    w["fc2_w"], w["fc2_b"] = sd["head.2.weight"], sd["head.2.bias"]
    w["aux_w"], w["aux_b"] = sd["aux.weight"], sd["aux.bias"]
    np.savez(path, **w)


def timeit(fn, warm=5, reps=30):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    q1, med, q3 = np.percentile(np.array(ts) * 1e3, [25, 50, 75])
    return {"median_ms": float(med), "iqr_ms": float(q3 - q1), "q1_ms": float(q1), "q3_ms": float(q3)}


# ---------- classic Monte Carlo (master_run scheme) vs surrogate ----------
COV = [[gen.SC ** 2, gen.RHO * gen.SC * gen.SP], [gen.RHO * gen.SC * gen.SP, gen.SP ** 2]]


def sample_master_run(N, seed=SEED):
    rng = np.random.default_rng(seed)
    s = rng.multivariate_normal([gen.C0, gen.PHI0], COV, N)
    c0, p0 = np.clip(s[:, 0], 10, 30), np.clip(s[:, 1], 20, 40)
    z, b = rng.uniform(8, 16, N), rng.uniform(26, 34, N)
    g = np.clip(rng.normal(gen.GAMMA, 0.5, N), 18.5, 21.0)     # clipped to the training range
    return c0, p0, g, z, b


def mc_classic(N, const, seed=SEED):
    c0, p0, g, z, b = sample_master_run(N, seed)
    fs = gen.run_oracle(c0, p0, g, z, b, gen.ALPHA, gen.DELTA, 0.0, np.ones(gen.T), None, 1.0, 1.0, const)["fs"]
    p5, p50, p95 = np.percentile(fs, [5, 50, 95], axis=0)
    tc, _ = gen.crossing_time(fs)
    return p50, p5, p95, tc


def mc_surrogate(N, sur, seed=SEED):
    c0, p0, g, z, b = sample_master_run(N, seed)
    X = gen.build_features(np.ones((N, gen.T)), 0.0, z[:, None], c0, p0, g, b,
                           gen.ALPHA, gen.DELTA, 0.0, 1.0)
    fs = sur.predict(X)
    p5, p50, p95 = np.percentile(fs, [5, 50, 95], axis=0)
    tc, _ = gen.crossing_time(fs)
    return p50, p5, p95, tc


def nominal_features():
    return gen.build_features(np.ones((1, gen.T)), 0.0, gen.Z, gen.C0, gen.PHI0, gen.GAMMA, gen.BETA,
                              gen.ALPHA, gen.DELTA, 0.0, 1.0)


def cortinas_features(ref):
    h = ref["cortinas_history"]
    return gen.build_features(np.array(h["p"])[None], np.array(h["rec"])[None], np.array(h["z_e"])[None],
                              gen.C0, gen.PHI0, gen.GAMMA, gen.BETA, gen.ALPHA, gen.DELTA, 0.0, 1.0)


def benchmark(sur, const):
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    ref_p50 = mc_classic(100_000, const)[0]
    nom_oracle = np.array(json.load(open(gen.REF_FILE))["nominal_v1"]["fs"])
    Xn = nominal_features()
    res = {"protocol": "perf_counter, 5 warm-up + 30 repetitions, median and IQR; same machine, CPU; "
                       "classic MC = master_run sampling (seed 42, rho=-0.62) on the vectorised oracle, full "
                       "pipeline (sampling + FS matrix N x 144 + P5/P50/P95 + t_crit); gamma clipped to [18.5, 21]",
           "reference": "P50 of classic MC with N=1e5"}
    tA = timeit(lambda: sur.predict(Xn))
    tB = timeit(lambda: sur.mc_dropout(Xn, K=50))
    fsA = sur.predict(Xn)[0]
    fsB = sur.mc_dropout(Xn, K=50)["mean"][0]
    res["surrogate_A_single_curve"] = {**tA, "rmse_vs_ref_p50": rmse(fsA, ref_p50), "rmse_vs_oracle_nominal": rmse(fsA, nom_oracle)}
    res["surrogate_B_mc_dropout_K50"] = {**tB, "rmse_vs_ref_p50": rmse(fsB, ref_p50), "rmse_vs_oracle_nominal": rmse(fsB, nom_oracle)}
    for N in (1000, 10000):
        tm = timeit(lambda: mc_classic(N, const))
        tc = timeit(lambda: mc_surrogate(N, sur), warm=2 if N > 1000 else 5)
        res[f"mc_classic_N{N}"] = {**tm, "rmse_p50_vs_ref": rmse(mc_classic(N, const)[0], ref_p50)}
        res[f"surrogate_C_batch_N{N}"] = {**tc, "rmse_p50_vs_ref": rmse(mc_surrogate(N, sur)[0], ref_p50)}
        res[f"speedup_N{N}"] = {"S_A": tm["median_ms"] / tA["median_ms"], "S_B": tm["median_ms"] / tB["median_ms"],
                                "S_C": tm["median_ms"] / tc["median_ms"]}
        print(f"  MC N={N}: classic {tm['median_ms']:.2f} ms | surrogate batch {tc['median_ms']:.2f} ms | "
              f"S_A={res[f'speedup_N{N}']['S_A']:.1f} S_B={res[f'speedup_N{N}']['S_B']:.1f} "
              f"S_C={res[f'speedup_N{N}']['S_C']:.2f}", flush=True)
    return res


def hardware():
    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "device": "cpu"}


# ============================== main ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lstm,gru,cnn,mlp")
    ap.add_argument("--eval-only", action="store_true", help="load saved .pt weights, skip training")
    ap.add_argument("--quick", action="store_true", help="smoke test: small subset, 2 epochs, scratch outputs")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--batch", type=int, default=256, help="mini-batch size (time lever: 512)")
    ap.add_argument("--lr", type=float, default=2e-3, help="initial learning rate (time lever: 3e-3)")
    a = ap.parse_args()
    out = Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    d = load_data()
    ref = json.load(open(gen.REF_FILE)); const = ref["constants"]
    sp = d["split"]
    if a.quick:
        keep = np.concatenate([np.where(sp == s)[0][:n] for s, n in ((0, 1500), (1, 300), (2, 300))])
        d = {k: v[keep] for k, v in d.items()}; sp = d["split"]
    sc = fit_scaler(d["X"][sp == 0])
    Xs = apply_scaler(d["X"], sc)
    data = {"Xtr": Xs[sp == 0], "Ytr": d["Y"][sp == 0], "Rtr": d["X"][sp == 0][:, :, 2].copy(),
            "TCtr": d["t_crit"][sp == 0], "Xva": Xs[sp == 1], "Yva": d["Y"][sp == 1]}
    Xte = Xs[sp == 2]
    json.dump(sc, open(out / "models" / "dl_scaler.json", "w"), indent=1)

    metrics_path = out / "dl_metrics.json"
    M = json.load(open(metrics_path)) if (a.eval_only and metrics_path.exists()) else {}
    M.setdefault("models", {})
    log_rows = []
    Xc = cortinas_features(ref)
    fs_c_oracle = np.array(ref["cortinas_history"]["fs"])
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    trained = {}
    for name in names:
        model = make_model(name)
        info = M["models"].get(name, {}).get("training", {})
        if a.eval_only:
            model.load_state_dict(torch.load(out / "models" / f"dl_{name}.pt", weights_only=True))
        else:
            print(f"Training {name} ({n_params(model)} parameters)", flush=True)
            info = train_model(name, model, data, log_rows, 2 if a.quick else MAX_EPOCHS[name], a.batch, a.lr)
            info.update({"batch": a.batch, "lr0": a.lr})
            torch.save(model.state_dict(), out / "models" / f"dl_{name}.pt")
        yhat = predict(model, Xte)
        met = evaluate(yhat, {k: d[k] for k in ("Y", "t_crit", "cross", "X", "regime", "split")})
        fs_c = predict(model, apply_scaler(Xc, sc))[0]
        err_c = np.abs(fs_c - fs_c_oracle)
        k_f = int(round(gen.T_F_HIST / gen.DT))
        met["cortinas_max_abs_err"] = float(err_c.max())
        met["cortinas_argmax_year"] = float(gen.YEAR0 + gen.t[int(err_c.argmax())])
        met["cortinas_max_abs_err_excl_failure_step"] = float(np.delete(err_c, k_f).max())
        met["cortinas_RMSE"] = float(np.sqrt(np.mean(err_c ** 2)))
        M["models"][name] = {"params": n_params(model), "training": info, "test": met}
        trained[name] = model
        print(f"  {name}: R2={met['R2']:.4f} RMSE={met['RMSE']:.4f} MAE={met['MAE']:.4f} "
              f"zone={met['RMSE_zone_1p1']:.4f} dt_med={met['dt_crit_median_yr']:.3f} "
              f"dt_MAE={met['dt_crit_MAE_yr']:.3f} mono={met['monotonic_violation_pct']:.2f}% "
              f"cortinas={met['cortinas_max_abs_err']:.4f}", flush=True)

    if log_rows:
        log_path = out / "dl_training_log.csv"
        old = [r for r in csv.DictReader(open(log_path))] if log_path.exists() else []
        old = [r for r in old if r["model"] not in names]
        with open(log_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            wr.writeheader(); wr.writerows(old + log_rows)

    if "lstm" in trained:
        lstm = trained["lstm"]
        export_lstm_npz(lstm, out / "models" / "dl_lstm.npz")
        sur = LSTMSurrogate.load(out / "models")
        sub = Xte[:200]
        fs_t = predict(lstm, sub)
        fs_n = sur.predict(d["X"][sp == 2][:200])
        mu, sd, p5, p95 = mc_dropout_torch(lstm, Xte)
        yte = d["Y"][sp == 2]
        L = M["models"]["lstm"]
        L["parity_numpy_torch_max_abs"] = float(np.abs(fs_t - fs_n).max())
        L["mc_dropout"] = {"K": 50, "PICP95_mu_pm_2sigma_pct": float(100 * np.mean((yte >= mu - 2 * sd) & (yte <= mu + 2 * sd))),
                           "coverage_P5_P95_pct": float(100 * np.mean((yte >= p5) & (yte <= p95))),
                           "mean_sigma": float(sd.mean()), "RMSE_of_mean": float(np.sqrt(np.mean((mu - yte) ** 2)))}
        L["latency_numpy"] = {"A_single_curve": timeit(lambda: sur.predict(nominal_features())),
                              "B_mc_dropout_K50": timeit(lambda: sur.mc_dropout(nominal_features(), K=50))}
        L["weights_npz_kB"] = round((out / "models" / "dl_lstm.npz").stat().st_size / 1024, 1)
        print(f"  parity numpy-torch {L['parity_numpy_torch_max_abs']:.2e} | PICP95 "
              f"{L['mc_dropout']['PICP95_mu_pm_2sigma_pct']:.1f}% | latency A "
              f"{L['latency_numpy']['A_single_curve']['median_ms']:.2f} ms B "
              f"{L['latency_numpy']['B_mc_dropout_K50']['median_ms']:.2f} ms", flush=True)
        if not a.quick:
            print("Benchmark MC vs surrogate", flush=True)
            M["benchmark"] = benchmark(sur, const)

    M["dataset"] = {"N": int(len(d["Y"])), "T": gen.T, "D": len(gen.FEATURES), "features": gen.FEATURES,
                    "split_train_val_test": [int((sp == s).sum()) for s in (0, 1, 2)],
                    "regime_B_pct": float(100 * d["regime"].mean()), "forcing_off_pct": float(100 * d["forcing_off"].mean()),
                    "t_crit_zero_pct": float(100 * (d["t_crit"] == 0).mean()), "seed": SEED}
    M["oracle_checks"] = gen.check(const, ref)
    M["calibration"] = {"constants": const, "residuals": ref["residuals"]}
    M["hardware"] = hardware()
    M["acceptance"] = acceptance(M)
    json.dump(M, open(metrics_path, "w"), indent=1)
    print("Acceptance:", json.dumps(M["acceptance"], indent=1))
    print("Metrics ->", metrics_path)


def acceptance(M):
    m = M["models"]
    acc = {}
    if "lstm" in m:
        L, t = m["lstm"], m["lstm"]["test"]
        acc.update({"C3_R2>=0.98": t["R2"] >= 0.98, "C3_RMSE<=0.03": t["RMSE"] <= 0.03, "C3_MAE<=0.02": t["MAE"] <= 0.02,
                    "C3_RMSE_zone<=0.03": t["RMSE_zone_1p1"] <= 0.03, "C3_dt_median<=0.25": t["dt_crit_median_yr"] <= 0.25,
                    "C3_dt_MAE<=0.40": t["dt_crit_MAE_yr"] <= 0.40, "C3_cross_acc>=0.97": t["cross_accuracy"] >= 0.97,
                    "C3_mono_viol<=1pct": t["monotonic_violation_pct"] <= 1.0,
                    "C5_cortinas_max_err<=0.05": t["cortinas_max_abs_err"] <= 0.05})
        if "parity_numpy_torch_max_abs" in L:
            acc["C6_parity<1e-4"] = L["parity_numpy_torch_max_abs"] < 1e-4
            acc["C7_latency_A<=15ms"] = L["latency_numpy"]["A_single_curve"]["median_ms"] <= 15
            acc["C7_latency_B<=60ms"] = L["latency_numpy"]["B_mc_dropout_K50"]["median_ms"] <= 60
        if "mlp" in m:
            acc["C4_RMSE_LSTM<RMSE_MLP"] = t["RMSE"] < m["mlp"]["test"]["RMSE"]
        if "gru" in m:
            acc["C4_note_GRU_vs_LSTM"] = "GRU better or equal" if m["gru"]["test"]["RMSE"] <= t["RMSE"] else "LSTM better"
    acc["C1_C2_oracle"] = M["oracle_checks"]["all_pass"]
    return acc


if __name__ == "__main__":
    main()
