"""
Evaluation figures of the AGD-Sentinel deep surrogate (300 DPI, Okabe-Ito colour-blind-safe palette).
Requires dl_dataset (data/), dl_cortinas_ref.json, models/ and dl_metrics.json / dl_training_log.csv
produced by dl_data_generator.py and dl_train_evaluate.py. NumPy inference only (no training framework).

  fig_dl_learning_curves.png    validation RMSE and training loss per epoch, 4 models
  fig_dl_fs_prediction.png      Cortinas historical: LSTM +- MC Dropout, oracle, MC P5-P95, +-50 % rate band
  fig_dl_benchmark_speedup.png  accuracy vs latency (log) and speedup S = t_MC / t_surrogate
  fig_dl_memory_states.png      cell states C_t vs damage / strength, forget gate vs pore pressure
  fig_dl_forcing_regimes.png    rainfall, r_u, c', phi' and FS of the Cortinas historical trajectory
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dl_data_generator as gen
from dl_inference import LSTMSurrogate

HERE = Path(__file__).resolve().parent
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "grey": "#8C8C8C"}
MODEL_STYLE = {"lstm": ("LSTM", OI["blue"]), "gru": ("GRU", OI["orange"]),
               "cnn": ("CNN-1D causal", OI["green"]), "mlp": ("Static MLP", OI["purple"])}
YEARS = gen.YEAR0 + gen.t
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#DDDDDD",
                     "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.6})


def load():
    ref = json.load(open(gen.REF_FILE))
    M = json.load(open(HERE / "dl_metrics.json"))
    log = list(csv.DictReader(open(HERE / "dl_training_log.csv")))
    return ref, M, log, LSTMSurrogate.load(HERE / "models")


def cortinas_X(ref, alpha=gen.ALPHA, delta=gen.DELTA):
    h = ref["cortinas_history"]
    return gen.build_features(np.array(h["p"])[None], np.array(h["rec"])[None], np.array(h["z_e"])[None],
                              gen.C0, gen.PHI0, gen.GAMMA, gen.BETA, alpha, delta, 0.0, 1.0)


def mark_threshold(ax):
    ax.axhline(gen.FS_CRIT, color=OI["red"], ls="--", lw=1.0)
    ax.axhline(1.0, color=OI["grey"], ls=":", lw=1.0)


def fig_learning_curves(log):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for name, (lab, col) in MODEL_STYLE.items():
        rows = [r for r in log if r["model"] == name]
        if not rows: continue
        ep = [int(r["epoch"]) for r in rows]
        axs[0].plot(ep, [float(r["val_rmse"]) for r in rows], color=col, label=lab)
        axs[1].plot(ep, [float(r["train_loss"]) for r in rows], color=col, label=lab)
    axs[0].set(yscale="log", xlabel="Epoch", ylabel="Validation RMSE of FS", title="(a) Validation RMSE")
    axs[1].set(yscale="log", xlabel="Epoch", ylabel="Composite training loss", title="(b) Training loss")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "fig_dl_learning_curves.png"); plt.close(fig)


def fig_fs_prediction(ref, sur, const):
    h = ref["cortinas_history"]
    fs_or = np.array(h["fs"])
    X = cortinas_X(ref)
    det = sur.predict(X)[0]
    mc = sur.mc_dropout(X, K=50, seed=0)
    # parametric MC on the oracle (master_run sampling), same forcing and failure date
    rng = np.random.default_rng(42)
    cov = [[gen.SC**2, gen.RHO * gen.SC * gen.SP], [gen.RHO * gen.SC * gen.SP, gen.SP**2]]
    s = rng.multivariate_normal([gen.C0, gen.PHI0], cov, 1000)
    c0, p0 = np.clip(s[:, 0], 10, 30), np.clip(s[:, 1], 20, 40)
    z, b, g = rng.uniform(8, 16, 1000), rng.uniform(26, 34, 1000), np.clip(rng.normal(gen.GAMMA, 0.5, 1000), 18.5, 21)
    kf = int(round(gen.T_F_HIST / gen.DT))
    fs_mc = gen.run_oracle(c0, p0, g, z, b, gen.ALPHA, gen.DELTA, 0.0, np.array(h["p"]), kf,
                           const["zeta"], 1.0, const)["fs"]
    p5, p95 = np.percentile(fs_mc, [5, 95], axis=0)
    rates = [gen.run_oracle(gen.C0, gen.PHI0, gen.GAMMA, gen.Z, gen.BETA, gen.ALPHA * f, gen.DELTA * f, 0.0,
                            np.array(h["p"]), kf, const["zeta"], 1.0, const)["fs"][0] for f in (0.5, 1.5)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(YEARS, p5, p95, color=OI["sky"], alpha=0.30, lw=0, label="Classic MC on oracle, P5-P95 (N=1000)")
    ax.fill_between(YEARS, np.minimum(*rates), np.maximum(*rates), facecolor="none", edgecolor=OI["grey"],
                    hatch="////", lw=0, label="Oracle, rates alpha, delta x0.5-x1.5")
    ax.fill_between(YEARS, mc["lo2s"][0], mc["hi2s"][0], color=OI["orange"], alpha=0.45, lw=0,
                    label="LSTM MC Dropout mean +- 2 sigma (K=50)")
    ax.plot(YEARS, fs_or, color=OI["black"], lw=1.6, label="Physical oracle (calibrated)")
    ax.plot(YEARS, det, color=OI["blue"], lw=1.6, ls="--", label="LSTM surrogate (deterministic)")
    camp = ref["campaigns"]
    ax.plot([int(y) for y in camp], [camp[y]["fs_spencer"] for y in camp], "s", color=OI["red"], ms=6,
            mec="white", mew=0.8, label="Campaign FS (Spencer LEM)", zorder=5)
    mark_threshold(ax)
    ax.axvline(gen.YEAR0 + gen.T_F_HIST, color=OI["grey"], lw=0.8)
    ax.text(gen.YEAR0 + gen.T_F_HIST + 0.08, 0.45, "failure / recovery regime (t_f = 2021.0)", fontsize=7, color="#444444")
    tc, cr = gen.crossing_time(det[None])
    ax.set(xlabel="Year", ylabel="Factor of safety FS(t)", xlim=(2017, 2029), ylim=(0.4, 1.8),
           title="Cortinas historical trajectory: LSTM surrogate vs physical oracle "
                 f"(max |error| = {np.abs(det - fs_or).max():.3f})")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout(); fig.savefig(HERE / "fig_dl_fs_prediction.png"); plt.close(fig)


def fig_benchmark(M):
    B = M["benchmark"]
    pts = [("Classic MC N=1000", B["mc_classic_N1000"], "rmse_p50_vs_ref", OI["black"], "o"),
           ("Classic MC N=10000", B["mc_classic_N10000"], "rmse_p50_vs_ref", OI["grey"], "o"),
           ("LSTM (A) 1 curve", B["surrogate_A_single_curve"], "rmse_vs_ref_p50", OI["blue"], "D"),
           ("LSTM (B) MC Dropout K=50", B["surrogate_B_mc_dropout_K50"], "rmse_vs_ref_p50", OI["orange"], "D"),
           ("LSTM (C) batch N=1000", B["surrogate_C_batch_N1000"], "rmse_p50_vs_ref", OI["green"], "s"),
           ("LSTM (C) batch N=10000", B["surrogate_C_batch_N10000"], "rmse_p50_vs_ref", OI["purple"], "s")]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.25, 1]})
    for lab, d, key, col, mk in pts:
        axs[0].errorbar(d["median_ms"], d[key], xerr=[[d["median_ms"] - d["q1_ms"]], [d["q3_ms"] - d["median_ms"]]],
                        fmt=mk, color=col, ms=7, mec="white", mew=0.8, capsize=2, label=lab)
    axs[0].set(xscale="log", yscale="log", xlabel="Latency per evaluation [ms] (median, IQR)",
               ylabel="RMSE of P50 vs MC reference (N=1e5)", title="(a) Accuracy vs latency")
    axs[0].legend(fontsize=6.5, loc="best")
    labels, vals, cols = [], [], []
    for N in (1000, 10000):
        S = B[f"speedup_N{N}"]
        for k, col in (("S_A", OI["blue"]), ("S_B", OI["orange"]), ("S_C", OI["green"])):
            labels.append(f"{k}\n{N // 1000}k"); vals.append(S[k]); cols.append(col)
    x = np.arange(len(vals))
    axs[1].bar(x, vals, color=cols, width=0.7)
    for xi, v in zip(x, vals):
        axs[1].text(xi, v * 1.08, f"{v:.2g}", ha="center", fontsize=7)
    axs[1].axhline(1.0, color=OI["red"], ls="--", lw=1)
    axs[1].set(yscale="log", xticks=x, ylabel="Speedup S = t_MC / t_surrogate", title="(b) Measured speedup (N = MC size)")
    axs[1].set_xticklabels(labels, fontsize=7)
    fig.tight_layout(); fig.savefig(HERE / "fig_dl_benchmark_speedup.png"); plt.close(fig)


def fig_memory_states(ref, sur):
    # unit selection on 400 test series (pooled correlation), display on Cortinas historical
    d = np.load(gen.DATA_FILE)
    te = np.where(d["split"] == 2)[0][:400]
    X, D, c, phi = d["X"][te], d["D"][te], d["c"][te], d["phi"][te]
    _, _, st = sur.forward(X, return_states=True)
    cands = []
    for layer in ("C1", "C2"):
        Ct = st[layer].reshape(-1, st[layer].shape[-1])
        for var_name, v in (("D", D), ("c'", c), ("phi'", phi)):
            vv = v.reshape(-1)
            r = np.array([np.corrcoef(Ct[:, j], vv)[0, 1] for j in range(Ct.shape[1])])
            for j in range(len(r)): cands.append((abs(r[j]), r[j], layer, j, var_name))
    cands.sort(reverse=True)
    chosen, used = [], set()
    for a, r, layer, j, var_name in cands:
        if (layer, j) not in used and var_name not in {cc[4] for cc in chosen}:
            chosen.append((a, r, layer, j, var_name)); used.add((layer, j))
        if len(chosen) == 3: break
    Xc = cortinas_X(ref)
    _, _, sc = sur.forward(Xc, return_states=True)
    h = ref["cortinas_history"]
    ref_var = {"D": np.array(h["D"]), "c'": np.array(h["c"]), "phi'": np.array(h["phi"])}
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.2))
    cols = [OI["blue"], OI["orange"], OI["green"]]
    for (a, r, layer, j, var_name), col in zip(chosen, cols):
        sgn = np.sign(r)
        axs[0].plot(YEARS, sgn * z(sc[layer][0, :, j]), color=col, label=f"{layer} unit {j} (sign-aligned), r={r:+.2f}")
        axs[0].plot(YEARS, z(ref_var[var_name]), color=col, ls=":", lw=1.2, label=f"{var_name}(t) oracle")
    axs[0].axvline(gen.YEAR0 + gen.T_F_HIST, color=OI["grey"], lw=0.8)
    axs[0].set(xlabel="Year", ylabel="Standardised value", title="(a) Cell state C_t vs soil state")
    axs[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    f_mean = sc["gates1"]["f"][0].mean(-1)
    ru = np.array(h["ru"])
    r_f = np.corrcoef(st["gates1"]["f"].mean(-1).reshape(-1), d["ru"][te].reshape(-1))[0, 1]
    axs[1].plot(YEARS, z(f_mean), color=OI["blue"], label=f"mean forget gate f_t (layer 1), pooled r={r_f:+.2f}")
    axs[1].plot(YEARS, z(ru), color=OI["red"], ls=":", lw=1.2, label="pore-pressure ratio r_u(t)")
    axs[1].set(xlabel="Year", ylabel="Standardised value", title="(b) Forget gate vs r_u")
    axs[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    fig.suptitle("Interpretation of LSTM memory on the Cortinas trajectory (analogy, not proof)", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "fig_dl_memory_states.png"); plt.close(fig)
    return [{"layer": l, "unit": int(j), "variable": v, "pooled_r": float(r)} for a, r, l, j, v in chosen], float(r_f)


def fig_forcing_regimes(ref):
    h = ref["cortinas_history"]
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 6.4), sharex=True)
    axs[0].bar(YEARS, np.array(h["p"]) * gen.P_BAR, width=1 / 12, color=OI["sky"], label="Rainfall (synthetic climatology + documented events)")
    axs[0].set(ylabel="Rain [mm/month]")
    axs[1].plot(YEARS, h["ru"], color=OI["blue"], label="r_u(t)")
    axs[1].set(ylabel="r_u [-]")
    axs[2].plot(YEARS, h["c"], color=OI["orange"], label="c'(t) [kPa]")
    axs[2].plot(YEARS, h["phi"], color=OI["green"], label="phi'(t) [deg]")
    camp = ref["campaigns"]
    axs[2].plot([int(y) for y in camp], [camp[y]["c"] for y in camp], "o", color=OI["orange"], mec="white", ms=6, label="c' campaigns")
    axs[2].plot([int(y) for y in camp], [camp[y]["phi"] for y in camp], "s", color=OI["green"], mec="white", ms=6, label="phi' campaigns")
    axs[2].set(ylabel="Strength")
    axs[3].plot(YEARS, h["fs"], color=OI["black"], label="FS(t) oracle")
    axs[3].plot([int(y) for y in camp], [camp[y]["fs_spencer"] for y in camp], "s", color=OI["red"], mec="white", ms=6, label="FS Spencer (campaigns)")
    mark_threshold(axs[3])
    axs[3].set(ylabel="FS", xlabel="Year", xlim=(2017, 2029))
    for ax in axs:
        ax.axvline(gen.YEAR0 + gen.T_F_HIST, color=OI["grey"], lw=0.8)
        ax.legend(fontsize=7, loc="best", ncol=2)
    axs[0].set_title("Cortinas historical trajectory: forcing, pore pressure, strength and FS (deterioration -> failure -> AGD recovery)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(HERE / "fig_dl_forcing_regimes.png"); plt.close(fig)


def main():
    ref, M, log, sur = load()
    const = ref["constants"]
    fig_learning_curves(log)
    fig_fs_prediction(ref, sur, const)
    fig_benchmark(M)
    units, r_f = fig_memory_states(ref, sur)
    fig_forcing_regimes(ref)
    M.setdefault("models", {}).setdefault("lstm", {})["memory_units"] = {"top_cell_units": units, "forget_gate_vs_ru_pooled_r": r_f}
    json.dump(M, open(HERE / "dl_metrics.json", "w"), indent=1)
    print("Figures: fig_dl_learning_curves.png, fig_dl_fs_prediction.png, fig_dl_benchmark_speedup.png, "
          "fig_dl_memory_states.png, fig_dl_forcing_regimes.png (300 DPI)")


if __name__ == "__main__":
    main()
