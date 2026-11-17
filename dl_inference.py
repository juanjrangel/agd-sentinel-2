"""
Pure-NumPy inference for the AGD-Sentinel LSTM surrogate (no deep-learning framework
needed at runtime). Reproduces the trained network exactly:

  x -> LSTM(12->64) -> LayerNorm -> Dropout -> LSTM(64->64) -> LayerNorm -> Dropout
    -> Linear(64->32) -> ReLU -> Linear(32->1) -> softplus = FS(t)
  auxiliary head: Linear(64->1) + sigmoid on h_T -> t_crit / T_h

Gate order follows the training framework convention (i, f, g, o):
  i_t = s(W_i x + U_i h + b_i)      f_t = s(W_f x + U_f h + b_f)
  g_t = tanh(W_g x + U_g h + b_g)   o_t = s(W_o x + U_o h + b_o)
  C_t = f_t * C_{t-1} + i_t * g_t   h_t = o_t * tanh(C_t)
MC Dropout: K stochastic passes evaluated in one batch (dropout active, LayerNorm is
deterministic) -> mean, std, mean +- 2 std and P5-P95 bands.
Weights: models/dl_lstm.npz (allow_pickle=False); scaler: models/dl_scaler.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
LN_EPS = 1e-5


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * w + b


def lstm_layer(x, w_ih, w_hh, b, return_states=False):
    """Single-layer LSTM over x (n, T, d). Input projection is precomputed for all steps."""
    n, T, _ = x.shape
    H = w_hh.shape[1]
    xp = x @ w_ih.T + b
    w_hhT = np.ascontiguousarray(w_hh.T)
    h = np.zeros((n, H), x.dtype); c = np.zeros((n, H), x.dtype)
    out = np.empty((n, T, H), x.dtype)
    if return_states:
        C = np.empty_like(out); G = {k: np.empty_like(out) for k in "ifgo"}
    for k in range(T):
        z = xp[:, k] + h @ w_hhT
        s = _sigmoid(z)
        i, f, o = s[:, :H], s[:, H:2 * H], s[:, 3 * H:]
        g = np.tanh(z[:, 2 * H:3 * H])
        c = f * c + i * g
        h = o * np.tanh(c)
        out[:, k] = h
        if return_states:
            C[:, k] = c
            for key, v in zip("ifgo", (i, f, g, o)): G[key][:, k] = v
    return (out, C, G) if return_states else out


class LSTMSurrogate:
    def __init__(self, weights, scaler):
        self.w = {k: np.asarray(v, np.float32) for k, v in weights.items()}
        self.mean = np.asarray(scaler["mean"], np.float32)
        self.std = np.asarray(scaler["std"], np.float32)
        self.scaled_idx = np.asarray(scaler["scaled_idx"], int)
        self.features = scaler["features"]
        self.th = float(scaler["T_h"])
        self.dropout = float(scaler.get("dropout", 0.10))

    @classmethod
    def load(cls, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        with np.load(model_dir / "dl_lstm.npz", allow_pickle=False) as z:
            weights = {k: z[k] for k in z.files}
        return cls(weights, json.load(open(model_dir / "dl_scaler.json")))

    def scale(self, X):
        Xs = np.array(X, np.float32, copy=True)
        Xs[..., self.scaled_idx] = (Xs[..., self.scaled_idx] - self.mean) / self.std
        return Xs

    def _drop(self, x, rng):
        if rng is None: return x
        keep = rng.random(x.shape, dtype=np.float32) >= self.dropout
        return x * keep / np.float32(1.0 - self.dropout)

    def forward(self, X, rng=None, return_states=False):
        """X: unscaled features (n, T, 12). rng != None activates dropout (MC pass)."""
        w = self.w
        x = self.scale(X)
        r1 = lstm_layer(x, w["l1_w_ih"], w["l1_w_hh"], w["l1_b"], return_states)
        h1 = r1[0] if return_states else r1
        a1 = self._drop(_layer_norm(h1, w["ln1_w"], w["ln1_b"]), rng)
        r2 = lstm_layer(a1, w["l2_w_ih"], w["l2_w_hh"], w["l2_b"], return_states)
        h2 = r2[0] if return_states else r2
        a2 = self._drop(_layer_norm(h2, w["ln2_w"], w["ln2_b"]), rng)
        hid = np.maximum(a2 @ w["fc1_w"].T + w["fc1_b"], 0.0)
        fs = np.logaddexp(0.0, (hid @ w["fc2_w"].T + w["fc2_b"])[..., 0])
        tc = _sigmoid(a2[:, -1] @ w["aux_w"].T + w["aux_b"])[:, 0] * self.th
        if not return_states:
            return fs, tc
        states = {"h1": h1, "C1": r1[1], "gates1": r1[2], "h2": h2, "C2": r2[1], "gates2": r2[2]}
        return fs, tc, states

    def predict(self, X):
        return self.forward(X)[0]

    def mc_dropout(self, X, K=50, seed=0):
        """K stochastic passes in a single batch. Returns dict of (n, T) arrays."""
        X = np.asarray(X, np.float32)
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        fs, _ = self.forward(np.repeat(X, K, axis=0), rng=rng)
        S = fs.reshape(n, K, -1)
        mu, sd = S.mean(1), S.std(1)
        p5, p95 = np.percentile(S, [5, 95], axis=1)
        return {"mean": mu, "std": sd, "lo2s": mu - 2 * sd, "hi2s": mu + 2 * sd,
                "p5": p5, "p95": p95, "samples": S}


def load_surrogate(model_dir=MODEL_DIR):
    """Returns the surrogate or None when the weights are not available."""
    try:
        return LSTMSurrogate.load(model_dir)
    except (OSError, KeyError, ValueError):
        return None
