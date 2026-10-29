#!/usr/bin/env python3
"""
Full replication pipeline for:
  Shen et al. (2024), "Intelligent crude oil price probability forecasting:
  Deep learning models and industry applications", Computers in Industry 163, 104150.

Datasets  : Brent and WTI, daily and weekly (EIA Excel files in data/)  -> 4 datasets
Models    : EEMD-SVR, EEMD-BiLSTM, EEMD-CNN, CNN-BiLSTM, EEMD-CNN-BiLSTM
Variants  : "point"  (RMSE / MAPE, Table 4)
            "qr"     (Pinball / CRPS / PINAW / PICP, Table 6)
Repeats   : 20 independent runs per neural model
Tests     : Friedman + Iman-Davenport + Nemenyi on MAPE (Table 5) and CRPS (Table 7)
Figures   : 5, 6, 7, 8, 9, 10, 11, 12, 13, 14

Commands (run from the folder that contains data/):
  python replicate_paper.py check
  python replicate_paper.py decompose --workers 6
  python replicate_paper.py train     --workers 4 --threads 2
  python replicate_paper.py report
  python replicate_paper.py all       --workers 4 --threads 2

Everything is resumable: finished pieces are stored under results/ and skipped.

The paper does not specify every detail. Each choice made here is listed in
CHOICES below and is printed at the start of every command.
"""
import os
import sys
import time
import json
import argparse
import traceback
import itertools
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import hashlib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

# =====================================================================
# CONFIGURATION  (values from the paper unless marked CHOICE)
# =====================================================================
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT = HERE / "results"

LAG = 5                    # paper: 5 past points predict the 6th
EPOCHS = 150               # paper: max epochs 150
BATCH = 16                 # paper: mini-batch 16
DROPOUT = 0.2              # paper
EEMD_TRIALS = 100          # paper: NE = 100
EEMD_NSTD = 0.2            # paper: Nstd = 0.2 x std of the series
EEMD_SEED = 42             # CHOICE: one fixed noise realisation for all decompositions
N_RUNS = 20                # paper: 20 runs
BASE_SEED = 1000           # CHOICE: run r uses seeds BASE_SEED + 100*r + component
SVR_PARAMS = dict(C=64, gamma=2, epsilon=0.001)   # paper (from Huang et al. 2022)

QUANTILES = np.round(np.arange(1, 100) / 100, 2).astype("float32")   # 0.01 ... 0.99
MEDIAN_IDX = 49

# CHOICE switches
QR_MODE = "joint"          # "joint": one net with 99 outputs. "per_tau": 99 separate nets
                           #   (Fig. 4 of the paper loops over tau; per_tau is ~99x slower)
CNN_TO_LSTM = "sequence"   # "flatten": Fig.1 flatten layer -> sequence of length 1 -> BiLSTM
                           # "sequence": keep the (5, 32) feature maps as the BiLSTM sequence
DELTA_TARGETS = False      # False = paper. True = predict the change from the last value

CHOICES = f"""
Choices the paper leaves open (change the constants at the top of this file to alter them):
  1. Scaling to [0,1] uses TRAINING min/max only (no test information).
  2. Training set is decomposed once; at every test step the history up to that step is
     re-decomposed (expanding window), and the last {LAG} values of each component are the
     network input. Training components and test components therefore come from different
     decompositions, as the paper describes.
  3. EEMD noise = {EEMD_NSTD} x std(series), converted to PyEMD's range-based noise_width.
     One fixed noise seed ({EEMD_SEED}); the 20 runs differ in network initialisation/shuffling.
  4. Components are matched to models by position; extra components are merged into the
     residue, missing ones padded with residue mean (not zeros).
  5. QR mode = {QR_MODE}; CNN->BiLSTM link = {CNN_TO_LSTM}; delta targets = {DELTA_TARGETS}.
  6. Point models use MSE loss and one output; "-QR" models use pinball loss.
     Every component is z-scored by its training std (rescaled after prediction).
  7. Quantiles are NOT sorted. Crossing is left visible in Pinball/CRPS so they reflect
     real model quality; sorting would hide it, not fix it.
  8. EEMD-SVR-QR = sum per-component point forecasts FIRST, then add residual quantiles
     from the SUMMED training residual. (Q_tau(sum) != sum(Q_tau) in general.)
     SVR is deterministic, so it is run once and counted 20 times.
  9. CRPS is the numerical integral of Eq. (16) using a Gaussian-KDE CDF fitted to the
     99 quantiles. PICP uses the 0.01-0.99 interval; PINAW divides by the training range.
 10. Adam with Keras defaults (learning rate 1e-3); no early stopping, no validation split.
"""

CFG_HASH = hashlib.md5(
    (CHOICES + f"|LAG={LAG}|EPOCHS={EPOCHS}|BATCH={BATCH}|DROPOUT={DROPOUT}"
     f"|QR={QR_MODE}|CNN2LSTM={CNN_TO_LSTM}|DELTA={DELTA_TARGETS}"
     f"|EEMD={EEMD_TRIALS},{EEMD_NSTD},{EEMD_SEED}").encode()
).hexdigest()[:8]

DATASETS = {
    "brent_daily":  dict(xls="RBRTEd", start="1987-06-01", end="2021-06-25",
                         n=8649, n_train=5767, label="Brent daily",  freq="daily"),
    "wti_daily":    dict(xls="RWTCd",  start="1986-01-06", end="2021-07-30",
                         n=8964, n_train=5978, label="WTI daily",    freq="daily"),
    "brent_weekly": dict(xls="RBRTEw", start="1987-06-05", end="2021-06-25",
                         n=1778, n_train=1185, label="Brent weekly", freq="weekly"),
    "wti_weekly":   dict(xls="RWTCw",  start="1986-01-10", end="2021-07-30",
                         n=1856, n_train=1237, label="WTI weekly",   freq="weekly"),
}

MODELS = {
    "EEMD-SVR":        dict(kind="svr",        eemd=True),
    "EEMD-BiLSTM":     dict(kind="bilstm",     eemd=True),
    "EEMD-CNN":        dict(kind="cnn",        eemd=True),
    "CNN-BiLSTM":      dict(kind="cnn_bilstm", eemd=False),
    "EEMD-CNN-BiLSTM": dict(kind="cnn_bilstm", eemd=True),
}
MODEL_ORDER = list(MODELS)
PROPOSED = "EEMD-CNN-BiLSTM"

# Values printed in the paper (Table 4: RMSE, MAPE; Table 6: Pinball, CRPS, PINAW, PICP)
PAPER_T4 = {
    "brent_daily":  {"EEMD-SVR": (4.3428, 4.5058), "EEMD-BiLSTM": (2.1119, 3.0148), "EEMD-CNN": (2.2875, 3.4101),
                     "CNN-BiLSTM": (2.5415, 3.4785), "EEMD-CNN-BiLSTM": (1.4278, 1.7517)},
    "brent_weekly": {"EEMD-SVR": (2.6718, 3.8490), "EEMD-BiLSTM": (2.1819, 3.0735), "EEMD-CNN": (2.2156, 2.9098),
                     "CNN-BiLSTM": (3.0074, 3.8380), "EEMD-CNN-BiLSTM": (1.8621, 2.6685)},
    "wti_daily":    {"EEMD-SVR": (3.6266, 4.3705), "EEMD-BiLSTM": (1.6310, 2.0272), "EEMD-CNN": (1.7608, 2.4030),
                     "CNN-BiLSTM": (2.4194, 2.8534), "EEMD-CNN-BiLSTM": (1.3954, 1.8661)},
    "wti_weekly":   {"EEMD-SVR": (2.5616, 4.0145), "EEMD-BiLSTM": (1.8327, 2.8837), "EEMD-CNN": (1.8747, 2.9215),
                     "CNN-BiLSTM": (2.6759, 4.3720), "EEMD-CNN-BiLSTM": (1.4390, 2.1581)},
}
PAPER_T6 = {
    "brent_daily":  {"EEMD-SVR": (1.6724, 2.5823, 0.3214, 85.4285), "EEMD-BiLSTM": (1.2056, 2.0516, 0.1806, 100.0),
                     "EEMD-CNN": (1.1729, 1.9811, 0.1775, 100.0), "CNN-BiLSTM": (1.2994, 2.1878, 0.1621, 78.1055),
                     "EEMD-CNN-BiLSTM": (1.0308, 1.7602, 0.1330, 100.0)},
    "brent_weekly": {"EEMD-SVR": (1.8295, 2.9230, 0.3941, 82.3126), "EEMD-BiLSTM": (1.4892, 2.5368, 0.2253, 100.0),
                     "EEMD-CNN": (1.2641, 2.1670, 0.2033, 99.1852), "CNN-BiLSTM": (1.4184, 2.2778, 0.1782, 76.6892),
                     "EEMD-CNN-BiLSTM": (1.2370, 2.0759, 0.1869, 100.0)},
    "wti_daily":    {"EEMD-SVR": (1.5926, 2.6518, 0.2891, 82.8236), "EEMD-BiLSTM": (1.1870, 2.1497, 0.2253, 99.9665),
                     "EEMD-CNN": (1.1918, 1.8323, 0.1551, 99.9665), "CNN-BiLSTM": (1.2386, 2.1761, 0.1221, 73.4851),
                     "EEMD-CNN-BiLSTM": (1.0211, 1.7062, 0.1256, 99.9665)},
    "wti_weekly":   {"EEMD-SVR": (1.8043, 2.7845, 0.3052, 86.1983), "EEMD-BiLSTM": (1.2387, 2.0836, 0.2154, 100.0),
                     "EEMD-CNN": (1.2590, 2.1601, 0.1880, 99.3752), "CNN-BiLSTM": (1.3234, 2.2423, 0.1732, 73.6487),
                     "EEMD-CNN-BiLSTM": (1.1152, 1.8879, 0.1811, 100.0)},
}


# =====================================================================
# DATA
# =====================================================================
def dsdir(ds):
    p = OUT / ds
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_df(ds, verbose=False):
    c = DATASETS[ds]
    xls = next((DATA_DIR / (c["xls"] + e) for e in (".xls", ".xlsx")
                if (DATA_DIR / (c["xls"] + e)).exists()), None)
    if xls is None:
        raise FileNotFoundError(f"Missing {c['xls']}.xls in {DATA_DIR} (EIA Excel download).")
    sys.path.insert(0, str(HERE))
    from convert_xls import read_eia
    _, _, df = read_eia(xls)
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    raw = len(df)
    df = df[(df["Date"] >= c["start"]) & (df["Date"] <= c["end"])].reset_index(drop=True)
    if verbose:
        print(f"  {c['label']:<13} file rows {raw:>5} | in paper window {len(df):>5} "
              f"| paper {c['n']:>5} | {df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}")
    return df


def prepare(ds):
    c = DATASETS[ds]
    df = load_df(ds)
    prices = df["Price"].to_numpy("float64")
    n = len(prices)
    if n != c["n"]:
        print(f"  WARNING [{ds}]: n={n} != paper_n={c['n']} "
              f"(short by {c['n'] - n}); using int(2/3 * n) split")
        split = int(n * 2 / 3)
    else:
        split = c["n_train"]
    lo, hi = prices[:split].min(), prices[:split].max()
    scaled = (prices - lo) / (hi - lo)
    return dict(df=df, prices=prices, n=n, split=split, lo=lo, hi=hi, scale=hi - lo,
                scaled=scaled, test_targets=np.arange(split, n))


def windows(series, targets, lag=LAG):
    X = np.stack([series[t - lag:t] for t in targets])
    return X.astype("float32"), series[targets].astype("float32")


# =====================================================================
# EEMD
# =====================================================================
def eemd_decompose(x, seed=EEMD_SEED):
    """EEMD with the paper's settings. Returns IMFs plus residue; rows sum to x."""
    from PyEMD import EEMD
    nw = EEMD_NSTD * x.std() / np.ptp(x)       # PyEMD scales noise by range; paper by std
    e = EEMD(trials=EEMD_TRIALS, noise_width=nw, parallel=False)
    e.noise_seed(seed)
    imfs = np.asarray(e.eemd(x, np.arange(len(x))))
    res = x - imfs.sum(axis=0)
    if np.max(np.abs(res)) > 1e-8:
        imfs = np.vstack([imfs, res[None, :]])
    return imfs


def align(comps, n):
    """Force a decomposition to have exactly n rows, residue last.
    If the test window produces FEWER components than training, pad in front of the
    residue with the residue's own mean (not zeros) so the network sees in-distribution
    inputs."""
    m = comps.shape[0]
    if m == n:
        return comps
    if m > n:
        return np.vstack([comps[:n - 1], comps[n - 1:].sum(axis=0, keepdims=True)])
    pad_val = float(comps[-1].mean())
    pad = np.full((n - m, comps.shape[1]), pad_val)
    return np.vstack([comps[:-1], pad, comps[-1:]])


_G = {}


def _init_decomp(scaled, split, n_comp):
    _G.update(scaled=scaled, split=split, n_comp=n_comp)


def _tail_chunk(args):
    path, j0, j1 = args
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    out = []
    for j in range(j0, j1):
        hist = _G["scaled"][:_G["split"] + j]
        c = align(eemd_decompose(hist), _G["n_comp"])
        out.append(c[:, -LAG:])
    tmp = path + ".tmp.npy"
    np.save(tmp, np.stack(out))
    os.replace(tmp, path)
    return path, time.time() - t0


def cmd_decompose(args):
    chunk = 25
    for ds in args.datasets:
        P = prepare(ds)
        d = dsdir(ds)
        print(f"\n=== EEMD: {DATASETS[ds]['label']} ===")
        tr_path = d / "decomp_train.npy"
        if tr_path.exists():
            comps = np.load(tr_path)
        else:
            t0 = time.time()
            comps = eemd_decompose(P["scaled"][:P["split"]])
            np.save(tr_path, comps)
            print(f"  training decomposition: {comps.shape[0]} components "
                  f"(IMFs + residue) in {time.time() - t0:.0f} s")
        n_comp = comps.shape[0]
        gap = np.max(np.abs(P["scaled"][:P["split"]] - comps.sum(axis=0)))
        print(f"  components: {n_comp}; max |signal - sum| = {gap:.2e}")
        assert gap < 1e-8, f"Decomposition does not reconstruct: gap={gap}"

        n_test = P["n"] - P["split"]
        tdir = d / "tails"
        tdir.mkdir(exist_ok=True)
        jobs = [(str(tdir / f"tail_{j0:05d}.npy"), j0, min(j0 + chunk, n_test))
                for j0 in range(0, n_test, chunk)]
        todo = [j for j in jobs if not os.path.exists(j[0])]
        print(f"  expanding-window test decompositions: {n_test} steps, "
              f"{len(todo)}/{len(jobs)} chunks to do")
        if todo:
            ctx = mp.get_context("spawn")
            t0, done = time.time(), 0
            with ctx.Pool(args.workers, initializer=_init_decomp,
                          initargs=(P["scaled"], P["split"], n_comp)) as pool:
                for path, sec in pool.imap_unordered(_tail_chunk, todo):
                    done += 1
                    el = time.time() - t0
                    eta = el / done * (len(todo) - done)
                    print(f"    chunk {done}/{len(todo)} ({Path(path).name}) "
                          f"elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min")
        parts = [np.load(p) for p, _, _ in jobs]
        tails = np.concatenate(parts, axis=0)
        assert tails.shape == (n_test, n_comp, LAG), tails.shape
        np.save(d / "tails.npy", tails)
        print(f"  saved tails.npy {tails.shape}")


# =====================================================================
# MODELS
# =====================================================================
def build_net(kind, taus):
    """taus=None -> point model (MSE, 1 output). Otherwise pinball over `taus`."""
    import tensorflow as tf
    from tensorflow import keras
    L = keras.layers
    n_out = 1 if taus is None else len(taus)

    inp = keras.Input(shape=(LAG, 1))
    x = inp
    if kind in ("cnn", "cnn_bilstm"):
        x = L.Conv1D(16, 5, padding="same", activation="relu")(x)
        x = L.MaxPooling1D(3, strides=1, padding="same")(x)
        x = L.Conv1D(32, 5, padding="same", activation="relu")(x)
        x = L.MaxPooling1D(3, strides=1, padding="same")(x)
        if kind == "cnn":
            x = L.Flatten()(x)
        elif CNN_TO_LSTM == "flatten":
            x = L.Flatten()(x)
            x = L.Reshape((1, -1))(x)
    if kind in ("bilstm", "cnn_bilstm"):
        x = L.Bidirectional(L.LSTM(200, return_sequences=True), merge_mode="sum")(x)   # Eq. (5)
        x = L.Dropout(DROPOUT)(x)
        x = L.Bidirectional(L.LSTM(150), merge_mode="sum")(x)
        x = L.Dropout(DROPOUT)(x)
    out = L.Dense(n_out)(x)
    model = keras.Model(inp, out)

    if taus is None:
        loss = "mse"
    else:
        q = tf.constant(np.asarray(taus, dtype="float32"))

        def loss(y_true, y_pred):
            y = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))
            e = y - tf.cast(y_pred, tf.float32)
            return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))
    model.compile(optimizer=keras.optimizers.Adam(), loss=loss)
    return model


def fit_predict_nn(series_list, test_X, split, kind, variant, seed0):
    from tensorflow import keras
    tr_targets = np.arange(LAG, split)
    total = 0.0
    for k, (s, Xte_raw) in enumerate(zip(series_list, test_X)):
        s = np.asarray(s)[:split]
        sd = max(float(s.std()), 1e-8)   # always normalize by training std (FIX 3)
        z = s / sd
        Xtr, ytr = windows(z, tr_targets)
        Xtr = Xtr[..., None]
        Xte = (np.asarray(Xte_raw) / sd).astype("float32")[..., None]
        shift_tr = Xtr[:, -1, 0] if DELTA_TARGETS else 0.0
        shift_te = Xte[:, -1, 0] if DELTA_TARGETS else 0.0
        y_fit = ytr - shift_tr

        if variant == "point":
            keras.utils.set_random_seed(seed0 + k)
            m = build_net(kind, None)
            m.fit(Xtr, y_fit, epochs=EPOCHS, batch_size=BATCH, shuffle=True, verbose=0)
            pred = m.predict(Xte, batch_size=1024, verbose=0)[:, 0] + shift_te
            pred = pred * sd
        elif QR_MODE == "joint":
            keras.utils.set_random_seed(seed0 + k)
            m = build_net(kind, QUANTILES)
            m.fit(Xtr, y_fit, epochs=EPOCHS, batch_size=BATCH, shuffle=True, verbose=0)
            pred = m.predict(Xte, batch_size=1024, verbose=0)
            pred = pred + (shift_te[:, None] if DELTA_TARGETS else 0.0)
            pred = pred * sd                # FIX 2a: no sort
        else:  # per_tau
            pred = np.zeros((len(Xte), len(QUANTILES)))
            for qi, tau in enumerate(QUANTILES):
                keras.utils.set_random_seed(seed0 + k * 1000 + qi)
                m = build_net(kind, [tau])
                m.fit(Xtr, y_fit, epochs=EPOCHS, batch_size=BATCH, shuffle=True, verbose=0)
                pred[:, qi] = (m.predict(Xte, batch_size=1024, verbose=0)[:, 0] + shift_te) * sd
            # FIX 2: no sort in per_tau either
        keras.backend.clear_session()
        total = total + pred
    return total                         # FIX 2b: no final sort


def fit_predict_svr(series_list, test_X, split, variant):
    """SVR per EEMD component; sum point forecasts FIRST, then add residual
    quantiles from the SUMMED training residual. Fixes Q_tau(sum) != sum(Q_tau)."""
    from sklearn.svm import SVR
    tr_targets = np.arange(LAG, split)

    total_point = 0.0
    total_train_pred = 0.0
    for s, Xte in zip(series_list, test_X):
        Xtr, ytr = windows(np.asarray(s)[:split], tr_targets)
        m = SVR(kernel="rbf", **SVR_PARAMS).fit(Xtr, ytr)
        total_point = total_point + m.predict(np.asarray(Xte, dtype="float32"))
        total_train_pred = total_train_pred + m.predict(Xtr)

    if variant == "point":
        return total_point

    y_train_sum = sum(np.asarray(s)[:split][tr_targets] for s in series_list)
    res_sum = y_train_sum - total_train_pred
    q_res = np.quantile(res_sum, QUANTILES)
    return total_point[:, None] + q_res[None, :]


def pred_path(ds, model, variant, run):
    p = dsdir(ds) / "preds"
    p.mkdir(exist_ok=True)
    return p / f"{model}_{variant}_run{run:02d}_{CFG_HASH}.npz"


def run_job(job):
    ds, mname, variant, run, threads = job
    out = pred_path(ds, mname, variant, run)
    if out.exists():
        return job, 0.0, "skip"
    try:
        if MODELS[mname]["kind"] != "svr":
            import tensorflow as tf
            try:
                tf.config.threading.set_intra_op_parallelism_threads(threads)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except RuntimeError:
                pass
        P = prepare(ds)
        cfg = MODELS[mname]
        t0 = time.time()
        if cfg["eemd"]:
            comps = np.load(dsdir(ds) / "decomp_train.npy")
            tails = np.load(dsdir(ds) / "tails.npy")
            series_list = [c for c in comps]
            test_X = [tails[:, k, :] for k in range(comps.shape[0])]
        else:
            series_list = [P["scaled"][:P["split"]]]
            test_X = [np.stack([P["scaled"][t - LAG:t] for t in P["test_targets"]])]
        if cfg["kind"] == "svr":
            pred = fit_predict_svr(series_list, test_X, P["split"], variant)
        else:
            seed0 = BASE_SEED + 100 * run
            pred = fit_predict_nn(series_list, test_X, P["split"], cfg["kind"], variant, seed0)
        sec = time.time() - t0
        tmp = str(out) + ".tmp.npz"
        np.savez(tmp, pred=pred.astype("float64"), seconds=sec)
        os.replace(tmp, out)
        return job, sec, "ok"
    except Exception:
        return job, 0.0, "ERROR\n" + traceback.format_exc()


def cmd_train(args):
    jobs = []
    for run in range(args.runs):                      # run-major so complete runs accrue early
        for ds in args.datasets:
            for mname in args.models:
                for variant in args.variants:
                    if MODELS[mname]["kind"] == "svr" and run > 0:
                        continue                      # deterministic: one run is enough
                    if not pred_path(ds, mname, variant, run).exists():
                        jobs.append((ds, mname, variant, run, args.threads))
    for ds in args.datasets:
        for f in ("decomp_train.npy", "tails.npy"):
            if any(MODELS[m]["eemd"] for m in args.models) and not (dsdir(ds) / f).exists():
                sys.exit(f"{ds}: {f} missing. Run `decompose` first.")
    print(f"{len(jobs)} training jobs to do ({args.workers} workers x {args.threads} threads)")
    if not jobs:
        return
    if args.workers == 1:
        # WORKER-FIX: macOS arm64 TF deadlocks inside multiprocessing.Pool workers.
        t0, done = time.time(), 0
        for job in jobs:
            job_r, sec, status = run_job(job)
            done += 1
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            ds, mname, variant, run, _ = job_r
            print(f"[{done}/{len(jobs)}] {ds} {mname} {variant} run{run:02d}: {status.splitlines()[0]} "
                  f"({sec / 60:.1f} min) | elapsed {el / 3600:.2f} h, ETA {eta / 3600:.2f} h")
            if status.startswith("ERROR"):
                print(status)
        return
    ctx = mp.get_context("spawn")
    t0, done = time.time(), 0
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for job, sec, status in pool.imap_unordered(run_job, jobs):
            done += 1
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            ds, mname, variant, run, _ = job
            print(f"[{done}/{len(jobs)}] {ds} {mname} {variant} run{run:02d}: {status.splitlines()[0]} "
                  f"({sec / 60:.1f} min) | elapsed {el / 3600:.2f} h, ETA {eta / 3600:.2f} h")
            if status.startswith("ERROR"):
                print(status)


# =====================================================================
# METRICS
# =====================================================================
def rmse(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mape(y, yh):
    return float(np.mean(np.abs((y - yh) / y)) * 100)


def pinball(y, q):
    d = y[:, None] - q
    return float(np.mean(np.maximum(QUANTILES * d, (QUANTILES - 1) * d)))


def picp(y, q):
    return float(np.mean((y >= q[:, 0]) & (y <= q[:, -1])) * 100)


def pinaw(y, q, scale):
    return float(np.mean(q[:, -1] - q[:, 0]) / scale)


_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def crps_kde(y, q, grid_n=2000):
    """Mean over time of Eq. (16): integral of (F(x) - H(x - y))^2 dx, with F a Gaussian-KDE
    CDF fitted to the 99 quantiles of each forecast."""
    from scipy.stats import norm
    T, m = q.shape
    out = np.empty(T)
    for t in range(T):
        s = q[t]
        h = max(float(s.std()) * m ** (-0.2), 1e-6)            # Scott's rule
        a = min(s.min(), y[t]) - 20 * h
        b = max(s.max(), y[t]) + 20 * h
        g = np.linspace(a, b, grid_n)
        F = norm.cdf((g[:, None] - s[None, :]) / h).mean(axis=1)
        H = (g >= y[t]).astype(float)
        out[t] = _trapezoid((F - H) ** 2, g)
    return float(out.mean())


def friedman_nemenyi(M):
    """M: (L runs, k models) of an error metric (lower = better). Eqs. (19)-(22)."""
    from scipy.stats import rankdata, f as fdist
    L, k = M.shape
    ranks = np.apply_along_axis(rankdata, 1, M)
    R = ranks.mean(axis=0)
    chi2 = 12 * L / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    denom = L * (k - 1) - chi2
    FF = (L - 1) * chi2 / denom if denom > 1e-12 else float("inf")
    crit = float(fdist.ppf(0.95, k - 1, (k - 1) * (L - 1)))
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164}[k]
    CD = q_alpha * np.sqrt(k * (k + 1) / (6 * L))
    return dict(R=R, chi2=float(chi2), FF=float(FF), crit=crit, q=q_alpha, CD=float(CD), L=L)


def available_runs(ds, mname, variant):
    if MODELS[mname]["kind"] == "svr":
        return [0] if pred_path(ds, mname, variant, 0).exists() else []
    return [r for r in range(N_RUNS) if pred_path(ds, mname, variant, r).exists()]


def metrics_for(ds, P, mname, variant, run):
    cache = dsdir(ds) / "metrics"
    cache.mkdir(exist_ok=True)
    cp = cache / f"{mname}_{variant}_run{run:02d}_{CFG_HASH}.json"
    if cp.exists():
        return json.loads(cp.read_text())
    z = np.load(pred_path(ds, mname, variant, run))
    pred = z["pred"] * P["scale"] + P["lo"]
    y = P["prices"][P["test_targets"]]
    if variant == "point":
        m = dict(RMSE=rmse(y, pred), MAPE=mape(y, pred))
    else:
        m = dict(Pinball=pinball(y, pred), CRPS=crps_kde(y, pred),
                 PINAW=pinaw(y, pred, P["scale"]), PICP=picp(y, pred))
    m["seconds"] = float(z["seconds"])
    cp.write_text(json.dumps(m))
    return m


def collect(ds, P, variant):
    """-> {model: [metric dicts, one per run]}. SVR (deterministic) is replicated N_RUNS times."""
    res = {}
    for mname in MODEL_ORDER:
        runs = available_runs(ds, mname, variant)
        if not runs:
            continue
        ms = [metrics_for(ds, P, mname, variant, r) for r in runs]
        if MODELS[mname]["kind"] == "svr":
            ms = ms * N_RUNS
        res[mname] = ms
    return res


# =====================================================================
# REPORT
# =====================================================================
def plot_cd(ax, names, R, CD, title):
    order = np.argsort(R)
    best = R[order[0]]
    for yi, i in enumerate(order):
        ax.plot([R[i], R[i] + CD], [yi, yi], lw=1.5)
        ax.plot(R[i], yi, "o")
    ax.axvline(best, ls="--", c="k", lw=0.8)
    ax.axvline(best + CD, ls="--", c="k", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean rank")
    ax.set_title(title, fontsize=9)


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    OUT.mkdir(exist_ok=True)
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    point_tables, qr_tables = {}, {}
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    for ds in args.datasets:
        P = prepare(ds)
        lab = DATASETS[ds]["label"]
        pt = collect(ds, P, "point")
        qr = collect(ds, P, "qr")
        point_tables[ds], qr_tables[ds] = pt, qr

        say(f"\n==================== {lab} ====================")
        say(f"Table 4 analogue (point forecasts; mean of runs; paper value in brackets)")
        say(f"{'model':<17}{'RMSE':>20}{'MAPE %':>20}{'runs':>6}{'sec/run':>10}")
        rows = []
        for m in MODEL_ORDER:
            if m not in pt:
                continue
            a = lambda key: np.mean([r[key] for r in pt[m]])
            pr, pm = PAPER_T4[ds][m]
            say(f"{m:<17}{a('RMSE'):>11.4f} [{pr:.4f}]{a('MAPE'):>11.4f} [{pm:.4f}]"
                f"{len(pt[m]):>6}{a('seconds'):>10.0f}")
            rows.append(dict(model=m, RMSE=a("RMSE"), MAPE=a("MAPE"),
                             RMSE_sd=np.std([r["RMSE"] for r in pt[m]]),
                             MAPE_sd=np.std([r["MAPE"] for r in pt[m]]), seconds=a("seconds"),
                             paper_RMSE=pr, paper_MAPE=pm))
        pd.DataFrame(rows).to_csv(OUT / f"table4_{ds}.csv", index=False)

        say(f"\nTable 6 analogue (probabilistic; mean of runs; paper value in brackets)")
        say(f"{'model':<21}{'Pinball':>17}{'CRPS':>17}{'PINAW':>17}{'PICP %':>19}")
        rows = []
        for m in MODEL_ORDER:
            if m not in qr:
                continue
            a = lambda key: np.mean([r[key] for r in qr[m]])
            pp, pc, pw, pk = PAPER_T6[ds][m]
            say(f"{m + '-QR':<21}{a('Pinball'):>8.4f} [{pp:.4f}]{a('CRPS'):>8.4f} [{pc:.4f}]"
                f"{a('PINAW'):>8.4f} [{pw:.4f}]{a('PICP'):>9.2f} [{pk:.2f}]")
            rows.append(dict(model=m + "-QR", Pinball=a("Pinball"), CRPS=a("CRPS"), PINAW=a("PINAW"),
                             PICP=a("PICP"), seconds=a("seconds"),
                             paper_Pinball=pp, paper_CRPS=pc, paper_PINAW=pw, paper_PICP=pk))
        pd.DataFrame(rows).to_csv(OUT / f"table6_{ds}.csv", index=False)

        # Friedman / Nemenyi (Tables 5 and 7, Figs. 8 and 10)
        for tag, tab, key, fig_id in (("MAPE (Table 5, Fig. 8)", pt, "MAPE", 8),
                                      ("CRPS (Table 7, Fig. 10)", qr, "CRPS", 10)):
            names = [m for m in MODEL_ORDER if m in tab]
            if len(names) < 3:
                continue
            Ls = {m: len(tab[m]) for m in names}
            if len(set(Ls.values())) > 1:
                print(f"  WARNING: ragged runs {Ls}; using L={min(Ls.values())} "
                      f"(results below are PRELIMINARY)")
            L = min(Ls.values())
            M = np.array([[tab[m][r][key] for m in names] for r in range(L)])
            st = friedman_nemenyi(M)
            say(f"\nFriedman + Nemenyi on {key}, {lab}: L={L} runs, k={len(names)} models")
            say(f"  chi2_F={st['chi2']:.4f}  F_F={st['FF']:.4f}  critical F={st['crit']:.4f}  "
                f"q0.05={st['q']}  CD={st['CD']:.3f}  -> "
                f"{'null rejected' if st['FF'] > st['crit'] else 'null NOT rejected'}")
            say("  mean ranks: " + ", ".join(f"{n}={r:.3f}" for n, r in zip(names, st["R"])))
            fig, ax = plt.subplots(figsize=(5, 2.6))
            plot_cd(ax, names, st["R"], st["CD"], f"{lab}: mean rank of {key}")
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig{fig_id}_{ds}_cd_{key}.png", dpi=150)
            plt.close(fig)

        # Fig. 5 (series) and Fig. 6 (decomposition)
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.plot(P["df"]["Date"], P["prices"], lw=0.8)
        ax.set_title(f"{lab} crude oil price")
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig5_{ds}.png", dpi=150)
        plt.close(fig)
        dp = dsdir(ds) / "decomp_train.npy"
        if dp.exists():
            comps = np.load(dp) * P["scale"]
            fig, axs = plt.subplots(len(comps), 1, figsize=(7, 1.0 * len(comps)), sharex=True)
            for i, ax in enumerate(np.atleast_1d(axs)):
                ax.plot(comps[i], "r" if i == len(comps) - 1 else "b", lw=0.6)
                ax.set_ylabel("R" if i == len(comps) - 1 else f"imf{i + 1}", fontsize=7)
                ax.tick_params(labelsize=6)
            fig.suptitle(f"{lab}: EEMD of the training set (USD)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig6_{ds}.png", dpi=150)
            plt.close(fig)

        # Figs. 11/13: prediction intervals; Figs. 12/14: densities (proposed model, run 00)
        pp = pred_path(ds, PROPOSED, "qr", 0)
        if pp.exists():
            q = np.load(pp)["pred"] * P["scale"] + P["lo"]
            y = P["prices"][P["test_targets"]]
            dates = P["df"]["Date"].iloc[P["test_targets"]].values
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.fill_between(dates, q[:, 0], q[:, -1], color="red", alpha=0.25, label="Interval")
            ax.plot(dates, y, lw=0.8, label="Actual")
            ax.plot(dates, q[:, MEDIAN_IDX], lw=0.8, label="Median")
            ax.legend()
            ax.set_title(f"{lab}: interval prediction (0.01-0.99) and observation")
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig{11 if DATASETS[ds]['freq'] == 'daily' else 13}_{ds}.png", dpi=150)
            plt.close(fig)

            idxs = ([1, 255, 510, 765, 1020, 1275, 1530, 1785, 2040, 2295, 2550, 2805]
                    if DATASETS[ds]["freq"] == "daily"
                    else [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
            fig, axs = plt.subplots(3, 4, figsize=(11, 6))
            for ax, i in zip(axs.ravel(), idxs):
                t = min(i - 1, len(y) - 1)
                kde = gaussian_kde(q[t])
                g = np.linspace(q[t].min() - 5, q[t].max() + 5, 300)
                ax.plot(g, kde(g), "r")
                ax.axvline(y[t], c="b")
                ax.set_title(str(i), fontsize=8)
                ax.tick_params(labelsize=6)
            fig.suptitle(f"{lab}: probability density of 12 test points (EEMD-CNN-BiLSTM-QR)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig{12 if DATASETS[ds]['freq'] == 'daily' else 14}_{ds}.png", dpi=150)
            plt.close(fig)

    # Fig. 7 (RMSE, MAPE bars) and Fig. 9 (CRPS bars)
    for fig_id, tab, keys in ((7, point_tables, ("RMSE", "MAPE")), (9, qr_tables, ("CRPS",))):
        for market in ("brent", "wti"):
            fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.2))
            for ax, key in zip(np.atleast_1d(axs), keys):
                w = 0.15
                for mi, m in enumerate(MODEL_ORDER):
                    vals = []
                    for freq in ("daily", "weekly"):
                        t = tab.get(f"{market}_{freq}", {}).get(m)
                        vals.append(np.mean([r[key] for r in t]) if t else np.nan)
                    ax.bar(np.arange(2) + (mi - 2) * w, vals, w,
                           label=m + ("-QR" if fig_id == 9 else ""))
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Daily", "Weekly"])
                ax.set_title(f"{market.upper()} {key}")
            np.atleast_1d(axs)[0].legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig{fig_id}_{market}.png", dpi=150)
            plt.close(fig)

    (OUT / "report.txt").write_text("\n".join(lines))
    print(f"\nSaved report.txt, table CSVs and figures in {OUT}")


# =====================================================================
# CHECK
# =====================================================================
def cmd_check(args):
    print(CHOICES)
    print("Data check (paper's sample size in the 'paper' column):")
    for ds in DATASETS:
        try:
            load_df(ds, verbose=True)
        except Exception as e:
            print(f"  {DATASETS[ds]['label']:<13} {e}")
    n_nn = sum(1 for m in MODEL_ORDER if MODELS[m]["kind"] != "svr")
    jobs = 4 * (n_nn * 2 * N_RUNS + 2)
    print(f"\nTotal training jobs for the complete study: {jobs} "
          f"({n_nn} neural models x 2 variants x {N_RUNS} runs x 4 datasets, plus SVR).")
    print("Each EEMD neural job trains one network per component (about 12 for daily data).")
    print("Daily jobs dominate the cost. The log prints each job's time and a running ETA.")


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "decompose", "train", "report", "all"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    ap.add_argument("--variants", nargs="+", default=["point", "qr"], choices=["point", "qr"])
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--threads", type=int, default=2, help="TensorFlow threads per training worker")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.command == "check":
        cmd_check(args)
    elif args.command == "decompose":
        print(CHOICES)
        cmd_decompose(args)
    elif args.command == "train":
        print(CHOICES)
        cmd_train(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        print(CHOICES)
        cmd_decompose(args)
        cmd_train(args)
        cmd_report(args)


if __name__ == "__main__":
    main()