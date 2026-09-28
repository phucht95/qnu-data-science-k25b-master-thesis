"""Section 3.3 (preview): reproduce Fig. 3 of Huang et al. (2013) on Cosexp, and extend it with HH.

Cosexp: 500 training and 500 test points; PWL-C-SVM with the additive map (11), M = 2, ..., 40
(M / n = 1, ..., 20; M = 2 is the linear classifier). gamma (= C) is tuned by 5-fold CV.
Every configuration is repeated on 10 independent draws of the data.
"""

import json
import os
import time

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import datasets2d as d2
import plotstyle as ps
from plotstyle import plt
from pwlsvm import PWLSVM

RESULTS = os.path.join(os.path.dirname(__file__), "results")
C_GRID = [2.0 ** k for k in range(-3, 12, 2)]
G_GRID = [2.0 ** k for k in range(-3, 10, 2)]
M_PER_DIM = list(range(1, 21))
N_REP = 10


def cv(seed):
    return StratifiedKFold(5, shuffle=True, random_state=seed)


def fit_time(params, X, y, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        PWLSVM(**params).fit(X, y)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run():
    rows, keep = [], {}
    for rep in range(N_REP):
        Xtr, ytr, Xte, yte = d2.make("cosexp", 500, 500, seed=100 + rep)
        rbf = GridSearchCV(make_pipeline(MinMaxScaler(), SVC(kernel="rbf")),
                           {"svc__C": C_GRID, "svc__gamma": G_GRID}, cv=cv(rep), n_jobs=-1).fit(Xtr, ytr)
        rows.append(dict(rep=rep, kind="rbf", mpd=0, acc=float((rbf.predict(Xte) == yte).mean()),
                         time=0.0, C=rbf.best_params_["svc__C"]))
        for kind in ("additive", "hh"):
            for mpd in M_PER_DIM:
                est = PWLSVM(kind=kind, m_per_dim=mpd, solver="libsvm", random_state=rep)
                gs = GridSearchCV(est, {"C": C_GRID}, cv=cv(rep), n_jobs=-1).fit(Xtr, ytr)
                best = gs.best_estimator_
                acc = float((best.predict(Xte) == yte).mean())
                rows.append(dict(rep=rep, kind=kind, mpd=mpd, acc=acc,
                                 time=fit_time(best.get_params(), Xtr, ytr), C=gs.best_params_["C"]))
                if rep == 0 and kind == "additive" and mpd in (6, 20):
                    keep[mpd] = best
        print("rep", rep, "done", flush=True)
    with open(os.path.join(RESULTS, "exp_cosexp.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows, keep


def plot(rows, keep):
    Xtr, ytr, _, _ = d2.make("cosexp", 500, 500, seed=100)
    fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, 4.9))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26)
    g1 = np.linspace(0, 5, 400); g2 = np.linspace(-1, 1, 300)
    G1, G2 = np.meshgrid(g1, g2)
    grid = np.c_[G1.ravel(), G2.ravel()]
    true = np.cos(0.5 * np.exp(0.7 * g1))
    for j, mpd in enumerate((6, 20)):
        ax = fig.add_subplot(gs[0, j])
        F = keep[mpd].decision_function(grid).reshape(G1.shape)
        ax.contourf(G1, G2, F, levels=[-1e9, 0, 1e9], colors=[ps.ORANGE_TINT, ps.BLUE_TINT])
        ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=4, color=ps.BLUE, lw=0)
        ax.scatter(Xtr[ytr == -1, 0], Xtr[ytr == -1, 1], s=6, color=ps.ORANGE, lw=0.5, marker="x")
        ax.plot(g1, true, color=ps.INK2, lw=0.9, ls="--")
        ax.contour(G1, G2, F, levels=[0], colors=ps.INK, linewidths=1.4)
        ax.set_xlim(0, 5); ax.set_ylim(-1, 1)
        ps.comma_axes(ax)
        ax.set_xlabel(r"$x(1)$"); ax.set_ylabel(r"$x(2)$", labelpad=0)
        ax.set_title(f"({'ab'[j]}) PWL cộng tính, $M={2 * mpd}$")

    ax = fig.add_subplot(gs[1, 0])
    M = 2 * np.array(M_PER_DIM)
    for kind, col, ls, lab in (("additive", ps.BLUE, "-", "cộng tính (11)"), ("hh", ps.ORANGE, "--", "HH (12)")):
        A = np.array([[r["acc"] for r in rows if r["kind"] == kind and r["mpd"] == m] for m in M_PER_DIM])
        mu, sd = A.mean(1), A.std(1, ddof=1)
        ax.fill_between(M, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.plot(M, mu, color=col, ls=ls, marker="o" if kind == "additive" else "s", ms=3, label=lab)
    rbf = np.mean([r["acc"] for r in rows if r["kind"] == "rbf"])
    ax.axhline(rbf, color=ps.INK2, lw=1.0, ls=":", label="SVM-RBF")
    ax.set_xlabel(r"số đặc trưng $M$"); ax.set_ylabel("độ chính xác kiểm tra")
    ax.set_xlim(0, 41); ps.comma_axes(ax)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("(c) Độ chính xác theo $M$")

    ax = fig.add_subplot(gs[1, 1])
    for kind, col, ls in (("additive", ps.BLUE, "-"), ("hh", ps.ORANGE, "--")):
        T = np.array([[r["time"] for r in rows if r["kind"] == kind and r["mpd"] == m] for m in M_PER_DIM])
        ax.plot(M, 1000 * np.median(T, 1), color=col, ls=ls, marker="o" if kind == "additive" else "s", ms=3)
    ax.set_xlabel(r"số đặc trưng $M$"); ax.set_ylabel("thời gian huấn luyện (ms)")
    ax.set_xlim(0, 41); ps.comma_axes(ax)
    ax.set_title("(d) Thời gian huấn luyện theo $M$")
    ps.save(fig, "fig_cosexp_M.pdf")


def summary(rows):
    out = {}
    for kind in ("additive", "hh"):
        for m in M_PER_DIM:
            a = [r["acc"] for r in rows if r["kind"] == kind and r["mpd"] == m]
            out[f"{kind}|{2*m}"] = (float(np.mean(a)), float(np.std(a, ddof=1)))
    a = [r["acc"] for r in rows if r["kind"] == "rbf"]
    out["rbf"] = (float(np.mean(a)), float(np.std(a, ddof=1)))
    with open(os.path.join(RESULTS, "exp_cosexp_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k in ("additive|2", "additive|12", "additive|20", "additive|40", "hh|20", "hh|40", "rbf"):
        print(k, "%.4f ± %.4f" % out[k])


if __name__ == "__main__":
    rows, keep = run()
    plot(rows, keep)
    summary(rows)
