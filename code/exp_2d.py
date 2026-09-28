"""Section 3.2: PWL-SVMs on four two-dimensional data sets (decision boundaries + accuracy table)."""

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
os.makedirs(RESULTS, exist_ok=True)

C_GRID = [2.0 ** k for k in range(-3, 12, 2)]          # 2^-3, 2^-1, ..., 2^11
G_GRID = [2.0 ** k for k in range(-3, 10, 2)]          # RBF width for [0,1]^2 inputs


def methods(seed):
    return {
        "linear": ("SVM tuyến tính",
                   make_pipeline(MinMaxScaler(), SVC(kernel="linear")),
                   {"svc__C": C_GRID}),
        "additive": ("PWL cộng tính",
                     PWLSVM(kind="additive", m_per_dim=10, solver="libsvm", random_state=seed),
                     {"C": C_GRID}),
        "hh": ("PWL-HH",
               PWLSVM(kind="hh", m_per_dim=50, solver="libsvm", random_state=seed),
               {"C": C_GRID}),
        "ghh": ("PWL-GHH",
                PWLSVM(kind="ghh", m_per_dim=25, solver="libsvm", random_state=seed),
                {"C": C_GRID}),
        "rbf": ("SVM-RBF",
                make_pipeline(MinMaxScaler(), SVC(kernel="rbf")),
                {"svc__C": C_GRID, "svc__gamma": G_GRID}),
    }


def fit_tuned(est, grid, X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(est, grid, cv=cv, n_jobs=-1).fit(X, y)
    return gs.best_estimator_, gs.best_params_


def run(n_rep=10, datasets=("moons", "circles", "checker", "spirals")):
    rows, keep = [], {}
    for name in datasets:
        for rep in range(n_rep):
            Xtr, ytr, Xte, yte = d2.make(name, 500, 500, seed=rep)
            for key, (label, est, grid) in methods(rep).items():
                t0 = time.perf_counter()
                model, best = fit_tuned(est, grid, Xtr, ytr, rep)
                acc = float((model.predict(Xte) == yte).mean())
                rows.append(dict(data=name, rep=rep, method=key, acc=acc,
                                 time=time.perf_counter() - t0, best=str(best)))
                if rep == 0:
                    keep[(name, key)] = (model, acc)
            print(name, rep, {r["method"]: round(r["acc"], 3) for r in rows[-5:]}, flush=True)
    with open(os.path.join(RESULTS, "exp_2d.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows, keep


def plot(keep, datasets=("moons", "circles", "checker", "spirals")):
    keys = list(methods(0).keys())
    labels = {k: v[0] for k, v in methods(0).items()}
    fig, axes = plt.subplots(len(datasets), len(keys), figsize=(ps.TEXTWIDTH_IN, 5.15),
                             gridspec_kw=dict(wspace=0.06, hspace=0.12))
    for r, name in enumerate(datasets):
        Xtr, ytr, Xte, yte = d2.make(name, 500, 500, seed=0)
        lo, hi = Xtr.min(0), Xtr.max(0)
        pad = 0.05 * (hi - lo)
        g1 = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 300)
        g2 = np.linspace(lo[1] - pad[1], hi[1] + pad[1], 300)
        G1, G2 = np.meshgrid(g1, g2)
        grid = np.c_[G1.ravel(), G2.ravel()]
        sub = np.random.default_rng(1).choice(len(yte), 220, replace=False)
        for c, key in enumerate(keys):
            ax = axes[r, c]
            model, acc = keep[(name, key)]
            F = model.decision_function(grid).reshape(G1.shape)
            ax.contourf(G1, G2, F, levels=[-1e9, 0, 1e9], colors=[ps.ORANGE_TINT, ps.BLUE_TINT])
            Xs, ys = Xte[sub], yte[sub]
            ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=3.5, color=ps.BLUE, lw=0, marker="o")
            ax.scatter(Xs[ys == -1, 0], Xs[ys == -1, 1], s=5, color=ps.ORANGE, lw=0.5, marker="x")
            ax.contour(G1, G2, F, levels=[0], colors=ps.INK, linewidths=1.1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(g1[0], g1[-1]); ax.set_ylim(g2[0], g2[-1])
            ax.text(0.97, 0.03, f"{100 * acc:.1f}%".replace(".", ","), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            for sp in ax.spines.values():
                sp.set_color(ps.MUTED)
            if r == 0:
                ax.set_title(labels[key], fontsize=9.5)
            if c == 0:
                ax.set_ylabel(d2.NAMES_VI[name], fontsize=9.5)
    ps.save(fig, "fig_demo_2d.pdf")


def table(rows, datasets=("moons", "circles", "checker", "spirals")):
    keys = list(methods(0).keys())
    out = {}
    for name in datasets:
        for key in keys:
            a = np.array([r["acc"] for r in rows if r["data"] == name and r["method"] == key])
            t = np.array([r["time"] for r in rows if r["data"] == name and r["method"] == key])
            out[(name, key)] = (a.mean(), a.std(ddof=1), t.mean())
    with open(os.path.join(RESULTS, "exp_2d_summary.json"), "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()}, f, indent=1)
    for name in datasets:
        print(name, "  ".join(f"{k}: {100*out[(name,k)][0]:.1f}±{100*out[(name,k)][1]:.1f}"
                              for k in keys))
    return out


if __name__ == "__main__":
    rows, keep = run()
    plot(keep)
    table(rows)
