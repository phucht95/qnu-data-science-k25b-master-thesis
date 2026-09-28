"""Figures for Chapter 2: hard/soft margin, loss functions, feature lifting, kNN and Ik-SVM boundaries."""

import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import plotstyle as ps
from plotstyle import plt
from pwlsvm import intersection_kernel


def _clean(ax, lim=(0, 1)):
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ps.GRID)


def _line(ax, w, b, c=0.0, **kw):
    """Draw {x: w^T x + b = c} inside the unit square."""
    s = np.linspace(-0.2, 1.2, 2)
    ax.plot(s, (c - b - w[0] * s) / w[1], **kw)


def separable_data(seed=8, m=12):
    rng = np.random.default_rng(seed)
    A = rng.normal([0.3, 0.66], 0.075, size=(m, 2))
    B = rng.normal([0.7, 0.34], 0.075, size=(m, 2))
    X = np.r_[A, B]; y = np.r_[np.ones(m), -np.ones(m)]
    return X, y


def random_separators(X, y, k=4, seed=0):
    """A few random lines that separate the data (to show that the separator is not unique)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < k:
        ang = rng.uniform(np.pi / 8, 3 * np.pi / 8)
        w = np.array([-np.cos(ang), np.sin(ang)]) * np.array([1.0, 1.0])
        w = np.array([-1.0, np.tan(ang)])
        c = rng.uniform(0.35, 0.65, size=2)
        bb = -w @ c
        if np.all(np.sign(X @ w + bb) == y):
            out.append((w, bb))
    return out


# ----------------------------------------------------------------------------
# Hình 2.1  hard margin: many separating lines; margin and SVs; convex hull view
# ----------------------------------------------------------------------------
def fig_hard_margin():
    X, y = separable_data()
    svm = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b = svm.coef_.ravel(), svm.intercept_[0]
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXTWIDTH_IN, 2.3), gridspec_kw=dict(wspace=0.08))
    for ax in axes:
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, color=ps.BLUE, lw=0, zorder=3)
        ax.scatter(X[y == -1, 0], X[y == -1, 1], s=22, color=ps.ORANGE, marker="x", lw=1.1, zorder=3)
    a = axes[0]
    for ww, bb in random_separators(X, y):
        _line(a, ww, bb, color=ps.MUTED, lw=1.0)
    a.set_title("(a) nhiều biên tách được")
    bm = axes[1]
    _line(bm, w, b, color=ps.INK, lw=1.6)
    _line(bm, w, b, 1.0, color=ps.INK2, lw=0.9, ls="--")
    _line(bm, w, b, -1.0, color=ps.INK2, lw=0.9, ls="--")
    sv = svm.support_vectors_
    bm.scatter(sv[:, 0], sv[:, 1], s=95, facecolors="none", edgecolors=ps.INK, lw=1.0, zorder=4)
    # margin arrow: from a point on w^T x + b = -1 to the line w^T x + b = +1 along w
    x_m = np.array([0.82, 0.0]); x_m[1] = (-1 - b - w[0] * x_m[0]) / w[1]
    x_p = x_m + 2 * w / (w @ w)
    bm.annotate("", xy=x_p, xytext=x_m, arrowprops=dict(arrowstyle="<->", color=ps.BLUE, lw=1.0))
    bm.text(x_p[0] + 0.02, x_p[1] - 0.02, r"$\frac{2}{\|w\|}$", color=ps.BLUE, fontsize=10)
    bm.set_title("(b) lề cực đại")
    c = axes[2]
    for lab, col, tint in ((1, ps.BLUE, ps.BLUE_TINT), (-1, ps.ORANGE, ps.ORANGE_TINT)):
        P = X[y == lab]; h = ConvexHull(P)
        c.add_patch(Polygon(P[h.vertices], closed=True, fc=tint, ec=col, lw=1.0, alpha=0.9, zorder=1))
    def edges(P):
        h = ConvexHull(P); V = P[h.vertices]
        return [(V[i], V[(i + 1) % len(V)]) for i in range(len(V))]
    def seg_pts(e, k=300):
        return e[0] + np.linspace(0, 1, k)[:, None] * (e[1] - e[0])
    A_pts = np.vstack([seg_pts(e) for e in edges(X[y == 1])])
    B_pts = np.vstack([seg_pts(e) for e in edges(X[y == -1])])
    d = np.linalg.norm(A_pts[:, None, :] - B_pts[None, :, :], axis=2)
    i, j = np.unravel_index(d.argmin(), d.shape)
    u, v = A_pts[i], B_pts[j]
    c.plot(*np.c_[u, v], color=ps.INK, lw=1.0, ls=":")
    c.plot(*u, "o", color=ps.INK, ms=3.5); c.plot(*v, "o", color=ps.INK, ms=3.5)
    _line(c, w, b, color=ps.INK, lw=1.6)
    c.set_title("(c) hai bao lồi")
    for ax in axes:
        _clean(ax)
    ps.save(fig, "fig_hard_margin.pdf")
    return svm


# ----------------------------------------------------------------------------
# Hình 2.2  soft margin with slack variables; loss functions
# ----------------------------------------------------------------------------
def fig_soft_margin():
    rng = np.random.default_rng(21)
    m = 16
    A = rng.normal([0.33, 0.66], 0.085, size=(m, 2))
    B = rng.normal([0.67, 0.34], 0.085, size=(m, 2))
    A = np.r_[A, [[0.58, 0.45]]]; B = np.r_[B, [[0.45, 0.58]]]          # two clear violators
    X = np.r_[A, B]; y = np.r_[np.ones(len(A)), -np.ones(len(B))]
    C = 2.0
    svm = SVC(kernel="linear", C=C).fit(X, y)
    w, b = svm.coef_.ravel(), svm.intercept_[0]
    alpha = np.zeros(len(y)); alpha[svm.support_] = np.abs(svm.dual_coef_.ravel())
    f = X @ w + b
    fig, (a, bx) = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.75),
                                gridspec_kw=dict(wspace=0.28, width_ratios=[1, 1.15]))
    _line(a, w, b, color=ps.INK, lw=1.6)
    _line(a, w, b, 1.0, color=ps.INK2, lw=0.9, ls="--")
    _line(a, w, b, -1.0, color=ps.INK2, lw=0.9, ls="--")
    a.scatter(X[y == 1, 0], X[y == 1, 1], s=16, color=ps.BLUE, lw=0, zorder=3)
    a.scatter(X[y == -1, 0], X[y == -1, 1], s=22, color=ps.ORANGE, marker="x", lw=1.1, zorder=3)
    free = (alpha > 1e-6) & (alpha < C - 1e-6)
    a.scatter(X[free, 0], X[free, 1], s=95, facecolors="none", edgecolors=ps.INK, lw=1.0, zorder=4)
    for k in np.where(y * f < 1 - 1e-6)[0]:
        target = X[k] + (y[k] - f[k]) / (w @ w) * w       # foot on the line y_k (w^T x + b) = 1
        a.plot(*np.c_[X[k], target], color=ps.INK2, lw=1.0, ls=":")
    _clean(a)
    a.set_title("(a) lề mềm và biến bù")
    t = np.linspace(-2, 3, 500)
    bx.plot(t, (t < 0).astype(float), color=ps.INK2, lw=1.4, drawstyle="steps-post", label="0–1")
    bx.plot(t, np.maximum(0, 1 - t), color=ps.BLUE, lw=1.8, label=r"bản lề $\max\{0,1-t\}$")
    bx.plot(t, (1 - t) ** 2, color=ps.ORANGE, lw=1.4, ls="--", label=r"bình phương $(1-t)^2$")
    bx.set_xlim(-2, 3); bx.set_ylim(-0.1, 3.2)
    bx.set_xlabel(r"$t=y\,f(x)$"); bx.set_ylabel("mất mát")
    bx.axvline(0, color=ps.GRID, lw=0.8, zorder=0); bx.axvline(1, color=ps.GRID, lw=0.8, zorder=0)
    ps.comma_axes(bx)
    bx.legend(loc="upper center", bbox_to_anchor=(0.56, 1.0), fontsize=8.5)
    bx.set_title("(b) các hàm mất mát")
    ps.save(fig, "fig_soft_margin.pdf")
    return int(free.sum()), int(((alpha >= C - 1e-6)).sum()), int((y * f < 0).sum())


# ----------------------------------------------------------------------------
# Hình 2.3  lifting: circles are separable after phi(x) = (x1, x2, x1^2 + x2^2)
# ----------------------------------------------------------------------------
def fig_lifting():
    X, y = make_circles(n_samples=240, noise=0.06, factor=0.45, random_state=2)
    y = np.where(y == 1, 1, -1)
    fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, 2.7))
    a = fig.add_subplot(1, 2, 1)
    a.scatter(X[y == 1, 0], X[y == 1, 1], s=10, color=ps.BLUE, lw=0)
    a.scatter(X[y == -1, 0], X[y == -1, 1], s=14, color=ps.ORANGE, marker="x", lw=0.8)
    a.set_aspect("equal"); a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1]); ps.comma_axes(a)
    a.set_xlabel(r"$x(1)$"); a.set_ylabel(r"$x(2)$", labelpad=0)
    a.set_title(r"(a) trong $\mathbb{R}^2$: không tách tuyến tính được")
    b = fig.add_subplot(1, 2, 2, projection="3d")
    z = X[:, 0] ** 2 + X[:, 1] ** 2
    b.scatter(X[y == 1, 0], X[y == 1, 1], z[y == 1], s=8, color=ps.BLUE, depthshade=False)
    b.scatter(X[y == -1, 0], X[y == -1, 1], z[y == -1], s=10, color=ps.ORANGE, marker="x", depthshade=False)
    g = np.linspace(-1.1, 1.1, 2); G1, G2 = np.meshgrid(g, g)
    b.plot_surface(G1, G2, np.full_like(G1, 0.5), color=ps.MUTED, alpha=0.25, linewidth=0)
    b.view_init(elev=12, azim=-60)
    b.set_xticks([-1, 0, 1]); b.set_yticks([-1, 0, 1]); b.set_zticks([0, 0.5, 1])
    b.set_xlabel(r"$x(1)$", labelpad=-6); b.set_ylabel(r"$x(2)$", labelpad=-6)
    b.set_zlabel(r"$z$", labelpad=-7)
    b.tick_params(pad=-3)
    for axis in (b.xaxis, b.yaxis, b.zaxis):
        axis.pane.set_facecolor("white"); axis.pane.set_edgecolor(ps.GRID)
    b.set_title(r"(b) sau ánh xạ $\varphi$: một mặt phẳng tách được", pad=0)
    ps.save(fig, "fig_lifting.pdf")


# ----------------------------------------------------------------------------
# Hình 2.8  boundaries of 1-NN and of the intersection-kernel SVM are PWL
# ----------------------------------------------------------------------------
def fig_knn_iksvm():
    rng = np.random.default_rng(5)
    X = rng.uniform(0.05, 0.95, size=(26, 2))
    y = np.where((X[:, 0] - 0.5) ** 2 + 1.5 * (X[:, 1] - 0.45) ** 2 < 0.06, 1, -1)
    y[rng.choice(len(y), 2, replace=False)] *= -1
    g = np.linspace(0, 1, 500); G1, G2 = np.meshgrid(g, g); G = np.c_[G1.ravel(), G2.ravel()]
    knn = KNeighborsClassifier(1).fit(X, y)
    Fk = knn.predict(G).reshape(G1.shape).astype(float)
    ik = SVC(kernel="precomputed", C=10.0).fit(intersection_kernel(X, X), y)
    Fi = (intersection_kernel(G, X[ik.support_]) @ ik.dual_coef_.ravel() + ik.intercept_[0]).reshape(G1.shape)
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN * 0.72, 2.45), gridspec_kw=dict(wspace=0.08))
    for ax, F, lev, title in ((axes[0], Fk, 0.0, "(a) kNN với $k=1$"), (axes[1], Fi, 0.0, "(b) SVM hạt nhân giao")):
        ax.contourf(G1, G2, F, levels=[-1e9, lev, 1e9], colors=[ps.ORANGE_TINT, ps.BLUE_TINT])
        ax.contour(G1, G2, F, levels=[lev], colors=ps.INK, linewidths=1.4)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, color=ps.BLUE, lw=0, zorder=3)
        ax.scatter(X[y == -1, 0], X[y == -1, 1], s=22, color=ps.ORANGE, marker="x", lw=1.1, zorder=3)
        _clean(ax)
        ax.set_title(title)
    ps.save(fig, "fig_knn_iksvm.pdf")


if __name__ == "__main__":
    svm = fig_hard_margin()
    print("hard margin: n_SV =", svm.n_support_.sum())
    print("soft margin: free SV, bounded SV =", fig_soft_margin())
    fig_lifting()
    fig_knn_iksvm()
