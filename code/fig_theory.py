"""Illustrative figures for Chapters 1-2 (all data and parameters generated here)."""

import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from sklearn.datasets import make_moons
from sklearn.svm import SVC

import plotstyle as ps
from plotstyle import plt
from pwlsvm import PWLFeatures, random_hyperplane

PASTEL = ["#dbe9fb", "#fbe3d8", "#d6f2e7", "#fcefc7", "#f8dde8", "#e4e0f5", "#e9e8e3", "#d9ecd9"]


def relu(t):
    return np.maximum(0.0, t)


# ----------------------------------------------------------------------------
# Fig. 1.x  hinge function and a univariate PWL function as a sum of hinges
# ----------------------------------------------------------------------------
def fig_hinge_1d():
    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.3),
                               gridspec_kw=dict(width_ratios=[1, 1.45], wspace=0.28))
    t = np.linspace(-1, 1, 401)
    a.axhline(0, color=ps.GRID, lw=0.8, zorder=0)
    a.axvline(0, color=ps.GRID, lw=0.8, zorder=0)
    a.plot(t, relu(t), color=ps.INK, lw=1.8, label=r"$h(t)=\max\{0,t\}$")
    a.plot(t, relu(t - 0.4), color=ps.BLUE, lw=1.2, ls="--", label=r"$h(t-0{,}4)$")
    a.legend(loc="upper left", handlelength=2.0, fontsize=8.5)
    a.set_xlim(-1, 1); a.set_ylim(-0.1, 1.05)
    a.set_xticks([-1, 0, 0.4, 1]); a.set_yticks([0, 0.5, 1])
    ps.comma_axes(a)
    a.set_xlabel(r"$t$"); a.set_title("(a) Hàm bản lề (ReLU)")

    t = np.linspace(0, 1, 401)
    t1, t2, c1, c2 = 0.35, 0.70, -2.0, 2.5
    lin = 0.3 + 0.8 * t
    f = lin + c1 * relu(t - t1) + c2 * relu(t - t2)
    b.axhline(0, color=ps.GRID, lw=0.8, zorder=0)
    for tk in (t1, t2):
        b.axvline(tk, color=ps.MUTED, lw=0.6, ls=":", zorder=0)
    b.plot(t, lin, color=ps.INK2, lw=1.0, ls="--", label=r"$0{,}3+0{,}8t$")
    b.plot(t, c1 * relu(t - t1), color=ps.BLUE, lw=1.1, ls="-.", label=r"$-2(t-0{,}35)_+$")
    b.plot(t, c2 * relu(t - t2), color=ps.ORANGE, lw=1.1, ls=(0, (1, 1.5)), label=r"$2{,}5(t-0{,}7)_+$")
    b.plot(t, f, color=ps.INK, lw=2.0, label=r"$f(t)$ (tổng)")
    b.plot([t1, t2], [0.3 + 0.8 * t1, 0.3 + 0.8 * t2 + c1 * (t2 - t1)], "o", ms=4,
           color=ps.INK, zorder=5)
    b.set_xlim(0, 1); b.set_ylim(-0.75, 1.15)
    b.set_xticks([0, t1, t2, 1]); b.set_yticks([-0.5, 0, 0.5, 1])
    ps.comma_axes(b)
    b.set_xlabel(r"$t$"); b.set_title("(b) Hàm PWL một biến = phần affine + các bản lề")
    b.legend(loc="upper left", ncol=2, handlelength=2.2, columnspacing=1.0, fontsize=8.5)
    ps.save(fig, "fig_hinge_1d.pdf")


# ----------------------------------------------------------------------------
# Fig. 1.y  a continuous PWL function of two variables and its polyhedral partition
# ----------------------------------------------------------------------------
LINES_2D = [(1.0, 1.0, -0.9), (1.0, 0.0, -0.6), (0.0, -1.0, 0.45)]   # a1 x1 + a2 x2 + b
COEF_2D = (0.15, 0.25, 0.20, [0.9, -1.2, 0.8])                        # w0, w1, w2, c_j


def pwl2d(x1, x2):
    w0, w1, w2, c = COEF_2D
    f = w0 + w1 * x1 + w2 * x2
    for cj, (a1, a2, b) in zip(c, LINES_2D):
        f = f + cj * relu(a1 * x1 + a2 * x2 + b)
    return f


def region_codes(x1, x2):
    code = np.zeros_like(x1, dtype=int)
    for j, (a1, a2, b) in enumerate(LINES_2D):
        code += (2 ** j) * (a1 * x1 + a2 * x2 + b > 0)
    return code



def _clip(poly, a1, a2, b, keep_pos):
    """Clip a convex polygon by the halfplane a1 x + a2 y + b >= 0 (or <= 0)."""
    sgn = 1.0 if keep_pos else -1.0
    out = []
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]
        fp = sgn * (a1 * P[0] + a2 * P[1] + b); fq = sgn * (a1 * Q[0] + a2 * Q[1] + b)
        if fp >= 0:
            out.append(P)
        if fp * fq < 0:
            t = fp / (fp - fq); out.append(P + t * (Q - P))
    return np.array(out)


def region_polygons():
    """Exact polygons of the linear regions of pwl2d inside the unit square."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    res = []
    for code_c in range(2 ** len(LINES_2D)):
        poly = square
        for j, (a1, a2, b) in enumerate(LINES_2D):
            poly = _clip(poly, a1, a2, b, bool(code_c >> j & 1))
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
            if area > 1e-9:
                res.append((code_c, poly))
    return res


_STRONG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#898781", "#008300"]


def strong_color(k):
    return np.array(to_rgb(_STRONG[k % len(_STRONG)]))


def fig_pwl_surface():
    g = np.linspace(0, 1, 241)
    x1, x2 = np.meshgrid(g, g)
    F = pwl2d(x1, x2)
    code = region_codes(x1, x2)
    uniq = sorted(np.unique(code))
    lut = {c: i for i, c in enumerate(uniq)}
    idx = np.vectorize(lut.get)(code)
    cols = np.array([to_rgb(PASTEL[i % len(PASTEL)]) for i in range(len(uniq))])
    darker = np.clip(cols * 0.93, 0, 1)

    fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.22)
    ax = fig.add_subplot(gs[0], projection="3d")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = region_polygons()
    zfloor = F.min() - 0.35
    light = np.array([-0.5, -0.6, 0.62]); light /= np.linalg.norm(light)
    tops, floors, fcs, flc = [], [], [], []
    for code_c, poly in polys:
        k = lut[code_c]
        z = pwl2d(poly[:, 0], poly[:, 1])
        top = np.c_[poly, z]
        nrm = np.cross(top[1] - top[0], top[2] - top[0]); nrm /= np.linalg.norm(nrm)
        nrm *= np.sign(nrm[2])
        shade = 0.72 + 0.28 * max(0.0, nrm @ light)
        base = 0.5 * cols[k] + 0.5 * strong_color(k)
        tops.append(top); fcs.append(np.clip(base * shade, 0, 1))
        floors.append(np.c_[poly, np.full(len(poly), zfloor)]); flc.append(cols[k])
    ax.add_collection3d(Poly3DCollection(floors, facecolors=flc, edgecolors=ps.INK2,
                                         linewidths=0.4, linestyles="--"))
    ax.add_collection3d(Poly3DCollection(tops, facecolors=fcs, edgecolors=ps.INK,
                                         linewidths=0.8, alpha=0.97))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(zfloor, F.max())
    ax.view_init(elev=33, azim=-148)
    ax.set_xlabel(r"$x(1)$", labelpad=-7); ax.set_ylabel(r"$x(2)$", labelpad=-7)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([])
    ax.tick_params(pad=-3)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white"); axis.pane.set_edgecolor(ps.GRID)
    ax.set_title("(a) Đồ thị của $f$: các mảnh phẳng nối liền", pad=0)

    b = fig.add_subplot(gs[1])
    b.imshow(cols[idx], origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
    names = []
    for k, c in enumerate(uniq):
        m = code == c
        cx, cy = x1[m].mean(), x2[m].mean()
        b.text(cx, cy, rf"$\Omega_{{{k + 1}}}$", ha="center", va="center", fontsize=10)
        names.append(c)
    for a1, a2, bb in LINES_2D:
        if a2 != 0:
            s = np.linspace(0, 1, 2); b.plot(s, -(a1 * s + bb) / a2, color=ps.INK2, lw=0.9, ls="--")
        else:
            b.axvline(-bb / a1, color=ps.INK2, lw=0.9, ls="--")
    ps.unit_square(b)
    b.set_title(f"(b) Phân hoạch đa diện ({len(uniq)} vùng)")
    ps.save(fig, "fig_pwl_surface.pdf")


# ----------------------------------------------------------------------------
# Fig. 2.x  two moons: a PWL boundary made of three segments
# ----------------------------------------------------------------------------
def moons_data():
    X, y = make_moons(n_samples=600, noise=0.07, random_state=3)
    Xr = np.c_[X[:, 1], -X[:, 0]]
    Xs = (Xr - Xr.min(0)) / (Xr.max(0) - Xr.min(0))
    return 0.02 + 0.96 * Xs, np.where(y == 1, 1, -1)


def moons_phi(Z):
    return np.c_[Z[:, 0], Z[:, 1], relu(Z[:, 1] - 1 / 3), relu(Z[:, 1] - 2 / 3)]


def fig_moons_pwl():
    X, y = moons_data()
    svm = SVC(kernel="linear", C=100).fit(moons_phi(X), y)
    w, w0 = svm.coef_.ravel(), svm.intercept_[0]
    acc = (svm.predict(moons_phi(X)) == y).mean()
    # segment k: c_k^T x + d_k = 0 on Omega_k
    segs = []
    for k, active in enumerate([(), (0,), (0, 1)]):
        c = np.array([w[0], w[1] + sum(w[2 + j] for j in active)])
        d = w0 - sum(w[2 + j] * q for j, q in zip(active, (1 / 3, 2 / 3)) if j in active)
        segs.append((c, d))
    print("moons: train acc", acc, "w", w.round(3), "w0", round(w0, 3), "nSV", svm.n_support_)
    for k, (c, d) in enumerate(segs):
        print(f"  segment {k + 1}: c = {c.round(3)}, d = {d:.3f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 3.0), gridspec_kw=dict(wspace=0.3))
    g = np.linspace(0, 1, 400); G1, G2 = np.meshgrid(g, g)
    F = svm.decision_function(moons_phi(np.c_[G1.ravel(), G2.ravel()])).reshape(G1.shape)
    for ax in (a, b):
        for q in (1 / 3, 2 / 3):
            ax.axhline(q, color=ps.INK2, lw=0.9, ls="--")
    a.contourf(G1, G2, F, levels=[-1e9, 0, 1e9], colors=[ps.ORANGE_TINT, ps.BLUE_TINT], alpha=0.8)
    a.scatter(X[y == 1, 0], X[y == 1, 1], s=7, marker="o", color=ps.BLUE, lw=0, label="lớp $+1$")
    a.scatter(X[y == -1, 0], X[y == -1, 1], s=10, marker="x", color=ps.ORANGE, lw=0.8, label="lớp $-1$")
    a.contour(G1, G2, F, levels=[0], colors=ps.INK, linewidths=1.8)
    ps.unit_square(a)
    a.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.3, fontsize=8.5,
             markerscale=1.4)
    a.set_title("(a) Dữ liệu và biên tuyến tính từng phần")

    ybands = [(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1)]
    for k, ((c, d), (lo, hi)) in enumerate(zip(segs, ybands)):
        s = np.linspace(0, 1, 2)
        full = -(c[0] * s + d) / c[1]
        b.plot(s, full, color=ps.MUTED, lw=0.7, ls=":")
        u = np.linspace(lo, hi, 2)
        b.plot(-(c[1] * u + d) / c[0], u, color=ps.INK, lw=2.0)
        b.text(0.06, (lo + hi) / 2, rf"$\Omega_{{{k + 1}}}$", fontsize=11, va="center")
        mid = lo + (0.35 if k == 1 else 0.5) * (hi - lo)
        xm = -(c[1] * mid + d) / c[0]
        b.text(xm + 0.05, mid, rf"$c_{{{k + 1}}}^\top x+d_{{{k + 1}}}=0$", fontsize=9,
               va="center", ha="left")
    ps.unit_square(b)
    b.set_title(r"(b) Trên mỗi $\Omega_k$ biên là một đoạn thẳng")
    ps.save(fig, "fig_moons_pwl.pdf")
    return w, w0, segs, acc, svm.n_support_


# ----------------------------------------------------------------------------
# Fig. 2.y  one random HH feature map, three different boundaries
# ----------------------------------------------------------------------------
def flex_features(seed=107, D=4):
    rng = np.random.default_rng(seed)
    lo, hi = np.zeros(2), np.ones(2)
    return [random_hyperplane(rng, lo, hi) for _ in range(D)]


def flex_eval(planes, w0, w, G1, G2):
    f = w0 + w[0] * G1 + w[1] * G2
    for (p, q), c in zip(planes, w[2:]):
        f = f + c * relu(p[0] * G1 + p[1] * G2 + q)
    return f


def fig_flexibility(weights):
    planes = flex_features()
    g = np.linspace(0, 1, 500); G1, G2 = np.meshgrid(g, g)
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXTWIDTH_IN, 2.25), gridspec_kw=dict(wspace=0.12))
    titles = ["(a) biên lồi", "(b) biên không lồi", "(c) biên không liên thông"]
    for ax, (w0, w), title in zip(axes, weights, titles):
        F = flex_eval(planes, w0, w, G1, G2)
        ax.contourf(G1, G2, F, levels=[-1e9, 0, 1e9], colors=[ps.ORANGE_TINT, ps.BLUE_TINT])
        for j, (p, q) in enumerate(planes):
            s = np.linspace(-0.2, 1.2, 2)
            ax.plot(s, -(p[0] * s + q) / p[1], color=ps.INK2, lw=0.8, ls="--")
        ax.contour(G1, G2, F, levels=[0], colors=ps.INK, linewidths=1.8)
        ps.unit_square(ax, labels=False)
        ax.set_title(title)
    axes[0].set_ylabel(r"$x(2)$", labelpad=1)
    for ax in axes:
        ax.set_xlabel(r"$x(1)$")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    ps.save(fig, "fig_flexibility.pdf")
    return planes


# Weights found by a small random search (kept in scratch notes); they are listed here so
# that the figure and the numbers quoted in the thesis are reproducible.
FLEX_WEIGHTS = [(0.4, np.array([-3.0, 1.6, 2.9, 0.5, 1.1, 1.9])),     # convex: hinge weights >= 0
                (0.3, np.array([-1.1, 2.3, -1.4, -3.0, 1.3, 1.1])),    # nonconvex
                (1.7, np.array([-2.3, 1.6, -2.9, -1.6, 2.2, -0.9]))]   # disconnected


# ----------------------------------------------------------------------------
# Fig. 2.z  linear regions induced by the three feature-map families
# ----------------------------------------------------------------------------
def region_image(codes):
    uniq, inv = np.unique(codes, return_inverse=True)
    perm = np.random.default_rng(1).permutation(len(uniq))
    cols = np.array([to_rgb(PASTEL[(perm[i]) % len(PASTEL)]) for i in range(len(uniq))])
    return cols[inv.reshape(codes.shape)], inv.reshape(codes.shape), len(uniq)


def fig_regions_maps():
    n_grid = 900
    g = np.linspace(0, 1, n_grid); G1, G2 = np.meshgrid(g, g)
    P = np.c_[G1.ravel(), G2.ravel()]
    corners = np.array([[0, 0], [1, 1]])
    maps = [("additive", 4, "(a) cộng tính, $M/n=4$"),
            ("hh", 4, "(b) HH, $M/n=4$"),
            ("ghh", 2.5, "(c) GHH, $M/n=2{,}5$")]
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXTWIDTH_IN, 2.35), gridspec_kw=dict(wspace=0.12))
    for ax, (kind, mpd, title) in zip(axes, maps):
        feat = PWLFeatures(kind, mpd, random_state=5).fit(corners)
        codes = feat.activation_pattern(P).reshape(G1.shape)
        img, lab, nreg = region_image(codes)
        ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
        for r in range(nreg):
            ax.contour(G1, G2, (lab == r).astype(float), levels=[0.5], colors=ps.INK2,
                       linewidths=0.8)
        ps.unit_square(ax, labels=False)
        ax.set_title(title)
        ax.text(0.97, 0.03, f"{nreg} vùng", ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        print(kind, "hinges", feat.n_hinges_, "regions", nreg)
    axes[0].set_ylabel(r"$x(2)$", labelpad=1)
    for ax in axes:
        ax.set_xlabel(r"$x(1)$")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    ps.save(fig, "fig_regions_maps.pdf")


if __name__ == "__main__":
    fig_hinge_1d()
    fig_pwl_surface()
    fig_moons_pwl()
    planes = fig_flexibility(FLEX_WEIGHTS)
    for p, q in planes:
        print("plane: p =", np.round(p, 3), "q =", round(q, 3))
    fig_regions_maps()
