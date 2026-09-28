"""Figures for Chapter 1: convex sets, polyhedra, separation, duality (all drawn from scratch)."""

import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

import plotstyle as ps
from plotstyle import plt


def _clean(ax, lim=(0, 1)):
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ps.GRID)


# ----------------------------------------------------------------------------
# Hình 1.1  convex set, nonconvex set, convex hull of a finite set
# ----------------------------------------------------------------------------
def fig_convex_sets():
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXTWIDTH_IN, 2.15), gridspec_kw=dict(wspace=0.08))
    # (a) convex: an ellipse-like polygon
    t = np.linspace(0, 2 * np.pi, 200)
    E = np.c_[0.5 + 0.36 * np.cos(t), 0.5 + 0.24 * np.sin(t + 0.3)]
    a = axes[0]
    a.add_patch(Polygon(E, closed=True, fc=ps.BLUE_TINT, ec=ps.BLUE, lw=1.2))
    P, Q = np.array([0.25, 0.42]), np.array([0.74, 0.62])
    a.plot(*np.c_[P, Q], color=ps.INK, lw=1.3); a.plot(*np.c_[P, Q], "o", color=ps.INK, ms=3.5)
    a.text(P[0] - 0.02, P[1] - 0.09, r"$x_1$", ha="center"); a.text(Q[0] + 0.02, Q[1] + 0.05, r"$x_2$", ha="center")
    a.set_title("(a) tập lồi")
    # (b) nonconvex: a crescent
    r1, r2 = 0.34, 0.27
    outer = np.c_[0.5 + r1 * np.cos(t), 0.5 + r1 * np.sin(t)]
    b = axes[1]
    from matplotlib.path import Path
    inner_c = np.array([0.62, 0.55])
    inside_outer = lambda X: np.hypot(X[:, 0] - 0.5, X[:, 1] - 0.5) <= r1
    inside_inner = lambda X: np.hypot(X[:, 0] - inner_c[0], X[:, 1] - inner_c[1]) <= r2
    g = np.linspace(0, 1, 500); G1, G2 = np.meshgrid(g, g); XY = np.c_[G1.ravel(), G2.ravel()]
    mask = (inside_outer(XY) & ~inside_inner(XY)).reshape(G1.shape)
    b.contourf(G1, G2, mask.astype(float), levels=[0.5, 1.5], colors=[ps.ORANGE_TINT])
    b.contour(G1, G2, mask.astype(float), levels=[0.5], colors=[ps.ORANGE], linewidths=1.2)
    P, Q = np.array([0.36, 0.78]), np.array([0.56, 0.2])
    b.plot(*np.c_[P, Q], color=ps.INK, lw=1.3); b.plot(*np.c_[P, Q], "o", color=ps.INK, ms=3.5)
    b.text(P[0] - 0.05, P[1] + 0.03, r"$x_1$"); b.text(Q[0] + 0.02, Q[1] - 0.07, r"$x_2$")
    b.set_title("(b) tập không lồi")
    # (c) convex hull of finitely many points
    rng = np.random.default_rng(4)
    pts = 0.15 + 0.7 * rng.uniform(size=(14, 2))
    hull = ConvexHull(pts)
    c = axes[2]
    c.add_patch(Polygon(pts[hull.vertices], closed=True, fc=ps.BLUE_TINT, ec=ps.BLUE, lw=1.2))
    c.plot(pts[:, 0], pts[:, 1], "o", color=ps.INK, ms=3.2)
    c.set_title(r"(c) bao lồi $\mathrm{conv}\,C$")
    for ax in axes:
        _clean(ax)
    ps.save(fig, "fig_convex_sets.pdf")


# ----------------------------------------------------------------------------
# Hình 1.2  polyhedron, separating hyperplane, supporting hyperplane
# ----------------------------------------------------------------------------
def fig_separation():
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXTWIDTH_IN, 2.25), gridspec_kw=dict(wspace=0.08))
    # (a) polyhedron as intersection of halfplanes a_j^T x <= b_j
    A = np.array([[0.0, -1.0], [1.0, -0.35], [0.55, 1.0], [-0.8, 0.9], [-1.0, -0.3]])
    b = np.array([-0.18, 0.62, 1.1, 0.35, -0.2])
    g = np.linspace(0, 1, 600); G1, G2 = np.meshgrid(g, g); XY = np.c_[G1.ravel(), G2.ravel()]
    feas = np.all(XY @ A.T <= b, axis=1).reshape(G1.shape)
    a = axes[0]
    a.contourf(G1, G2, feas.astype(float), levels=[0.5, 1.5], colors=[ps.BLUE_TINT])
    for (a1, a2), bj in zip(A, b):
        s = np.linspace(-0.2, 1.2, 50)
        if abs(a2) > 1e-9:
            a.plot(s, (bj - a1 * s) / a2, color=ps.INK2, lw=0.8, ls="--")
        else:
            a.axvline(bj / a1, color=ps.INK2, lw=0.8, ls="--")
    a.contour(G1, G2, feas.astype(float), levels=[0.5], colors=[ps.BLUE], linewidths=1.3)
    # outward normals at the middle of each edge
    for (a1, a2), bj in zip(A, b):
        on = np.abs(XY @ np.array([a1, a2]) - bj) < 3e-3
        cand = XY[on & np.all(XY @ A.T <= b + 3e-3, axis=1)]
        if len(cand):
            m = cand.mean(0); nv = np.array([a1, a2]) / np.hypot(a1, a2)
            a.annotate("", xy=m + 0.12 * nv, xytext=m, arrowprops=dict(arrowstyle="->", color=ps.INK, lw=0.9))
    a.text(0.5, 0.5, r"$P$", ha="center", va="center", fontsize=12)
    a.set_title("(a) đa diện")
    # (b) separating hyperplane between two disjoint convex sets
    t = np.linspace(0, 2 * np.pi, 200)
    Cset = np.c_[0.3 + 0.17 * np.cos(t), 0.66 + 0.21 * np.sin(t)]
    Dset = np.array([[0.55, 0.12], [0.9, 0.2], [0.93, 0.5], [0.7, 0.55], [0.52, 0.35]])
    bb = axes[1]
    bb.add_patch(Polygon(Cset, closed=True, fc=ps.BLUE_TINT, ec=ps.BLUE, lw=1.2))
    bb.add_patch(Polygon(Dset, closed=True, fc=ps.ORANGE_TINT, ec=ps.ORANGE, lw=1.2))
    # closest points: brute force on dense boundaries
    def dense(poly, k=400):
        pts = []
        for i in range(len(poly)):
            P, Q = poly[i], poly[(i + 1) % len(poly)]
            pts.append(P + np.linspace(0, 1, k)[:, None] * (Q - P))
        return np.vstack(pts)
    Cd, Dd = dense(Cset, 20), dense(Dset)
    dd = np.linalg.norm(Cd[:, None, :] - Dd[None, :, :], axis=2)
    i, j = np.unravel_index(dd.argmin(), dd.shape)
    c0, d0 = Cd[i], Dd[j]
    bb.plot(*np.c_[c0, d0], color=ps.INK, lw=0.9, ls=":")
    bb.plot(*c0, "o", color=ps.INK, ms=3.5); bb.plot(*d0, "o", color=ps.INK, ms=3.5)
    bb.text(c0[0] - 0.02, c0[1] - 0.09, r"$c$", ha="center"); bb.text(d0[0] + 0.05, d0[1] + 0.01, r"$d$")
    mid, nv = (c0 + d0) / 2, (d0 - c0) / np.linalg.norm(d0 - c0)
    tv = np.array([-nv[1], nv[0]])
    L = np.c_[mid - 0.8 * tv, mid + 0.8 * tv]
    bb.plot(L[0], L[1], color=ps.INK, lw=1.4)
    bb.text(0.27, 0.66, r"$C$", ha="center", va="center", fontsize=12)
    bb.text(0.76, 0.33, r"$D$", ha="center", va="center", fontsize=12)
    bb.set_title("(b) siêu phẳng phân tách")
    # (c) supporting hyperplane at a boundary point
    cc = axes[2]
    Eset = np.c_[0.47 + 0.3 * np.cos(t), 0.45 + 0.22 * np.sin(t)]
    cc.add_patch(Polygon(Eset, closed=True, fc=ps.BLUE_TINT, ec=ps.BLUE, lw=1.2))
    th = 1.0
    x0 = np.array([0.47 + 0.3 * np.cos(th), 0.45 + 0.22 * np.sin(th)])
    nrm = np.array([np.cos(th) / 0.3, np.sin(th) / 0.22]); nrm /= np.linalg.norm(nrm)
    tv = np.array([-nrm[1], nrm[0]])
    L = np.c_[x0 - 0.6 * tv, x0 + 0.6 * tv]
    cc.plot(L[0], L[1], color=ps.INK, lw=1.4)
    cc.plot(*x0, "o", color=ps.INK, ms=3.5)
    cc.annotate("", xy=x0 + 0.17 * nrm, xytext=x0, arrowprops=dict(arrowstyle="->", color=ps.INK, lw=0.9))
    cc.text(*(x0 + 0.2 * nrm + np.array([0.02, 0.0])), r"$a$")
    cc.text(x0[0] - 0.08, x0[1] - 0.02, r"$x_0$", ha="center")
    cc.text(0.43, 0.4, r"$C$", ha="center", va="center", fontsize=12)
    cc.set_title("(c) siêu phẳng tựa")
    for ax in axes:
        _clean(ax)
    ps.save(fig, "fig_separation.pdf")


# ----------------------------------------------------------------------------
# Hình 1.3  distance to a hyperplane; weak duality on a toy problem
# ----------------------------------------------------------------------------
def fig_duality():
    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.5), gridspec_kw=dict(wspace=0.3))
    # (a) projection of x0 onto the hyperplane w^T x + b = 0
    w, b0 = np.array([1.0, 1.6]), -1.3
    s = np.linspace(-0.2, 1.4, 2)
    a.plot(s, -(w[0] * s + b0) / w[1], color=ps.INK, lw=1.4)
    x0 = np.array([0.95, 0.95])
    xp = x0 - (w @ x0 + b0) / (w @ w) * w
    a.plot(*np.c_[x0, xp], color=ps.BLUE, lw=1.2, ls="--")
    a.plot(*x0, "o", color=ps.INK, ms=4); a.plot(*xp, "o", color=ps.BLUE, ms=4)
    a.annotate("", xy=xp + 0.28 * w / np.linalg.norm(w), xytext=xp,
               arrowprops=dict(arrowstyle="->", color=ps.INK2, lw=0.9))
    a.text(x0[0] + 0.03, x0[1] + 0.03, r"$x_0$")
    tip = xp + 0.28 * w / np.linalg.norm(w)
    a.text(tip[0] - 0.12, tip[1] + 0.0, r"$w$", color=ps.INK2)
    a.text(xp[0] - 0.13, xp[1] - 0.1, r"$x^\ast$", color=ps.BLUE)
    a.text(0.02, 0.9, r"$w^\top x+b=0$", fontsize=9)
    a.text(0.02, 0.1, r"$\frac{|w^\top x_0+b|}{\|w\|}$ = độ dài nét đứt", fontsize=9, color=ps.BLUE)
    a.set_xlim(0, 1.2); a.set_ylim(0, 1.2); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(ps.GRID)
    a.set_title("(a) khoảng cách đến siêu phẳng")
    # (b) min x^2 s.t. x >= 1:  g(lambda) = lambda - lambda^2/4 <= p* = 1
    lam = np.linspace(0, 4.2, 300)
    b.axhline(1.0, color=ps.INK2, lw=1.0, ls="--")
    b.plot(lam, lam - lam ** 2 / 4, color=ps.BLUE, lw=1.8)
    b.plot([2], [1], "o", color=ps.BLUE, ms=4)
    b.text(3.0, 1.06, r"$p^\ast=1$", color=ps.INK2)
    b.text(0.15, -0.42, r"$g(\lambda)=\lambda-\lambda^2/4$", color=ps.BLUE)
    b.text(2.2, 0.72, r"$\lambda^\ast=2$", color=ps.BLUE, fontsize=9)
    b.set_xlim(0, 4.2); b.set_ylim(-0.6, 1.3)
    b.set_xlabel(r"$\lambda$"); b.set_ylabel("giá trị")
    b.set_xticks([0, 1, 2, 3, 4]); b.set_yticks([-0.5, 0, 0.5, 1])
    ps.comma_axes(b)
    b.set_title(r"(b) hàm đối ngẫu: $g(\lambda)\leq p^\ast$")
    ps.save(fig, "fig_duality.pdf")


if __name__ == "__main__":
    fig_convex_sets()
    fig_separation()
    fig_duality()
