"""Figures of Chapter 3 built from the JSON files in results/ (run after the experiments)."""

import json
import os
import sys

import matplotlib
import numpy as np

import plotstyle as ps
from plotstyle import plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda name: json.load(open(os.path.join(HERE, "results", name)))

ORDER = ["Haberman", "Transfusion", "Banknote", "Pima", "Breast", "Magic", "Parkinsons",
         "Ionosphere", "Spambase", "Sonar"]
LAB = {"linear": "SVM tuyến tính", "rbf": "SVM-RBF", "lssvm": "LS-SVM-RBF", "knn": "kNN ($k=1$)",
       "iksvm": "Ik-SVM", "mlp": "MLP-ReLU", "pwl_c": "PWL-C-SVM cộng tính",
       "pwl_ls": "PWL-LS-SVM cộng tính", "pwl_hh": "PWL-C-SVM HH"}


def mean_acc(rows, data, method):
    return np.mean([r["acc"] for r in rows if r["data"] == data and r["method"] == method])


# ----------------------------------------------------------------------------
def fig_benchmark_diff(xmin=-12.0, xmax=4.0):
    rows = R("benchmark.json")
    names = [d for d in ORDER if any(r["data"] == d for r in rows)]
    series = [("linear", ps.MUTED, "o"), ("pwl_c", ps.BLUE, "s"), ("pwl_hh", ps.ORANGE, "^"),
              ("mlp", ps.AQUA, "D")]
    fig, ax = plt.subplots(figsize=(ps.TEXTWIDTH_IN, 3.7))
    ypos = np.arange(len(names))[::-1]
    ax.axvline(0, color=ps.INK2, lw=1.0)
    for y in ypos:
        ax.axhline(y, color=ps.GRID, lw=0.6, zorder=0)
    for j, (m, col, mk) in enumerate(series):
        d = np.array([100 * (mean_acc(rows, nm, m) - mean_acc(rows, nm, "rbf")) for nm in names])
        yy = ypos + (j - 1.5) * 0.14
        inside = d >= xmin
        ax.plot(d[inside], yy[inside], mk, color=col, ms=5.5, label=LAB[m], lw=0)
        for dv, yv in zip(d[~inside], yy[~inside]):
            ax.annotate(f"{dv:.1f}".replace(".", ",").replace("-", "\u2212"), xy=(xmin, yv), xytext=(xmin + 1.6, yv),
                        fontsize=8, color=col, va="center",
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
    ax.set_yticks(ypos); ax.set_yticklabels(names)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("chênh lệch độ chính xác so với SVM-RBF (điểm phần trăm)")
    ps.comma_axes(ax, y=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8.5, handletextpad=0.3, columnspacing=1.0)
    ps.save(fig, "fig_benchmark_diff.pdf")


# ----------------------------------------------------------------------------
def fig_sensitivity():
    rows = R("sensitivity.json")
    names = ["Ionosphere", "Magic", "Spambase"]
    var = [("add_uni", "cộng tính, lưới đều", ps.BLUE, "-", "o"),
           ("add_q", "cộng tính, phân vị", ps.AQUA, "-", "s"),
           ("hh", r"HH, $|p(1)|=1$", ps.ORANGE, "--", "^"),
           ("hh_norm", r"HH, $\|p\|_2=1$", ps.INK2, ":", "v")]
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXTWIDTH_IN, 2.55), gridspec_kw=dict(wspace=0.3))
    for ax, nm in zip(axes, names):
        for key, lab, col, ls, mk in var:
            mpd = sorted({r["mpd"] for r in rows if r["data"] == nm and r["variant"] == key})
            mu = [100 * np.mean([r["acc"] for r in rows if r["data"] == nm and r["variant"] == key and r["mpd"] == m]) for m in mpd]
            ax.plot(mpd, mu, ls=ls, marker=mk, ms=3.5, color=col, label=lab, lw=1.3)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_xlabel(r"$M/n$"); ax.set_title(nm)
        ps.comma_axes(ax, x=False)
    axes[0].set_ylabel("độ chính xác (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8.5, columnspacing=1.2)
    ps.save(fig, "fig_sensitivity.pdf")


# ----------------------------------------------------------------------------
def place_labels(ax, xy, labels, fontsize=7.5):
    """Direct labels next to scatter points: for each point, the candidate offset with the least overlap."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ab = ax.get_window_extent(rend)
    pts = [ax.transData.transform(p) for p in xy]
    taken = [matplotlib.transforms.Bbox.from_bounds(px - 4, py - 4, 8, 8) for px, py in pts]
    cands = [((5, 3), "left", "bottom"), ((5, -3), "left", "top"), ((-5, 3), "right", "bottom"),
             ((-5, -3), "right", "top"), ((0, 7), "center", "bottom"), ((0, -7), "center", "top"),
             ((8, 0), "left", "center"), ((-8, 0), "right", "center"), ((6, 14), "left", "bottom"),
             ((-6, -14), "right", "top"), ((6, -14), "left", "top"), ((-6, 14), "right", "bottom"),
             ((0, 18), "center", "bottom"), ((0, -18), "center", "top")]

    def area(a, b):
        w = min(a.x1, b.x1) - max(a.x0, b.x0); h = min(a.y1, b.y1) - max(a.y0, b.y0)
        return max(w, 0) * max(h, 0)

    for k in np.argsort([p[0] for p in pts]):
        best, best_cost = None, None
        for (dx, dy), ha, va in cands:
            t = ax.annotate(labels[k], xy[k], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=ps.INK)
            bb = t.get_window_extent(rend)
            out = (max(ab.x0 - bb.x0, 0) + max(bb.x1 - ab.x1, 0)) * bb.height + \
                  (max(ab.y0 - bb.y0, 0) + max(bb.y1 - ab.y1, 0)) * bb.width
            cost = sum(area(bb, o) for o in taken) + 3 * out
            if best_cost is None or cost < best_cost:
                if best is not None:
                    best.remove()
                best, best_cost, best_bb = t, cost, bb
            else:
                t.remove()
            if cost == 0:
                break
        taken.append(best_bb)


def fig_cost():
    rows = R("cost.json")
    lab = {"linear": "SVM tuyến tính", "logit": "hồi quy logistic", "logit_bin": "thẻ điểm", "pwl_c": "PWL-C-SVM",
           "pwl_ls": "PWL-LS-SVM", "pwl_hh": "PWL-LS-SVM HH", "rbf": "SVM-RBF", "mlp": "MLP-ReLU",
           "hgb": "tăng cường cây"}
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 3.1), gridspec_kw=dict(wspace=0.3))
    for ax, (data, metric, ylab) in zip(axes, (("Magic", "acc", "độ chính xác (%)"), ("Credit", "auc", "AUC"))):
        pts = [r for r in rows if r["data"] == data]
        xy, labels = [], []
        for r in pts:
            pwl = r["method"].startswith("pwl")
            v = 100 * r[metric] if metric == "acc" else r[metric]
            ax.plot(1e6 * r["t_pred"], v, "s" if pwl else "o", color=ps.BLUE if pwl else ps.INK2,
                    ms=6 if pwl else 5, zorder=3)
            xy.append((1e6 * r["t_pred"], v)); labels.append(lab[r["method"]])
        ax.set_xscale("log")
        xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
        ax.set_xlim(min(xs) / 3, max(xs) * 6)
        pad = 0.16 * (max(ys) - min(ys))
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_xlabel(r"$\mu$s mỗi mẫu (thang log)")
        ax.set_ylabel(ylab); ax.set_title(data if data == "Magic" else "Dữ liệu tín dụng")
        ps.comma_axes(ax, x=False)
        place_labels(ax, xy, labels)
    ps.save(fig, "fig_cost.pdf")


# ----------------------------------------------------------------------------
def fig_credit_overview():
    sys.path.insert(0, HERE)
    import credit_data
    df, y = credit_data.load_raw()
    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.6), gridspec_kw=dict(wspace=0.3))
    vals = np.arange(-2, 9)
    rate = [100 * np.mean(y[df["PAY_0"] == v] == 1) for v in vals]
    cnt = [int(np.sum(df["PAY_0"] == v)) for v in vals]
    bars = a.bar(vals, rate, color=[ps.BLUE if c >= 100 else ps.BLUE_TINT for c in cnt],
                 edgecolor=ps.BLUE, lw=0.6)
    a.axhline(100 * np.mean(y == 1), color=ps.INK2, lw=0.9, ls="--")
    a.text(-2.4, 100 * np.mean(y == 1) + 2.5, "trung bình 22,1%", fontsize=8, color=ps.INK2)
    a.set_xticks(vals); a.set_xlabel("PAY_0 (tháng 9/2005)"); a.set_ylabel("tỉ lệ vỡ nợ (%)")
    a.set_title("(a) Tỉ lệ vỡ nợ theo trạng thái trả nợ")
    ps.comma_axes(a, x=False)
    lim = df["LIMIT_BAL"].to_numpy() / 1000
    b.hist(lim, bins=60, color=ps.BLUE_TINT, edgecolor=ps.BLUE, lw=0.4)
    lo, hi = lim.min(), lim.max()
    for s in range(1, 10):
        b.axvline(lo + (hi - lo) * s / 10, color=ps.ORANGE, lw=0.9, ls="--")
        b.axvline(np.quantile(lim, s / 10), color=ps.INK, lw=0.9)
    b.plot([], [], color=ps.ORANGE, ls="--", label="nút lưới đều")
    b.plot([], [], color=ps.INK, label="nút phân vị")
    b.legend(fontsize=8, loc="center right", framealpha=0.95, frameon=True, edgecolor="none")
    b.set_xlabel("LIMIT_BAL (nghìn Đài tệ)"); b.set_ylabel("số khách hàng")
    b.set_title("(b) Hạn mức tín dụng và vị trí nút")
    ps.comma_axes(b)
    ps.save(fig, "fig_credit_overview.pdf")


# ----------------------------------------------------------------------------
def fig_credit_roc():
    from make_tables import credit_results
    res = credit_results()
    fig, ax = plt.subplots(figsize=(ps.TEXTWIDTH_IN * 0.55, 2.9))
    ax.plot([0, 1], [0, 1], color=ps.GRID, lw=0.8)
    series = [("logit", "hồi quy logistic", ps.INK2, ":"), ("logit_bin", "thẻ điểm", ps.ORANGE, "--"),
              ("pwl_ls_q", "PWL-LS-SVM (phân vị)", ps.BLUE, "-"), ("hgb", "tăng cường cây", ps.AQUA, "-.")]
    auc = {m: np.mean([r["auc"] for r in res["rows"] if r["method"] == m]) for m, *_ in series}
    for m, lab, col, ls in series:
        ax.plot(res["roc"][m]["fpr"], res["roc"][m]["tpr"], color=col, ls=ls, lw=1.5,
                label=f"{lab} ({auc[m]:.3f})".replace(".", ","))
    ax.set_xlabel("tỉ lệ dương tính giả"); ax.set_ylabel("tỉ lệ dương tính thật")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ps.comma_axes(ax)
    ax.legend(loc="lower right", fontsize=7.5)
    ps.save(fig, "fig_credit_roc.pdf")


# ----------------------------------------------------------------------------
MONTH = {1: 9, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4}
PRETTY = {"LIMIT_BAL": "LIMIT_BAL (nghìn Đài tệ)", "AGE": "AGE (tuổi)", "PAY_0": "PAY_0 (tháng 9)"}
PRETTY.update({f"PAY_{k}": f"PAY_{k} (tháng {MONTH[k]})" for k in range(2, 7)})
PRETTY.update({f"BILL_AMT{k}": f"BILL_AMT{k} (nghìn Đài tệ)" for k in range(1, 7)})
PRETTY.update({f"PAY_AMT{k}": f"PAY_AMT{k} (nghìn Đài tệ)" for k in range(1, 7)})
MONEY = {"LIMIT_BAL"} | {f"BILL_AMT{k}" for k in range(1, 7)} | {f"PAY_AMT{k}" for k in range(1, 7)}


def fig_credit_shapes(k=6):
    it = R("credit_interpret.json")
    numeric = [nm for nm in it["names"] if not nm.startswith(("EDU_", "FEMALE", "MARRIED", "SINGLE"))]
    top = sorted(numeric, key=lambda nm: -it["importance"][nm])[:k]
    path = os.path.join(HERE, "results", "credit_shapes_all.json")
    allsp = json.load(open(path)) if os.path.exists(path) else None
    fig, axes = plt.subplots(2, 3, figsize=(ps.TEXTWIDTH_IN, 4.3), gridspec_kw=dict(wspace=0.32, hspace=0.55))
    for ax, nm in zip(axes.ravel(), top):
        sh = it["shapes"][nm]
        x = np.array(sh["x"]); h = np.array(sh["h"])
        sc = 1000 if nm in MONEY else 1
        if allsp is not None:                      # the other four splits, thin lines
            xg = np.array(allsp["grid"][nm])
            keep = (xg >= x.min()) & (xg <= x.max())
            for spl in allsp["splits"][1:]:
                ax.plot(xg[keep] / sc, np.array(spl["h"][nm])[keep], color=ps.BLUE_TINT2, lw=0.9, zorder=1)
        if len(x) <= 15:
            ax.plot(x / sc, h, "-o", color=ps.BLUE, ms=3.5, lw=1.4, zorder=3)
        else:
            ax.plot(x / sc, h, color=ps.BLUE, lw=1.6, zorder=3)
        ax.axhline(0, color=ps.GRID, lw=0.8, zorder=0)
        for q in sh["knots"]:
            if x.min() <= q <= x.max():
                ax.axvline(q / sc, color=ps.MUTED, lw=0.5, ls=":", zorder=0)
        ax.set_title(PRETTY.get(nm, nm), fontsize=9)
        ps.comma_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"đóng góp $h_i$")
    ps.save(fig, "fig_credit_shapes.pdf")
    return top


def fig_credit_explain():
    it = R("credit_interpret.json")
    imp = it["importance"]
    groups = {"EDUCATION": ["EDU_GRAD", "EDU_UNIV", "EDU_HIGH"], "MARRIAGE": ["MARRIED", "SINGLE"],
              "SEX": ["FEMALE"]}
    imp2 = {nm: v for nm, v in imp.items() if not any(nm in g for g in groups.values())}
    cl = it["client"]["contrib"]
    cl2 = {nm: v for nm, v in cl.items() if not any(nm in g for g in groups.values())}
    for g, cols in groups.items():
        imp2[g] = it["importance_groups"][g]          # std of the summed contribution of the indicators
        cl2[g] = float(sum(cl[c] for c in cols))
    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 3.3), gridspec_kw=dict(wspace=0.75))
    top = sorted(imp2, key=lambda k: -imp2[k])[:10][::-1]
    a.barh(range(len(top)), [imp2[k] for k in top], color=ps.BLUE, height=0.6)
    a.set_yticks(range(len(top))); a.set_yticklabels(top, fontsize=8)
    a.set_xlabel("độ lệch chuẩn của đóng góp"); a.set_title("(a) Mức độ ảnh hưởng của biến")
    ps.comma_axes(a, y=False)
    topc = sorted(cl2, key=lambda k: -abs(cl2[k]))[:8][::-1]
    vals = [cl2[k] for k in topc]
    raw = it["client"]["values"]

    def label(k):                                   # "PAY_0 = 2", "LIMIT_BAL = 20.000"
        if k in groups:
            return k
        v = raw[k]
        txt = f"{int(round(v)):,}".replace(",", ".") if abs(v - round(v)) < 1e-9 else f"{v:.1f}".replace(".", ",")
        return f"{k} = {txt}"
    b.barh(range(len(topc)), vals, color=[ps.ORANGE if v > 0 else ps.BLUE for v in vals], height=0.6)
    b.axvline(0, color=ps.INK2, lw=0.8)
    b.set_yticks(range(len(topc))); b.set_yticklabels([label(k) for k in topc], fontsize=8)
    b.set_xlabel("đóng góp so với trung bình"); b.set_title("(b) Giải thích một hồ sơ rủi ro cao")
    ps.comma_axes(b, y=False)
    ps.save(fig, "fig_credit_explain.pdf")


def fig_credit_monotone():
    mo = R("credit_monotone.json")
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXTWIDTH_IN, 2.45), gridspec_kw=dict(wspace=0.36))
    for ax, nm in zip(axes, ["PAY_0", "PAY_4", "LIMIT_BAL"]):
        sc = 1000 if nm in MONEY else 1
        for key, col, ls, mk, lab in (("free", ps.BLUE, "-", "o", "không ràng buộc"),
                                      ("monotone", ps.ORANGE, "--", "s", "ràng buộc đơn điệu")):
            sh = mo["shapes"][key][nm]
            x, h = np.array(sh["x"]) / sc, np.array(sh["h"])
            if len(x) <= 15:
                ax.plot(x, h, ls=ls, marker=mk, ms=3.2, color=col, lw=1.4, label=lab)
            else:
                ax.plot(x, h, ls=ls, color=col, lw=1.5, label=lab)
        for q in mo["shapes"]["free"][nm]["knots"]:
            if x.min() <= q / sc <= x.max():
                ax.axvline(q / sc, color=ps.MUTED, lw=0.5, ls=":", zorder=0)
        if nm.startswith("PAY_"):                  # values of the status that are rare in the data
            ax.axvspan(2.5, x.max() + 0.5, color=ps.GRID, alpha=0.6, lw=0, zorder=0)
            ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.axhline(0, color=ps.GRID, lw=0.8, zorder=0)
        ax.set_title(PRETTY.get(nm, nm), fontsize=9)
        ps.comma_axes(ax)
    axes[0].set_ylabel(r"đóng góp $h_i$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8.5, handlelength=2.4)
    ps.save(fig, "fig_credit_monotone.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["benchmark", "sensitivity", "cost", "overview", "roc", "shapes", "explain", "monotone"]
    fns = {"benchmark": fig_benchmark_diff, "sensitivity": fig_sensitivity, "cost": fig_cost,
           "overview": fig_credit_overview, "roc": fig_credit_roc, "shapes": fig_credit_shapes,
           "explain": fig_credit_explain, "monotone": fig_credit_monotone}
    for w in which:
        out = fns[w]()
        if out is not None:
            print(w, out)
