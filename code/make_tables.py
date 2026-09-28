"""LaTeX tables of Chapter 3 and of the appendix, generated from results/*.json."""

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "de_an", "chapters")
R = lambda name: json.load(open(os.path.join(HERE, "results", name)))

ORDER = ["Haberman", "Transfusion", "Banknote", "Pima", "Breast", "Magic", "Parkinsons",
         "Ionosphere", "Spambase", "Sonar"]
METH = ["linear", "rbf", "lssvm", "knn", "iksvm", "mlp", "pwl_c", "pwl_ls", "pwl_hh"]
HEAD = {"linear": "SVM\\\\tuyến tính", "rbf": "SVM-\\\\RBF", "lssvm": "LS-SVM-\\\\RBF", "knn": "kNN\\\\$(k=1)$",
        "iksvm": "Ik-\\\\SVM", "mlp": "MLP-\\\\ReLU", "pwl_c": "PWL-C\\\\cộng tính", "pwl_ls": "PWL-LS\\\\cộng tính",
        "pwl_hh": "PWL-C\\\\HH"}
NAME = {"linear": "SVM tuyến tính", "rbf": "SVM-RBF", "lssvm": "LS-SVM-RBF", "knn": "kNN ($k=1$)",
        "iksvm": "Ik-SVM", "mlp": "MLP-ReLU", "pwl_c": "PWL-C-SVM cộng tính",
        "pwl_ls": "PWL-LS-SVM cộng tính", "pwl_hh": "PWL-C-SVM HH"}


def f1(x, nd=1):
    return f"{x:.{nd}f}".replace(".", "{,}")


def thousands(x):
    return f"{int(round(x)):,}".replace(",", ".")


def fmt_sec(t):
    """Training time in seconds with about two significant digits."""
    return f1(t, 2) if t < 1 else f1(t, 1) if t < 10 else f1(t, 0)


def fmt_us(t):
    """Prediction time in microseconds, with two significant digits for small values."""
    x = 1e6 * t
    return f1(x, 2) if x < 1 else f1(x, 1) if x < 100 else thousands(x)


def stack(h):
    return "\\begin{tabular}[b]{@{}c@{}}" + h + "\\end{tabular}"


def bench_stats():
    rows = R("benchmark.json")
    names = [d for d in ORDER if any(r["data"] == d for r in rows)]
    M = {(d, m): np.array([r["acc"] for r in rows if r["data"] == d and r["method"] == m]) for d in names for m in METH}
    P = {(d, m): np.median([r["params"] for r in rows if r["data"] == d and r["method"] == m]) for d in names for m in METH}
    return rows, names, M, P


def ranks(names, M):
    rk = []
    for d in names:
        v = np.array([100 * M[(d, m)].mean() for m in METH])
        v = np.round(v, 1)
        order = np.argsort(-v)
        r = np.empty(len(v))
        # average ranks for ties
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        rk.append(r)
    return np.mean(rk, axis=0)


def table_benchmark():
    rows, names, M, P = bench_stats()
    avg = ranks(names, M)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption[Độ chính xác trên mười bộ dữ liệu chuẩn]{Độ chính xác trung bình trên tập kiểm tra (\%) qua $10$ lần chia ngẫu nhiên ($5$ lần với Spambase). In đậm: giá trị cao nhất mỗi hàng. Dòng cuối là hạng trung bình trên mười bộ dữ liệu ($1$ là tốt nhất). Độ lệch chuẩn được cho ở \cref{tab:appendix-bench} (Phụ lục~B).}",
         r"\label{tab:bench}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.18}",
         r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{@{}l" + " r" * len(METH) + r"@{}}", r"\toprule",
         r"\textbf{Dữ liệu} & " + " & ".join(stack(HEAD[m]) for m in METH) + r"\\", r"\midrule"]
    for d in names:
        v = [100 * M[(d, m)].mean() for m in METH]
        best = max(np.round(v, 1))
        cells = [(r"\textbf{" + f1(x) + "}") if round(x, 1) == best else f1(x) for x in v]
        L.append(d + " & " + " & ".join(cells) + r"\\")
    L.append(r"\midrule")
    L.append(r"Hạng trung bình & " + " & ".join(f1(r, 2) for r in avg) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(OUT, "tab_bench.tex"), "w").write("\n".join(L) + "\n")
    return names, M, avg


def table_wilcoxon():
    rows, names, M, P = bench_stats()
    ref = "pwl_c"
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption[So sánh theo cặp với PWL-C-SVM cộng tính]{So sánh theo cặp giữa PWL-C-SVM cộng tính và từng phương pháp khác trên mười bộ dữ liệu: số bộ mà PWL-C-SVM cao hơn / bằng / thấp hơn (theo độ chính xác trung bình làm tròn đến $0{,}1$ điểm phần trăm), chênh lệch trung bình, và giá trị $p$ của kiểm định dấu hạng Wilcoxon hai phía.}",
         r"\label{tab:wilcoxon}", r"\small", r"\renewcommand{\arraystretch}{1.18}",
         r"\begin{tabular}{@{}l c r r@{}}", r"\toprule",
         r"\textbf{Phương pháp so sánh} & \textbf{Cao hơn / bằng / thấp hơn} & \textbf{Chênh lệch TB} & $p$\\", r"\midrule"]
    out = {}
    for m in METH:
        if m == ref:
            continue
        a = np.array([np.round(100 * M[(d, ref)].mean(), 1) for d in names])
        b = np.array([np.round(100 * M[(d, m)].mean(), 1) for d in names])
        w, t, l = int((a > b).sum()), int((a == b).sum()), int((a < b).sum())
        diff = a - b
        p = wilcoxon(a, b, zero_method="wilcox").pvalue if np.any(diff != 0) else 1.0
        out[m] = (w, t, l, diff.mean(), p)
        L.append(f"{NAME[m]} & {w} / {t} / {l} & ${'+' if diff.mean() >= 0 else '-'}{f1(abs(diff.mean()), 2)}$ & {f1(p, 3)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_wilcoxon.tex"), "w").write("\n".join(L) + "\n")
    return out


def table_appendix_bench():
    rows, names, M, P = bench_stats()
    parts = [METH[:5], METH[5:]]
    L = [r"\begin{table}[p]", r"\centering",
         r"\caption[Độ chính xác đầy đủ trên mười bộ dữ liệu chuẩn]{Độ chính xác trên tập kiểm tra (\%, trung bình $\pm$ độ lệch chuẩn qua $10$ lần chia, $5$ lần với Spambase) và kích thước mô hình (số thực cần lưu, trung vị).}",
         r"\label{tab:appendix-bench}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.15}"]
    for part in parts:
        L += [r"\begin{tabular}{@{}l" + " c" * len(part) + r"@{}}", r"\toprule",
              r"\textbf{Dữ liệu} & " + " & ".join(stack(HEAD[m]) for m in part) + r"\\", r"\midrule"]
        for d in names:
            L.append(d + " & " + " & ".join(f"${f1(100 * M[(d, m)].mean())}\\pm{f1(100 * M[(d, m)].std(ddof=1)) if len(M[(d, m)]) > 1 else '-'}$" for m in part) + r"\\")
        L.append(r"\midrule")
        for d in names:
            L.append(r"\quad số thực, " + d + " & " + " & ".join(thousands(P[(d, m)]) for m in part) + r"\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\par\medskip"]
    L += [r"\end{table}"]
    open(os.path.join(OUT, "tab_appendix_bench.tex"), "w").write("\n".join(L) + "\n")


def credit_results():
    """Rows of exp_credit.py, exp_credit_scorecard.py and exp_monotone.py (constrained model = "pwl_ls_mono")."""
    res = R("credit.json")
    rows, roc = list(res["rows"]), dict(res["roc"])
    if os.path.exists(os.path.join(HERE, "results", "credit_scorecard.json")):
        sc = R("credit_scorecard.json"); rows += sc["rows"]; roc.update(sc["roc"])
    if os.path.exists(os.path.join(HERE, "results", "credit_monotone.json")):
        rows += [dict(r, method="pwl_ls_mono") for r in R("credit_monotone.json")["rows"] if r["method"] == "monotone"]
    return dict(rows=rows, roc=roc)


def table_credit():
    res = credit_results()["rows"]
    meth = [("logit", "Hồi quy logistic"), ("logit_bin", "Thẻ điểm (logistic, biến rời rạc hóa)"),
            ("linear", "SVM tuyến tính"),
            ("pwl_ls_uni", "PWL-LS-SVM, cộng tính, lưới đều"), ("pwl_ls_q", "PWL-LS-SVM, cộng tính, phân vị"),
            ("pwl_ls_mono", "\\quad và ràng buộc đơn điệu (Mục~\\ref{sec:monotone})"),
            ("pwl_c_q", "PWL-C-SVM, cộng tính, phân vị"), ("pwl_ls_hh", "PWL-LS-SVM, HH"),
            ("rbf", "SVM-RBF"), ("mlp", "MLP-ReLU"), ("hgb", "Tổ hợp cây tăng cường")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption[Kết quả trên dữ liệu vỡ nợ thẻ tín dụng]{Kết quả trên dữ liệu vỡ nợ thẻ tín dụng (trung bình $\pm$ độ lệch chuẩn qua $5$ lần chia $70/30$). AUC: diện tích dưới đường ROC; ĐCX-CB: độ chính xác cân bằng; F1 tính cho lớp vỡ nợ; ngưỡng được chọn trên tập huấn luyện sao cho tỉ lệ dự báo vỡ nợ bằng tỉ lệ vỡ nợ quan sát. Cột cuối: số thực cần lưu (trung vị); thời gian dự đoán được đo riêng ở \cref{sec:cost}.}",
         r"\label{tab:credit}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.18}",
         r"\begin{tabular}{@{}l c c c r@{}}", r"\toprule",
         r"\textbf{Mô hình} & \textbf{AUC} & \textbf{ĐCX-CB (\%)} & \textbf{F1 (\%)} & \textbf{Số thực}\\", r"\midrule"]
    summ = {}
    for m, lab in meth:
        rr = [r for r in res if r["method"] == m]
        if not rr:
            continue
        auc = np.array([r["auc"] for r in rr]); ba = np.array([r["bacc"] for r in rr]); f = np.array([r["f1"] for r in rr])
        par = np.median([r["params"] for r in rr])
        summ[m] = dict(auc=auc.mean(), auc_sd=auc.std(ddof=1), bacc=ba.mean(), f1=f.mean(), params=par)
        L.append(f"{lab} & ${f1(auc.mean(), 3)}\\pm{f1(auc.std(ddof=1), 3)}$ & ${f1(100 * ba.mean())}\\pm{f1(100 * ba.std(ddof=1))}$ & "
                 f"${f1(100 * f.mean())}\\pm{f1(100 * f.std(ddof=1))}$ & {thousands(par)}\\\\")
        if m in ("linear", "pwl_ls_hh"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_credit.tex"), "w").write("\n".join(L) + "\n")
    return summ


COST_HEADS = [stack(r"\textbf{Huấn}\\\textbf{luyện (s)}"), stack(r"\textbf{Dự đoán}\\\textbf{($\mu$s/mẫu)}"),
              stack(r"\textbf{Số}\\\textbf{thực}")]


def table_cost():
    rows = R("cost.json")
    lab = {"linear": "SVM tuyến tính", "logit": "Hồi quy logistic", "logit_bin": "Thẻ điểm",
           "pwl_c": "PWL-C-SVM, cộng tính", "pwl_ls": "PWL-LS-SVM, cộng tính", "pwl_hh": "PWL-LS-SVM, HH",
           "rbf": "SVM-RBF", "mlp": "MLP-ReLU", "hgb": "Tổ hợp cây tăng cường"}
    order = ["linear", "logit", "logit_bin", "pwl_c", "pwl_ls", "pwl_hh", "rbf", "mlp", "hgb"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption[Chi phí huấn luyện, dự đoán và bộ nhớ]{Chi phí trên Magic ($9.510$ mẫu huấn luyện) và dữ liệu tín dụng ($21.000$ mẫu huấn luyện), đo trên một luồng xử lý: thời gian huấn luyện một mô hình với tham số đã chọn, thời gian dự đoán mỗi mẫu, số thực cần lưu, và độ chính xác (Magic) hoặc AUC (tín dụng) trên tập kiểm tra. Đây là một lần chia riêng, với tham số chọn trên mẫu con (và với Magic, tập huấn luyện lớn hơn nhiều), nên độ chính xác khác với ở \cref{tab:bench,tab:credit}.}",
         r"\label{tab:cost-measured}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.18}",
         r"\begin{tabular}{@{}l r r r c r r r c@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{\textbf{Magic}} & \multicolumn{4}{c}{\textbf{Tín dụng}}\\",
         r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
         r"\textbf{Mô hình} & " + " & ".join(COST_HEADS + [stack(r"\textbf{ĐCX}\\\textbf{(\%)}")] + COST_HEADS + [r"\textbf{AUC}"]) + r"\\",
         r"\midrule"]
    get = lambda d, m: next((r for r in rows if r["data"] == d and r["method"] == m), None)
    for m in order:
        a, b = get("Magic", m), get("Credit", m)
        if a is None or b is None:
            continue
        tf = fmt_sec
        L.append(f"{lab[m]} & {tf(a['t_fit'])} & {fmt_us(a['t_pred'])} & {thousands(a['params'])} & {f1(100 * a['acc'])} & "
                 f"{tf(b['t_fit'])} & {fmt_us(b['t_pred'])} & {thousands(b['params'])} & {f1(b['auc'], 3)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_cost.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    which = sys.argv[1:] or ["bench", "credit", "cost"]
    if "bench" in which:
        names, M, avg = table_benchmark()
        print("avg ranks", {m: round(r, 2) for m, r in zip(METH, avg)})
        print("wilcoxon", {m: (v[0], v[1], v[2], round(v[3], 2), round(v[4], 3)) for m, v in table_wilcoxon().items()})
        table_appendix_bench()
    if "credit" in which:
        print("credit", table_credit())
    if "cost" in which:
        table_cost()
