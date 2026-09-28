"""Write the LaTeX tables of Section 3.3.2 from results/exp_uci_preview_summary.json."""

import json
import os

import numpy as np

from exp_uci_preview import DATA, PAPER

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "results", "exp_uci_preview_summary.json")))
R = json.load(open(os.path.join(HERE, "results", "exp_uci_preview.json")))

METHODS = [("linear", "SVM tuyến tính"), ("rbf", "SVM-RBF"), ("lssvm", "LS-SVM-RBF"),
           ("knn", "kNN ($k=1$)"), ("iksvm", "Ik-SVM"),
           ("pwl_c", "PWL-C-SVM, cộng tính"), ("pwl_ls", "PWL-LS-SVM, cộng tính"),
           ("pwl_hh", "PWL-C-SVM, HH")]
NAMES = list(DATA.keys())


def fmt(x, nd=1):
    return f"{x:.{nd}f}".replace(".", "{,}")


def dims(name):
    r = next(r for r in R if r["data"] == name)
    return r["n"], r["N"]


lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption[Độ chính xác trên sáu bộ dữ liệu UCI (sơ bộ)]{Độ chính xác trên tập kiểm tra (\%, trung bình $\pm$ độ lệch chuẩn qua $10$ lần chia ngẫu nhiên $50/50$) trên sáu bộ dữ liệu UCI. In đậm: giá trị trung bình cao nhất mỗi cột. Hai dòng cuối là số liệu của bài báo gốc~\cite[Bảng~1]{huang2013} với một lần chia.}")
lines.append(r"\label{tab:uci}")
lines.append(r"\footnotesize")
lines.append(r"\setlength{\tabcolsep}{3.2pt}")
lines.append(r"\renewcommand{\arraystretch}{1.18}")
lines.append(r"\begin{tabular}{@{}l" + " c" * len(NAMES) + r"@{}}")
lines.append(r"\toprule")
lines.append(r"\textbf{Phương pháp} & " + " & ".join(r"\textbf{" + n + "}" for n in NAMES) + r"\\")
lines.append(r" & " + " & ".join(f"$n={dims(n)[0]}$" for n in NAMES) + r"\\")
lines.append(r"\midrule")
best = {n: max(S[f"{n}|{k}"]["mean"] for k, _ in METHODS) for n in NAMES}
for k, lab in METHODS:
    cells = []
    for n in NAMES:
        m, s = S[f"{n}|{k}"]["mean"], S[f"{n}|{k}"]["std"]
        mm = fmt(100 * m)
        if abs(m - best[n]) < 1e-12:
            mm = r"\mathbf{" + mm + "}"
        cells.append(f"${mm}\\pm{fmt(100 * s)}$")
    lines.append(lab + " & " + " & ".join(cells) + r"\\")
lines.append(r"\midrule")
for k, lab in (("rbf", "Bài gốc: C-SVM-RBF"), ("pwl_c", "Bài gốc: PWL-C-SVM")):
    lines.append(r"\textit{" + lab + "} & " + " & ".join(f"${fmt(100 * PAPER[n][k])}$" for n in NAMES) + r"\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

# model size table
lines.append("")
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption[Kích thước mô hình trên sáu bộ dữ liệu UCI]{Số thực cần lưu để dự đoán (trung vị qua $10$ lần chia), gồm cả tham số chuẩn hóa. Với SVM-RBF, con số này tỉ lệ với số vectơ hỗ trợ.}")
lines.append(r"\label{tab:ucisize}")
lines.append(r"\footnotesize")
lines.append(r"\setlength{\tabcolsep}{4pt}")
lines.append(r"\renewcommand{\arraystretch}{1.18}")
lines.append(r"\begin{tabular}{@{}l" + " r" * len(NAMES) + r"@{}}")
lines.append(r"\toprule")
lines.append(r"\textbf{Mô hình} & " + " & ".join(r"\textbf{" + n + "}" for n in NAMES) + r"\\")
lines.append(r"\midrule")
for k, lab in (("linear", "SVM tuyến tính"), ("rbf", "SVM-RBF"), ("pwl_c", "PWL-C-SVM, cộng tính"),
               ("pwl_hh", "PWL-C-SVM, HH")):
    cells = [f"{int(round(S[f'{n}|{k}']['params'])):,}".replace(",", ".") for n in NAMES]
    lines.append(lab + " & " + " & ".join(cells) + r"\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

out = os.path.join(HERE, "..", "de_an", "chapters", "chuong3_uci_tables.tex")
open(out, "w").write("\n".join(lines) + "\n")
print("written", os.path.normpath(out))

# a compact text summary for writing the discussion
for n in NAMES:
    print(n, dims(n), " ".join(f"{k}={100*S[f'{n}|{k}']['mean']:.1f}±{100*S[f'{n}|{k}']['std']:.1f}" for k, _ in METHODS))
for n in NAMES:
    print(n, "params:", {k: int(S[f'{n}|{k}']['params']) for k, _ in METHODS})
ranks = []
for n in NAMES:
    vals = np.array([S[f"{n}|{k}"]["mean"] for k, _ in METHODS])
    order = (-vals).argsort().argsort() + 1
    ranks.append(order)
print("mean rank:", {k: round(float(r), 2) for (k, _), r in zip(METHODS, np.mean(ranks, axis=0))})
