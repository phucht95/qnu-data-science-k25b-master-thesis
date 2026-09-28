"""Fill the {{TOKENS}} of the chapter templates (*.tpl) with numbers computed from results/*.json.

Every number quoted in the text of Sections 3.3.2-3.6 and of the Conclusion comes from here, so that
the text cannot drift away from the tables.  Qualitative statements of the text ("significant",
"lower than the majority rate", ...) are checked by assertions: if a rerun changes them, the script
stops and the sentence has to be rewritten by hand.
"""

import json
import os
import re

import numpy as np

from make_tables import METH, bench_stats, f1, fmt_sec, fmt_us, ranks, thousands, table_wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "..", "de_an", "chapters")
R = lambda name: json.load(open(os.path.join(HERE, "results", name)))
WORD = {1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười"}


def vn(x, nd):
    return f1(x, nd)


def ratio(x):
    """Round a ratio for the text: one decimal below 10, integer below 100, tens above."""
    if x < 10:
        return str(int(round(x))) if abs(x - round(x)) < 0.05 else vn(x, 1)
    if x < 100:
        return str(int(round(x)))
    return thousands(round(x, -1))


T = {}

# ---------------------------------------------------------------- benchmark (Section 3.3.2)
rows, names, M, P = bench_stats()
assert len(names) == 10
avg = dict(zip(METH, ranks(names, M)))
mean = {(d, m): round(100 * M[(d, m)].mean(), 1) for d in names for m in METH}
for key, m in [("R_LSSVM", "lssvm"), ("R_RBF", "rbf"), ("R_MLP", "mlp"), ("R_PWLC", "pwl_c"),
               ("R_LIN", "linear"), ("R_KNN", "knn"), ("R_HH", "pwl_hh")]:
    T[key] = vn(avg[m], 2)
NAMEV = {"linear": "SVM tuyến tính", "rbf": "SVM hạt nhân Gauss", "lssvm": "LS-SVM hạt nhân Gauss", "knn": "kNN",
         "iksvm": "Ik-SVM", "mlp": "mạng nơ-ron", "pwl_c": "PWL-C-SVM cộng tính", "pwl_ls": "PWL-LS-SVM cộng tính",
         "pwl_hh": "PWL-C-SVM HH"}


def and_list(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " và " + xs[-1]


top3 = sorted(METH, key=lambda m: avg[m])[:3]
assert "pwl_c" not in top3
T["TOP3_TEXT"] = (and_list([NAMEV[m] for m in top3]) + ", với hạng trung bình lần lượt "
                  + and_list([f"${vn(avg[m], 2)}$" for m in top3]))
assert avg["pwl_c"] < avg["linear"] and avg["pwl_c"] < avg["knn"]

a = np.array([mean[(d, "pwl_c")] for d in names])
T["W_RBF_DIFF"] = vn(np.mean(np.array([mean[(d, "rbf")] for d in names]) - a), 1)
T["W_LIN_DIFF"] = vn(np.mean(a - np.array([mean[(d, "linear")] for d in names])), 1)
W = table_wilcoxon()                                               # (wins, ties, losses, mean diff, p)
pv = lambda m: f"$p={vn(W[m][4], 3)}$"
others = [m for m in ("rbf", "lssvm", "mlp", "iksvm", "knn") if m in W]
worse = [m for m in others if W[m][4] < 0.05 and W[m][3] < 0]
better = [m for m in others if W[m][4] < 0.05 and W[m][3] > 0]
nonsig = [m for m in others if W[m][4] >= 0.05]
parts = []
if worse:
    parts.append("Ở mức ý nghĩa $5\\%$, PWL-C-SVM cộng tính kém " + and_list([NAMEV[m] for m in worse])
                 + " một cách có ý nghĩa (" + and_list([pv(m) for m in worse]) + ")"
                 + (", và tốt hơn " + and_list([NAMEV[m] for m in better]) + " (" + and_list([pv(m) for m in better]) + ")" if better else "")
                 + ".")
elif better:
    parts.append("Ở mức ý nghĩa $5\\%$, PWL-C-SVM cộng tính tốt hơn " + and_list([NAMEV[m] for m in better]) + " một cách có ý nghĩa ("
                 + and_list([pv(m) for m in better]) + ").")
wl = W["linear"]
lin = f"Nó cao hơn SVM tuyến tính trên {WORD[wl[0]]} trong mười bộ, với chênh lệch trung bình ${T['W_LIN_DIFF']}$ điểm"
lin += (f", và khác biệt này có ý nghĩa thống kê ({pv('linear')})" if wl[4] < 0.05 else
        f", nhưng giá trị {pv('linear')} chưa đủ để kết luận ở mức $5\\%$")
nonsig_rest = [m for m in nonsig if m in ("iksvm", "mlp", "rbf", "lssvm")]
lin += ("; khác biệt so với " + and_list([NAMEV[m] for m in nonsig_rest]) + " không có ý nghĩa thống kê." if nonsig_rest else ".")
parts.append(lin)
T["WILCOX_TEXT"] = " ".join(parts)
T["RBF_SIG_PHRASE"] = ", một khác biệt có ý nghĩa thống kê" if W["rbf"][4] < 0.05 else ", dù khác biệt này chưa có ý nghĩa thống kê"
T["LIN_SIG_PHRASE"] = "" if W["linear"][4] < 0.05 else " (chưa có ý nghĩa thống kê)"
assert W["linear"][3] > 0 and W["rbf"][3] < 0

T["MAGIC_C"], T["MAGIC_RBF"] = vn(mean[("Magic", "pwl_c")], 1), vn(mean[("Magic", "rbf")], 1)
T["P_C_MAGIC"], T["P_RBF_MAGIC"] = thousands(P[("Magic", "pwl_c")]), thousands(P[("Magic", "rbf")])
T["RATIO_MAGIC"] = ratio(P[("Magic", "rbf")] / P[("Magic", "pwl_c")])
rat = [P[(d, "rbf")] / P[(d, "pwl_c")] for d in names]
T["RATIO_MIN"], T["RATIO_MAX"] = ratio(min(rat)), ratio(max(rat))
assert max(rat) == P[("Magic", "rbf")] / P[("Magic", "pwl_c")]     # "Magic ... about RATIO_MAX times"
pc = [P[(d, "pwl_c")] for d in names]
T["PWLC_PMIN"], T["PWLC_PMAX"] = thousands(min(pc)), thousands(max(pc))

# two groups of data sets (statements of the second paragraph of 3.3.2)
svm_based = ["linear", "rbf", "lssvm", "iksvm", "pwl_c", "pwl_ls", "pwl_hh"]
for d in ["Haberman", "Transfusion", "Banknote", "Pima", "Breast"]:
    v = [mean[(d, m)] for m in svm_based]
    assert max(v) - min(v) <= 1.55, d
g2 = ["Magic", "Parkinsons", "Ionosphere", "Sonar"]
gap = [mean[(d, "rbf")] - mean[(d, "linear")] for d in g2]
assert 4 <= min(gap) and max(gap) <= 9.5, gap

hh = {r["rep"]: 100 * r["acc"] for r in rows if r["data"] == "Spambase" and r["method"] == "pwl_hh"}
assert sorted(hh) == [0, 1, 2, 3, 4]
fail = [k for k in hh if hh[k] < 60.6]                              # below the majority rate 60.6 %
ok = [hh[k] for k in hh if k not in fail]
assert 0 in fail                                                    # the diagnostics below are those of split 1
if fail == [0]:
    T["HH_SPAM_SENT"] = (f"ở lần chia đầu tiên, mô hình thất bại hoàn toàn (${vn(hh[0], 1)}\\%$, thấp hơn cả tỉ lệ "
                         f"$60{{,}}6\\%$ của lớp đa số), trong khi ở bốn lần chia còn lại nó đạt từ ${vn(min(ok), 1)}\\%$ "
                         f"đến ${vn(max(ok), 1)}\\%$")
else:
    T["HH_SPAM_SENT"] = (f"mô hình thất bại hoàn toàn ở {WORD[len(fail)]} trong năm lần chia (độ chính xác thấp hơn cả tỉ lệ "
                         f"$60{{,}}6\\%$ của lớp đa số)" + (f", trong khi ở các lần chia còn lại nó đạt từ ${vn(min(ok), 1)}\\%$ "
                         f"đến ${vn(max(ok), 1)}\\%$" if ok else ""))
dg = R("spam_hh_diag.json")
raw = [d for d in dg if not d["normalize"]]
pct = lambda v: vn(100 * v, 0) + "\\%"
T["DEAD_MIN"], T["DEAD_MAX"] = pct(min(d["dead"] for d in raw)), pct(max(d["dead"] for d in raw))
T["ALW_MIN"], T["ALW_MAX"] = pct(min(d["always"] for d in raw)), pct(max(d["always"] for d in raw))


def sci(x):
    if x < 1e4:
        return thousands(x)
    e = int(np.floor(np.log10(x)))
    return f"{vn(x / 10 ** e, 1)}\\cdot10^{{{e}}}"


T["MAXH_MIN"], T["MAXH_MAX"] = sci(min(d["max_feat"] for d in raw)), sci(max(d["max_feat"] for d in raw))
assert max(raw, key=lambda d: d["max_feat"])["rep"] == 0
T["MAXP0"] = sci(next(d["max_abs_p"] for d in raw if d["rep"] == 0))
T["NORM_MAXH"] = vn(np.ceil(10 * max(d["max_feat"] for d in dg if d["normalize"])) / 10, 1)
T["N_MLP_HH"] = WORD[sum(mean[(d, "mlp")] > mean[(d, "pwl_hh")] for d in names)]
LABEL = {"linear": "SVM tuyến tính", "pwl_c": "PWL-C-SVM cộng tính", "pwl_hh": "PWL-C-SVM HH", "mlp": "MLP-ReLU"}
out_of_frame = [f"{LABEL[m]} trên {d}" for m in LABEL for d in names
                if 100 * (M[(d, m)].mean() - M[(d, "rbf")].mean()) < -12.0]      # xmin of fig_benchmark_diff
T["BENCH_DIFF_NOTE"] = (" Mũi tên chỉ giá trị nằm ngoài khung hình: " + ", ".join(out_of_frame) + ".") if out_of_frame else ""
sp = R("spam_hhnorm.json")
acc = 100 * np.array([r["acc"] for r in sp])
T["SPAMHHNORM"] = f"${vn(acc.mean(), 1)}\\pm{vn(acc.std(ddof=1), 1)}\\%$"
T["SPAMHHNORM_PLAIN"] = vn(acc.mean(), 1)

# ---------------------------------------------------------------- credit (Section 3.5)
from make_tables import credit_results
cr = credit_results()["rows"]
for m in ("pwl_ls_q", "logit_bin", "pwl_ls_mono", "hgb"):
    assert len({r["rep"] for r in cr if r["method"] == m}) == 5, m
cm = lambda m, k: np.mean([r[k] for r in cr if r["method"] == m])
cmed = lambda m, k: np.median([r[k] for r in cr if r["method"] == m])
per = lambda m: np.array([next(r["auc"] for r in cr if r["method"] == m and r["rep"] == k) for k in range(5)])
for key, m in [("A_Q", "pwl_ls_q"), ("A_HGB", "hgb"), ("A_LOGIT", "logit"), ("A_LIN", "linear"),
               ("A_RBF", "rbf"), ("A_MLP", "mlp"), ("A_UNI", "pwl_ls_uni"), ("A_CQ", "pwl_c_q"),
               ("A_HH", "pwl_ls_hh"), ("A_BIN", "logit_bin"), ("A_MONO", "pwl_ls_mono")]:
    T[key] = vn(cm(m, "auc"), 3)
T["D_Q_LOGIT"] = vn(cm("pwl_ls_q", "auc") - cm("logit", "auc"), 3)
T["D_Q_BIN"] = vn(cm("pwl_ls_q", "auc") - cm("logit_bin", "auc"), 3)
assert cm("pwl_ls_q", "auc") > cm("logit_bin", "auc")                 # "higher than the scorecard"
k_gt = int(np.sum(per("pwl_ls_q") > per("logit_bin")))
assert k_gt >= 3, k_gt
T["N_Q_GT_BIN"] = "cả năm lần chia" if k_gt == 5 else WORD[k_gt] + " trong năm lần chia"
T["SD_Q"] = vn(np.std(per("pwl_ls_q"), ddof=1), 3)
for key, m in (("P_Q", "pwl_ls_q"), ("P_UNI", "pwl_ls_uni"), ("P_RBF", "rbf"), ("P_HGB", "hgb"), ("P_BIN", "logit_bin")):
    T[key] = thousands(cmed(m, "params"))
tr_ = R("credit_hgb_trees.json")
assert [t["params"] for t in tr_] == [next(r["params"] for r in cr if r["method"] == "hgb" and r["rep"] == t["rep"]) for t in tr_]
T["HGB_TREES_MIN"], T["HGB_TREES_MAX"] = str(min(t["trees"] for t in tr_)), str(max(t["trees"] for t in tr_))
auc = {m: cm(m, "auc") for m in {r["method"] for r in cr}}
lead, lower = ("pwl_ls_q", "hgb", "pwl_ls_uni", "logit_bin"), ("logit", "linear", "rbf")
assert max(auc[m] for m in lower) < auc["mlp"] < min(auc[m] for m in lead)          # two groups, MLP between
assert max(auc, key=auc.get) in ("pwl_ls_q", "hgb")
assert auc["pwl_ls_q"] > auc["pwl_ls_uni"] and cmed("pwl_ls_q", "params") < cmed("pwl_ls_uni", "params")
assert auc["pwl_c_q"] < auc["pwl_ls_q"] - 0.01 and auc["pwl_ls_hh"] < auc["pwl_ls_q"]
assert 0.5 * cmed("logit_bin", "params") < cmed("pwl_ls_q", "params") < 2 * cmed("logit_bin", "params")  # "comparable size"
d = auc["pwl_ls_q"] - auc["hgb"]
if abs(d) < 0.0015:
    T["VS_HGB"] = "ngang với"; T["SENT_BEST"] = "ngang với tổ hợp cây tăng cường"
elif d > 0:
    T["VS_HGB"] = "nhỉnh hơn"; T["SENT_BEST"] = "cao nhất, nhỉnh hơn tổ hợp cây tăng cường"
else:
    T["VS_HGB"] = "chỉ kém một chút so với"; T["SENT_BEST"] = "chỉ kém tổ hợp cây tăng cường một chút"
T["VS_HGB2"] = T["VS_HGB3"] = T["VS_HGB"]
for k in ("bacc", "f1"):                                            # "the same conclusion"
    assert min(cm(m, k) for m in lead) > max(cm(m, k) for m in lower) + 0.005, k
    assert max(cm(m, k) for m in lead) - min(cm(m, k) for m in lead) < 0.01, k   # "almost equal"
LEADNAME = {"pwl_ls_q": "PWL-LS-SVM cộng tính với nút phân vị", "hgb": "tổ hợp cây tăng cường",
            "pwl_ls_uni": "PWL-LS-SVM cộng tính với lưới đều", "logit_bin": "thẻ điểm logistic trên các biến đã rời rạc hóa"}
lead_sorted = sorted(lead, key=lambda m: -auc[m])
T["LEAD_TEXT"] = and_list([f"{LEADNAME[m]} ({'AUC trung bình ' if j == 0 else ''}{vn(auc[m], 3)})"
                           for j, m in enumerate(lead_sorted)])
# monotone constraints (Section 3.5.5)
mo = R("credit_monotone.json")
assert mo["max_coef_diff_free_vs_ridge"] < 1e-9
free = np.array([r["auc"] for r in mo["rows"] if r["method"] == "free"])
assert np.allclose(free, per("pwl_ls_q"), atol=1e-6)               # the "free" model is the model of Table 3.x
d_mb = auc["pwl_ls_mono"] - auc["logit_bin"]
T["MONO_VS_BIN"] = "ngang với" if abs(d_mb) < 0.0015 else "cao hơn" if d_mb > 0 else "thấp hơn một chút so với"
dm = auc["pwl_ls_mono"] - auc["pwl_ls_q"]
assert dm > -0.005                                                   # at most a very small cost
if dm <= -0.0005:
    T["MONO_EFFECT_LONG"] = f"Cái giá về độ chính xác rất nhỏ: AUC trung bình chỉ giảm từ {T['A_Q']} xuống {T['A_MONO']}"
    T["MONO_EFFECT_SHORT"] = f"AUC chỉ giảm từ {T['A_Q']} xuống {T['A_MONO']}"
    T["MONO_EFFECT_TAIL"] = "mà hầu như không mất độ chính xác"
elif dm < 0.0005:
    T["MONO_EFFECT_LONG"] = f"Độ chính xác hầu như không thay đổi: AUC trung bình là {T['A_MONO']}, so với {T['A_Q']} của mô hình không ràng buộc"
    T["MONO_EFFECT_SHORT"] = f"AUC hầu như không đổi ({T['A_MONO']} so với {T['A_Q']})"
    T["MONO_EFFECT_TAIL"] = "mà không mất độ chính xác"
else:
    T["MONO_EFFECT_LONG"] = (f"Độ chính xác không những không giảm mà còn tăng nhẹ: AUC trung bình tăng từ {T['A_Q']} lên {T['A_MONO']}, "
                             "vì ràng buộc loại bỏ những đoạn được ước lượng từ quá ít dữ liệu và do đó đóng vai trò như một dạng chính quy hóa có cơ sở")
    T["MONO_EFFECT_SHORT"] = f"AUC còn tăng nhẹ, từ {T['A_Q']} lên {T['A_MONO']}"
    T["MONO_EFFECT_TAIL"] = "mà không phải hy sinh độ chính xác"
kp = sum(any(v["bad"][nm] > 0 for nm in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")) for v in mo["violations"])
kl = sum(v["bad"]["LIMIT_BAL"] > 0 for v in mo["violations"])
assert kp >= 1
T["VIOL_ANY"] = "cả năm lần chia" if kp == 5 else WORD.get(kp, str(kp)) + " trong năm lần chia"
T["VIOL_LIMIT"] = "cả năm lần chia" if kl == 5 else WORD.get(kl, str(kl)) + " trong năm lần chia"
assert kl >= 1                                                       # otherwise rewrite the LIMIT_BAL clause
for nm in ("PAY_0", "PAY_4"):                                        # Fig. 3.x: flat after two months, same elsewhere
    xf, hf = np.array(mo["shapes"]["free"][nm]["x"]), np.array(mo["shapes"]["free"][nm]["h"])
    hm = np.array(mo["shapes"]["monotone"][nm]["h"])
    assert np.ptp(hm[xf >= 2]) < 1e-6, nm
    assert np.abs(hm - hf)[xf <= 2].max() < 0.15 * np.ptp(hf), nm
hm = np.array(mo["shapes"]["monotone"]["LIMIT_BAL"]["h"])
assert np.all(np.diff(hm) <= 1e-9)
sh = mo["share_ge3"]
_sh = np.ceil(100 * max(sh["PAY_0"], sh["PAY_4"]) * 2) / 2
T["SHARE_GE3"] = "chưa tới " + (str(int(_sh)) if _sh == int(_sh) else vn(_sh, 1)) + "\\%"
T["SHARE_P0_GE3"] = vn(100 * mo["share_pay0_ge3"], 1) + "\\%"
dr = mo["default_rate_pay0"]; cnt = mo["count_pay0"]
for key, v in (("DR_M2", "-2"), ("DR_M1", "-1"), ("DR_0", "0"), ("DR_1", "1"), ("DR_2", "2")):
    T[key] = vn(100 * dr[v], 1) + "\\%"
ge3 = [v for v in dr if int(v) >= 3]
dr_ge3 = sum(dr[v] * cnt[v] for v in ge3) / sum(cnt[v] for v in ge3)
assert dr_ge3 >= dr["2"]                                             # "not lower than the two-month group"
T["DR_GE3"] = vn(100 * dr_ge3, 1) + "\\%"
assert dr["-1"] > dr["-2"] and dr["-1"] > dr["0"]
# shape functions of the first split (Section 3.5.4)
it = R("credit_interpret.json")
shp = it["shapes"]
hp = dict(zip(shp["PAY_0"]["x"], shp["PAY_0"]["h"]))
jump = hp[2.0] - hp[0.0]
T["J_PAY0"] = vn(jump, 2)
amp = {nm: max(v["h"]) - min(v["h"]) for nm, v in shp.items() if nm != "PAY_0"}
assert jump > max(amp.values()), max(amp.items(), key=lambda kv: kv[1])
xl, hl = np.array(shp["LIMIT_BAL"]["x"]), np.array(shp["LIMIT_BAL"]["h"])
h100 = np.interp(100000, xl, hl)
share = (hl[0] - h100) / (hl[0] - hl[-1])
T["LIM_SHARE"] = "khoảng " + str(int(5 * round(20 * share))) + "\\%"
assert hl[0] > 0 > hl[-1]
assert shp["BILL_AMT3"]["h"][-1] > shp["BILL_AMT3"]["h"][0] and shp["BILL_AMT4"]["h"][-1] < shp["BILL_AMT4"]["h"][0]
imp = dict(it["importance"]); imp.update(it["importance_groups"])
for g in ("EDUCATION", "MARRIAGE", "SEX", "AGE"):
    assert imp[g] < 0.5 * imp["LIMIT_BAL"], g                        # "negligible demographic variables"
single = {k: v for k, v in imp.items() if not k.startswith(("EDU_", "MARRIED", "SINGLE", "FEMALE"))}
order = sorted(single, key=lambda k: -single[k])
assert order[0] == "PAY_0" and order[1] == "LIMIT_BAL" and single["PAY_0"] > 2.5 * single["LIMIT_BAL"]
cl = it["client"]
T["C_SCORE"], T["C_BASE"], T["C_THR"] = vn(cl["score"], 2), vn(cl["base"], 2), vn(it["threshold"], 2)
T["C_EXCESS"] = vn(cl["score"] - cl["base"], 2)
T["C_PAY0"] = vn(cl["contrib"]["PAY_0"], 2)
T["C_PAYPREV"] = vn(sum(cl["contrib"][f"PAY_{k}"] for k in range(2, 7)), 2)
T["C_LIMIT"] = vn(cl["contrib"]["LIMIT_BAL"], 2)
T["C_LIMIT_VAL"] = thousands(cl["values"]["LIMIT_BAL"])
assert cl["values"]["PAY_0"] == 2 and all(cl["values"][f"PAY_{k}"] >= 1 for k in range(2, 7))
st = R("credit_shapes_all.json")
cb = st["corr_bill_consecutive"]
T["CORR_BILL_MIN"], T["CORR_BILL_MAX"] = vn(min(cb), 2), vn(max(cb), 2)
# stability of the shape functions over the five splits
assert len(st["splits"]) == 5
num = [nm for nm in st["names"] if not nm.startswith(("EDU_", "FEMALE", "MARRIED", "SINGLE"))]
tops = [sorted(num, key=lambda k: -sp["importance"][k])[:6] for sp in st["splits"]]
same6 = sum(set(t) == set(tops[0]) for t in tops[1:])               # among the four other splits
assert {t[0] for t in tops} == {"PAY_0"}
if {t[1] for t in tops} == {"LIMIT_BAL"}:
    T["STAB_TOP"] = "ở cả năm lần chia, PAY\\_0 là biến có ảnh hưởng lớn nhất và LIMIT\\_BAL đứng thứ hai"
else:
    T["STAB_TOP"] = "ở cả năm lần chia, PAY\\_0 là biến có ảnh hưởng lớn nhất"
if same6 == 4:
    T["STAB_TOP"] += ", và sáu biến quan trọng nhất là như nhau ở mọi lần chia"
elif same6 >= 2:
    T["STAB_TOP"] += f", và sáu biến quan trọng nhất trùng với lần chia đầu tiên ở {WORD[same6]} trong bốn lần chia còn lại"
else:
    T["STAB_TOP"] += "; thứ tự của các biến tiếp theo, vốn có mức ảnh hưởng gần nhau, thì thay đổi giữa các lần chia"
gl = np.array(st["grid"]["LIMIT_BAL"])
msk = (gl >= st["q05"]["LIMIT_BAL"]) & (gl <= st["q95"]["LIMIT_BAL"])
HL = np.array([sp["h"]["LIMIT_BAL"] for sp in st["splits"]])[:, msk]
dev = np.abs(HL[1:] - HL[0]).max() / (HL[0].max() - HL[0].min())
T["STAB_LIM"] = f"{int(5 * np.ceil(20 * dev))}\\%"
x0 = np.array(st["grid"]["PAY_0"]); H0 = np.array([sp["h"]["PAY_0"] for sp in st["splits"]])[:, x0 <= 2]
dev0 = np.abs(H0[1:] - H0[0]).max() / (H0[0].max() - H0[0].min())
T["STAB_P0"] = f"{int(5 * np.ceil(20 * dev0))}\\%"
assert dev < 0.3 and dev0 < 0.3, (dev, dev0)                         # "almost unchanged"
lo, hi = [], []
for nm in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
    xg = np.array(st["grid"][nm]); sd = np.array([sp["h"][nm] for sp in st["splits"]]).std(axis=0, ddof=1)
    lo.append(sd[xg <= 2].mean()); hi.append(sd[xg >= 3].mean())
rs = np.mean(hi) / np.mean(lo)
assert rs > 2, rs
T["STAB_RATIO"] = ("khoảng " + str(int(round(rs)))) if rs < 10 else ("khoảng " + str(int(5 * round(rs / 5))))

# ---------------------------------------------------------------- cost (Section 3.4)
co = R("cost.json")
g = lambda data, m, k: next(r[k] for r in co if r["data"] == data and r["method"] == m)
T["T_LS_MAGIC"] = fmt_us(g("Magic", "pwl_ls", "t_pred"))
T["T_LS_CREDIT"] = fmt_us(g("Credit", "pwl_ls", "t_pred"))
T["RT_MAGIC"] = ratio(g("Magic", "rbf", "t_pred") / g("Magic", "pwl_ls", "t_pred"))
T["RT_CREDIT"] = ratio(g("Credit", "rbf", "t_pred") / g("Credit", "pwl_ls", "t_pred"))
n_credit = 26
T["P_RBF_CREDIT"] = thousands(g("Credit", "rbf", "params"))
T["NSV_CREDIT"] = thousands((g("Credit", "rbf", "params") - 1 - 2 * n_credit) / (n_credit + 1))
T["P_LS_CREDIT"] = thousands(g("Credit", "pwl_ls", "params"))
T["P_MLP_CREDIT"] = thousands(g("Credit", "mlp", "params"))
T["P_HGB_CREDIT"] = thousands(g("Credit", "hgb", "params"))
T["RP_CREDIT"] = ratio(g("Credit", "rbf", "params") / g("Credit", "pwl_ls", "params"))
assert g("Credit", "rbf", "params") / g("Credit", "pwl_ls", "params") >= 100   # "hundreds of times smaller"
for data, suf in (("Magic", "MAGIC"), ("Credit", "CREDIT")):
    for m, key in (("pwl_ls", "LS"), ("pwl_c", "C"), ("rbf", "RBF"), ("mlp", "MLP")):
        T[f"TF_{key}_{suf}"] = fmt_sec(g(data, m, "t_fit"))


def rel_phrase(x, ref=" PWL-SVM cộng tính"):
    """x = t_pred(model) / t_pred(PWL-LS-SVM)."""
    if x < 1 / 1.5:
        return f"nhanh hơn{ref} khoảng {ratio(1 / x)} lần"
    if x <= 1.5:
        return f"nhanh ngang{ref}" if ref else "nhanh tương đương"
    return f"chậm hơn{ref} khoảng {ratio(x)} lần"


rel = lambda m: g("Credit", m, "t_pred") / g("Credit", "pwl_ls", "t_pred")
assert g("Credit", "mlp", "params") > 5 * g("Credit", "pwl_ls", "params")
assert g("Credit", "hgb", "params") > 5 * g("Credit", "pwl_ls", "params")
txt = (f"Trên dữ liệu tín dụng, mạng nơ-ron dự đoán {rel_phrase(rel('mlp'))}, còn tổ hợp cây tăng cường "
       f"{rel_phrase(rel('hgb'), '')}; cả hai mô hình cũng lớn hơn nhiều, với ${T['P_MLP_CREDIT']}$ và khoảng "
       f"${T['P_HGB_CREDIT']}$ số thực. ")
if rel("mlp") < 1:
    # operation counts per prediction: additive PWL-SVM 2D + 2M, one-hidden-layer MLP 2nH + 4H (H = 9n)
    nC, H = 26, 9 * 26
    Dk = (g("Credit", "pwl_ls", "params") - 3 * nC - 1) / 2            # params = D knots + M weights + 1 + 2n
    ops_pwl, ops_mlp = 2 * Dk + 2 * (nC + Dk), 2 * nC * H + 4 * H
    txt += (f"Việc mạng nơ-ron dự đoán nhanh hơn không mâu thuẫn với phân tích ở \\cref{{tab:cost}}: tính theo số phép toán, "
            f"nó cần khoảng ${thousands(ops_mlp)}$ phép tính cho mỗi mẫu, nhiều gấp khoảng {ratio(ops_mlp / ops_pwl)} lần "
            f"PWL-SVM cộng tính (khoảng ${thousands(ops_pwl)}$), nhưng phép nhân ma trận của nó được thư viện BLAS tối ưu hóa, "
            f"còn cài đặt PWL-SVM bằng NumPy tạo ra nhiều mảng trung gian. ")
pb, pl = g("Credit", "logit_bin", "params"), g("Credit", "pwl_ls", "params")
assert 0.5 < pb / pl < 2
txt += (f"Thẻ điểm có kích thước tương đương PWL-SVM cộng tính (${thousands(pb)}$ so với ${thousands(pl)}$ số thực), "
        f"và trong cài đặt dựa trên scikit-learn, nó dự đoán {rel_phrase(rel('logit_bin'), ' PWL-SVM')}. ")
for m in ("linear", "logit"):
    for data in ("Magic", "Credit"):
        assert g(data, m, "t_pred") < g(data, "pwl_ls", "t_pred") and g(data, m, "params") < g(data, "pwl_ls", "params"), (m, data)
    assert g("Magic", "pwl_ls", "acc") - g("Magic", m, "acc") > 0.02 and g("Credit", "pwl_ls", "auc") - g("Credit", m, "auc") > 0.02, m
for m in ("logit_bin", "pwl_c", "pwl_hh", "rbf", "mlp", "hgb"):       # "only the linear models are smaller AND faster"
    for data in ("Magic", "Credit"):
        assert not (g(data, m, "t_pred") < 0.9 * g(data, "pwl_ls", "t_pred") and g(data, m, "params") < g(data, "pwl_ls", "params")), (m, data)
txt += (f"Chỉ các mô hình tuyến tính trên biến gốc là nhỏ và nhanh hơn PWL-SVM, với cái giá là độ chính xác thấp hơn rõ rệt: "
        f"trên Magic, hồi quy logistic đạt ${vn(100 * g('Magic', 'logit', 'acc'), 1)}\\%$ so với "
        f"${vn(100 * g('Magic', 'pwl_ls', 'acc'), 1)}\\%$ của PWL-LS-SVM cộng tính.")
T["COST_OTHERS"] = txt
fastest = True
for data in ("Magic", "Credit"):
    others = {m: g(data, m, "t_fit") for m in ("pwl_c", "rbf", "mlp", "hgb", "logit_bin")}
    fastest &= g(data, "pwl_ls", "t_fit") < min(others.values())
    assert g(data, "pwl_ls", "t_fit") < 3 * min(others.values()), (data, others)
    assert g(data, "pwl_ls", "t_pred") < g(data, "rbf", "t_pred")
if fastest:
    T["TRAIN_LEAD"] = "PWL-LS-SVM là mô hình phi tuyến huấn luyện nhanh nhất"
    T["TRAIN_CH2"] = "PWL-LS-SVM còn là mô hình phi tuyến huấn luyện nhanh nhất."
else:
    T["TRAIN_LEAD"] = "PWL-LS-SVM thuộc nhóm mô hình phi tuyến huấn luyện nhanh nhất, cùng với thẻ điểm"
    T["TRAIN_CH2"] = "PWL-LS-SVM còn thuộc nhóm mô hình phi tuyến huấn luyện nhanh nhất."
assert g("Credit", "pwl_ls", "t_fit") < 1.0                          # "retraining takes less than a second"

# ---------------------------------------------------------------- free text written after looking at the figures
TEXT = {}
for key, fn in (("SHAPES_TEXT", "shapes_text.tpl"), ("EXPLAIN_TEXT", "explain_text.tpl"), ("STAB_TEXT", "stab_text.tpl")):
    TEXT[key] = open(os.path.join(CH, fn)).read().strip()

# ---------------------------------------------------------------- render templates
for tpl in ("chuong3_bench_b", "chuong3_cost", "chuong3_credit_b", "chuong3_end", "ketluan"):
    s = open(os.path.join(CH, tpl + ".tpl")).read()
    for _ in range(2):                                   # text blocks may contain further text blocks
        s = re.sub(r"\{\{([A-Z]+_TEXT)\}\}", lambda m: TEXT.get(m.group(1), m.group(0)), s)
    missing = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", s)) - set(T))
    if missing:
        raise SystemExit(f"{tpl}: missing tokens {missing}")
    s = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: T[m.group(1)], s)
    open(os.path.join(CH, tpl + ".tex"), "w").write(
        "% Generated by code/fill_numbers.py from " + tpl + ".tpl -- edit the template, not this file.\n" + s)
json.dump(T, open(os.path.join(HERE, "results", "numbers_in_text.json"), "w"), ensure_ascii=False, indent=1)
print(T)
