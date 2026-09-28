"""Quick feasibility probe for application option (C): default of credit card clients.

Not part of the thesis results -- only used to judge whether the case study is promising.
Data: Yeh & Lien (2009), 30,000 clients, 23 features, ~22% defaults (OpenML id 42477).
"""

import time
import warnings

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from pwlsvm import PWLSVM

warnings.filterwarnings("ignore")
d = fetch_openml(data_id=42477, as_frame=True, parser="auto")
X = d.data.apply(lambda c: c.astype(float)).to_numpy()
y = np.where(d.target.astype(str).to_numpy() == "1", 1, -1)
print("data", X.shape, "default rate", (y == 1).mean().round(3))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
sub = np.random.default_rng(0).choice(len(ytr), 5000, replace=False)   # RBF-SVM on a subsample


def report(name, model, Xf, yf, score):
    t0 = time.perf_counter(); model.fit(Xf, yf); tfit = time.perf_counter() - t0
    t0 = time.perf_counter(); s = score(model, Xte); tpred = (time.perf_counter() - t0) / len(yte)
    auc = roc_auc_score(yte, s)
    bacc = balanced_accuracy_score(yte, np.where(s > np.quantile(s, 1 - (ytr == 1).mean()), 1, -1))
    print(f"{name:28s} AUC={auc:.3f}  bal.acc={bacc:.3f}  fit={tfit:7.2f}s  predict={1e6*tpred:7.2f} us/sample", flush=True)


df = lambda m, Z: m.decision_function(Z)
pp = lambda m, Z: m.predict_proba(Z)[:, 1]
report("logistic regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xtr, ytr, df)
report("PWL-LS-SVM additive M/n=10", PWLSVM(kind="additive", m_per_dim=10, loss="squared", C=1.0), Xtr, ytr, df)
report("PWL-C-SVM additive M/n=10", PWLSVM(kind="additive", m_per_dim=10, loss="hinge", solver="liblinear", C=0.1), Xtr, ytr, df)
report("PWL-C-SVM HH M/n=10", PWLSVM(kind="hh", m_per_dim=10, loss="hinge", solver="liblinear", C=0.1, random_state=0), Xtr, ytr, df)
report("SVM-RBF (5k subsample)", make_pipeline(MinMaxScaler(), SVC(C=1.0, gamma="scale")), Xtr[sub], ytr[sub], df)
report("SVM-RBF (full 15k)", make_pipeline(MinMaxScaler(), SVC(C=1.0, gamma="scale")), Xtr, ytr, df)
report("gradient boosting (reference)", HistGradientBoostingClassifier(random_state=0), Xtr, ytr, pp)
