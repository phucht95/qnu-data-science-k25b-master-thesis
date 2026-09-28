"""Benchmark data sets (OpenML copies of UCI data sets) used in Chapter 3."""

import numpy as np
from sklearn.datasets import fetch_openml

# name: (OpenML id, positive label, number of training points or None for a 50/50 split)
DATA = {
    "Haberman":    (43, "2", None),
    "Transfusion": (1464, "2", None),
    "Banknote":    (1462, "2", None),
    "Pima":        (37, "tested_positive", None),
    "Breast":      (15, "malignant", None),
    "Magic":       (1120, "h", 2000),          # 2000 / 17020 as in Huang et al. (2013)
    "Parkinsons":  (1488, "2", None),
    "Ionosphere":  (59, "g", None),
    "Spambase":    (44, "1", None),
    "Sonar":       (40, "Mine", None),
}

DESCRIPTION_VI = {
    "Haberman": "sống sót sau phẫu thuật ung thư vú",
    "Transfusion": "hiến máu tình nguyện",
    "Banknote": "xác thực tiền giấy",
    "Pima": "bệnh tiểu đường",
    "Breast": "ung thư vú Wisconsin (bản gốc)",
    "Magic": "kính thiên văn tia gamma",
    "Parkinsons": "bệnh Parkinson (giọng nói)",
    "Ionosphere": "tín hiệu radar tầng điện li",
    "Spambase": "thư rác",
    "Sonar": "tín hiệu sonar (đá hay mìn)",
}


def load(name):
    """Return X (float, missing values imputed by the median, constant columns removed) and y in {-1,+1}."""
    did, pos, _ = DATA[name]
    d = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X = d.data.apply(lambda c: c.astype(float))
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0].to_numpy()
    y = np.where(d.target.astype(str).to_numpy() == pos, 1, -1)
    return X, y
