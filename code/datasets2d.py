"""Two-dimensional benchmark sets used for the illustrations in Chapter 3."""

import numpy as np
from sklearn.datasets import make_circles, make_moons


def moons(n, rng):
    X, y = make_moons(n_samples=n, noise=0.15, random_state=int(rng.integers(2**31)))
    return X, np.where(y == 1, 1, -1)


def circles(n, rng):
    X, y = make_circles(n_samples=n, noise=0.08, factor=0.5, random_state=int(rng.integers(2**31)))
    return X, np.where(y == 1, 1, -1)


def checker(n, rng, cells=4):
    """Uniform points on [0,1]^2 labelled by the colour of a cells x cells checkerboard."""
    X = rng.uniform(0, 1, size=(n, 2))
    idx = np.floor(X * cells).astype(int)
    return X, np.where((idx[:, 0] + idx[:, 1]) % 2 == 0, 1, -1)


def spirals(n, rng, turns=1.6, noise=0.06):
    m = n // 2
    t = np.sqrt(rng.uniform(0.05, 1, size=m)) * turns * 2 * np.pi
    a = np.c_[t * np.cos(t), t * np.sin(t)] / (turns * 2 * np.pi)
    X = np.r_[a, -a] + noise * rng.standard_normal((2 * m, 2))
    return X, np.r_[np.ones(m), -np.ones(m)].astype(int)


def cosexp(n, rng, maxx=5.0, freq=0.7):
    """Data set 'Cosexp' of the SVM-KM toolbox (Canu et al., 2005), as used by Huang et al."""
    xi = rng.uniform(0, 1, size=(n, 2))
    X = np.c_[maxx * xi[:, 0], 2 * xi[:, 1] - 1]
    y = np.sign(np.cos(0.5 * np.exp(freq * X[:, 0])) - X[:, 1])
    y[y == 0] = 1
    return X, y.astype(int)


GENERATORS = {"moons": moons, "circles": circles, "checker": checker, "spirals": spirals,
              "cosexp": cosexp}
NAMES_VI = {"moons": "Hai mặt trăng", "circles": "Hai vòng tròn", "checker": "Bàn cờ 4×4",
            "spirals": "Hai xoắn ốc", "cosexp": "Cosexp"}


def make(name, n_train=500, n_test=500, seed=0):
    rng = np.random.default_rng(seed)
    X, y = GENERATORS[name](n_train + n_test, rng)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
