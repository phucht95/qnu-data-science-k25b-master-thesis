import numpy as np
from data_loader.data_loader import DataLoader


class CosExp(DataLoader):
    def load(self):
        self.name = "Cosexp"
        n_samples = self.train_size + self.test_size
        x1 = np.random.uniform(-2, 2, n_samples)
        x2 = np.cos(x1) + np.exp(-x1**2) + np.random.normal(0, 0.1, n_samples)
        self.X = np.vstack((x1, x2)).T
        self.y = (x2 > np.cos(x1)).astype(int)