import numpy as np
from data_loader.data_loader import DataLoader


class Clowns(DataLoader):
    def load(self):
        self.name = "Clowns"
        n_samples = self.train_size + self.test_size
        noise = 0.2

        x1_pos = np.random.randn(n_samples // 2) * noise
        x2_pos = x1_pos**2 + np.random.randn(n_samples // 2) * noise

        x1_neg = np.random.randn(n_samples // 2) * noise
        x2_neg = -x1_neg**2 + np.random.randn(n_samples // 2) * noise

        x1 = np.concatenate([x1_pos, x1_neg])
        x2 = np.concatenate([x2_pos, x2_neg])
        self.X = np.vstack((x1, x2)).T
        self.y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))