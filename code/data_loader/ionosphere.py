import numpy as np
from sklearn.datasets import fetch_openml
from data_loader.data_loader import DataLoader


class Ionosphere(DataLoader):
    def load(self):
        self.name = "Ionosphere"
        data = fetch_openml(name='ionosphere', version=1, as_frame=False)
        self.X = data.data
        self.y = np.where(data.target == 'g', 1, 0)