import numpy as np
from sklearn.datasets import fetch_openml
from data_loader.data_loader import DataLoader


class Pima(DataLoader):
    def load(self):
        self.name = "Pima Indians Diabetes"
        data = fetch_openml(name='diabetes', version=1, as_frame=False)
        self.X = data.data
        self.y = np.where(data.target == 'tested_positive', 1, 0)