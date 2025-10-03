import numpy as np
from data_loader.data_loader import DataLoader
from ucimlrepo import fetch_ucirepo


class Parkinsons(DataLoader):
    def load(self):
        self.name = "Parkinsons"
        data = fetch_ucirepo(id=174)
        self.X = data.data.features
        self.y = data.data.targets

        self.X = self.X.values
        self.y = self.y.values.flatten()
