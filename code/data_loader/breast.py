from sklearn.datasets import load_breast_cancer
from ucimlrepo import fetch_ucirepo
from data_loader.data_loader import DataLoader


class Breast(DataLoader):
    def load(self):
        self.name = "Breast Cancer Wisconsin (Original)"
        data = fetch_ucirepo(id=15)
        X = data.data.features
        y = data.data.targets

        # Process NaN Value
        mean_value = X.Bare_nuclei.mean()
        X.loc[:, 'Bare_nuclei'] = X['Bare_nuclei'].fillna(mean_value)

        self.X = X.values
        self.y = (y == 4).astype(int).values.flatten()
