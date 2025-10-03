import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ModelRunner:
    def __init__(self, dataset):
        self.name = dataset.name
        self.X_train = dataset.X_train
        self.X_test = dataset.X_test
        self.y_train = dataset.y_train
        self.y_test = dataset.y_test
        self.clf = None
        self.accuracy = None
        self.runtime = None

    def run(self):
        raise NotImplementedError("Phương thức run() phải được định nghĩa trong class con.")
