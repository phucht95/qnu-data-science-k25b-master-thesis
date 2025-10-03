import numpy as np
import time
from neo_ls_svm import NeoLSSVM
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from models.model_runner import ModelRunner


class PWLLSSVM(ModelRunner):
    def __init__(self, dataset, M=5):
        super().__init__(dataset)
        self.M = M
        self.X_train, self.X_test = self.apply_plfm(self.X_train, self.X_test)

    def apply_plfm(self, X_train, X_test):
        kmeans = KMeans(n_clusters=self.M, random_state=42)
        regions_train = kmeans.fit_predict(X_train)
        regions_test = kmeans.predict(X_test)
        return self.map(X_train, regions_train), self.map(X_test, regions_test)

    def map(self, X, regions):
        n_samples, n_features = X.shape
        mapped = np.zeros((n_samples, n_features * self.M))
        for i in range(n_samples):
            r = regions[i]
            mapped[i, r * n_features:(r + 1) * n_features] = X[i]
        return mapped

    def run(self):
        start = time.time()

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train.ravel()
        y_test = self.y_test.ravel()

        self.clf = NeoLSSVM()
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        if y_pred is None:
            raise ValueError("❌ PWL-LS-SVM.predict() trả về None. Kiểm tra lại dữ liệu đầu vào và bước fit().")

        self.accuracy = accuracy_score(y_test, y_pred)
        self.runtime = round(time.time() - start, 4)
        print(f"🧩 PWL LS-SVM (neo) Accuracy on {self.name}: {self.accuracy:.4f} | Time: {self.runtime} s")
