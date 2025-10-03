import time
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import accuracy_score
from models.model_runner import ModelRunner


class PWLCSVM(ModelRunner):
    def __init__(self, dataset, M):
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
        self.clf = SVC(kernel='linear')
        self.clf.fit(self.X_train, self.y_train)
        self.accuracy = accuracy_score(self.y_test, self.clf.predict(self.X_test))
        self.runtime = round(time.time() - start, 4)
        print(f"🧩 PWL C-SVM Accuracy on {self.name}: {self.accuracy:.4f} | Time: {self.runtime} s")