import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class C_SVM:
    def __init__(self, X, y, train_size, test_size, M=5):
        self.X = StandardScaler().fit_transform(X)
        self.y = y
        self.train_size = train_size
        self.test_size = test_size
        self.M = M
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_custom()

    def split_custom(self):
        np.random.seed(42)
        indices = np.random.permutation(len(self.X))
        train_idx = indices[:self.train_size]
        test_idx = indices[self.train_size:self.train_size + self.test_size]
        return self.X[train_idx], self.X[test_idx], self.y[train_idx], self.y[test_idx]
    
    def evaluate(self):
        clf_svm = SVC(kernel='linear')
        clf_svm.fit(self.X_train, self.y_train)
        acc_svm = accuracy_score(self.y_test, clf_svm.predict(self.X_test))
        return acc_svm