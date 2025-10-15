from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


class DataLoader:
    """
    1. Cosexp
    2. Pima Indians Diabetes
    3. Ionosphere
    4. Parkinsons
    5. Breast Cancer
    """
    def __init__(self, train_size=100, test_size=50):
        self.train_size = train_size
        self.test_size = test_size
        self.name = None
        self.X = None
        self.y = None
        self.load()
        self.preprocess()
        self.split()

    def load(self):
        raise NotImplementedError("Phương thức load() phải được định nghĩa trong class con.")

    def preprocess(self):   
        self.X = StandardScaler().fit_transform(self.X)
        self.y = self.y.astype(int)

    def split(self):
        np.random.seed(42)
        indices = np.random.permutation(len(self.X))
        train_idx = indices[:self.train_size]
        test_idx = indices[self.train_size:self.train_size + self.test_size]
        self.X_train = self.X[train_idx]
        self.X_test = self.X[test_idx]
        self.y_train = self.y[train_idx]
        self.y_test = self.y[test_idx]

        # self.X_train[:, 0] = (self.X_train[:, 0] - self.X_train[:, 0].min())/ (self.X_train[:, 0].max() - self.X_train[:, 0].min())
        # self.X_train[:, 1] = (self.X_train[:, 1] - self.X_train[:, 1].min())/ (self.X_train[:, 1].max() - self.X_train[:, 1].min())
        # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr')
        # # plt.scatter(x_train[:, 0], x_train[:, 1], color="green", marker="*", cmap='bwr')
        # # plt.scatter(x_train[:, 0], x_train[:, 1], color="red", marker="+", cmap='bwr')
        # plt.scatter(self.X_test[self.y_test == 1, 0], self.X_test[self.y_test == 1, 1], c='g', marker="*", label='Class 1')
        # plt.scatter(self.X_test[self.y_test == 0, 0], self.X_test[self.y_test == 0, 1], c='r', marker="+", label='Class 0')
        # plt.xlabel('x(1)')
        # plt.ylabel('x(2)')
        # # plt.title('Iris Dataset (Setosa vs. Non-Setosa)')
        # plt.show()
    


    
    
    