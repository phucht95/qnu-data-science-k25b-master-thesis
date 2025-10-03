from sklearn.preprocessing import StandardScaler
import numpy as np


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
    


    
    
    