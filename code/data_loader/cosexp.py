import numpy as np
from data_loader.data_loader import DataLoader


class CosExp(DataLoader):
    def load(self):
        self.name = "Cosexp"
        # n_samples = self.train_size + self.test_size
        # x1 = np.random.uniform(-2, 2, n_samples)
        # x2 = np.cos(x1) + np.exp(-x1**2) + np.random.normal(0, 0.1, n_samples)
        # self.X = np.vstack((x1, x2)).T
        # self.y = (x2 > np.cos(x1)).astype(int)
        maxx = 5
        freq = 0.7
        self.n_test = int(np.floor(np.sqrt(self.test_size)))

        # Tạo dữ liệu huấn luyện
        xi = np.random.rand(self.train_size, 2)
        x_train = np.column_stack((maxx * xi[:, 0], 2 * xi[:, 1] - 1))
        y_train = (np.cos(0.5 * np.exp(freq * x_train[:, 0])) > x_train[:, 1]).astype(int)

        # Tạo dữ liệu kiểm tra dạng lưới
        x1_test, x2_test = np.meshgrid(np.linspace(0, maxx, self.n_test),
                                       np.linspace(-1, 1, self.n_test))
        x_test = np.column_stack((x1_test.ravel(), x2_test.ravel()))
        y_test = (np.cos(0.5 * np.exp(freq * x_test[:, 0])) > x_test[:, 1]).astype(int)

        self.X_train, self.y_train = x_train, y_train
        self.X_test, self.y_test = x_test, y_test
        self.X = np.vstack((x_train, x_test))
        self.y = np.concatenate((y_train, y_test))
        