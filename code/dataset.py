import numpy as np


class DatasetGenerator:
    def __init__(self, n_train=500, n_test=500, sigma=0.1):
        self.n_train = n_train
        self.n_test = n_test
        self.sigma = sigma
        # self.x_train = []
        # self.y_train = []
        # self.x_test = []
        # self.y_test = []

    def generate(self, dataset_name="cosexp"):
        default = "cosexp"
        return getattr(self, dataset_name, lambda: default)()

    def cosexp(self):
        maxx = 5
        freq = 0.7
        self.n_test = int(np.floor(np.sqrt(self.n_test)))
        xi = np.random.rand(self.n_train, 2)
        y_train = np.sign(np.cos(0.5 * (np.exp(freq * maxx * xi[:, 0]))) - (2 * xi[:, 1] - 1))
        x_train = np.column_stack((maxx * xi[:, 0], (2 * xi[:, 1] - 1)))
        x_test1, x_test2 = np.meshgrid(np.linspace(0, maxx, self.n_test), np.linspace(-1, 1, self.n_test))
        x_test = np.column_stack((x_test1.ravel(), x_test2.ravel()))
        y_test = np.sign(np.cos(0.5 * (np.exp(freq * maxx * x_test[:, 0]))) - (2 * x_test[:, 1] - 1))
        return x_train, y_train, x_test, y_test
                
    def gaussian(self):
        pass
    
    
# maxx = 5
# freq = 0.7
# nbtest = int(np.floor(np.sqrt(nbtest)))
# xi = np.random.rand(nbapp, 2)
# yapp = np.sign(np.cos(0.5 * (np.exp(freq * maxx * xi[:, 0]))) - (2 * xi[:, 1] - 1))
# xapp = np.column_stack((maxx * xi[:, 0], (2 * xi[:, 1] - 1)))
# xtest1, xtest2 = np.meshgrid(np.linspace(0, maxx, nbtest), np.linspace(-1, 1, nbtest))
# nn = xtest1.size
# xtest = np.column_stack((xtest1.ravel(), xtest2.ravel()))
# ytest = np.sign(np.cos(0.5 * (np.exp(freq * maxx * xtest[:, 0]))) - (2 * xtest[:, 1] - 1))