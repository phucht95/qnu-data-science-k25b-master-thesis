import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from dataset import DatasetGenerator
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time

# mat_contents = sio.loadmat('data/cosexp.mat')
# print(mat_contents.keys())
# x_train = mat_contents['xapp']
# y_train = mat_contents['yapp']
# x_test = mat_contents['xtest']
# y_test = mat_contents['ytest']

# print(x_train, type(x_train), x_train.shape)
dataset_generator = DatasetGenerator()
x_train, y_train, x_test, y_test = dataset_generator.generate()
x_train[:, 0] = (x_train[:, 0] - x_train[:, 0].min())/ (x_train[:, 0].max() - x_train[:, 0].min())
x_train[:, 1] = (x_train[:, 1] - x_train[:, 1].min())/ (x_train[:, 1].max() - x_train[:, 1].min())
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr')
# plt.scatter(x_train[:, 0], x_train[:, 1], color="green", marker="*", cmap='bwr')
# plt.scatter(x_train[:, 0], x_train[:, 1], color="red", marker="+", cmap='bwr')
plt.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], c='g', marker="*", label='Class 1')
plt.scatter(x_train[y_train==-1, 0], x_train[y_train==-1, 1], c='r', marker="+", label='Class -1')
plt.xlabel('x(1)')
plt.ylabel('x(2)')
# plt.title('Iris Dataset (Setosa vs. Non-Setosa)')
plt.show()

start = time.time()
clf = SVC(kernel = 'rbf', C = 100)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
runtime = round(time.time() - start, 4)
print(f"C-SVM (RBF) Accuracy on CosExp: {accuracy:.4f} | Time: {runtime}s")
