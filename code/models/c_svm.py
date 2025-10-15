from models.model_runner import ModelRunner
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time


class CSVM(ModelRunner):
    # def run(self):
    #     start = time.time()
    #     self.clf = SVC(kernel = 'rbf', C = 100)
    #     self.clf.fit(self.X_train, self.y_train)
    #     y_pred = self.clf.predict(self.X_test)
    #     self.accuracy = accuracy_score(self.y_test, y_pred)
    #     self.runtime = round(time.time() - start, 4)
    #     print(f"🎯 C-SVM (RBF) Accuracy on {self.name}: {self.accuracy:.4f} | Time: {self.runtime}s")
    def run(self):
        start = time.time()

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.01, 0.1, 1, 'scale']
        }

        grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=10)
        grid.fit(self.X_train, self.y_train)

        self.clf = grid.best_estimator_
        y_pred = self.clf.predict(self.X_test)

        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.runtime = round(time.time() - start, 4)
        self.best_params = grid.best_params_

        print(f"🎯 C-SVM (RBF) Accuracy on {self.name}: {self.accuracy:.4f} | Time: {self.runtime}s")
        print("Best Params:", self.best_params)