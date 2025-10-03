from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from models.model_runner import ModelRunner
from sklearn.metrics import accuracy_score
import time


class AdaBoost(ModelRunner):
    def run(self):
        start = time.time()
        base = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
        self.clf = AdaBoostClassifier(estimator=base, n_estimators=50)
        self.clf.fit(self.X_train, self.y_train)
        self.accuracy = accuracy_score(self.y_test, self.clf.predict(self.X_test))
        self.runtime = round(time.time() - start, 4)
        print(f"🚀 AdaBoost Accuracy on {self.name}: {self.accuracy:.4f} | Time: {self.runtime} s")
