from sklearn.metrics import accuracy_score
from neo_ls_svm import NeoLSSVM
import time
from models.model_runner import ModelRunner


class LSSVM(ModelRunner):
    def run(self):
        start = time.time()

        self.clf = NeoLSSVM()
        self.clf.fit(self.X_train, self.y_train)
        y_pred = self.clf.predict(self.X_test)

        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.runtime = round(time.time() - start, 4)
        print(f"🧮 LS-SVM (neo) Accuracy on {self.name}: {self.accuracy:.4f} | Time: {self.runtime} s")