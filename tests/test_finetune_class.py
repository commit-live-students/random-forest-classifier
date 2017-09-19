from unittest import TestCase
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TestFinetune_class(TestCase):
    def test_finetune_class(self):
        from build import finetune_class
        dataset = loadtxt('./data/diabetes.csv', delimiter=',', skiprows=1)
        X = dataset[:, 0:8]
        y = dataset[:, 8]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)

        param_grid = {"criterion": ['gini', 'entropy'],
                      "n_estimators": [10, 20, 30],
                      "max_depth": [None, 6, 8, 10],
                      "max_leaf_nodes": [None, 5, 10, 20],
                      "min_impurity_split": [0.1, 0.2, 0.3]}

        y_pred, _ = finetune_class(X_train, X_test, y_train, param_grid)
        acc = accuracy_score(y_pred, y_test)

        self.assertGreaterEqual(acc, 0.7)
