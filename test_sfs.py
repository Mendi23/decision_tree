import pandas as pd
import numpy as np

from unittest import TestCase
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from operator import itemgetter
from sklearn.naive_bayes import MultinomialNB

from sfs import sfs


class TestSfs(TestCase):
    def real_sfs(self, x, y, k, clf, score):
        raise NotImplementedError()

    def sfs_wrapper(self, x, y, k, clf, score):
        return sfs(clf=clf, k=k, score=score, x=x.values.tolist(), y=y.tolist())

    def cross_score(self, clf, x, y):
        return np.average(cross_val_score(clf, x, y, cv=4))

    def setUp(self):
        def create_array(cat_num, size, error_rate):
            data = np.array(np.concatenate([np.array([i]*int(size/cat_num)) for i in range(cat_num)]))
            for i, error in filter(itemgetter(1), enumerate(np.random.binomial(1, error_rate/100.0, size))):
                data[i] = cat_num - data[i] - 1
            return data

        self.size = 9600
        arrays = [[create_array(i, self.size, 30 + (6-i)*6)] for i in range(6, 1, -1)] + [[create_array(i, self.size, 66 + (6-i)*6)] for i in range(6, 1, -1)]
        self.X = np.concatenate(arrays).T

        self.y = np.array([0]*int(self.size/2) + [1]*int(self.size/2))
        self.X = pd.DataFrame(self.X).sample(frac=1)
        self.simple_clf = DecisionTreeClassifier()

    def test_sfs_one_of_one(self):
        col = list(self.X.columns)[0]
        self.assertEqual([col], self.sfs_wrapper(clf=self.simple_clf, k=1, x=self.X[[col]], y=self.y, score=self.cross_score))

    def test_sfs_one_of_two(self):
        col1 = list(self.X.columns)[0]
        col2 = list(self.X.columns)[1]
        x = self.X[[col1, col2]]
        selected = self.sfs_wrapper(clf=self.simple_clf, k=1, x=x, y=self.y, score=self.cross_score)[0]
        self.assertGreaterEqual(self.cross_score(self.simple_clf, self.X[[selected]], self.y),
                                self.cross_score(self.simple_clf, self.X[[col1]], self.y))
        self.assertGreaterEqual(self.cross_score(self.simple_clf, self.X[[selected]], self.y),
                                self.cross_score(self.simple_clf, self.X[[col2]], self.y))

    def test_sfs_five_of_ten(self):
        self.assertSetEqual(set(self.real_sfs(self.X, self.y, 5, self.simple_clf, self.cross_score)),
                             set(self.sfs_wrapper(self.X, self.y, 5, self.simple_clf, self.cross_score)))

    def test_sfs_different_classifier(self):
        self.simple_clf = MultinomialNB()
        self.test_sfs_five_of_ten()

    def test_sfs_different_score(self):
        self.cross_score = lambda clf, x, y: np.average(cross_val_score(clf, x, y, cv=4, scoring='f1_micro'))
        self.test_sfs_five_of_ten()

    def test_sfs_contain_lowers(self):
        known = set()
        for i in range(1, self.X.shape[1]+1):
            results = set(self.sfs_wrapper(self.X, self.y, i, self.simple_clf, self.cross_score))
            self.assertTrue(results.issuperset(known))
            known = results
