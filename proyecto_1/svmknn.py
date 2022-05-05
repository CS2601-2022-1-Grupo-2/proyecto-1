import os

from .image import get_vectors
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class SvmKnn(object):

    def __init__(self, directory: str, method: str, k: int):
        self.method = method
        self.k      = k

        for file in os.scandir(directory):
            if file.name == "Train":
                self.train_path: str = file.path
            elif file.name == "Test":
                self.test_path: str = file.path

    def knnQuery(self, indices):
        return [Counter([self.y[i] for i in ii]).most_common(1)[0][0] for ii in indices]

    def train(self):
        X, self.y = get_vectors(self.train_path, True)

        if self.method == "knn":
            self.nbrs = NearestNeighbors(n_neighbors=self.k).fit(X)
        elif self.method == "svm":
            print("TODO")

    def test(self):
        X, _ = get_vectors(self.test_path, False)

        if self.method == "knn":
            _, indices = self.nbrs.kneighbors(X)
            print(self.knnQuery(indices))
        elif self.method == "svm":
            print("TODO")
