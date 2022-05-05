import os

from .image import get_vectors
from sklearn.neighbors import NearestNeighbors

class SvmKnn(object):

    def __init__(self, directory: str, method: str):
        self.method = method

        for file in os.scandir(directory):
            if file.name == "Train":
                self.train_path: str = file.path
            elif file.name == "Test":
                self.test_path: str = file.path

    def train(self):
        X, self.y = get_vectors(self.train_path, True)

        if self.method == "knn":
            self.nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        elif self.method == "svm":
            print("TODO")

    def test(self):
        X, _ = get_vectors(self.test_path, False)

        if self.method == "knn":
            _, indices = self.nbrs.kneighbors(X)
            print([[self.y[i] for i in ii] for ii in indices])
        elif self.method == "svm":
            print("TODO")
