import os

from .image import get_vectors
from sklearn.neighbors import NearestNeighbors

class SvmAndKnn(object):

    def __init__(self, directory: str):
        for file in os.scandir(directory):
            if file.name == "Train":
                self.train_path: str = file.path
            elif file.name == "Test":
                self.test_path: str = file.path

    def train(self):
        X, self.y = get_vectors(self.train_path, True)
        self.nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    def test(self):
        X, _ = get_vectors(self.test_path, False)
        _, indices = self.nbrs.kneighbors(X)
        print([[self.y[i] for i in ii] for ii in indices])
