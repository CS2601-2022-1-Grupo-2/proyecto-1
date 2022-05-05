import os

import numpy as np
import pandas as pd

from .image import process_image
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class SvmKnn(object):

    def __init__(self, directory: str, method: str, k: int):
        self.directory = directory
        self.method    = method
        self.k         = k

        for file in os.scandir(directory):
            if file.name == "Train.csv":
                self.train_path: str = file.path
            elif file.name == "Test.csv":
                self.test_path: str = file.path


    def get_vectors(self, csv_path: str):
        v = []
        y = []

        df = pd.read_csv(csv_path)

        if isinstance(df, pd.DataFrame):
            for i in range(len(df)):
                v.append(process_image(os.path.join(self.directory, df.at[i, "Path"])))
                y.append(df.at[i, "ClassId"])

        return (np.array(v), y)

    def knnQuery(self, indices):
        return [Counter([self.y[i] for i in ii]).most_common(1)[0][0] for ii in indices]

    def train(self):
        X, self.y = self.get_vectors(self.train_path)

        if self.method == "knn":
            self.nbrs = NearestNeighbors(n_neighbors=self.k).fit(X)
        elif self.method == "svm":
            print("TODO")

    def test(self):
        X, true_y = self.get_vectors(self.test_path)

        if self.method == "knn":
            _, indices = self.nbrs.kneighbors(X)
            print(self.knnQuery(indices))
        elif self.method == "svm":
            print("TODO")
