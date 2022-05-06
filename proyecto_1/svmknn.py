import os

import numpy as np
import pandas as pd

from .image import process_image
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

class SvmKnn(object):

    def __init__(self, directory: str, method: str, k: int, kernel: str, seed: int):
        self.directory = directory
        self.method    = method
        self.k         = k
        self.kernel    = kernel
        self.seed      = seed
        self.scaler    = preprocessing.MinMaxScaler()


        for file in os.scandir(directory):
            if file.name == "Train.csv":
                self.train_path: str = file.path
            elif file.name == "Test.csv":
                self.test_path: str = file.path


    def get_vectors(self, csv_path: str, fit: bool = False):
        v = []
        y = []

        df = pd.read_csv(csv_path)

        if isinstance(df, pd.DataFrame):
            for i in range(len(df)):
                v.append(process_image(os.path.join(self.directory, df.at[i, "Path"])))
                y.append(df.at[i, "ClassId"])

        if fit:
            v = self.scaler.fit_transform(np.array(v))
        else:
            v = self.scaler.transform(np.array(v))

        return (v, y)

    def knn_query(self, indices):
        return [Counter([self.y[i] for i in ii]).most_common(1)[0][0] for ii in indices]

    def train(self):
        X, self.y = self.get_vectors(self.train_path, True)

        if self.method == "knn":
            self.nbrs = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(X)
        elif self.method == "svm":
            self.svc = SVC(
                kernel=self.kernel,
                probability=True,
                random_state=self.seed).fit(X, self.y)

    def test(self):
        X, true_y = self.get_vectors(self.test_path)
        y_pred = []

        if self.method == "knn":
            _, indices = self.nbrs.kneighbors(X)
            y_pred = self.knn_query(indices)
        elif self.method == "svm":
            y_pred = self.svc.predict(X)

        print(confusion_matrix(true_y, y_pred))
