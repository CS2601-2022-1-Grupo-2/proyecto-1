import os

import numpy as np
import pandas as pd

from .image import process_image
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

class SvmKnn(object):

    def get_vectors(self, csv_paths: list[str], fit: bool = False):
        v = []
        y = []

        for csv_path in csv_paths:
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

        self.X, self.y = self.get_vectors([self.train_path, self.test_path], True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=self.seed)

    def knn_query(self, indices):
        return [Counter([self.y_train[i] for i in ii]).most_common(1)[0][0] for ii in indices]

    def train(self):
        if self.method == "knn":
            self.nbrs = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(self.X_train)
        elif self.method == "svm":
            self.svc = SVC(
                kernel=self.kernel,
                probability=True,
                random_state=self.seed).fit(self.X_train, self.y_train)

    def test(self):
        y_pred = []

        if self.method == "knn":
            _, indices = self.nbrs.kneighbors(self.X_test)
            y_pred = self.knn_query(indices)
        elif self.method == "svm":
            y_pred = self.svc.predict(self.X_test)

        np.set_printoptions(precision=2)
        print(confusion_matrix(self.y_test, y_pred, normalize="true"))
        print(zero_one_loss(self.y_test, y_pred))

        if self.method == "svm":
            print(cross_val_score(self.svc, self.X, self.y))
