import os

import numpy as np
import pandas as pd

from .image import process_image
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from statistics import mean

class SvmKnn(object):

    def get_vectors(self, csv_paths, fit, split):
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

        y = np.array(y)

        if split != 1.0:
            _, v, _, y = train_test_split(v,y, test_size=split)

        return (v, y)

    def __init__(self, directory, method, k, kernel, seed, split):
        np.set_printoptions(precision=2)

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

        self.X, self.y = self.get_vectors([self.train_path, self.test_path], True, split)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=self.seed)

        if self.method == "knn":
            self.model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        elif self.method == "svm":
            self.model = SVC(kernel=kernel, probability=True, random_state=seed)

    def ebv(self, X, y):
        error = 0
        bias = 0
        variance = 0
        errors = []

        for i, p in enumerate(self.predictor.predict_proba(X)):
            errors.append(1.0 - p[y[i]])

        error = mean(errors)
        bias = mean([0.0 if e <= 0.5 else 1.0 for e in errors])
        variance = error - bias

        return error, bias, variance

    def train(self):
        self.predictor = self.model.fit(self.X_train, self.y_train)

    def test(self):
        y_pred = self.predictor.predict(self.X_test)

        print(confusion_matrix(self.y_test, y_pred, normalize="true"))
        #print(zero_one_loss(self.y_test, y_pred))

        error, bias, variance = self.ebv(self.X, self.y)
        print(f"Error: {error}")
        print(f"Bias: {bias}")
        print(f"Variance: {variance}")
