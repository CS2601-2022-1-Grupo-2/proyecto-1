import os
import numpy as np
import pandas as pd
from numpy import absolute
from .image import process_image
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split

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

        np.set_printoptions(precision=2)
        print(confusion_matrix(true_y, y_pred, normalize="true"))
        print(accuracy_score(true_y, y_pred))
    
    def cross(self):
      X = self[['x1', 'x2']]
      y = self['y']
      cv = KFold(n_splits=10,random_state=1,shuffle=True)
      model = LinearRegression()
      scores = cross_val_score(model, X, y,    scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
      valid= numpy.mean(absolute(scores))
      return valid


    def bootstrap(self):
      values=self.values
      n_iterations = 1000
      n_size = int(len(self) * 0.50)
      stats = list()
      for i in range(n_iterations):
	      train = resample(values, n_samples=n_size)
	      test = numpy.array([x for x in values if x.tolist() not in train.tolist()])

	      model = DecisionTreeClassifier()
	      model.fit(train[:,:-1], train[:,-1])

	      predictions = model.predict(test[:,:-1])
	      score = accuracy_score(test[:,-1], predictions)
	      print(score)
	      stats.append(score)

      pyplot.hist(stats)
      pyplot.show()

      alpha = 0.95
      p = ((1.0-alpha)/2.0) * 100
      lower = max(0.0, numpy.percentile(stats, p))
      p = (alpha+((1.0-alpha)/2.0)) * 100
      upper = min(1.0, numpy.percentile(stats, p))
      print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


    def biasvar(self):
      data = self.values
      X, y = data[:, :-1], data[:, -1]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
      model = LinearRegression()
      mse, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=1)
      print('MSE: %.3f' % mse)
      print('Bias: %.3f' % bias)
      print('Variance: %.3f' % var)
