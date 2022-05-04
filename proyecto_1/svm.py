import numpy as np
import os

from .image import get_vectors

class Svm(object):

    def __init__(self, directory: str):
        for file in os.scandir(directory):
            if file.name == "Train":
                self.train_path: str = file.path
            elif file.name == "Test":
                self.test_path: str = file.path

    def train(self):
        print(get_vectors(self.train_path, True))

    def test(self):
        print(get_vectors(self.test_path, False))
