import os
from .image import process_image

class Svm(object):

    def __init__(self, directory: str):
        for file in os.scandir(directory):
            if file.name == "Train":
                self.train_path: str = file.path
            elif file.name == "Test":
                self.test_path: str = file.path

    def train(self):
        for characteristic in os.scandir(self.train_path):
            for file in os.scandir(characteristic):
                print(process_image(file.path))

    def test(self):
        for characteristic in os.scandir(self.train_path):
            for file in os.scandir(characteristic):
                print(process_image(file.path))
