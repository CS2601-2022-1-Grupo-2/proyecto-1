import os

class svm(object):

    def __init__(self, directory: str):
        for file in os.listdir(directory):
            if file == "Train":
                self.train_path: str = os.path.join(directory, file)
            elif file == "Test":
                self.test_path: str = os.path.join(directory, file)

    def train(self):
        for charact in os.listdir(self.train_path):
            for file in os.listdir(os.path.join(self.train_path, charact)):
                print(file)

    def test(self):
        for charact in os.listdir(self.train_path):
            for file in os.listdir(os.path.join(self.train_path, charact)):
                print(file)
