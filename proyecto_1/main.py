import argparse

from .svmknn import SvmKnn

def main():
    parser = argparse.ArgumentParser(description="Transit sign detector")

    parser.add_argument(
        "directory",
        metavar = "D",
        type    = str,
        help    = "Path of the directory with the train and test data"
    )

    parser.add_argument(
        "-m",
        "--method",
        type    = str,
        help    = "svm or knn",
        choices = ["svm", "knn"],
        default = "knn"
    )

    parser.add_argument(
        "-k",
        "--neighbors",
        type    = int,
        help    = "K for knn",
        metavar = "K",
        default = 5
    )

    args = parser.parse_args()

    svm = SvmKnn(args.directory, args.method, args.neighbors)
    svm.train()
    svm.test()
