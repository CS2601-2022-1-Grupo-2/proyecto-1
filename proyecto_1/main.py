import argparse
import time

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

    parser.add_argument(
        "-K",
        "--kernel",
        type    = str,
        help    = "The kernel used by the SVM",
        choices = ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        default = "linear"
    )

    parser.add_argument(
        "-s",
        "--seed",
        type    = int,
        help    = "The seed of the  pseudo random number generator.",
        default = time.time_ns() % 2**32
    )

    args = parser.parse_args()

    from .svmknn import SvmKnn

    svm = SvmKnn(args.directory, args.method, args.neighbors, args.kernel, args.seed)
    svm.train()
    svm.test()
