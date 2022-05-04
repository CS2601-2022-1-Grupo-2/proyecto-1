import argparse

from .svm import SvmAndKnn

def main():
    parser = argparse.ArgumentParser(description="Transit sign detector")

    parser.add_argument(
        "directory",
        metavar = "D",
        type    = str,
        help    = "Path of the directory with the train and test data"
    )

    args = parser.parse_args()

    svm = SvmAndKnn(args.directory)
    svm.train()
    svm.test()
