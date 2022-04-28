import argparse

def main():
    parser = argparse.ArgumentParser(description="Transit sign detector")

    parser.add_argument(
        "directories",
        metavar = "D",
        type    = str,
        nargs   = "+",
        help    = "Path of the directory with the train and test data"
    )

    args = parser.parse_args()

    for path in args.directories:
        print(path)
