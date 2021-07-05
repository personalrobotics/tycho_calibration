import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to data file to be split")
    parser.add_argument("train_path", type=str, help="Path where train data will be saved")
    parser.add_argument("test_path", type=str, help="Path where test data will be saved")
    parser.add_argument("train_portion", type=float,
                        help="In the range [0,1], determining what percentage of the data is used for training")
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.data_path)
    train, test = train_test_split(data, train_size=args.train_portion)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(args.train_path)
    test.to_csv(args.test_path)


if __name__ == "__main__":
    main()
