#!/usr/bin/python

import argparse

import torch

from src.crnn import CRNN
from src.dataset import Dataset


def main():
    parser = argparse.ArgumentParser(
        prog="pov-textt", description="Text transcript using CRNN"
    )
    _ = parser.add_argument(
        "-d",
        "--dataset",
        default="dataset",
        type=str,
        help="Path to the dataset directory",
    )
    args = parser.parse_args()

    data = Dataset.parse_datasets(args.dataset)

    classes = Dataset.join_classes(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(classes)
    crnn = CRNN(72, len(classes), device).to(device)


if __name__ == "__main__":
    main()
