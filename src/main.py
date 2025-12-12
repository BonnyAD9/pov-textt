#!/usr/bin/python

import argparse


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


if __name__ == "__main__":
    main()
