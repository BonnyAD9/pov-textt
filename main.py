#!/usr/bin/python

import argparse

from src.train import train


def main():
    parser = argparse.ArgumentParser(
        prog="pov-textt", description="Text transcript using CRNN"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Choose a mode to run"
    )

    train_parser = subparsers.add_parser("train", help="Train the CRNN model")
    train_parser.add_argument(
        "-d",
        "--dataset",
        default="dataset",
        type=str,
        help="Path to the dataset directory",
    )
    train_parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "-b",
        "--batch",
        default=64,
        type=int,
        help="Batch size used for training",
    )
    train_parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help="Path to a pretrained model to resume training",
    )
    train_parser.add_argument(
        "-o",
        "--output",
        default="train",
        help="Directory to save the trained model into",
    )

    run_parser = subparsers.add_parser(
        "run", help="Runs the CRNN model on given image"
    )
    run_parser.add_argument(
        "-i", "--image", help="Image to transcribe the text from"
    )
    train_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to a pretrained model to resume training",
    )
    # TODO

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "run":
        pass


if __name__ == "__main__":
    main()
