#!/usr/bin/python

import argparse
from pathlib import Path


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

    data = parse_dataset(args.dataset)


def parse_dataset(dataset):
    data_path = Path(dataset) / "data"
    if not data_path.exists():
        return []

    data = []
    for sub_dir in data_path.iterdir():
        txt_file = list(sub_dir.glob("*.txt"))
        if not txt_file:
            continue
        data.append({"dir": sub_dir, "data": parse_dataset_txt(txt_file[0])})

    return data


def parse_dataset_txt(path):
    data = []
    file = open(path, "r")
    for line in file:
        line = line.strip()
        if not line:
            continue

        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue

        data.append(
            {"image": parts[0], "num": int(parts[1]), "text": parts[2]}
        )

    file.close()
    return data


if __name__ == "__main__":
    main()
