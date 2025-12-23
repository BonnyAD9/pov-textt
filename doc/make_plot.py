#!/usr/bin/python

import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Create plot from CSV.")
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-s", "--scale", type=str)
    
    args = parser.parse_args()
    data = pd.read_csv(args.file)
    
    plt.plot(data[data.columns[0]], data[data.columns[1]])
    if args.scale is not None:
        plt.yscale(args.scale)
    plt.xlabel(str(data.columns[0]))
    plt.ylabel(str(data.columns[1]))
    plt.savefig(args.output)

if __name__ == "__main__":
    main()
