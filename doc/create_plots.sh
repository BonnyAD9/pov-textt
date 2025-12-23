#!/usr/bin/sh

set -e

./make_plot.py -f <(awk '
    BEGIN {
        printf "epoch,loss\n"
    }
    match($0, /epoch ([^ ]*) loss: ([^ ]*)/, r) {
        printf r[1] "," r[2] "\n"
    }
' $1) -o assets/loss-plot.png -s log

./make_plot.py -f <(awk '
    BEGIN {
        printf "epoch,accuracy\n"
    }
    match($0, /epoch ([^ ]*).*acc: ([^ ]*)/, r) {
        printf r[1] "," r[2] "\n"
    }
' $1) -o assets/acc-plot.png