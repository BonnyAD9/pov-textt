#/usr/bin/bash

set -e

typst c doc/doc.typ
cp doc/doc.pdf doc.pdf
zip team-xsleza26.zip -r main.py src doc.pdf README.md

rm doc.pdf
