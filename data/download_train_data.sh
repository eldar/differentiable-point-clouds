#!/bin/sh

FILE=$1-renders.tar.gz

curl -L -O https://datasets.d2.mpi-inf.mpg.de/unsupervised-shape-pose/$FILE
mkdir -p renders

echo "unpacking" $FILE
tar -xzf $1-renders.tar.gz -C renders
