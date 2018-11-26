#!/bin/sh

FILE=$1-gt.tar.gz

curl -L -O https://datasets.d2.mpi-inf.mpg.de/unsupervised-shape-pose/$FILE
mkdir -p gt/downsampled

echo "unpacking" $FILE
tar -xzf $FILE -C gt/downsampled
