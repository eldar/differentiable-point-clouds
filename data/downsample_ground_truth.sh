#!/bin/sh

python ../dpc/densify/downsample_gt.py \
--inp_dir=gt/dense \
--out_dir=gt/downsampled \
--synth_set=$1
