import startup

import sys
import os
import glob
import argparse

import numpy as np
import scipy.io

from util.fs import mkdir_if_missing

import open3d


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--synth_set", type=str, default="03001627")
    parser.add_argument("--downsample_voxel_size", type=float, default=0.01)
    return parser.parse_args(sys.argv[1:])


def downsample_point_clouds():
    cfg = parse_arguments()

    vox_size = cfg.downsample_voxel_size
    synth_set = cfg.synth_set

    inp_dir = os.path.join(cfg.inp_dir, synth_set)
    files = glob.glob('{}/*.mat'.format(inp_dir))

    out_dir = cfg.out_dir
    out_synthset = os.path.join(out_dir, cfg.synth_set)
    mkdir_if_missing(out_synthset)

    for k, model_file in enumerate(files):
        print("{}/{}".format(k, len(files)))

        file_name = os.path.basename(model_file)
        sample_name, _ = os.path.splitext(file_name)

        obj = scipy.io.loadmat(model_file)

        out_filename = "{}/{}.mat".format(out_synthset, sample_name)
        if os.path.isfile(out_filename):
            print("already exists:", sample_name)
            continue

        Vgt = obj["points"]

        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(Vgt)
        downpcd = open3d.voxel_down_sample(pcd, voxel_size=vox_size)
        down_xyz = np.asarray(downpcd.points)
        scipy.io.savemat(out_filename, {"points": down_xyz})


if __name__ == '__main__':
    downsample_point_clouds()
