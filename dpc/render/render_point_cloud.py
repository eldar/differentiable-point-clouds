import os
import tempfile
import subprocess

from easydict import EasyDict as edict

import numpy as np
import imageio

from util.common import build_command_line_args


script_dir = os.path.dirname(os.path.realpath(__file__))

blender_exec = f'{script_dir}/../../external/blender/blender'
python_script = f'{script_dir}/render_point_cloud_blender.py'.format(script_dir)


def render_point_cloud(point_cloud, cfg):
    """
    Wraps the call to blender to render the image
    """
    cfg = edict(cfg)
    temp_dir = tempfile._get_default_tempdir()

    temp_name = next(tempfile._get_candidate_names())
    in_file = f"{temp_dir}/{temp_name}.npz"
    point_cloud_save = np.reshape(point_cloud, (1, -1, 3))
    np.savez(in_file, point_cloud_save)

    temp_name = next(tempfile._get_candidate_names())
    out_file = f"{temp_dir}/{temp_name}.png"

    args = build_command_line_args([["in_file", in_file],
                                    ["out_file", out_file],
                                    ["vis_azimuth", cfg.vis_azimuth],
                                    ["vis_elevation", cfg.vis_elevation],
                                    ["vis_dist", cfg.vis_dist],
                                    ["cycles_samples", cfg.render_cycles_samples],
                                    ["like_train_data", True],
                                    ["voxels", False],
                                    ["colored_subsets", False],
                                    ["image_size", cfg.render_image_size]],
                                   as_string=False)

    full_args = [blender_exec, "--background", "-P", python_script, "--"] + args
    subprocess.check_call(full_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    image = imageio.imread(out_file)
    os.remove(in_file)
    os.remove(out_file)

    return image