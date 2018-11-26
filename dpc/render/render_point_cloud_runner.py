import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("{}/..".format(script_dir))

import tensorflow as tf

from util.common import build_command_line_args, parse_lines
from util.app_config import config as app_config
from util.simple_dataset import Dataset3D
from util.fs import mkdir_if_missing

blender_exec = f'{script_dir}/../../external/blender/blender'
python_script = f'{script_dir}/render_point_cloud_blender.py'


def main(_):
    cfg = app_config

    exp_dir = cfg.checkpoint_dir
    out_dir = os.path.join(exp_dir, 'render')
    mkdir_if_missing(out_dir)
    inp_dir = os.path.join(exp_dir, cfg.save_predictions_dir)

    if cfg.models_list:
        models = parse_lines(cfg.models_list)
    else:
        dataset = Dataset3D(cfg)
        models = [sample.name for sample in dataset.data]

    for model_name in models:
        in_file = "{}/{}_pc.mat".format(inp_dir, model_name)
        if not os.path.isfile(in_file):
            in_file = "{}/{}_pc.npz".format(inp_dir, model_name)
            assert os.path.isfile(in_file), "no input file with saved point cloud"

        out_file = "{}/{}.png".format(out_dir, model_name)

        if os.path.isfile(out_file):
            print("{} already rendered".format(model_name))
            continue

        args = build_command_line_args([["in_file", in_file],
                                        ["out_file", out_file],
                                        ["vis_azimuth", cfg.vis_azimuth],
                                        ["vis_elevation", cfg.vis_elevation],
                                        ["vis_dist", cfg.vis_dist],
                                        ["cycles_samples", cfg.render_cycles_samples],
                                        ["voxels", False],
                                        ["colored_subsets", cfg.render_colored_subsets],
                                        ["image_size", cfg.render_image_size]]
                                       )
        render_cmd = "{} --background  -P {} -- {}".format(blender_exec, python_script, args)

        os.system(render_cmd)


if __name__ == '__main__':
    tf.app.run()
