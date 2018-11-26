#!/usr/bin/env python

import startup

import os

import tensorflow as tf

from util.app_config import config as app_config

from run.predict import compute_predictions as compute_predictions_pc
from run.eval_chamfer import run_eval
from run.eval_camera_pose import run_eval as run_camera_pose_eval


def compute_eval():
    cfg = app_config

    compute_predictions_pc()

    if cfg.predict_pose and cfg.eval_split == "val":
        import subprocess
        import sys
        # need to use subprocess, because optimal_alignment uses eager execution
        # and it cannot be mixed with the graph mode within the same process
        script_dir = os.path.dirname(os.path.realpath(__file__))
        args = " ".join(sys.argv[1:])
        cmd = f"python {script_dir}/compute_alignment.py {args}"
        subprocess.call(cmd, shell=True)

    run_eval()

    if cfg.predict_pose:
        run_camera_pose_eval()


def main(_):
    compute_eval()


if __name__ == '__main__':
    tf.app.run()
