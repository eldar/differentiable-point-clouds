#!/usr/bin/env python

import startup

import os

import numpy as np
import scipy.io
import tensorflow as tf

from util.point_cloud import chamfer_distance
from util.simple_dataset import Dataset3D
from util.app_config import config as app_config
from util.tools import partition_range, to_np_object
from util.quaternion import quaternion_rotate


def compute_distance(cfg, sess, min_dist, idx, source, target, source_np, target_np):
    """
    compute projection from source to target
    """
    num_parts = cfg.pc_eval_chamfer_num_parts
    partition = partition_range(source_np.shape[0], num_parts)
    min_dist_np = np.zeros((0,))
    idx_np = np.zeros((0,))
    for k in range(num_parts):
        r = partition[k, :]
        src = source_np[r[0]:r[1]]
        (min_dist_0_np, idx_0_np) = sess.run([min_dist, idx],
                                             feed_dict={source: src,
                                                       target: target_np})
        min_dist_np = np.concatenate((min_dist_np, min_dist_0_np), axis=0)
        idx_np = np.concatenate((idx_np, idx_0_np), axis=0)
    return min_dist_np, idx_np


def run_eval():
    config = tf.ConfigProto(
        device_count={'GPU': 1}
    )

    cfg = app_config
    exp_dir = cfg.checkpoint_dir
    num_views = cfg.num_views
    eval_unsup = cfg.eval_unsupervised_shape

    gt_dir = os.path.join(cfg.gt_pc_dir, cfg.synth_set)

    g = tf.Graph()
    with g.as_default():
        source_pc = tf.placeholder(dtype=tf.float64, shape=[None, 3])
        target_pc = tf.placeholder(dtype=tf.float64, shape=[None, 3])
        quat_tf = tf.placeholder(dtype=tf.float64, shape=[1, 4])

        _, min_dist, min_idx = chamfer_distance(source_pc, target_pc)

        source_pc_2 = tf.placeholder(dtype=tf.float64, shape=[1, None, 3])
        rotated_pc = quaternion_rotate(source_pc_2, quat_tf)

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    save_pred_name = "{}_{}".format(cfg.save_predictions_dir, cfg.eval_split)
    save_dir = os.path.join(exp_dir, cfg.save_predictions_dir)

    if eval_unsup:
        reference_rotation = scipy.io.loadmat("{}/final_reference_rotation.mat".format(exp_dir))["rotation"]

    dataset = Dataset3D(cfg)

    num_models = dataset.num_samples()

    model_names = []
    chamfer_dists = np.zeros((0, num_views, 2), dtype=np.float64)
    for k in range(num_models):
        sample = dataset.data[k]

        print("{}/{}".format(k, num_models))
        print(sample.name)

        gt_filename = "{}/{}.mat".format(gt_dir, sample.name)
        if not os.path.isfile(gt_filename):
            continue

        model_names.append(sample.name)
        mat_filename = "{}/{}_pc.mat".format(save_dir, sample.name)
        if os.path.isfile(mat_filename):
            data = scipy.io.loadmat(mat_filename)
            all_pcs = np.squeeze(data["points"])
            if "num_points" in data:
                all_pcs_nums = np.squeeze(data["num_points"])
                has_number = True
            else:
                has_number = False
        else:
            data = np.load("{}/{}_pc.npz".format(save_dir, sample.name))
            all_pcs = np.squeeze(data["arr_0"])
            if 'arr_1' in list(data.keys()):
                all_pcs_nums = np.squeeze(data["arr_1"])
                has_number = True
            else:
                has_number = False

        obj = scipy.io.loadmat(gt_filename)
        Vgt = obj["points"]

        chamfer_dists_current = np.zeros((num_views, 2), dtype=np.float64)
        for i in range(num_views):
            pred = all_pcs[i, :, :]
            if has_number:
                pred = pred[0:all_pcs_nums[i], :]

            if eval_unsup:
                pred = np.expand_dims(pred, 0)
                pred = sess.run(rotated_pc, feed_dict={source_pc_2: pred,
                                                       quat_tf: reference_rotation})
                pred = np.squeeze(pred)

            pred_to_gt, idx_np = compute_distance(cfg, sess, min_dist, min_idx, source_pc, target_pc, pred, Vgt)
            gt_to_pred, _ = compute_distance(cfg, sess, min_dist, min_idx, source_pc, target_pc, Vgt, pred)
            chamfer_dists_current[i, 0] = np.mean(pred_to_gt)
            chamfer_dists_current[i, 1] = np.mean(gt_to_pred)

            is_nan = np.isnan(pred_to_gt)
            assert(not np.any(is_nan))

        current_mean = np.mean(chamfer_dists_current, 0)
        print("total:", current_mean)
        chamfer_dists = np.concatenate((chamfer_dists, np.expand_dims(chamfer_dists_current, 0)))

    final = np.mean(chamfer_dists, axis=(0, 1)) * 100
    print(final)

    scipy.io.savemat(os.path.join(exp_dir, "chamfer_{}.mat".format(save_pred_name)),
                     {"chamfer": chamfer_dists,
                      "model_names": to_np_object(model_names)})

    file = open(os.path.join(exp_dir, "chamfer_{}.txt".format(save_pred_name)), "w")
    file.write("{} {}\n".format(final[0], final[1]))
    file.close()


def main(_):
    run_eval()


if __name__ == '__main__':
    tf.app.run()
