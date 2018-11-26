import startup

import os

import numpy as np
import scipy.io
import tensorflow as tf

from util.simple_dataset import Dataset3D
from util.app_config import config as app_config
from util.quaternion import quaternion_multiply, quaternion_conjugate
from util.camera import quaternion_from_campos


def run_eval():
    config = tf.ConfigProto(
        device_count={'GPU': 1}
    )

    cfg = app_config
    exp_dir = cfg.checkpoint_dir
    num_views = cfg.num_views

    g = tf.Graph()
    with g.as_default():
        quat_inp = tf.placeholder(dtype=tf.float64, shape=[1, 4])
        quat_inp_2 = tf.placeholder(dtype=tf.float64, shape=[1, 4])

        quat_conj = quaternion_conjugate(quat_inp)
        quat_mul = quaternion_multiply(quat_inp, quat_inp_2)

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    save_pred_name = "{}_{}".format(cfg.save_predictions_dir, cfg.eval_split)
    save_dir = os.path.join(exp_dir, cfg.save_predictions_dir)

    reference_rotation = scipy.io.loadmat("{}/final_reference_rotation.mat".format(exp_dir))["rotation"]
    ref_conj_np = sess.run(quat_conj, feed_dict={quat_inp: reference_rotation})

    dataset = Dataset3D(cfg)

    num_models = dataset.num_samples()

    model_names = []

    angle_error = np.zeros((num_models, num_views), dtype=np.float64)

    for model_idx in range(num_models):
        sample = dataset.data[model_idx]

        print("{}/{}".format(model_idx, num_models))
        print(sample.name)
        model_names.append(sample.name)

        mat_filename = "{}/{}_pc.mat".format(save_dir, sample.name)
        data = scipy.io.loadmat(mat_filename)
        all_cameras = data["camera_pose"]

        for view_idx in range(num_views):
            cam_pos = sample.cam_pos[view_idx, :]
            gt_quat_np = quaternion_from_campos(cam_pos)
            gt_quat_np = np.expand_dims(gt_quat_np, 0)
            pred_quat_np = all_cameras[view_idx, :]
            pred_quat_np /= np.linalg.norm(pred_quat_np)
            pred_quat_np = np.expand_dims(pred_quat_np, 0)

            pred_quat_aligned_np = sess.run(quat_mul, feed_dict={
                quat_inp: pred_quat_np,
                quat_inp_2: ref_conj_np
            })

            q1 = gt_quat_np
            q2 = pred_quat_aligned_np

            q1_conj = sess.run(quat_conj, feed_dict={quat_inp: q1})
            q_diff = sess.run(quat_mul, feed_dict={quat_inp: q1_conj, quat_inp_2: q2})

            ang_diff = 2 * np.arccos(q_diff[0, 0])
            if ang_diff > np.pi:
                ang_diff -= 2*np.pi

            angle_error[model_idx, view_idx] = np.fabs(ang_diff)

    all_errors = np.reshape(angle_error, (-1))
    angle_thresh_rad = cfg.pose_accuracy_threshold / 180.0 * np.pi
    correct = all_errors < angle_thresh_rad
    num_predictions = correct.shape[0]
    accuracy = np.count_nonzero(correct) / num_predictions
    median_error = np.sort(all_errors)[num_predictions // 2]
    median_error = median_error / np.pi * 180
    print("accuracy:", accuracy, "median angular error:", median_error)

    scipy.io.savemat(os.path.join(exp_dir, "pose_error_{}.mat".format(save_pred_name)),
                     {"angle_error": angle_error,
                      "accuracy": accuracy,
                      "median_error": median_error})

    f = open(os.path.join(exp_dir, "pose_error_{}.txt".format(save_pred_name)), "w")
    f.write("{} {}\n".format(accuracy, median_error))
    f.close()


def main(_):
    run_eval()


if __name__ == '__main__':
    tf.app.run()
