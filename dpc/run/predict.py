import startup

import os

import numpy as np
import imageio
import scipy.io

import tensorflow as tf
import tensorflow.contrib.slim as slim

from models import model_pc

from util.common import parse_lines
from util.app_config import config as app_config
from util.system import setup_environment
from util.simple_dataset import Dataset3D
from util.fs import mkdir_if_missing
from util.camera import get_full_camera, quaternion_from_campos
from util.visualise import vis_pc, merge_grid, mask4vis
from util.point_cloud import pointcloud2voxels, smoothen_voxels3d, pointcloud2voxels3d_fast, pointcloud_project_fast
from util.quaternion import as_rotation_matrix, quaternion_rotate


def build_model(model):
    cfg = model.cfg()
    batch_size = cfg.batch_size
    inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, cfg.image_size, cfg.image_size, 3])
    camera_extr_src = tf.placeholder(dtype=tf.float32, shape=[4, 4])
    cam_matrix = get_full_camera(cfg, camera_extr_src, inverted=False)
    cam_matrix = tf.reshape(cam_matrix, shape=[batch_size, 4, 4])
    cam_quaternion = tf.placeholder(dtype=tf.float32, shape=[batch_size, 4])

    model_fn = model.get_model_fn(is_training=False, reuse=False)
    code = 'images' if cfg.predict_pose else 'images_1'
    input = {code: inputs,
             'matrices': cam_matrix,
             'camera_quaternion': cam_quaternion}
    outputs = model_fn(input)
    cam_transform = outputs['poses'] if cfg.predict_pose else tf.no_op()
    outputs["inputs"] = inputs
    outputs["camera_extr_src"] = camera_extr_src
    outputs["cam_quaternion"] = cam_quaternion
    outputs["cam_transform"] = cam_transform
    return outputs


def model_student(inputs, model):
    cfg = model.cfg()
    outputs = model.model_predict(inputs, is_training=False,
                                  reuse=True, predict_for_all=False)
    points = outputs["points_1"]
    camera_pose = outputs["pose_student"]
    rgb = None
    transl = outputs["predicted_translation"] if cfg.predict_translation else None
    proj_out = pointcloud_project_fast(model.cfg(), points, camera_pose, transl,
                                       rgb, model.gauss_kernel())
    proj = proj_out["proj_depth"]

    return proj, camera_pose


def model_unrotate_points(cfg):
    """
    un_q = quat_gt^(-1) * predicted_quat
    pc_unrot = un_q * pc_np * un_q^(-1)
    """

    from util.quaternion import quaternion_normalise, quaternion_conjugate, \
        quaternion_rotate, quaternion_multiply
    input_pc = tf.placeholder(dtype=tf.float32, shape=[1, cfg.pc_num_points, 3])
    pred_quat = tf.placeholder(dtype=tf.float32, shape=[1, 4])
    gt_quat = tf.placeholder(dtype=tf.float32, shape=[1, 4])

    pred_quat_n = quaternion_normalise(pred_quat)
    gt_quat_n = quaternion_normalise(gt_quat)

    un_q = quaternion_multiply(quaternion_conjugate(gt_quat_n), pred_quat_n)
    pc_unrot = quaternion_rotate(input_pc, un_q)

    return input_pc, pred_quat, gt_quat, pc_unrot


def normalise_depthmap(depth_map):
    depth_map = np.clip(depth_map, 1.5, 2.5)
    depth_map -= 1.5
    return depth_map


def compute_predictions():
    cfg = app_config

    setup_environment(cfg)

    exp_dir = cfg.checkpoint_dir

    cfg.batch_size = 1
    cfg.step_size = 1

    pc_num_points = cfg.pc_num_points
    vox_size = cfg.vox_size
    save_pred = cfg.save_predictions
    save_voxels = cfg.save_voxels
    fast_conversion = True

    pose_student = cfg.pose_predictor_student and cfg.predict_pose

    g = tf.Graph()
    with g.as_default():
        model = model_pc.ModelPointCloud(cfg)

        out = build_model(model)
        input_image = out["inputs"]
        cam_matrix = out["camera_extr_src"]
        cam_quaternion = out["cam_quaternion"]
        point_cloud = out["points_1"]
        rgb = out["rgb_1"] if cfg.pc_rgb else tf.no_op()
        projs = out["projs"]
        projs_rgb = out["projs_rgb"]
        projs_depth = out["projs_depth"]
        cam_transform = out["cam_transform"]
        z_latent = out["z_latent"]

        if pose_student:
            proj_student, camera_pose_student = model_student(input_image, model)

        input_pc = tf.placeholder(tf.float32, [cfg.batch_size, None, 3])
        if save_voxels:
            if fast_conversion:
                voxels, _ = pointcloud2voxels3d_fast(cfg, input_pc, None)
                voxels = tf.expand_dims(voxels, axis=-1)
                voxels = smoothen_voxels3d(cfg, voxels, model.gauss_kernel())
            else:
                voxels = pointcloud2voxels(cfg, input_pc, model.gauss_sigma())

        q_inp = tf.placeholder(tf.float32, [1, 4])
        q_matrix = as_rotation_matrix(q_inp)

        input_pc, pred_quat, gt_quat, pc_unrot = model_unrotate_points(cfg)
        pc_rot = quaternion_rotate(input_pc, pred_quat)

        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )
        config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        variables_to_restore = slim.get_variables_to_restore(exclude=["meta"])

    restorer = tf.train.Saver(variables_to_restore)
    checkpoint_file = tf.train.latest_checkpoint(exp_dir)
    print("restoring checkpoint", checkpoint_file)
    restorer.restore(sess, checkpoint_file)

    save_dir = os.path.join(exp_dir, '{}_vis_proj'.format(cfg.save_predictions_dir))
    mkdir_if_missing(save_dir)
    save_pred_dir = os.path.join(exp_dir, cfg.save_predictions_dir)
    mkdir_if_missing(save_pred_dir)

    vis_size = cfg.vis_size

    dataset = Dataset3D(cfg)

    pose_num_candidates = cfg.pose_predict_num_candidates
    num_views = cfg.num_views
    plot_h = 4
    plot_w = 6
    num_views = int(min(num_views, plot_h * plot_w / 2))

    if cfg.models_list:
        model_names = parse_lines(cfg.models_list)
    else:
        model_names = [sample.name for sample in dataset.data]

    num_models = len(model_names)
    for k in range(num_models):
        model_name = model_names[k]
        sample = dataset.sample_by_name(model_name)

        images = sample.image
        masks = sample.mask
        if cfg.saved_camera:
            cameras = sample.camera
            cam_pos = sample.cam_pos
        if cfg.vis_depth_projs:
            depths = sample.depth
        if cfg.variable_num_views:
            num_views = sample.num_views

        print("{}/{} {}".format(k, num_models, model_name))

        if pose_num_candidates == 1:
            grid = np.empty((plot_h, plot_w), dtype=object)
        else:
            plot_w = pose_num_candidates + 1
            if pose_student:
                plot_w += 1
            grid = np.empty((num_views, plot_w), dtype=object)

        if save_pred:
            all_pcs = np.zeros((num_views, pc_num_points, 3))
            all_cameras = np.zeros((num_views, 4))
            all_voxels = np.zeros((num_views, vox_size, vox_size, vox_size))
            all_z_latent = np.zeros((num_views, cfg.fc_dim))

        for view_idx in range(num_views):
            input_image_np = images[[view_idx], :, :, :]
            gt_mask_np = masks[[view_idx], :, :, :]
            if cfg.saved_camera:
                extr_mtr = cameras[view_idx, :, :]
                cam_quaternion_np = quaternion_from_campos(cam_pos[view_idx, :])
                cam_quaternion_np = np.expand_dims(cam_quaternion_np, axis=0)
            else:
                extr_mtr = np.zeros((4, 4))

            if cfg.pc_rgb:
                proj_tensor = projs_rgb
            elif cfg.vis_depth_projs:
                proj_tensor = projs_depth
            else:
                proj_tensor = projs
            (pc_np, rgb_np, proj_np, cam_transf_np, z_latent_np) = sess.run([point_cloud, rgb, proj_tensor, cam_transform, z_latent],
                                                               feed_dict={input_image: input_image_np,
                                                                          cam_matrix: extr_mtr,
                                                                          cam_quaternion: cam_quaternion_np})

            if pose_student:
                (proj_student_np, camera_student_np) = sess.run([proj_student, camera_pose_student],
                                                                feed_dict={input_image: input_image_np})
                predicted_camera = camera_student_np
            else:
                predicted_camera = cam_transf_np

            if cfg.vis_depth_projs:
                proj_np = normalise_depthmap(proj_np)
                if depths is not None:
                    depth_np = depths[view_idx, :, :, :]
                    depth_np = normalise_depthmap(depth_np)
                else:
                    depth_np = 1.0 - np.squeeze(gt_mask_np)
                if pose_student:
                    proj_student_np = normalise_depthmap(proj_student_np)

            if cfg.predict_pose:
                if cfg.save_rotated_points:
                    ref_rot = scipy.io.loadmat("{}/final_reference_rotation.mat".format(exp_dir))
                    ref_rot = ref_rot["rotation"]
                    pc_np_unrot = sess.run(pc_rot, feed_dict={input_pc: pc_np,
                                                                pred_quat: ref_rot})
                    pc_np = pc_np_unrot

            if cfg.pc_rgb:
                gt_image = input_image_np
            elif cfg.vis_depth_projs:
                gt_image = depth_np
            else:
                gt_image = gt_mask_np

            if pose_num_candidates == 1:
                view_j = view_idx * 2 // plot_w
                view_i = view_idx * 2 % plot_w

                gt_image = np.squeeze(gt_image)
                grid[view_j, view_i] = mask4vis(cfg, gt_image, vis_size)

                curr_img = np.squeeze(proj_np)
                grid[view_j, view_i + 1] = mask4vis(cfg, curr_img, vis_size)

                if cfg.save_individual_images:
                    curr_dir = os.path.join(save_dir, sample.name)
                    if not os.path.exists(curr_dir):
                        os.makedirs(curr_dir)
                    imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'rgb_gt')),
                                    mask4vis(cfg, np.squeeze(input_image_np), vis_size))
                    imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'mask_pred')),
                                    mask4vis(cfg, np.squeeze(proj_np), vis_size))
            else:
                view_j = view_idx

                gt_image = np.squeeze(gt_image)
                grid[view_j, 0] = mask4vis(cfg, gt_image, vis_size)

                for kk in range(pose_num_candidates):
                    curr_img = np.squeeze(proj_np[kk, :, :, :])
                    grid[view_j, kk + 1] = mask4vis(cfg, curr_img, vis_size)

                    if cfg.save_individual_images:
                        curr_dir = os.path.join(save_dir, sample.name)
                        if not os.path.exists(curr_dir):
                            os.makedirs(curr_dir)
                        imageio.imwrite(os.path.join(curr_dir, '{}_{}_{}.png'.format(view_idx, kk, 'mask_pred')),
                                        mask4vis(cfg, np.squeeze(curr_img), vis_size))

                if cfg.save_individual_images:
                    imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'mask_gt')),
                                    mask4vis(cfg, np.squeeze(gt_mask_np), vis_size))

                if pose_student:
                    grid[view_j, -1] = mask4vis(cfg, np.squeeze(proj_student_np), vis_size)

            if save_pred:
                all_pcs[view_idx, :, :] = np.squeeze(pc_np)
                all_z_latent[view_idx] = z_latent_np
                if cfg.predict_pose:
                    all_cameras[view_idx, :] = predicted_camera
                if save_voxels:
                    # multiplying by two is necessary because
                    # pc->voxel conversion expects points in [-1, 1] range
                    pc_np_range = pc_np
                    if not fast_conversion:
                        pc_np_range *= 2.0
                    voxels_np = sess.run(voxels, feed_dict={input_pc: pc_np_range})
                    all_voxels[view_idx, :, :, :] = np.squeeze(voxels_np)

            vis_view = view_idx == 0 or cfg.vis_all_views
            if cfg.vis_voxels and vis_view:
                rgb_np = np.squeeze(rgb_np) if cfg.pc_rgb else None
                vis_pc(np.squeeze(pc_np), rgb=rgb_np)

        grid_merged = merge_grid(cfg, grid)
        imageio.imwrite("{}/{}_proj.png".format(save_dir, sample.name), grid_merged)

        if save_pred:
            if cfg.save_as_mat:
                save_dict = {"points": all_pcs,
                             "z_latent": all_z_latent}
                if cfg.predict_pose:
                    save_dict["camera_pose"] = all_cameras
                scipy.io.savemat("{}/{}_pc".format(save_pred_dir, sample.name),
                                 mdict=save_dict)
            else:
                np.savez("{}/{}_pc".format(save_pred_dir, sample.name), all_pcs)

            if save_voxels:
                np.savez("{}/{}_vox".format(save_pred_dir, sample.name), all_voxels)

    sess.close()


def main(_):
    compute_predictions()


if __name__ == '__main__':
    tf.app.run()
