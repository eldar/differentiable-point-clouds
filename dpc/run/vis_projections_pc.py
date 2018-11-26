import startup

import os
import time

import numpy as np
import scipy.io

import tensorflow as tf
from skimage.transform import resize as imresize

import matplotlib.pyplot as plt
from matplotlib import cm

from models import model_pc

from util.app_config import config as app_config
from util.data import tf_record_options
from util.voxel import voxel2pc
from util.visualise import vis_voxels as visualise_voxels, merge_grid
from util.point_cloud import pointcloud2voxels, pointcloud_project, subsample_points
from util.point_cloud import pointcloud_project_fast, pointcloud2voxels3d_fast
from util.camera import camera_from_blender, intrinsic_matrix, quaternion_from_campos
from util.gauss_kernel import smoothing_kernel, gauss_smoothen_image


def sample_points_from_voxels(voxels, num_points):
    xyz, _ = voxel2pc(voxels, 0.5)
    xyz_s = subsample_points(xyz, num_points)

    max_perturb = 1.0 / voxels.shape[0]
    noise = np.random.uniform(-max_perturb, max_perturb, xyz_s.shape)
    #xyz_s += noise0

    return xyz_s


def pc_colors(xyz, color_axis=2):
    axis_vis = xyz[:, color_axis]
    min_ = np.min(axis_vis)
    max_ = np.max(axis_vis)

    return cm.gist_rainbow((axis_vis - min_) / (max_ - min_))[:, 0:3]


def main(_):
    cfg = app_config

    config = tf.ConfigProto(
        device_count={'GPU': 1}
    )
    sess = tf.Session(config=config)
    print("vox size", cfg.vox_size)

    tfrecords_filename = "/home/eldar/src/3dshape/ptn/train_data/03001627_val.tfrecords"

    synth_set = cfg.synth_set
    data_split = cfg.eval_split
    tfrecords_filename = "{}/{}_{}.tfrecords".format(cfg.inp_dir, synth_set, data_split)

    # tfrecords_filename = "/home/eldar/src/3dshape/ptn/train_data_drc_2/03001627_val.tfrecords"

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename,
                                                      options=tf_record_options(cfg))
    src_data_dir = "/home/eldar/src/3dshape/cachedir/blenderRender_128x128_24/03001627/"
    shapenet_dir = "/home/eldar/data/ShapeNetCore.v1/03001627/"

    surface_gt_root = "/home/eldar/storage/3dshape/surface/"
    surface_gt_root = "/BS/eldar-3dshape/work/dataset/shapenet_surface_gt/orig_coord"

    vis_voxels = True
    vis_images = False
    vis_projections = True

    num_points = 8000

    has_rgb = cfg.pc_rgb

    num_views = cfg.num_views
    image_size = cfg.image_size
    vox_size = cfg.vox_size
    vis_size = cfg.vis_size

    sigma_rel = cfg.pc_relative_sigma
    sigma = sigma_rel / vox_size

    surface_gt_path = os.path.join(surface_gt_root, cfg.synth_set)

    model = model_pc.ModelPointCloud(cfg)

    input_pc = tf.placeholder(dtype=tf.float32, shape=[1, num_points, 3])
    input_rgb = tf.placeholder(dtype=tf.float32, shape=[1, num_points, 3]) if has_rgb else None
    pc_voxels = pointcloud2voxels(cfg, input_pc, sigma)
    pc_voxels_fast, _ = pointcloud2voxels3d_fast(cfg, input_pc, None)
    pc_voxels_fast = tf.transpose(pc_voxels_fast, [0, 2, 1, 3])

    transform_matrix_pc = tf.placeholder(tf.float32, [1, 4, 4])
    transform_quaternion = tf.placeholder(tf.float32, [1, 4])
    if cfg.pose_quaternion:
        cam_transform = transform_quaternion
    else:
        cam_transform = transform_matrix_pc

    if not cfg.pc_fast:
        proj_of_pc, voxels_tr = pointcloud_project(cfg, input_pc, cam_transform, sigma)

    translation = tf.placeholder(shape=[1, 3], dtype=tf.float32)
    gauss_kernel = smoothing_kernel(cfg, sigma_rel)
    out = pointcloud_project_fast(cfg, input_pc, cam_transform, translation, input_rgb, gauss_kernel)
    proj_of_pc_fast = out["proj"]
    tr_pc_fast = out["tr_pc"]
    proj_rgb = out["proj_rgb"]
    proj_depth = out["proj_depth"]

    img_2d_input = tf.placeholder(tf.float32, [1, cfg.vis_size, cfg.vis_size, 1])
    img_2d_smoothed = gauss_smoothen_image(cfg, img_2d_input, sigma_rel)

    # Initialize session
    sess.run(tf.initialize_all_variables())

    k = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        byte_list = example.features.feature['name'].bytes_list.value[0]
        model_name = str(byte_list)
        print("model_name", k, model_name)
        #if model_name != "a9e72050a2964ba31030665054ebb2a9":
        #    continue

        float_list = example.features.feature['mask'].float_list.value
        masks_1d = np.array(float_list)
        masks = masks_1d.reshape((num_views, image_size, image_size, -1))

        float_list = example.features.feature['image'].float_list.value
        images_1d = np.array(float_list)
        images = images_1d.reshape((num_views, image_size, image_size, -1))

        stuff = scipy.io.loadmat("{}/{}.mat".format(surface_gt_path, model_name))
        orig_points = stuff["points"]

        xyz_s = subsample_points(orig_points, num_points)
        rgb_s = pc_colors(xyz_s, color_axis=2)
        #vis_pc(xyz_s, color_axis=2)

        if False:
            xyz_s = xyz_s[:, [0, 2, 1]]
        # rgb_s = np.zeros_like(xyz_s)
        # rgb_s[:, 0] = 1.0

        if 'depth' in list(example.features.feature.keys()):
            float_list = example.features.feature['depth'].float_list.value
            depths_1d = np.array(float_list)
            depths = depths_1d.reshape((num_views, image_size, image_size, -1))

        if False:
            float_list = example.features.feature['vox'].float_list.value
            vox_1d = np.array(float_list)
            vox_data_size = int(round(vox_1d.shape[0] ** (1.0/3.0)))
            voxels = vox_1d.reshape((vox_data_size, vox_data_size, vox_data_size))
            voxels1 = vox_1d.reshape((1, vox_data_size, vox_data_size, vox_data_size, 1))

            #xyz_s = sample_points_from_voxels(voxels, num_points=num_points)
            #xyz_s = xyz_s.astype(np.float32)

        if cfg.saved_camera:
            float_list = example.features.feature['extrinsic'].float_list.value
            cam_1d = np.array(float_list)
            cameras = cam_1d.reshape((num_views, 4, 4))

            float_list = example.features.feature['cam_pos'].float_list.value
            cam_pos_1d = np.array(float_list)
            cam_pos = cam_pos_1d.reshape((num_views, 3))

        #vis_pc(np.squeeze(xyz_s), 2)
        #continue

        xyz_s = np.expand_dims(xyz_s, axis=0)
        rgb_s = np.expand_dims(rgb_s, axis=0)

        if False:
            start = time.time()
            voxels_new = sess.run(pc_voxels, feed_dict={input_pc: xyz_s})
            voxels_new_fast = sess.run(pc_voxels_fast, feed_dict={input_pc: xyz_s})
            end = time.time()
            #print(end - start)

            #visualise_voxels(cfg, voxels)
            #vis_pc(xyz_s, 1)
            visualise_voxels(cfg, voxels_new, vis_axis=0)
            visualise_voxels(cfg, voxels_new_fast, vis_axis=0)

        if vis_projections:
            subplot_height = 4
            subplot_width = 4
            num_plots = subplot_width * subplot_height
            grid = np.empty((subplot_height, subplot_width), dtype=object)

            start = time.time()

            for j in range(num_plots):
                plot_j = j // subplot_width
                plot_i = j % subplot_width

                if j % 2 == 0:
                    view_idx = j // 2
                    orig_img = images if has_rgb else masks
                    curr_img = np.squeeze(orig_img[view_idx, :, :, :])
                else:
                    view_idx = j // 2

                    extrinsic = cameras[view_idx, :, :]
                    extrinsic = camera_from_blender(extrinsic)
                    intrinsic = intrinsic_matrix(cfg, dims=4)
                    full_cam_matrix = np.matmul(intrinsic, extrinsic)
                    full_cam_matrix = np.expand_dims(full_cam_matrix, axis=0)

                    pos = cam_pos[view_idx, :]
                    quat = quaternion_from_campos(pos)
                    #matr = rot_matrix_from_q(quat)
                    quat_np = np.expand_dims(quat, axis=0)

                    trans_np = np.zeros((1, 3))
                    if view_idx % 4 == 0:
                        trans_np[0, 1] = -0.1
                    elif view_idx % 4 == 1:
                        trans_np[0, 1] = 0.1
                    elif view_idx % 4 == 2:
                        trans_np[0, 2] = -0.1
                    else:
                        trans_np[0, 2] = 0.1
                    trans_np[0, 0] = 0.2
                    trans_np = np.zeros((1, 3))

                    if cfg.pc_fast:
                        if has_rgb:
                            (proj_xyz_np, tr_pc_fast_np, proj_rgb_np) = sess.run([proj_of_pc_fast, tr_pc_fast, proj_rgb],
                                                                              feed_dict={input_pc: xyz_s,
                                                                                         input_rgb: rgb_s,
                                                                                         transform_matrix_pc: full_cam_matrix,
                                                                                         transform_quaternion: quat_np})
                        else:
                            (proj_xyz_np, tr_pc_fast_np, proj_depth_np, tr_voxels_np) = sess.run([proj_of_pc_fast, tr_pc_fast, proj_depth, out["voxels"]],
                                                                      feed_dict={input_pc: xyz_s,
                                                                                 transform_matrix_pc: full_cam_matrix,
                                                                                 transform_quaternion: quat_np,
                                                                                 translation: trans_np})
                    else:
                        (proj_xyz_np, tr_voxels_np) = sess.run([proj_of_pc, voxels_tr], feed_dict={input_pc: xyz_s,
                                                                               transform_matrix_pc: full_cam_matrix})


                    print(tr_voxels_np.shape)
                    print(np.histogram(tr_voxels_np.flatten()))
                    print("sum", np.sum(tr_voxels_np.flatten()))
                    #plt.hist(tr_voxels_np.flatten(), bins=10)
                    #plt.show()

                    proj = proj_xyz_np

                    proj = np.squeeze(proj)
                    if has_rgb:
                        proj_rgb_np = np.squeeze(proj_rgb_np)
                    curr_img = proj_rgb_np if has_rgb else proj

                curr_img = np.clip(curr_img, 0.0, 1.0)
                curr_img = imresize(curr_img, (vis_size, vis_size), order=3)

                if cfg.pc_gauss_filter_gt:
                    if j % 2 == 0:
                        img_prc = np.reshape(curr_img, (1, vis_size, vis_size, 1))
                        img_smoothed_np = sess.run(img_2d_smoothed, feed_dict={img_2d_input: img_prc})
                        curr_img = np.squeeze(img_smoothed_np)

                if curr_img.shape[-1] != 3:
                    curr_img = 255-np.clip(curr_img * 255, 0, 255).astype(dtype=np.uint8)

                grid[plot_j, plot_i] = curr_img

            end = time.time()
            print("time:", end - start)

            grid_merged = merge_grid(cfg, grid)

            plt.imshow(grid_merged)
            plt.show()

        k += 1


if __name__ == '__main__':
    tf.app.run()
