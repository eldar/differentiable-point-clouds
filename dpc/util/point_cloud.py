import numpy as np
import tensorflow as tf

import util.drc
from util.quaternion import quaternion_rotate
from util.camera import intrinsic_matrix
from util.point_cloud_distance import *


def multi_expand(inp, axis, num):
    inp_big = inp
    for i in range(num):
        inp_big = tf.expand_dims(inp_big, axis)
    return inp_big


def pointcloud2voxels(cfg, input_pc, sigma):  # [B,N,3]
    # TODO replace with split or tf.unstack
    x = input_pc[:, :, 0]
    y = input_pc[:, :, 1]
    z = input_pc[:, :, 2]

    vox_size = cfg.vox_size

    rng = tf.linspace(-1.0, 1.0, vox_size)
    xg, yg, zg = tf.meshgrid(rng, rng, rng)  # [G,G,G]

    x_big = multi_expand(x, -1, 3)  # [B,N,1,1,1]
    y_big = multi_expand(y, -1, 3)  # [B,N,1,1,1]
    z_big = multi_expand(z, -1, 3)  # [B,N,1,1,1]

    xg = multi_expand(xg, 0, 2)  # [1,1,G,G,G]
    yg = multi_expand(yg, 0, 2)  # [1,1,G,G,G]
    zg = multi_expand(zg, 0, 2)  # [1,1,G,G,G]

    # squared distance
    sq_distance = tf.square(x_big - xg) + tf.square(y_big - yg) + tf.square(z_big - zg)

    # compute gaussian
    func = tf.exp(-sq_distance / (2.0 * sigma * sigma))  # [B,N,G,G,G]

    # normalise gaussian
    if cfg.pc_normalise_gauss:
        normaliser = tf.reduce_sum(func, [2, 3, 4], keep_dims=True)
        func /= normaliser
    elif cfg.pc_normalise_gauss_analytical:
        # should work with any grid sizes
        magic_factor = 1.78984352254  # see estimate_gauss_normaliser
        sigma_normalised = sigma * vox_size
        normaliser = 1.0 / (magic_factor * tf.pow(sigma_normalised, 3))
        func *= normaliser

    summed = tf.reduce_sum(func, axis=1)  # [B,G,G G]
    voxels = tf.clip_by_value(summed, 0.0, 1.0)
    voxels = tf.expand_dims(voxels, axis=-1)  # [B,G,G,G,1]

    return voxels


def pointcloud2voxels3d_fast(cfg, pc, rgb):  # [B,N,3]
    vox_size = cfg.vox_size
    if cfg.vox_size_z != -1:
        vox_size_z = cfg.vox_size_z
    else:
        vox_size_z = vox_size

    batch_size = pc.shape[0]
    num_points = tf.shape(pc)[1]

    has_rgb = rgb is not None

    grid_size = 1.0
    half_size = grid_size / 2

    filter_outliers = True
    valid = tf.logical_and(pc >= -half_size, pc <= half_size)
    valid = tf.reduce_all(valid, axis=-1)

    vox_size_tf = tf.constant([[[vox_size_z, vox_size, vox_size]]], dtype=tf.float32)
    pc_grid = (pc + half_size) * (vox_size_tf - 1)
    indices_floor = tf.floor(pc_grid)
    indices_int = tf.cast(indices_floor, tf.int32)
    batch_indices = tf.range(0, batch_size, 1)
    batch_indices = tf.expand_dims(batch_indices, -1)
    batch_indices = tf.tile(batch_indices, [1, num_points])
    batch_indices = tf.expand_dims(batch_indices, -1)

    indices = tf.concat([batch_indices, indices_int], axis=2)
    indices = tf.reshape(indices, [-1, 4])

    r = pc_grid - indices_floor  # fractional part
    rr = [1.0 - r, r]

    if filter_outliers:
        valid = tf.reshape(valid, [-1])
        indices = tf.boolean_mask(indices, valid)

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][:, :, 0] * rr[pos[1]][:, :, 1] * rr[pos[2]][:, :, 2]
        updates = tf.reshape(updates_raw, [-1])
        if filter_outliers:
            updates = tf.boolean_mask(updates, valid)

        indices_loc = indices
        indices_shift = tf.constant([[0] + pos])
        num_updates = tf.shape(indices_loc)[0]
        indices_shift = tf.tile(indices_shift, [num_updates, 1])
        indices_loc = indices_loc + indices_shift

        voxels = tf.scatter_nd(indices_loc, updates, [batch_size, vox_size_z, vox_size, vox_size])
        if has_rgb:
            if cfg.pc_rgb_stop_points_gradient:
                updates_raw = tf.stop_gradient(updates_raw)
            updates_rgb = tf.expand_dims(updates_raw, axis=-1) * rgb
            updates_rgb = tf.reshape(updates_rgb, [-1, 3])
            if filter_outliers:
                updates_rgb = tf.boolean_mask(updates_rgb, valid)
            voxels_rgb = tf.scatter_nd(indices_loc, updates_rgb, [batch_size, vox_size_z, vox_size, vox_size, 3])
        else:
            voxels_rgb = None

        return voxels, voxels_rgb

    voxels = []
    voxels_rgb = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                vx, vx_rgb = interpolate_scatter3d([k, j, i])
                voxels.append(vx)
                voxels_rgb.append(vx_rgb)

    voxels = tf.add_n(voxels)
    voxels_rgb = tf.add_n(voxels_rgb) if has_rgb else None

    return voxels, voxels_rgb


def smoothen_voxels3d(cfg, voxels, kernel):
    if cfg.pc_separable_gauss_filter:
        for krnl in kernel:
            voxels = tf.nn.conv3d(voxels, krnl, [1, 1, 1, 1, 1], padding="SAME")
    else:
        voxels = tf.nn.conv3d(voxels, kernel, [1, 1, 1, 1, 1], padding="SAME")
    return voxels


def convolve_rgb(cfg, voxels_rgb, kernel):
    channels = [voxels_rgb[:, :, :, :, k:k+1] for k in range(3)]
    for krnl in kernel:
        for i in range(3):
            channels[i] = tf.nn.conv3d(channels[i], krnl, [1, 1, 1, 1, 1], padding="SAME")
    out = tf.concat(channels, axis=4)
    return out


def pc_perspective_transform(cfg, point_cloud,
                             transform, predicted_translation=None,
                             focal_length=None):
    """
    :param cfg:
    :param point_cloud: [B, N, 3]
    :param transform: [B, 4] if quaternion or [B, 4, 4] if camera matrix
    :param predicted_translation: [B, 3] translation vector
    :return:
    """
    camera_distance = cfg.camera_distance

    if focal_length is None:
        focal_length = cfg.focal_length
    else:
        focal_length = tf.expand_dims(focal_length, axis=-1)

    if cfg.pose_quaternion:
        pc2 = quaternion_rotate(point_cloud, transform)

        if predicted_translation is not None:
            predicted_translation = tf.expand_dims(predicted_translation, axis=1)
            pc2 += predicted_translation

        xs = tf.slice(pc2, [0, 0, 2], [-1, -1, 1])
        ys = tf.slice(pc2, [0, 0, 1], [-1, -1, 1])
        zs = tf.slice(pc2, [0, 0, 0], [-1, -1, 1])

        # translation part of extrinsic camera
        zs += camera_distance
        # intrinsic transform
        xs *= focal_length
        ys *= focal_length
    else:
        xyz1 = tf.pad(point_cloud, tf.constant([[0, 0], [0, 0], [0, 1]]), "CONSTANT", constant_values=1.0)

        extrinsic = transform
        intr = intrinsic_matrix(cfg, dims=4)
        intrinsic = tf.convert_to_tensor(intr)
        intrinsic = tf.expand_dims(intrinsic, axis=0)
        intrinsic = tf.tile(intrinsic, [tf.shape(extrinsic)[0], 1, 1])
        full_cam_matrix = tf.matmul(intrinsic, extrinsic)

        pc2 = tf.matmul(xyz1, tf.transpose(full_cam_matrix, [0, 2, 1]))

        # TODO unstack instead of split
        xs = tf.slice(pc2, [0, 0, 2], [-1, -1, 1])
        ys = tf.slice(pc2, [0, 0, 1], [-1, -1, 1])
        zs = tf.slice(pc2, [0, 0, 0], [-1, -1, 1])

    xs /= zs
    ys /= zs

    zs -= camera_distance
    if predicted_translation is not None:
        zt = tf.slice(predicted_translation, [0, 0, 0], [-1, -1, 1])
        zs -= zt

    xyz2 = tf.concat([zs, ys, xs], axis=2)
    return xyz2


def pointcloud_project(cfg, point_cloud, transform, sigma):
    tr_pc = pc_perspective_transform(cfg, point_cloud, transform)
    voxels = pointcloud2voxels(cfg, tr_pc, sigma)
    voxels = tf.transpose(voxels, [0, 2, 1, 3, 4])

    proj, probs = util.drc.drc_projection(voxels, cfg)
    proj = tf.reverse(proj, [1])
    return proj, voxels


def pointcloud_project_fast(cfg, point_cloud, transform, predicted_translation,
                            all_rgb, kernel=None, scaling_factor=None, focal_length=None):
    has_rgb = all_rgb is not None

    tr_pc = pc_perspective_transform(cfg, point_cloud,
                                     transform, predicted_translation,
                                     focal_length)
    voxels, voxels_rgb = pointcloud2voxels3d_fast(cfg, tr_pc, all_rgb)
    voxels = tf.expand_dims(voxels, axis=-1)
    voxels_raw = voxels

    voxels = tf.clip_by_value(voxels, 0.0, 1.0)

    if kernel is not None:
        voxels = smoothen_voxels3d(cfg, voxels, kernel)
        if has_rgb:
            if not cfg.pc_rgb_clip_after_conv:
                voxels_rgb = tf.clip_by_value(voxels_rgb, 0.0, 1.0)
            voxels_rgb = convolve_rgb(cfg, voxels_rgb, kernel)

    if scaling_factor is not None:
        sz = scaling_factor.shape[0]
        scaling_factor = tf.reshape(scaling_factor, [sz, 1, 1, 1, 1])
        voxels = voxels * scaling_factor
        voxels = tf.clip_by_value(voxels, 0.0, 1.0)

    if has_rgb:
        if cfg.pc_rgb_divide_by_occupancies:
            voxels_div = tf.stop_gradient(voxels_raw)
            voxels_div = smoothen_voxels3d(cfg, voxels_div, kernel)
            voxels_rgb = voxels_rgb / (voxels_div + cfg.pc_rgb_divide_by_occupancies_epsilon)

        if cfg.pc_rgb_clip_after_conv:
            voxels_rgb = tf.clip_by_value(voxels_rgb, 0.0, 1.0)

    if cfg.ptn_max_projection:
        proj = tf.reduce_max(voxels, [1])
        drc_probs = None
        proj_depth = None
    else:
        proj, drc_probs = util.drc.drc_projection(voxels, cfg)
        drc_probs = tf.reverse(drc_probs, [2])
        proj_depth = util.drc.drc_depth_projection(drc_probs, cfg)

    proj = tf.reverse(proj, [1])

    if voxels_rgb is not None:
        voxels_rgb = tf.reverse(voxels_rgb, [2])
        proj_rgb = util.drc.project_volume_rgb_integral(cfg, drc_probs, voxels_rgb)
    else:
        proj_rgb = None

    output = {
        "proj": proj,
        "voxels": voxels,
        "tr_pc": tr_pc,
        "voxels_rgb": voxels_rgb,
        "proj_rgb": proj_rgb,
        "drc_probs": drc_probs,
        "proj_depth": proj_depth
    }
    return output


def pc_point_dropout(points, rgb, keep_prob):
    shape = points.shape.as_list()
    num_input_points = shape[1]
    batch_size = shape[0]
    num_channels = shape[2]
    num_output_points = tf.cast(num_input_points * keep_prob, tf.int32)

    def sampler(num_output_points_np):
        all_inds = []
        for k in range(batch_size):
            ind = np.random.choice(num_input_points, num_output_points_np, replace=False)
            ind = np.expand_dims(ind, axis=-1)
            ks = np.ones_like(ind) * k
            inds = np.concatenate((ks, ind), axis=1)
            all_inds.append(np.expand_dims(inds, 0))
        return np.concatenate(tuple(all_inds), 0).astype(np.int64)

    selected_indices = tf.py_func(sampler, [num_output_points], tf.int64)
    out_points = tf.gather_nd(points, selected_indices)
    out_points = tf.reshape(out_points, [batch_size, num_output_points, num_channels])
    if rgb is not None:
        num_rgb_channels = rgb.shape.as_list()[2]
        out_rgb = tf.gather_nd(rgb, selected_indices)
        out_rgb = tf.reshape(out_rgb, [batch_size, num_output_points, num_rgb_channels])
    else:
        out_rgb = None
    return out_points, out_rgb


def subsample_points(xyz, num_points):
    idxs = np.random.choice(xyz.shape[0], num_points)
    xyz_s = xyz[idxs, :]
    return xyz_s
