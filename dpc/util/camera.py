import numpy as np
import tensorflow as tf


def intrinsic_matrix(cfg, dims=3, inverse=False):
    focal_length = cfg.focal_length
    val = float(focal_length)
    if inverse:
        val = 1.0 / val
    intrinsic_matrix = np.eye(dims, dtype=np.float32)
    intrinsic_matrix[1, 1] = val
    intrinsic_matrix[2, 2] = val
    return intrinsic_matrix


def camera_from_blender(their):
    our = np.zeros((4, 4), dtype=np.float32)
    our[0, 0] = -their[2, 0]
    our[0, 1] =  their[2, 2]
    our[0, 2] =  their[2, 1]

    our[1, 0] =  their[1, 0]
    our[1, 1] = -their[1, 2]
    our[1, 2] = -their[1, 1]

    our[2, 0] = -their[0, 0]
    our[2, 1] =  their[0, 2]
    our[2, 2] =  their[0, 1]

    our[0, 3] =  their[2, 3]
    our[1, 3] =  their[1, 3]
    our[2, 3] =  their[0, 3]

    our[3, 3] =  their[3, 3]

    return our


def get_full_camera(cfg, cam, inverted):
    def fix_matrix(extr):
        return camera_from_blender(extr)
    extr_tf = tf.py_func(fix_matrix, [cam], tf.float32)
    extr_tf = tf.reshape(extr_tf, shape=[4, 4])
    return extr_tf


def ypr_from_campos_blender(pos):
    from util.euler import ypr_from_campos

    yaw, pitch, roll = ypr_from_campos(pos[0], pos[1], pos[2])
    yaw = yaw + np.pi

    return yaw, pitch, roll


def quaternion_from_campos(cam_pos):
    from util.euler import quaternionFromYawPitchRoll

    yaw, pitch, roll = ypr_from_campos_blender(cam_pos)
    return quaternionFromYawPitchRoll(yaw, pitch, roll)
