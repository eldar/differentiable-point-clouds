import math
import numpy as np

import tensorflow as tf
from util.quaternion import quaternion_multiply_np as q_mul_np


def quaternion2euler_full(q, rotseq, print_all=False):
    def twoaxisrot(r11, r12, r21, r31, r32):
        res = np.zeros(3, np.float32)
        res[0] = math.atan2(r11, r12)
        res[1] = math.acos(r21)
        res[2] = math.atan2(r31, r32)
        return res

    def threeaxisrot(r11, r12, r21, r31, r32):
        res = np.zeros(3, np.float32)
        res[0] = math.atan2(r31, r32)
        res[1] = math.asin(np.clip(r21, -1.0, 1.0))
        res[2] = math.atan2(r11, r12)
        return res

    all = {
        "zyx": threeaxisrot(2 * (q[1] * q[2] + q[0] * q[3]),
                            q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3],
                            -2 * (q[1] * q[3] - q[0] * q[2]),
                            2 * (q[2] * q[3] + q[0] * q[1]),
                            q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
                            ),
        "zyz": twoaxisrot(2 * (q[2] * q[3] - q[0] * q[1]),
                          2 * (q[1] * q[3] + q[0] * q[2]),
                          q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
                          2 * (q[2] * q[3] + q[0] * q[1]),
                          -2 * (q[1] * q[3] - q[0] * q[2])
                          ),
        "zxy": threeaxisrot(-2 * (q[1] * q[2] - q[0] * q[3]),
                            q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3],
                            2 * (q[2] * q[3] + q[0] * q[1]),
                            -2 * (q[1] * q[3] - q[0] * q[2]),
                            q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]),
        "zxz": twoaxisrot(2 * (q[1] * q[3] + q[0] * q[2]),
                          -2 * (q[2] * q[3] - q[0] * q[1]),
                          q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
                          2 * (q[1] * q[3] - q[0] * q[2]),
                          2 * (q[2] * q[3] + q[0] * q[1])),
        "yxz": threeaxisrot(2 * (q[1] * q[3] + q[0] * q[2]),
                            q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
                            -2 * (q[2] * q[3] - q[0] * q[1]),
                            2 * (q[1] * q[2] + q[0] * q[3]),
                            q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]),
        "yxy": twoaxisrot(2 * (q[1] * q[2] - q[0] * q[3]),
                          2 * (q[2] * q[3] + q[0] * q[1]),
                          q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3],
                          2 * (q[1] * q[2] + q[0] * q[3]),
                          -2 * (q[2] * q[3] - q[0] * q[1])),
        "yzx": threeaxisrot(-2 * (q[1] * q[3] - q[0] * q[2]),
                            q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3],
                            2 * (q[1] * q[2] + q[0] * q[3]),
                            -2 * (q[2] * q[3] - q[0] * q[1]),
                            q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]),
        "yzy": twoaxisrot(2 * (q[2] * q[3] + q[0] * q[1]),
                          -2 * (q[1] * q[2] - q[0] * q[3]),
                          q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3],
                          2 * (q[2] * q[3] - q[0] * q[1]),
                          2 * (q[1] * q[2] + q[0] * q[3])),
        "xyz": threeaxisrot(-2 * (q[2] * q[3] - q[0] * q[1]),
                            q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
                            2 * (q[1] * q[3] + q[0] * q[2]),
                            -2 * (q[1] * q[2] - q[0] * q[3]),
                            q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]),
        "xyx": twoaxisrot(2 * (q[1] * q[2] + q[0] * q[3]),
                          -2 * (q[1] * q[3] - q[0] * q[2]),
                          q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3],
                          2 * (q[1] * q[2] - q[0] * q[3]),
                          2 * (q[1] * q[3] + q[0] * q[2])),
        "xzy": threeaxisrot(2 * (q[2] * q[3] + q[0] * q[1]),
                            q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3],
                            -2 * (q[1] * q[2] - q[0] * q[3]),
                            2 * (q[1] * q[3] + q[0] * q[2]),
                            q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]),
        "xzx": twoaxisrot(2 * (q[1] * q[3] - q[0] * q[2]),
                          2 * (q[1] * q[2] + q[0] * q[3]),
                          q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3],
                          2 * (q[1] * q[3] + q[0] * q[2]),
                          -2 * (q[1] * q[2] - q[0] * q[3]))
    }

    if print_all:
        for k, v in all.items():
            print(k, v[0], v[1], v[2])

    return all[rotseq]


def quaternion2euler_full_tf(q, rotseq="yzy"):
    def twoaxisrot_tf(r11, r12, r21, r31, r32):
        a0 = tf.atan2(r11, r12)
        a1 = tf.acos(r21)
        a2 = tf.atan2(r31, r32)
        return tf.stack([a0, a1, a2], axis=-1)

    def threeaxisrot_tf(r11, r12, r21, r31, r32):
        a0 = tf.atan2(r31, r32)
        a1 = tf.asin(tf.clip_by_value(r21, -1.0, 1.0))
        a2 = tf.atan2(r11, r12)
        return tf.stack([a0, a1, a2], axis=-1)

    q_norm = tf.expand_dims(tf.norm(q, axis=-1), axis=-1)
    q /= q_norm

    if rotseq == "yzy":
        angles = twoaxisrot_tf(2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1]),
                               -2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3]),
                               q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3],
                               2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]),
                               2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3]))
        yaw = angles[:, 2]
        pitch = angles[:, 1]
    elif rotseq == "xzy":
        angles = threeaxisrot_tf(2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1]),
                                 q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3],
                                 -2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3]),
                                 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2]),
                                 q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3])
        yaw = angles[:, 0]
        pitch = angles[:, 1]
    elif rotseq == "zxy":
        angles = threeaxisrot_tf(-2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3]),
                                 q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3],
                                 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1]),
                                 -2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2]),
                                 q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
        yaw = angles[:, 0]
        pitch = angles[:, 2]

    return yaw, pitch


def ypr_from_campos(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(tx)
    if ty > 0:
        yaw = 2 * math.pi - yaw

    roll = 0
    pitch = math.asin(cz)

    return yaw, pitch, roll


def axis_angle_quaternion(angle, axis):
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    q = np.zeros(4)
    q[0] = c
    q[1:4] = s * axis
    return q


def quaternion2euler(quat):
    return quaternion2euler_full(quat, "xzy")


def quaternionFromYawPitchRoll(yaw, pitch, roll):
    # reverse transformation is ypr = quaternion2euler(quat)
    q_yaw = axis_angle_quaternion(yaw, np.array([0, 1, 0]))
    q_pitch = axis_angle_quaternion(pitch, np.array([0, 0, 1]))
    q_roll = axis_angle_quaternion(roll, np.array([1, 0, 0]))
    return q_mul_np(q_roll, q_mul_np(q_pitch, q_yaw))
