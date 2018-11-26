# Copyright Philipp Jund (jundp@cs.uni-freiburg.de) and Eldar Insafutdinov, 2018.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Website: https://github.com/PhilJd/tf-quaternion

import tensorflow as tf
import numpy as np


def validate_shape(x):
    """Raise a value error if x.shape ist not (..., 4)."""
    error_msg = ("Can't create a quaternion from a tensor with shape {}."
                 "The last dimension must be 4.")
    # Check is performed during graph construction. If your dimension
    # is unknown, tf.reshape(x, (-1, 4)) might work.
    if x.shape[-1] != 4:
        raise ValueError(error_msg.format(x.shape))


def vector3d_to_quaternion(x):
    """Convert a tensor of 3D vectors to a quaternion.
    Prepends a 0 to the last dimension, i.e. [[1,2,3]] -> [[0,1,2,3]].
    Args:
        x: A `tf.Tensor` of rank R, the last dimension must be 3.
    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
    Raises:
        ValueError, if the last dimension of x is not 3.
    """
    x = tf.convert_to_tensor(x)
    if x.shape[-1] != 3:
        raise ValueError("The last dimension of x must be 3.")
    return tf.pad(x, (len(x.shape) - 1) * [[0, 0]] + [[1, 0]])


def _prepare_tensor_for_div_mul(x):
    """Prepare the tensor x for division/multiplication.
    This function
    a) converts x to a tensor if necessary,
    b) prepends a 0 in the last dimension if the last dimension is 3,
    c) validates the type and shape.
    """
    x = tf.convert_to_tensor(x)
    if x.shape[-1] == 3:
        x = vector3d_to_quaternion(x)
    validate_shape(x)
    return x


def quaternion_multiply(a, b):
    """Multiply two quaternion tensors.
    Note that this differs from tf.multiply and is not commutative.
    Args:
        a, b: A `tf.Tensor` with shape (..., 4).
    Returns:
        A `Quaternion`.
    """
    a = _prepare_tensor_for_div_mul(a)
    b = _prepare_tensor_for_div_mul(b)
    w1, x1, y1, z1 = tf.unstack(a, axis=-1)
    w2, x2, y2, z2 = tf.unstack(b, axis=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return tf.stack((w, x, y, z), axis=-1)


def quaternion_conjugate(q):
    """Compute the conjugate of q, i.e. [q.w, -q.x, -q.y, -q.z]."""
    return tf.multiply(q, [1.0, -1.0, -1.0, -1.0])


def quaternion_normalise(q):
    """Normalises quaternion to use as a rotation quaternion
    Args:
        q: [..., 4] quaternion
    Returns:
        q / ||q||_2
    """
    return q / tf.norm(q, axis=-1, keepdims=True)


def quaternion_rotate(pc, q, inverse=False):
    """rotates a set of 3D points by a rotation,
    represented as a quaternion
    Args:
        pc: [B,N,3] point cloud
        q: [B,4] rotation quaternion
    Returns:
        q * pc * q'
    """
    q_norm = tf.expand_dims(tf.norm(q, axis=-1), axis=-1)
    q /= q_norm
    q = tf.expand_dims(q, axis=1)  # [B,1,4]
    q_ = quaternion_conjugate(q)
    qmul = quaternion_multiply
    if not inverse:
        wxyz = qmul(qmul(q, pc), q_)  # [B,N,4]
    else:
        wxyz = qmul(qmul(q_, pc), q)  # [B,N,4]
    if len(wxyz.shape) == 2: # bug with batch size of 1
        wxyz = tf.expand_dims(wxyz, axis=0)
    xyz = wxyz[:, :, 1:4]  # [B,N,3]
    return xyz


def normalized(q):
    q_norm = tf.expand_dims(tf.norm(q, axis=-1), axis=-1)
    q /= q_norm
    return q


def as_rotation_matrix(q):
    """Calculate the corresponding rotation matrix.

    See
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/

    Returns:
        A `tf.Tensor` with R+1 dimensions and
        shape [d_1, ..., d_(R-1), 3, 3], the rotation matrix
    """
    # helper functions
    def diag(a, b):  # computes the diagonal entries,  1 - 2*a**2 - 2*b**2
        return 1 - 2 * tf.pow(a, 2) - 2 * tf.pow(b, 2)

    def tr_add(a, b, c, d):  # computes triangle entries with addition
        return 2 * a * b + 2 * c * d

    def tr_sub(a, b, c, d):  # computes triangle entries with subtraction
        return 2 * a * b - 2 * c * d

    w, x, y, z = tf.unstack(normalized(q), axis=-1)
    m = [[diag(y, z), tr_sub(x, y, z, w), tr_add(x, z, y, w)],
         [tr_add(x, y, z, w), diag(x, z), tr_sub(y, z, x, w)],
         [tr_sub(x, z, y, w), tr_add(y, z, x, w), diag(x, y)]]
    return tf.stack([tf.stack(m[i], axis=-1) for i in range(3)], axis=-2)


def from_rotation_matrix(mtr):
    """
    See
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    mtr = tf.convert_to_tensor(mtr)
    def m(j, i):
        shape = mtr.shape.as_list()
        begin = [0 for _ in range(len(shape))]
        begin[-2] = j
        begin[-1] = i
        size = [s for s in shape]
        size[-2] = 1
        size[-1] = 1
        v = tf.slice(mtr, begin=begin, size=size)
        v = tf.squeeze(v, axis=[-1, -2])
        return v

    w = tf.sqrt(1.0 + m(0, 0) + m(1, 1) + m(2, 2)) / 2
    x = (m(2, 1) - m(1, 2)) / (4 * w)
    y = (m(0, 2) - m(2, 0)) / (4 * w)
    z = (m(1, 0) - m(0, 1)) / (4 * w)
    q = tf.stack([w, x, y, z], axis=-1)
    return q


def quaternion_multiply_np(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])