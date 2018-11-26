import numpy as np
import tensorflow as tf

from util.camera import camera_from_blender, quaternion_from_campos


def pool_single_view(cfg, tensor, view_idx):
    indices = tf.range(cfg.batch_size) * cfg.step_size + view_idx
    indices = tf.expand_dims(indices, axis=-1)
    return tf.gather_nd(tensor, indices)


class ModelBase(object):  # pylint:disable=invalid-name

    def __init__(self, cfg):
        self._params = cfg

    def cfg(self):
        return self._params

    def preprocess(self, raw_inputs, step_size, random_views=True):
        """Selects the subset of viewpoints to train on."""
        cfg = self.cfg()

        var_num_views = cfg.variable_num_views

        num_views = raw_inputs['image'].get_shape().as_list()[1]
        quantity = cfg.batch_size
        if cfg.num_views_to_use == -1:
            max_num_views = num_views
        else:
            max_num_views = cfg.num_views_to_use

        inputs = dict()

        def batch_sampler(all_num_views):
            out = np.zeros((0, 2), dtype=np.int64)
            valid_samples = np.zeros((0), dtype=np.float32)
            for n in range(quantity):
                valid_samples_m = np.ones((step_size), dtype=np.float32)
                if var_num_views:
                    num_actual_views = int(all_num_views[n, 0])
                    ids = np.random.choice(num_actual_views, min(step_size, num_actual_views), replace=False)
                    if num_actual_views < step_size:
                        to_fill = step_size - num_actual_views
                        ids = np.concatenate((ids, np.zeros((to_fill), dtype=ids.dtype)))
                        valid_samples_m[num_actual_views:] = 0.0
                elif random_views:
                    ids = np.random.choice(max_num_views, step_size, replace=False)
                else:
                    ids = np.arange(0, step_size).astype(np.int64)

                ids = np.expand_dims(ids, axis=-1)
                batch_ids = np.full((step_size, 1), n, dtype=np.int64)
                full_ids = np.concatenate((batch_ids, ids), axis=-1)
                out = np.concatenate((out, full_ids), axis=0)

                valid_samples = np.concatenate((valid_samples, valid_samples_m), axis=0)

            return out, valid_samples

        num_actual_views = raw_inputs['num_views'] if var_num_views else tf.constant([0])

        indices, valid_samples = tf.py_func(batch_sampler, [num_actual_views], [tf.int64, tf.float32])
        indices = tf.reshape(indices, [step_size*quantity, 2])
        inputs['valid_samples'] = tf.reshape(valid_samples, [step_size*quantity])

        inputs['masks'] = tf.gather_nd(raw_inputs['mask'], indices)
        inputs['images'] = tf.gather_nd(raw_inputs['image'], indices)
        if cfg.saved_depth:
            inputs['depths'] = tf.gather_nd(raw_inputs['depth'], indices)
        inputs['images_1'] = pool_single_view(cfg, inputs['images'], 0)

        def fix_matrix(extr):
            out = np.zeros_like(extr)
            num_matrices = extr.shape[0]
            for k in range(num_matrices):
                out[k, :, :] = camera_from_blender(extr[k, :, :])
            return out

        def quaternion_from_campos_wrapper(campos):
            num = campos.shape[0]
            out = np.zeros([num, 4], dtype=np.float32)
            for k in range(num):
                out[k, :] = quaternion_from_campos(campos[k, :])
            return out

        if cfg.saved_camera:
            matrices = tf.gather_nd(raw_inputs['extrinsic'], indices)
            orig_shape = matrices.shape
            extr_tf = tf.py_func(fix_matrix, [matrices], tf.float32)
            inputs['matrices'] = tf.reshape(extr_tf, shape=orig_shape)

            cam_pos = tf.gather_nd(raw_inputs['cam_pos'], indices)
            orig_shape = cam_pos.shape
            quaternion = tf.py_func(quaternion_from_campos_wrapper, [cam_pos], tf.float32)
            inputs['camera_quaternion'] = tf.reshape(quaternion, shape=[orig_shape[0], 4])

        return inputs
