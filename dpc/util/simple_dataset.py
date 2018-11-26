import os
import sys
from collections import namedtuple

import numpy as np
import tensorflow as tf

from util.data import tf_record_options


def tf_records_dataset_filename(cfg):
    filename = '{}_{}.tfrecords'.format(cfg.synth_set, cfg.eval_split)
    return os.path.join(cfg.inp_dir, filename)


Model3D = namedtuple('Model3D', 'id, name, voxels, mask, image, camera, cam_pos, depth, num_views')


class Dataset3D:
    def __init__(self, cfg):
        self.quickie = None
        self.data = self.load_data(cfg)
        self.current_idx = 0
        self.epoch = None

    def load_data(self, cfg):
        image_size = cfg.image_size
        num_views = cfg.num_views

        tfrecords_filename = tf_records_dataset_filename(cfg)
        options = tf_record_options(cfg)
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename, options=options)

        data = []
        quickie = {}
        num_samples = cfg.num_dataset_samples

        for k, string_record in enumerate(record_iterator):
            if num_samples != -1 and k == num_samples:
                break

            example = tf.train.Example()
            example.ParseFromString(string_record)

            byte_list = example.features.feature['name'].bytes_list.value[0]
            model_name = byte_list.decode('UTF-8')
            # print("model", model_name)
            sys.stdout.write('.')
            sys.stdout.flush()

            float_list = example.features.feature['vox'].float_list.value
            vox_1d = np.array(float_list)
            vox_data_size = int(round(vox_1d.shape[0] ** (1.0/3.0)))
            voxels = vox_1d.reshape((vox_data_size, vox_data_size, vox_data_size))

            float_list = example.features.feature['image'].float_list.value

            images_1d = np.array(float_list)
            images = images_1d.reshape((num_views, image_size, image_size, -1))

            float_list = example.features.feature['mask'].float_list.value

            masks_1d = np.array(float_list)
            masks = masks_1d.reshape((num_views, image_size, image_size, -1))

            if 'depth' in list(example.features.feature.keys()):
                float_list = example.features.feature['depth'].float_list.value

                depths_1d = np.array(float_list)
                depths = depths_1d.reshape((num_views, image_size, image_size, -1))
            else:
                depths = None

            if 'cam_pos' in list(example.features.feature.keys()):
                float_list = example.features.feature['cam_pos'].float_list.value
                cam_pos_1d = np.array(float_list)
                cam_pos = cam_pos_1d.reshape((num_views, 3))
            else:
                cam_pos = None

            if cfg.saved_camera:
                float_list = example.features.feature['extrinsic'].float_list.value
                cam_1d = np.array(float_list)
                cameras = cam_1d.reshape((num_views, 4, 4))
            else:
                cameras = None

            if cfg.variable_num_views:
                float_list = example.features.feature['num_views'].float_list.value
                num_views_1d = np.array(float_list)
                num_views_actual = int(num_views_1d[0])
            else:
                num_views_actual = num_views

            model = Model3D(id=k, name=model_name, voxels=voxels,
                            mask=masks, image=images, camera=cameras, cam_pos=cam_pos,
                            depth=depths, num_views=num_views_actual)
            quickie[model_name] = model
            data.append(model)

        sys.stdout.write('\n')
        sys.stdout.flush()

        self.quickie = quickie
        return data

    def sample_by_name(self, key):
        return self.quickie[key]

    def num_samples(self):
        return len(self.data)

    def get_sample(self, idx):
        return self.data[idx]
