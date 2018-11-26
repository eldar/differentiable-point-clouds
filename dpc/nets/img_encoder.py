import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


def _preprocess(images):
    return images * 2 - 1


def model(images, cfg, is_training):
    """Model encoding the images into view-invariant embedding."""
    del is_training  # Unused
    image_size = images.get_shape().as_list()[1]
    target_spatial_size = 4

    f_dim = cfg.f_dim
    fc_dim = cfg.fc_dim
    z_dim = cfg.z_dim
    outputs = dict()

    act_func = tf.nn.leaky_relu

    images = _preprocess(images)
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
        batch_size = images.shape[0]
        hf = slim.conv2d(images, f_dim, [5, 5], stride=2, activation_fn=act_func)

        num_blocks = int(np.log2(image_size / target_spatial_size) - 1)

        for k in range(num_blocks):
            f_dim = f_dim * 2
            hf = slim.conv2d(hf, f_dim, [3, 3], stride=2, activation_fn=act_func)
            hf = slim.conv2d(hf, f_dim, [3, 3], stride=1, activation_fn=act_func)

        # Reshape layer
        rshp0 = tf.reshape(hf, [batch_size, -1])
        outputs["conv_features"] = rshp0

        fc1 = slim.fully_connected(rshp0, fc_dim, activation_fn=act_func)
        fc2 = slim.fully_connected(fc1, fc_dim, activation_fn=act_func)
        fc3 = slim.fully_connected(fc2, z_dim, activation_fn=act_func)

        outputs["z_latent"] = fc1
        outputs['ids'] = fc3
        if cfg.predict_pose:
            outputs['poses'] = slim.fully_connected(fc2, z_dim)
    return outputs


def decoder_part(input, cfg):
    batch_size = input.shape.as_list()[0]
    fake_input = tf.zeros([batch_size, 128*4*4])
    act_func = tf.nn.leaky_relu

    fc_dim = cfg.fc_dim
    z_dim = cfg.z_dim

    # this is unused but needed to match the FC layers in the encoder function
    fc1 = slim.fully_connected(fake_input, fc_dim, activation_fn=act_func)

    fc2 = slim.fully_connected(input, fc_dim, activation_fn=act_func)
    fc3 = slim.fully_connected(fc2, z_dim, activation_fn=act_func)
    return fc3
