import tensorflow as tf
import tensorflow.contrib.slim as slim


def pose_branch(inputs, cfg):
    num_layers = cfg.pose_candidates_num_layers
    f_dim = 32
    t = inputs
    for k in range(num_layers):
        if k == (num_layers - 1):
            out_dim = 4
            act_func = None
        else:
            out_dim = f_dim
            act_func = tf.nn.leaky_relu
        t = slim.fully_connected(t, out_dim, activation_fn=act_func)
    return t


def model(inputs, cfg):
    """predict pose quaternions
    inputs: [B,Z]
    """

    w_init = tf.contrib.layers.variance_scaling_initializer()

    out = {}
    with slim.arg_scope(
            [slim.fully_connected],
            weights_initializer=w_init):
        with tf.variable_scope('predict_pose', reuse=tf.AUTO_REUSE):
            num_candidates = cfg.pose_predict_num_candidates
            if num_candidates > 1:
                outs = [pose_branch(inputs, cfg) for _ in range(num_candidates)]
                q = tf.concat(outs, axis=1)
                q = tf.reshape(q, [-1, 4])
                if cfg.pose_predictor_student:
                    out["pose_student"] = pose_branch(inputs, cfg)
            else:
                q = slim.fully_connected(inputs, 4, activation_fn=None)

            if cfg.predict_translation:
                trans_init_stddev = cfg.predict_translation_init_stddev
                w_trans_init = tf.truncated_normal_initializer(stddev=trans_init_stddev, seed=1)
                t = slim.fully_connected(inputs, 3,
                                         activation_fn=None,
                                         weights_initializer=w_trans_init)
                if cfg.predict_translation_tanh:
                    t = tf.tanh(t) * cfg.predict_translation_scaling_factor
            else:
                t = None

    out["poses"] = q
    out["predicted_translation"] = t

    return out
