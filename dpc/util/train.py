import tensorflow as tf


def get_trainable_variables(scopes):
    is_trainable = lambda x: x in tf.trainable_variables()

    var_list = []

    for scope in scopes:
        var_list_raw = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        var_list_scope = list(filter(is_trainable, var_list_raw))
        var_list.extend(var_list_scope)

    return var_list


def get_learning_rate(cfg, global_step, add_summary=True):
    step_val = cfg.learning_rate_step * cfg.max_number_of_steps
    global_step = tf.cast(global_step, tf.float32)
    lr = tf.where(tf.less(global_step, step_val), cfg.learning_rate, cfg.learning_rate_2)
    if add_summary:
        tf.contrib.summary.scalar("learning_rate", lr)
    return lr
