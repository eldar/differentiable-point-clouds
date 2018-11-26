#!/usr/bin/env python

import startup

import os
import time

import tensorflow as tf

from models import model_pc

from util.app_config import config as app_config
from util.system import setup_environment
from util.train import get_trainable_variables, get_learning_rate
from util.losses import regularization_loss
from util.fs import mkdir_if_missing
from util.data import tf_record_compression

tfsum = tf.contrib.summary


def parse_tf_records(cfg, serialized):
    num_views = cfg.num_views
    image_size = cfg.image_size

    # A dictionary from TF-Example keys to tf.FixedLenFeature instance.
    features = {
        'image': tf.FixedLenFeature([num_views, image_size, image_size, 3], tf.float32),
        'mask': tf.FixedLenFeature([num_views, image_size, image_size, 1], tf.float32),
    }

    if cfg.saved_camera:
        features.update(
            {'extrinsic': tf.FixedLenFeature([num_views, 4, 4], tf.float32),
             'cam_pos': tf.FixedLenFeature([num_views, 3], tf.float32)})
    if cfg.saved_depth:
        features.update(
            {'depth': tf.FixedLenFeature([num_views, image_size, image_size, 1], tf.float32)})

    return tf.parse_single_example(serialized, features)


def train():
    cfg = app_config

    setup_environment(cfg)

    train_dir = cfg.checkpoint_dir
    mkdir_if_missing(train_dir)

    tf.logging.set_verbosity(tf.logging.INFO)

    split_name = "train"
    dataset_file = os.path.join(cfg.inp_dir, f"{cfg.synth_set}_{split_name}.tfrecords")

    dataset = tf.data.TFRecordDataset(dataset_file, compression_type=tf_record_compression(cfg))
    if cfg.shuffle_dataset:
        dataset = dataset.shuffle(7000)
    dataset = dataset.map(lambda rec: parse_tf_records(cfg, rec), num_parallel_calls=4) \
        .batch(cfg.batch_size) \
        .prefetch(buffer_size=100) \
        .repeat()

    iterator = dataset.make_one_shot_iterator()
    train_data = iterator.get_next()

    summary_writer = tfsum.create_file_writer(train_dir, flush_millis=10000)

    with summary_writer.as_default(), tfsum.record_summaries_every_n_global_steps(10):
        global_step = tf.train.get_or_create_global_step()
        model = model_pc.ModelPointCloud(cfg, global_step)
        inputs = model.preprocess(train_data, cfg.step_size)

        model_fn = model.get_model_fn(
            is_training=True, reuse=False, run_projection=True)
        outputs = model_fn(inputs)

        # train_scopes
        train_scopes = ['encoder', 'decoder']

        # loss
        task_loss = model.get_loss(inputs, outputs)
        reg_loss = regularization_loss(train_scopes, cfg)
        loss = task_loss + reg_loss

        # summary op
        summary_op = tfsum.all_summary_ops()

        # optimizer
        var_list = get_trainable_variables(train_scopes)
        optimizer = tf.train.AdamOptimizer(get_learning_rate(cfg, global_step))
        train_op = optimizer.minimize(loss, global_step, var_list)

    # saver
    max_to_keep = 2
    saver = tf.train.Saver(max_to_keep=max_to_keep)

    session_config = tf.ConfigProto(
        log_device_placement=False)
    session_config.gpu_options.allow_growth = cfg.gpu_allow_growth
    session_config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction

    sess = tf.Session(config=session_config)
    with sess, summary_writer.as_default():
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        tfsum.initialize(graph=tf.get_default_graph())

        global_step_val = 0
        while global_step_val < cfg.max_number_of_steps:
            t0 = time.perf_counter()
            _, loss_val, global_step_val, summary = sess.run([train_op, loss, global_step, summary_op])
            t1 = time.perf_counter()
            dt = t1 - t0
            print(f"step: {global_step_val}, loss = {loss_val:.4f} ({dt:.3f} sec/step)")
            if global_step_val % 5000 == 0:
                saver.save(sess, f"{train_dir}/model", global_step=global_step_val)


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
