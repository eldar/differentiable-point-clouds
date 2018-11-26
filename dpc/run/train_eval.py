#!/usr/bin/env python

import startup

import tensorflow as tf

from run import train
from run.predict_eval import compute_eval


def main(_):
    train.train()
    compute_eval()


if __name__ == '__main__':
    tf.app.run()
