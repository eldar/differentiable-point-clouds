import os


def mkdir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)
