import os


def setup_environment(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
