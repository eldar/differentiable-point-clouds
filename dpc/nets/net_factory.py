import importlib


def get_network(name):
    m = importlib.import_module("nets.{}".format(name))
    return m.model
