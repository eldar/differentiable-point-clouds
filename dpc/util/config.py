import os
import sys
import yaml
from easydict import EasyDict as edict


def _merge_a_into_b(a, b, type_conversion=False):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if not (type(a) is edict or type(a) is dict):
        raise TypeError(f"parameter a must be of dict type, got {type(a)} instead.")

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        param_type = type(b[k])
        if type_conversion:
            # still raise error if we supplied bool to a non-bool parameter:
            if type(v) == bool and param_type != bool:
                raise TypeError(
                    f"The type of the parameter \"{k}\" is \"{param_type.__name__}\", but a value {v} is supplied.")
            v = param_type(v)
        else:
            # the parameter types must match those in the default config file
            if type(v) != param_type:
                raise TypeError(f"The type of the parameter \"{k}\" is \"{param_type.__name__}\", but a value {v} is supplied.")

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

    return b


def config_from_file(filename):
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg


def parse_cmd_args(argv):
    args = {}
    for arg in argv:
        assert(arg[:2] == "--")
        arg = arg[2:]
        idx = arg.find("=")
        arg_name = arg[:idx]
        arg_val = arg[idx+1:]
        args[arg_name] = arg_val
    return args


def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_bool(s):
    s_l = s.lower()
    return s_l == "true" or s_l == "false"


def typify_args(cfg):
    for k, v in cfg.items():
        if _is_int(v):
            cfg[k] = int(v)
        elif _is_float(v):
            cfg[k] = float(v)
        elif _is_bool(v):
            v_l = v.lower()
            cfg[k] = True if v_l == "true" else False
        # leave as string in the remaining cases
    return cfg


def typify_args_bool_only(cfg):
    for k, v in cfg.items():
        if _is_bool(v):
            v_l = v.lower()
            cfg[k] = True if v_l == "true" else False
        # leave as string in the remaining cases
    return cfg


def merge_configs_recursive(configs):
    """
    Recursively merges configs starting with the default one
    """
    cfg = configs[-1]
    while "config" in cfg:
        filename = cfg["config"]
        cfg = config_from_file(filename)
        configs.append(cfg)

    config = config_from_file(DEFAULT_CONFIG)
    for cfg in reversed(configs):
        config = _merge_a_into_b(cfg, config)

    return config


def print_config(cfg):
    for k in sorted(cfg.keys()):
        print(f"{k} = {cfg[k]}")


def setup_config(cfg):
    return merge_configs_recursive([cfg])


def setup_config_with_cmd_args():
    """
    This is the main function that constructs a config object.
    When setting a value for the parameter it is first looked up in the command
    line arguments in the form --param_name=param_value . If not found, it is
    then looked up in yaml config file. Finally, if not found in the config
    file, a default parameter defined in default_config.py is used.
    """
    args = parse_cmd_args(sys.argv[1:])
    configs = [{}]
    if "config" in args:
        configs.append(config_from_file(args["config"]))
    elif os.path.isfile(CONFIG_DEFAULT_NAME):
        # try load default config file in the directory
        configs.append(config_from_file(CONFIG_DEFAULT_NAME))
    config = merge_configs_recursive(configs)

    cfg = typify_args_bool_only(args)
    config = _merge_a_into_b(cfg, config, type_conversion=True)
    print_config(config)
    return config


script_dir = os.path.dirname(os.path.realpath(__file__))
code_root = "{}/..".format(script_dir)
DEFAULT_CONFIG = os.path.join(code_root, "resources", "default_config.yaml")
CONFIG_DEFAULT_NAME = "config.yaml"
