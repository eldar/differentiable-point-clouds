import startup

import os
import sys
import os.path
import argparse

from util.common import parse_lines

script_dir = os.path.dirname(os.path.realpath(__file__))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_path", type=str)
    parser.add_argument("--python_interpreter", type=str, default="python3")
    parser.add_argument("--synth_set", type=str)
    parser.add_argument("--subset", type=str, default="val")
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args(sys.argv[1:])


def generate_commands():
    args = parse_arguments()
    synth_set = args.synth_set
    output_dir = os.path.join(os.getcwd(), args.output_dir)
    script_path = os.path.join(script_dir, "densify_single.py")

    prefix = "{} {} {} {} {} ".format(args.python_interpreter, script_path, args.shapenet_path, output_dir, synth_set)

    model_list = "splits/{}_{}.txt".format(synth_set, args.subset)
    models = parse_lines(model_list)

    with open("densify_{}_{}.txt".format(synth_set, args.subset), "w") as file:
        for l in models:
            file.write(prefix + l + "\n")


if __name__ == '__main__':
    generate_commands()