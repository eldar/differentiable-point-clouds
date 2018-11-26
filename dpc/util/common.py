def build_command_line_args(pairs, as_string=True):
    if as_string:
        s = ""
    else:
        s = []
    for p in pairs:
        arg = None
        if type(p[1]) == bool:
            if p[1]:
                arg = f"--{p[0]}"
        else:
            arg = f"--{p[0]}={p[1]}"
        if arg:
            if as_string:
                s += arg + " "
            else:
                s.append(arg)
    return s


def parse_lines(filename):
    f = open(filename, "r")
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines