import numpy as np
import math


def partition_range(total, num_parts):
    ranges = np.zeros((num_parts, 2), dtype=np.int32)
    size = int(math.ceil(total / num_parts))
    for k in range(num_parts):
        ranges[k, 0] = k * size
        ranges[k, 1] = min((k+1) * size, total)
    return ranges


def to_np_object(stuff):
    length = len(stuff)
    my_list = np.zeros((length,), dtype=np.object)
    my_list[:] = stuff
    return my_list
