"""
Authors: Chen-Hsuan Lin
https://github.com/chenhsuanlin/3D-point-cloud-generation/
"""

import os
import sys
import utils as util
import time
import os.path

import scipy.io
import numpy as np


densifyN = 100000
new_format = True
shapenet_v2 = False

if shapenet_v2:
    model_file = "models/model_normalized.obj"
else:
    model_file = "model.obj"


def densify_model(model_name, output_path, shapenet_dir, category):
    save_filename = "{0}/{1}.mat".format(output_path, model_name)
    if os.path.isfile(save_filename):
        print("already computed", model_name)
        return

    timeStart = time.time()

    shape_file = os.path.join(shapenet_dir, category, model_name, model_file)
    print("processing file", shape_file)
    V,E,F = util.parseObj(shape_file)
    F = util.removeWeirdDuplicate(F)
    Vorig, Eorig, Forig = V.copy(), E.copy(), F.copy()

    # sort by length (maintain a priority queue)
    Elist = list(range(len(E)))
    Elist.sort(key=lambda i: util.edgeLength(V, E, i), reverse=True)

    # create edge-to-triangle and triangle-to-edge lists
    EtoF = [[] for j in range(len(E))]
    FtoE = [[] for j in range(len(F))]
    for f in range(len(F)):
        v = F[f]
        util.pushEtoFandFtoE(EtoF,FtoE,E,f,v[0],v[1])
        util.pushEtoFandFtoE(EtoF,FtoE,E,f,v[0],v[2])
        util.pushEtoFandFtoE(EtoF,FtoE,E,f,v[1],v[2])
    V,E,F = list(V),list(E),list(F)

    # repeat densification
    for z in range(densifyN):
        util.densify(V, E, F, EtoF, FtoE, Elist)

    densifyV = np.array(V[-densifyN:])

    if new_format:
        Vgt = np.concatenate([Vorig, densifyV], axis=0)
        scipy.io.savemat(save_filename, {"points": Vgt})
    else:
        scipy.io.savemat(save_filename, {
            "V": Vorig,
            "E": Eorig,
            "F": Forig,
            "Vd": densifyV
        })

    print("{0} done, time = {1:.6f} sec".format(model_name, time.time() - timeStart))


if __name__ == "__main__":
    SHAPENET_PATH = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    CATEGORY = sys.argv[3]
    MODEL_NAME = sys.argv[4]

    output_path = os.path.join(OUTPUT_DIR, CATEGORY)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    densify_model(MODEL_NAME, output_path, SHAPENET_PATH, CATEGORY)
