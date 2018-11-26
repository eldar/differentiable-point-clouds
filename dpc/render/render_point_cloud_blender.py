"""
Code adapted from the rendering code by Maxim Tatarchenko
Original version here: https://github.com/lmb-freiburg/ogn/blob/master/python/rendering/render_model.py 
"""

import sys
import os.path
import argparse
import math
import numpy as np
import bmesh
import bpy


particle_materials = []
particle_prototypes = []


def clear_selection():
    bpy.context.scene.objects.active = None
    for o in bpy.data.objects:
        o.select = False


def select_object(obj):
    clear_selection()
    bpy.context.selected_objects.clear()
    bpy.context.scene.objects.active = obj
    obj.select = True
    return obj


def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)


def setup_camera(azimuth=140.0, elevation=15.0, dist=1.2):
    # set position
    scene = bpy.data.scenes["Scene"]
    x, y, z = obj_centened_camera_pos(dist, azimuth, elevation)
    scene.camera.location.x = y
    scene.camera.location.y = x
    scene.camera.location.z = z
    #scene.camera.location.x = 0.5
    #scene.camera.location.y = -1.0
    #scene.camera.location.z = 0.5

    bpy.data.cameras["Camera"].clip_end = 10000.

    # track invisible cube at (0, 0, 0)
    bpy.data.objects['Cube'].hide_render = True
    ttc = scene.camera.constraints.new(type='TRACK_TO')
    ttc.target = bpy.data.objects['Cube']
    ttc.track_axis = 'TRACK_NEGATIVE_Z'
    ttc.up_axis = 'UP_Y'


def setup_general(filename, im_size, cycles_samples, like_train_data):
    bpy.data.worlds["World"].horizon_color = (1, 1, 1)
    bpy.data.scenes["Scene"].render.engine = "CYCLES"
    bpy.data.scenes["Scene"].cycles.samples = cycles_samples
    bpy.data.scenes["Scene"].use_nodes = True
    bpy.data.scenes["Scene"].render.filepath = filename
    bpy.data.scenes["Scene"].render.use_compositing = False
    bpy.data.scenes["Scene"].render.layers["RenderLayer"].use_pass_z = False

    #prefs = bpy.context.user_preferences.addons['cycles'].preferences
    #print("device type", prefs.compute_device_type)
    #prefs.compute_device_type = "CUDA"
    #bpy.context.scene.cycles.device = 'GPU'

    #for d in prefs.devices:
    #    print(d.name)

    bpy.data.scenes["Scene"].render.resolution_x = im_size * (100 / bpy.context.scene.render.resolution_percentage)
    bpy.data.scenes["Scene"].render.resolution_y = im_size * (100 / bpy.context.scene.render.resolution_percentage)

    if like_train_data:
        camObj = bpy.data.objects['Camera']
        camObj.data.lens = 60  # 60 mm focal length
        camObj.data.sensor_height = 32.0
        camObj.data.sensor_width = float(camObj.data.sensor_height) / im_size * im_size

    bpy.data.screens['Default'].scene = bpy.data.scenes['Scene']

    bpy.ops.render.render(write_still=True)


def setup_particle_prototypes(colors):
    col = 0.5
    color = (col, col, col)
    add_prototype(0, color)
    if colors is not None:
        for i in range(colors.shape[0]):
            color = colors[i, :]
            add_prototype(i+1, color)


def add_prototype(level, color):
    mat = bpy.data.materials.new('cube_material_%02d' % (level))
    mat.diffuse_color = color
    particle_materials.append(mat)
    """
    bpy.ops.mesh.primitive_cube_add(location=(-1000, -1000, -1000),
                                    rotation=(0, 0, 0),
                                    radius=1 - .00625)
    """
    bpy.ops.mesh.primitive_uv_sphere_add(location=(-1000, -1000, -1000),
                                         rotation=(0, 0, 0))

    prototype = bpy.context.object
    select_object(prototype)
    prototype.name = 'proto_cube_level_%01d' % (level)
    bpy.ops.object.material_slot_add()
    prototype.material_slots[0].material = mat
    bpy.ops.object.modifier_add(type='BEVEL')
    prototype.modifiers['Bevel'].width = 0.2
    prototype.modifiers['Bevel'].segments = 3
    particle_prototypes.append(prototype)


def load_data(file_name, out_dir, subset_indices):
    file_type = os.path.splitext(file_name)[1]
    if file_type == ".mat":
        import scipy.io
        all_pcs = scipy.io.loadmat(file_name)["points"]
    else:
        all_pcs = np.load(file_name)["arr_0"]
    vis_idx = 0
    model = np.squeeze(all_pcs[vis_idx, :, :])

    DEFAULT_SIZE = 0.01

    if subset_indices is None:
        add_points(model, 0, DEFAULT_SIZE, out_dir)
    else:
        any_coloured = np.any(subset_indices, axis=0)
        the_rest_ids = np.logical_not(any_coloured)
        the_rest = model[the_rest_ids, :]
        add_points(the_rest, 0, DEFAULT_SIZE * 0.75, out_dir)
        for i in range(subset_indices.shape[0]):
            add_points(model[subset_indices[i, :], :], i + 1, DEFAULT_SIZE, out_dir)


def add_points(model, proto_id, point_size, out_dir):
    is_mvc = True
    if is_mvc:
        # model = model[:, [2, 1, 0]]
        ax = 0
        model[:, 0] = -model[:, 0]

    global_scale = 1.0

    ply_vertices = []
    for k in range(model.shape[0]):
        x = model[k, 2]
        y = model[k, 0]
        z = model[k, 1]
        ply_vertices.append('%f %f %f' % (x * global_scale, y * global_scale, z * global_scale))

    ply_template = """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
end_header
%s"""

    path = "{}/tmp-ply.ply".format(out_dir)
    with open(path, 'w') as f:
        f.write(ply_template % (len(ply_vertices),
                                '\n'.join(ply_vertices)))
    bpy.ops.import_mesh.ply(filepath=path)
    bpy.ops.object.particle_system_add()

    os.remove(path)

    ps = bpy.data.particles[-1]
    ps.count = len(ply_vertices)
    ps.emit_from = 'VERT'
    ps.use_emit_random = False
    ps.normal_factor = 0.0
    ps.physics_type = 'NO'
    ps.use_render_emitter = False
    ps.show_unborn = True
    ps.use_dead = False
    ps.lifetime = 250
    ps.render_type = 'OBJECT'
    ps.dupli_object = particle_prototypes[proto_id]
    ps.particle_size = point_size * global_scale


def voxel2pc(voxels, threshold):
    voxels = np.squeeze(voxels)
    vox = voxels > threshold
    vox = np.squeeze(vox)
    vox_size = vox.shape[0]

    # generate some neat n times 3 matrix using a variant of sync function
    x = np.linspace(-0.5, 0.5, vox_size)
    mesh_x, mesh_y, mesh_z = np.meshgrid(x, x, x)
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(mesh_z, -1)

    occupancies = np.reshape(vox, -1)
    xyz = xyz[occupancies, :]
    return xyz, occupancies


def load_voxels(file_name, vis_threshold, out_dir):
    stuff = np.load(file_name)["arr_0"]

    vis_idx = 0
    voxels = np.squeeze(stuff[vis_idx, :, :, :, :])

    model, _ = voxel2pc(voxels, vis_threshold)

    DEFAULT_SIZE = 0.01
    add_points(model, 0, DEFAULT_SIZE, out_dir)


def parse_arguments():
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--vis_azimuth", type=float, default=140.0)
    parser.add_argument("--vis_elevation", type=float, default=15.0)
    parser.add_argument("--vis_dist", type=float, default=1.2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--cycles_samples", type=int, default=100)
    parser.add_argument("--colored_subsets", action="store_true")
    parser.add_argument("--voxels", action="store_true")
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--like_train_data", action="store_true")

    return parser.parse_args(argv)


def main():
    args = parse_arguments()

    in_file = args.in_file
    out_file = args.out_file

    in_dir = os.path.dirname(in_file)
    out_dir = os.path.dirname(out_file)

    if args.colored_subsets:
        stuff = np.load("{}/coloured_subsets.npz".format(in_dir))
        subset_indices = stuff["arr_0"]
        subset_colors = stuff["arr_1"]
    else:
        subset_indices = None
        subset_colors = None

    setup_camera(args.vis_azimuth, args.vis_elevation, args.vis_dist)
    setup_particle_prototypes(subset_colors)
    if args.voxels:
        load_voxels(in_file, args.vis_threshold, out_dir)
    else:
        load_data(in_file, out_dir, subset_indices)
    setup_general(out_file, args.image_size, args.cycles_samples, args.like_train_data)


if __name__ == '__main__':
    main()