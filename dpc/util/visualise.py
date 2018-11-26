import numpy as np

from skimage.transform import resize as imresize

from matplotlib import cm

from util.voxel import voxel2pc


def vis_voxels_matplotlib(voxels, vis_threshold):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    cond = voxels > vis_threshold
    colors[cond] = 'red'

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(cond, facecolors=colors, edgecolor='k')

    plt.show()


def vis_voxels(cfg, voxels, rgb=None, vis_axis=0):
    # TODO move to the other module and do import in the module
    import open3d
    threshold = cfg.vis_threshold
    xyz, occupancies = voxel2pc(voxels, threshold)

    # Pass xyz to Open3D.PointCloud and visualize
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(xyz)

    if rgb is not None:
        rgbs = np.reshape(rgb, (-1, 3))

        colors = rgbs[occupancies, :]
        colors = np.clip(colors, 0.0, 1.0)

        pcd.colors = open3d.Vector3dVector(colors)
    else:
        voxels = np.squeeze(voxels)
        sh = voxels.shape
        rgb = np.zeros((sh[0], sh[1], sh[2], 3), dtype=np.float32)
        for k in range(sh[0]):
            color = cm.gist_rainbow(float(k) / (sh[0] - 1))[:3]
            if vis_axis == 0:
                rgb[k, :, :, :] = color
            elif vis_axis == 1:
                rgb[:, k, :, :] = color
            elif vis_axis == 2:
                rgb[:, :, k, :] = color
            else:
                assert(False)

        rgbs = np.reshape(rgb, (-1, 3))
        colors = rgbs[occupancies, :]
        pcd.colors = open3d.Vector3dVector(colors)

    if False:
        axis_vis = xyz[:, 0]
        min_ = np.min(axis_vis)
        max_ = np.max(axis_vis)
        colors = cm.gist_rainbow((axis_vis - min_) / (max_ - min_))[:, 0:3]
        pcd.colors = open3d.Vector3dVector(colors)

    # open3d.write_point_cloud("sync.ply", pcd)

    # Load saved point cloud and transform it into NumPy array
    # pcd_load = open3d.read_point_cloud("sync.ply")
    # xyz_load = np.asarray(pcd_load.points)
    # print(xyz_load)

    # visualization
    open3d.draw_geometries([pcd])


def vis_pc(xyz, color_axis=-1, rgb=None):
    # TODO move to the other module and do import in the module
    import open3d
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(xyz)

    if color_axis >= 0:
        if color_axis == 3:
            axis_vis = np.arange(0, xyz.shape[0], dtype=np.float32)
        else:
            axis_vis = xyz[:, color_axis]
        min_ = np.min(axis_vis)
        max_ = np.max(axis_vis)

        colors = cm.gist_rainbow((axis_vis - min_) / (max_ - min_))[:, 0:3]
        pcd.colors = open3d.Vector3dVector(colors)
    if rgb is not None:
        pcd.colors = open3d.Vector3dVector(rgb)

    open3d.draw_geometries([pcd])


def mask4vis(cfg, curr_img, vis_size):
    curr_img = np.clip(curr_img, 0.0, 1.0)
    curr_img = imresize(curr_img, (vis_size, vis_size), order=3)
    curr_img = np.clip(curr_img * 255, 0, 255).astype(dtype=np.uint8)
    if curr_img.shape[-1] != 3 and not cfg.vis_depth_projs:
        curr_img = 255 - curr_img
    return curr_img


def merge_grid(cfg, grid):
    vis_size = grid[0, 0].shape[0]
    empty = np.ones((vis_size, vis_size, 3), dtype=np.uint8) * 255
    for j in range(grid.shape[0]):
        out_row = np.zeros((vis_size, 0, 3), dtype=np.uint8)
        for i in range(grid.shape[1]):
            img = grid[j, i]
            if img is None:
                img = empty
            elif img.shape[-1] != 3:
                img = np.expand_dims(img, axis=2)
                img = np.concatenate([img]*3, axis=2)
            out_row = np.concatenate([out_row, img], axis=1)
        if j == 0:
            out = out_row
        else:
            out = np.concatenate([out, out_row], axis=0)
    return out


def list_to_grid(grid, row_major=True):
    num_rows = len(grid)
    num_cols = len(grid[0])
    if not row_major:
        num_rows, num_cols = num_cols, num_rows
    grid_out = np.empty((num_rows, num_cols), dtype=object)

    for j in range(num_rows):
        for i in range(num_cols):
            grid_out[j, i] = grid[j][i] if row_major else grid[i][j]
    return grid_out