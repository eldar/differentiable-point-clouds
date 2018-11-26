import tensorflow as tf

from util.gauss_kernel import gauss_smoothen_image


def regularization_loss(scopes, cfg, postfix=""):
    reg_loss = tf.zeros(dtype=tf.float32, shape=[])
    if cfg.weight_decay > 0:
        def is_weights(x):
            return 'kernel' in x.name or 'weights' in x.name

        for scope in scopes:
            scope_vars = [x for x in tf.trainable_variables() if x.name.startswith(scope)]
            scope_vars_w = list(filter(is_weights, scope_vars))
            if scope_vars_w:
                reg_loss += tf.add_n([tf.nn.l2_loss(var) for var in scope_vars_w])

    tf.contrib.summary.scalar("losses/reg_loss" + postfix, reg_loss)
    reg_loss *= cfg.weight_decay
    return reg_loss


def drc_loss(cfg, probs, gt_proj):
    gt_proj2 = tf.expand_dims(gt_proj, axis=0)
    gt_proj_fg = 1-tf.tile(gt_proj2, [cfg.vox_size, 1, 1, 1, 1])
    gt_proj_bg = gt_proj2
    psi = tf.concat([gt_proj_fg, gt_proj_bg], axis=0)

    return tf.reduce_sum(probs * psi) #/ tf.to_float(vox_size)


def drc_rgb_loss(cfg, probs, rgb, gt):
    vox_size = cfg.vox_size
    gt_proj2 = tf.expand_dims(gt, axis=1)
    gt_vol = tf.tile(gt_proj2, [1, vox_size+1, 1, 1, 1])

    num_samples = rgb.shape[0]
    white_bg = tf.ones([num_samples, 1, vox_size, vox_size, 3])
    rgb_pred = tf.concat([rgb, white_bg], axis=1)

    probs = tf.transpose(probs, [1, 0, 2, 3, 4])

    psi = tf.square(gt_vol - rgb_pred)
    psi = tf.reduce_sum(psi, axis=4, keep_dims=True)

    return tf.reduce_sum(probs * psi) #/ tf.to_float(vox_size)


def add_drc_loss(cfg, inputs, outputs, weight_scale, add_summary=True):
    """Computes the projection loss of voxel generation model.
    """
    gt = inputs['masks']
    pred = outputs['drc_probs']
    num_samples = gt.shape[0]

    gt_size = gt.shape[1]
    pred_size = pred.shape[2]
    if gt_size != pred_size:
        gt = tf.image.resize_images(gt, [pred_size, pred_size])

    loss = drc_loss(cfg, pred, gt)
    loss /= tf.to_float(num_samples)
    if add_summary:
        tf.contrib.summary.scalar("losses/drc_loss", loss)
    loss *= weight_scale
    return loss


def add_proj_rgb_loss(cfg, inputs, outputs, weight_scale, add_summary=True, sigma=None):
    gt = inputs['images']
    pred = outputs['projs_rgb']
    num_samples = pred.shape[0]

    gt_size = gt.shape[1]
    pred_size = pred.shape[1]
    if gt_size != pred_size:
        gt = tf.image.resize_images(gt, [pred_size, pred_size])
    if cfg.pc_gauss_filter_gt_rgb:
        smoothed = gauss_smoothen_image(cfg, gt, sigma)
        if cfg.pc_gauss_filter_gt_switch_off:
            gt = tf.where(tf.less(sigma, 1.0), gt, smoothed)
        else:
            gt = smoothed

    proj_loss = tf.nn.l2_loss(gt - pred)
    proj_loss /= tf.to_float(num_samples)
    if add_summary:
        tf.contrib.summary.scalar("losses/proj_rgb_loss", proj_loss)
    proj_loss *= weight_scale
    return proj_loss


def add_drc_rgb_loss(cfg, inputs, outputs, weight_scale, add_summary=True):
    gt = inputs['images']
    drc_probs = outputs['drc_probs']
    pred = outputs['voxels_rgb']

    num_samples = pred.shape[0]

    gt_size = gt.shape[1]
    pred_size = pred.shape[1]
    if gt_size != pred_size:
        gt = tf.image.resize_images(gt, [pred_size, pred_size])

    loss = drc_rgb_loss(cfg, drc_probs, pred, gt)
    loss /= tf.to_float(num_samples)
    if add_summary:
        tf.contrib.summary.scalar("losses/drc_rgb_loss", loss)
    loss *= weight_scale
    return loss


def add_proj_depth_loss(cfg, inputs, outputs, weight_scale, sigma_rel, add_summary=True):
    gt = inputs['depths']
    pred = outputs['projs_depth']
    num_samples = pred.shape[0]

    gt_size = gt.shape[1]
    pred_size = pred.shape[1]

    if cfg.max_depth != cfg.max_dataset_depth:
        gt_pos = tf.cast(tf.not_equal(gt, cfg.max_dataset_depth), tf.float32)
        gt_neg = tf.cast(tf.equal(gt, cfg.max_dataset_depth), tf.float32)
        gt = gt_pos * gt + gt_neg * cfg.max_depth

    if gt_size != pred_size:
        gt = tf.image.resize_images(gt, [pred_size, pred_size], method=tf.ResizeMethod.NEAREST_NEIGHBOR)
    if cfg.pc_gauss_filter_gt:
        gt = gauss_smoothen_image(cfg, gt, sigma_rel)

    proj_loss = tf.nn.l2_loss(gt - pred)
    proj_loss /= tf.to_float(num_samples)
    if add_summary:
        tf.contrib.summary.scalar("losses/proj_loss", proj_loss)
    proj_loss *= weight_scale
    return proj_loss
