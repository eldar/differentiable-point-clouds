import numpy as np


def white_background(rgb, mask):
    mask_fg = np.repeat(mask, 3, 2)
    mask_bg = 1.0 - mask_fg
    return rgb * mask_fg + np.ones(rgb.shape) * 255.0 * mask_bg


def preprocess_input_image(image):
    rgb = image[:, :, 0:3]
    mask = image[:, :, [3]]
    mask = mask / 255.0
    rgb = white_background(rgb, mask)
    rgb = rgb / 255.0
    return rgb, mask