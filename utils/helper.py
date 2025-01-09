#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
from torchvision.transforms import v2


def rgb_anno_normalize(rgb, config, norm=True):
    """
    Resize the input image to specified size. Keep same aspect ratio and pad the gap.
    i.e. first 752x480 -> 640x408, then padding the top and bottom to 640x480.
    :param norm:
    :param rgb:
    :param config:
    :return:
    """
    if norm is True:
        mean = config['Dataset']['transforms']['image_mean']
        std = config['Dataset']['transforms']['image_mean']
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    rgb_orig = rgb.copy()
    w_orig, h_orig = rgb_orig.size  # PIL (752, 480)
    rgb_set_size = config['mlnet_input_size']  # from config (640, 480)

    width_ratio = w_orig / rgb_set_size[0]  # 752/640=1.175
    height_ratio = h_orig / rgb_set_size[1]  # 480/480=1

    if width_ratio > height_ratio:  # width is reference, padding height (top and bottom)
        new_size = (int(rgb_set_size[1] / width_ratio), int(rgb_set_size[0]))  # (408, 640)
        rgb_normalize = v2.Compose([
            v2.Resize(new_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.Pad((0, int((rgb_set_size[1] - new_size[0]) / 2),  # pad (left,top,right,bottom)->(0,36,0,36)
                    0, int((rgb_set_size[1] - new_size[0]) / 2))),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std)
        ])
        anno_resize = v2.Compose([
            v2.Resize(new_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.Pad((0, int((rgb_set_size[1] - new_size[0]) / 2),  # pad (left,top,right,bottom)->(0,36,0,36)
                    0, int((rgb_set_size[1] - new_size[0]) / 2))),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    else:  # height is reference, padding width (left and right)
        new_size = (int(rgb_set_size[1]), int(rgb_set_size[0] / height_ratio))
        rgb_normalize = v2.Compose([
            v2.Resize(new_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.Pad((int((rgb_set_size[0] - new_size[1]) / 2), 0,
                    int((rgb_set_size[0] - new_size[1]) / 2), 0)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std)
        ])

        anno_resize = v2.Compose([
            v2.Resize(new_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.Pad((int((rgb_set_size[0] - new_size[1]) / 2), 0,
                    int((rgb_set_size[0] - new_size[1]) / 2), 0)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    return rgb_normalize, anno_resize


def save_heatmap(heatmap, rgb_img, size, save_path):
    """
    Save the neural network's heatmap output as single PNG for visualization.
    Save two kinds of result, one is grayscale and one is overlaying.
    Same path and directory tree as the input.
    :param size: input size as defined in config
    :param heatmap: float32 HxWx1
    :return:
    """
    heatmap = heatmap / np.max(heatmap + 1e-6) * 255
    heatmap = cv2.resize(heatmap, size, interpolation=cv2.INTER_LINEAR)
    heatmap = np.tile(np.expand_dims(np.uint8(heatmap), axis=-1), (1, 1, 3))
    grayscale_path = save_path.replace('datasets/smirk', 'outputs/smirk_heatmap')
    if not os.path.exists(os.path.dirname(grayscale_path)):
        os.makedirs(os.path.dirname(grayscale_path))
    cv2.imwrite(grayscale_path, heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    overlay = image_overlay(np.uint8(rgb_img), heatmap_color)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_path = save_path.replace('datasets/smirk', 'outputs/smirk_overlay')
    if not os.path.exists(os.path.dirname(overlay_path)):
        os.makedirs(os.path.dirname(overlay_path))
    cv2.imwrite(overlay_path, overlay)


def image_overlay(image, layer_image):
    alpha = 0.3  # how much transparency to apply
    beta = 1 - alpha  # alpha + beta should equal 1
    gamma = 0  # scalar added to each sum
    cv2.addWeighted(layer_image, alpha, image, beta, gamma, image)
    return image

#
# def creat_dir(data_list, root_path):
#     """
#     :param root_path: the root path for different dataset
#     :param data_list: The array of all files
#     """
#     for path in data_list:
#         file_path = os.path.join(root_path, path).replace('datasets', 'outputs')
#         dir_path = os.path.dirname(file_path)
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)


