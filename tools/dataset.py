#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataloader python script

Created on 29th Dec. 2024
"""
import os
import sys
import cv2
import random
import numpy as np
from glob import glob
from PIL import Image

import torch
from torchvision.transforms import v2
import torchvision.transforms.functional as TF


class Dataset(object):
    def __init__(self, config, path=None):
        np.random.seed(789)
        self.config = config

        list_examples_file = open(path, 'r')
        self.list_examples_cam = np.array(list_examples_file.read().splitlines())
        list_examples_file.close()

    def rgb_anno_normalize(self, rgb):
        """
        Resize the input image to specified size. Keep same aspect ratio and pad the gap.
        i.e. first 752x480 -> 640x408, then padding the top and bottom to 640x480.
        :param rgb:
        :return:
        """
        rgb_orig = rgb.copy()
        w_orig, h_orig = rgb_orig.size  # PIL (752, 480)
        rgb_set_size = self.config['mlnet_input_size']  # from config (640, 480)

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
                v2.Normalize(mean=self.config['Dataset']['transforms']['image_mean'],
                             std=self.config['Dataset']['transforms']['image_mean'])
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
                v2.Normalize(mean=self.config['Dataset']['transforms']['image_mean'],
                             std=self.config['Dataset']['transforms']['image_mean'])])

            anno_resize = v2.Compose([
                v2.Resize(new_size, interpolation=v2.InterpolationMode.BILINEAR),
                v2.Pad((int((rgb_set_size[0] - new_size[1]) / 2), 0,
                        int((rgb_set_size[0] - new_size[1]) / 2), 0)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ])
        return rgb_normalize, anno_resize

    def __len__(self):
        return len(self.list_examples_cam)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        dataroot = './dataset/smirk/'

        if self.config['Dataset']['name'] == 'smirk':
            cam_path = os.path.join(dataroot, self.list_examples_cam[idx])
            anno_path = cam_path.replace('.png', '.labels.png')

            # smirk rgb and anno is in 752x480
            rgb = Image.open(cam_path).convert('RGB')

            anno = Image.open(anno_path)

        elif self.config['Dataset']['name'] == 'zod':
            sys.exit()

        else:
            sys.exit("['Dataset']['name'] must be specified smirk or zod")

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        anno_name = anno_path.split('/')[-1].split('.')[0]
        assert (rgb_name == anno_name), "rgb and anno input not matching"

        rgb_orig = rgb.copy()

        rgb_norm, anno_resize = self.rgb_anno_normalize(rgb)
        rgb = rgb_norm(rgb)
        anno = anno_resize(anno)

        return {'rgb': rgb, 'rgb_orig': rgb_orig, 'anno': anno}
