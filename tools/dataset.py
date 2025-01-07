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

from utils.helper import rgb_anno_normalize


class Dataset(object):
    def __init__(self, config, path=None):
        np.random.seed(789)
        self.config = config

        list_examples_file = open(path, 'r')
        self.list_examples_cam = np.array(list_examples_file.read().splitlines())
        list_examples_file.close()

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

        rgb_norm, anno_resize = rgb_anno_normalize(rgb)
        rgb = rgb_norm(rgb)
        anno = anno_resize(anno)

        return {'rgb': rgb, 'rgb_orig': rgb_orig, 'anno': anno}
