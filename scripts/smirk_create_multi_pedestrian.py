#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script to read frame from smirk dataset, mirror the pedestrian (smirk only has single pedestrian),
then save the result to have 'synthetic' multi-pedestrian sequence.

Created on 12nd Jan, 2025
"""

import os
import sys
import cv2
import argparse

import numpy as np


def run(rgb_path_dir):
    rgb_path = os.path.join('../datasets/smirk', rgb_path_dir)
    anno_path = rgb_path.replace('.png', '.labels.png')
    bgr = cv2.imread(rgb_path)
    anno_gray = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
    ret, anno_binary = cv2.threshold(anno_gray, 0, 255, cv2.THRESH_BINARY)

    bgr_flip = cv2.flip(bgr, 1)
    anno_flip = cv2.flip(anno_binary, 1)

    new_anno = np.zeros_like(anno_flip)
    ped_mask_orig = anno_binary == 255
    new_anno[ped_mask_orig] = 255
    ped_mask_flip = anno_flip == 255
    new_anno[ped_mask_flip] = 255
    new_anno_3_channel = cv2.merge((new_anno, new_anno, new_anno))

    bgr[ped_mask_flip] = bgr_flip[ped_mask_flip]

    new_bgr_path = rgb_path.replace('smirk', 'smirk_multi_ped')
    new_anno_path = anno_path.replace('smirk', 'smirk_multi_ped')
    if not os.path.exists(os.path.dirname(new_bgr_path)):
        os.makedirs(os.path.dirname(new_bgr_path))
    cv2.imwrite(new_bgr_path, bgr)
    cv2.imwrite(new_anno_path, new_anno_3_channel)


if __name__ == '__main__':
    run(sys.argv[1])

