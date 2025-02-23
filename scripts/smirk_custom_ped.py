#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21nd Feb, 2025
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

df = pd.read_csv('../test.csv')
print(df)
seq_path = '9jJxC2pmKULPI8arYKQ5O'

for i in range(100):
    blankimg = np.zeros((480, 752, 3), np.uint8)
    blankimg_file = 'cam' + '%06d' % i + '.png'
    blankimg_path = os.path.join('../datasets/smirk_custom_ped/child/', seq_path, blankimg_file)
    if not os.path.exists(os.path.dirname(blankimg_path)):
        os.makedirs(os.path.dirname(blankimg_path))
    cv2.imwrite(blankimg_path, blankimg)

for idx, row in df.iterrows():
    start_x = row['start_x']
    end_x = row['end_x']
    start_y = row['start_y']
    end_y = row['end_y']
    frame_length = row['end_frame'] - row['start_frame']
    x_interval = int(abs(row['start_x'] - row['end_x']) / frame_length)
    y_interval = int(abs(row['start_y'] - row['end_y']) / frame_length)

    for i in range(frame_length+1):
        rgb_file = 'cam' + '%06d' % (row['start_frame'] + i) + '.png'
        rgb_path = os.path.join('../datasets/smirk/child/', seq_path, rgb_file)
        anno_path = rgb_path.replace('.png', '.labels.png')
        blank_path = rgb_path.replace('smirk', 'smirk_custom_ped')
        blank_img = cv2.imread(blank_path)
        bgr = cv2.imread(rgb_path)
        anno_gray = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
        ret, anno_binary = cv2.threshold(anno_gray, 0, 255, cv2.THRESH_BINARY)

        mask_binary = anno_binary == 255
        black_bg_mask = np.zeros_like(bgr)
        black_bg_mask[mask_binary] = bgr[mask_binary]

        pixel_cord = np.argwhere(anno_binary == 255)  # find all pixel coordination for ped contour
        min_y, min_x = np.min(pixel_cord[:], axis=0)
        max_y, max_x = np.max(pixel_cord[:], axis=0)
        height = max_y - min_y
        width = max_x - min_x

        bbox_bgr = bgr[min_y:max_y, min_x:max_x, :]
        bbox_binary = anno_binary[min_y:max_y, min_x:max_x]
        bbox_binary_bool = bbox_binary[:, :] == 255
        bbox_binary_bool_stack = np.stack([bbox_binary_bool, bbox_binary_bool, bbox_binary_bool], axis=2)
        bbox_black_bg = bbox_bgr * bbox_binary_bool_stack
        h, w = bbox_black_bg.shape[0], bbox_black_bg.shape[1]

        if start_x > end_x:
            cord_y = start_y - y_interval * i
            cord_x = start_x - x_interval * i
        else:
            cord_y = start_y + y_interval * i
            cord_x = start_x + x_interval * i
        blank_img[cord_y:cord_y+h, cord_x:cord_x+w, :] = blank_img[cord_y:cord_y+h, cord_x:cord_x+w, :] * \
            ~bbox_binary_bool_stack + bbox_black_bg
        # TODO: add anno for smirk_custom_ped
        # new_anno_path = anno_path.replace('smirk', 'smirk_custom_ped')
        cv2.imwrite(blank_path, blank_img)
        # cv2.imwrite(new_anno_path, new_anno_3_channel)

