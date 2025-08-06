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

df = pd.read_csv('./better_smirk_creat.csv')
print(df)
#seq_path = 'N09pgUrFChEH8GM6APzJ0'  # left to right
seq_path = 'aITx4TyRncKnardhTgCzz'  # right to left

for i in range(100):
    # env_img = cv2.resize(cv2.imread('../datasets/empty_smirk_roi_draw.png'), (640, 480))
    env_img = cv2.resize(cv2.imread('../datasets/empty_smirk.png'), (640, 480))
    anno_img = np.zeros_like(env_img)
    envimg_file = '%03d' % i + '.png'
    envimg_path = os.path.join('../datasets/MoreSMIRK/raw_data/event_8/evt_8', envimg_file)
    annoimg_path = envimg_path.replace('.png', '.labels.png').replace('evt_8', 'evt_8_anno')
    if not os.path.exists(os.path.dirname(envimg_path)):
        os.makedirs(os.path.dirname(envimg_path))
    if not os.path.exists(os.path.dirname(annoimg_path)):
        os.makedirs(os.path.dirname(annoimg_path))
    cv2.imwrite(envimg_path, env_img)
    cv2.imwrite(annoimg_path, anno_img)

for idx, row in df.iterrows():
    start_x = row['start_x']
    end_x = row['end_x']
    start_y = row['start_y']
    end_y = row['end_y']
    frame_length = row['end_frame'] - row['start_frame']
    x_interval = abs(row['start_x'] - row['end_x']) / frame_length
    y_interval = abs(row['start_y'] - row['end_y']) / frame_length

    for i in range(frame_length+1):
        print(i)
        rgb_file = 'cam' + '%06d' % (row['start_frame'] + i) + '.png'
        rgb_path = os.path.join('../datasets/smirk', seq_path, rgb_file)
        anno_path = rgb_path.replace('.png', '.labels.png')
        result_rgb_file = '%03d' % (row['start_frame'] + i) + '.png'
        result_rgb_path = os.path.join('../datasets/MoreSMIRK/raw_data/event_8/evt_8', result_rgb_file)
        result_anno_path = result_rgb_path.replace('.png', '.labels.png').replace('evt_8', 'evt_8_anno')
        env_img = cv2.imread(result_rgb_path)
        bgr = cv2.imread(rgb_path)
        bgr = cv2.resize(bgr, (640, 480))
        anno = cv2.imread(anno_path)
        anno = cv2.resize(anno, (640, 480))
        anno_gray = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
        anno_gray = cv2.resize(anno_gray, (640, 480))
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
            cord_y = int(start_y - y_interval * i)
            cord_x = int(start_x - x_interval * i)
        else:
            cord_y = int(start_y + y_interval * i)
            cord_x = int(start_x + x_interval * i)
        env_img[cord_y:cord_y+h, cord_x:cord_x+w, :] = env_img[cord_y:cord_y+h, cord_x:cord_x+w, :] * \
            ~bbox_binary_bool_stack + bbox_black_bg

        result_anno = cv2.imread(result_anno_path)
        result_anno[cord_y:cord_y+h, cord_x:cord_x+w, :] = anno[min_y:max_y, min_x:max_x, :]

        cv2.imwrite(result_rgb_path, env_img)
        cv2.imwrite(result_anno_path, result_anno)

