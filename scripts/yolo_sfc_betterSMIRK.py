#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the script of the yolo-sfc pipeline for betterSMIRK datasets.
The yolo prediction results will be saved as PNGs, also the sfc input (6 cell values) as CSV
and morton code chart of the whole sequence.

Created on 14th Mar, 2025
"""

import zCurve as z
import csv
import cv2
import os
import numpy as np
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt

from ultralytics import YOLO


def calculate_morton_from_list_with_zcurve(values):
    # Cap floating point numbers to one decimal place and convert to integers
    int_values = [int(round(value, 1) * 10) for value in values]
    value = z.interlace(*int_values, dims=len(int_values))
    return value


def overlap(cell_cord, yolo_cord): # [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    x_left = max(cell_cord[0], yolo_cord[0])
    x_right = min(cell_cord[2], yolo_cord[2])
    y_top = max(cell_cord[1], yolo_cord[1])
    y_bottom = min(cell_cord[3], yolo_cord[3])

    if x_right < x_left or y_bottom < y_top:
        return 0, 0, 0, 0, 0
    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    return overlap_area, x_left, y_top, x_right, y_bottom


seq_path = '../datasets/better_smirk/left2right/three_cells/sq_00/'
config_path = '../config.json'

with open(config_path, 'r') as _f:
    config = json.load(_f)

attention_cells_start_cord = np.array(config['attention_grid']['grid_left_top_coord'])
attention_cell_width = config['attention_grid']['width']
attention_cell_height = config['attention_grid']['height']
attention_cells_end_cord = [start_piont + np.array([attention_cell_width, attention_cell_height]) \
                            for start_piont in attention_cells_start_cord]

morton_codes = []
sfc_input = []
for anno_frame in sorted(glob.glob(os.path.join(seq_path, '*.labels.png'))):
    rgb_frame = anno_frame.replace('.labels.png', '.png')
    yolo_start_point = []
    yolo_end_point = []
    yolo = YOLO("yolo11x.pt")
    results = yolo.predict(rgb_frame, verbose=False)
    for result in results:
        result = result.boxes.cpu().numpy()
        if result.data.size > 0:
            xyxy = result.xyxy.astype(np.int64)
            xywh = result.xywh.astype(np.int64)
            for i in range(len(xyxy)):
                yolo_start_point.append([xyxy[i][0], xyxy[i][1]])
                yolo_end_point.append([xyxy[i][2], xyxy[i][3]])
        else:
            yolo_start_point = None
            yolo_end_point = None

    if yolo_start_point is not None and yolo_end_point is not None:
        yolo_coord_all = np.concatenate([yolo_start_point, yolo_end_point], axis=1)
    else:
        yolo_coord_all = [0, 0, 0, 0]

    cell_coord_all = np.concatenate([attention_cells_start_cord, attention_cells_end_cord], axis=1)

    bgr = cv2.imread(rgb_frame)

    input_zcurve = np.zeros(6)
    for i in range(len(cell_coord_all)):
        cell_coord = cell_coord_all[i]
        overlap_cum = 0
        for j in yolo_coord_all:
            overlap_area, x_left, y_top, x_right, y_bottom = overlap(cell_coord, j)
            color = list(np.random.random(size=3) * 256)
            bgr_overlap = cv2.rectangle(bgr, [x_left, y_top], [x_right, y_bottom], color, 1)
            overlap_cum += overlap_area

        cell_area = config['attention_grid']['width'] * config['attention_grid']['height']
        if overlap_cum > 0:
            input_zcurve[i] = overlap_cum / cell_area

    overlap_path = rgb_frame.replace('datasets/better_smirk', 'outputs/yolo_better_smirk')
    if not os.path.exists(os.path.dirname(overlap_path)):
        os.makedirs(os.path.dirname(overlap_path))
    cv2.imwrite(overlap_path, bgr_overlap)

    sfc_input.append({'frame': rgb_frame, 'cell_0': input_zcurve[0], 'cell_1': input_zcurve[1],
                                             'cell_2': input_zcurve[2], 'cell_3': input_zcurve[3],
                                             'cell_4': input_zcurve[4], 'cell_5': input_zcurve[5]})
    morton_bgr = calculate_morton_from_list_with_zcurve(input_zcurve)
    morton_codes.append({'frame': rgb_frame, 'morton': morton_bgr})

morton_codes = pd.DataFrame(morton_codes, columns=['frame', 'morton'])
sfc_input = pd.DataFrame(sfc_input, columns=['frame', 'cell_0', 'cell_1', 'cell_2',
                                                      'cell_3', 'cell_4', 'cell_5'])
morton_codes_path = '../outputs/morton_codes_betterSMIRK/left2right/three_cell_sq00.csv'
if not os.path.exists(os.path.dirname(morton_codes_path)):
    os.makedirs(os.path.dirname(morton_codes_path))
morton_codes.to_csv(morton_codes_path, sep=';', index=False)

sfc_input_path = morton_codes_path.replace('morton_codes_betterSMIRK', 'sfc_input_betterSMIRK')
if not os.path.exists(os.path.dirname(sfc_input_path)):
    os.makedirs(os.path.dirname(sfc_input_path))
sfc_input.to_csv(sfc_input_path, sep=';', index=False)

morton_codes_seq = pd.read_csv('../outputs/morton_codes_betterSMIRK/left2right/three_cell_sq00.csv', sep=';')
morton_codes_seq = morton_codes_seq.to_numpy()
mortons_seq = []
for i in range(len(morton_codes_seq)):
    frame, morton = morton_codes_seq[i]
    mortons_seq.append(int(morton)/10000)

fig, ax1 = plt.subplots()
ax1.set_xlabel("Morton")
ax1.set_ylabel("Frequency")
ax1.set_ylim((0, 1))
ax1.eventplot(mortons_seq, orientation='horizontal', colors="red",lineoffsets=0.5)

ax2 = ax1.twinx()
for frame_id in range(len(mortons_seq)):
    x=mortons_seq[frame_id]
    y=frame_id+1
    ax2.scatter(x, y, s=5, color='green')
    ax2.set_ylabel("Frame Number")

mortons_seq_chart_path = '../outputs/morton_charts_betterSMIRK/left2right/three_cell_sq00.png'
if not os.path.exists(os.path.dirname(mortons_seq_chart_path)):
    os.makedirs(os.path.dirname(mortons_seq_chart_path))
fig.savefig(mortons_seq_chart_path)
