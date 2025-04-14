#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 3rd March, 2025
"""

import os
import sys
import cv2
import json
import glob
import zCurve
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools.yolo import Yolo
from utils import helper


def run(args, config):
    input_path = args.input_path
    sfc_input_df = []
    morton_code_df = []
    yolo = Yolo(args, config)
    input_glob = sorted(glob.glob(os.path.join(input_path, '*.png')))
    if args.mode == 'detect':
        sfc_input_df, morton_code_df = yolo.yolo_detect(input_glob)
    elif args.mode == 'track':
        sfc_input_df, morton_code_df = yolo.yolo_track(input_glob)
    else:
        sys.exit('A mode must be specified for yolo! (detect or track)')

    if sfc_input_df is not None:
        sfc_input_df = pd.DataFrame(sfc_input_df, columns=['time_stamp_ms', 'frame', 'cell_0', 'cell_1', 'cell_2',
                                                           'cell_3', 'cell_4', 'cell_5'])
        sfc_csv_name = input_path.split('/')[-2] + '_sfc_input' + '.csv'
        sfc_csv_path = os.path.join(args.output_path, sfc_csv_name)
        helper.dir_path_check(sfc_csv_path)
        sfc_input_df.to_csv(sfc_csv_path, sep=';', index=False)

    if morton_code_df is not None:
        morton_code_df = pd.DataFrame(morton_code_df, columns=['time_stamp_ms', 'frame', 'morton'])
        morton_csv_name = input_path.split('/')[-2] + '_morton' + '.csv'
        morton_csv_path = os.path.join(args.output_path, morton_csv_name)
        helper.dir_path_check(morton_csv_path)
        morton_code_df.to_csv(morton_csv_path, sep=';', index=False)





# def roi_cell_compute(roi_grid_cfg):
#     attention_cells_start_cord = np.array(roi_grid_cfg['grid_left_top_coord'])
#     attention_cell_width = roi_grid_cfg['width']
#     attention_cell_height = roi_grid_cfg['height']
#     attention_cells_end_cord = [start_piont + np.array([attention_cell_width, attention_cell_height]) for
#                                 start_piont in
#                                 attention_cells_start_cord]
#
#     cell_coord_all = np.concatenate([attention_cells_start_cord, attention_cells_end_cord], axis=1)
#
#     return cell_coord_all


# def compute_valid_overlap_area(overlap_without_zeros):
#     empty_mask = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint8)
#     for i in overlap_without_zeros:
#         empty_mask[i[1]:i[3], i[0]:i[2]] = 1
#     valid_overlap_area = np.count_nonzero(empty_mask == 1)
#
#     return valid_overlap_area
#
#
# def find_yolo_roi_overlap(roi_coord_all, yolo_coord_all):
#     """
#     :param roi_coord_all: 6 pre-defined ROI cell coordinates [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
#     :param yolo_coord_all: All YOLO pedestrian detection coordinates
#                             [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
#     :return roi_yolo_overlap_all: overlap size-and-coordinates of all yolo_ped_bbox to each pre-defined ROI cell
#     """
#     roi_yolo_overlap_all = [None] * len(roi_coord_all)
#     for i in range(len(roi_yolo_overlap_all)):
#         roi_coord = roi_coord_all[i]
#         roi_yolo_overlap = [None] * len(yolo_coord_all)
#         for j in range(len(yolo_coord_all)):
#             yolo_coord = yolo_coord_all[j]
#             overlap_area, x_left, y_top, x_right, y_bottom = find_overlap_for_two_bbox(roi_coord, yolo_coord)
#             roi_yolo_overlap[j] = [overlap_area, x_left, y_top, x_right, y_bottom]
#
#         roi_yolo_overlap_all[i] = roi_yolo_overlap
#
#     return roi_yolo_overlap_all
#
#
# def find_overlap_for_two_bbox(cord_0, cord_1):
#     """
#     :param cord_0: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
#     :param cord_1: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
#     :return:
#     """
#     x_left = max(cord_0[0], cord_1[0])
#     x_right = min(cord_0[2], cord_1[2])
#     y_top = max(cord_0[1], cord_1[1])
#     y_bottom = min(cord_0[3], cord_1[3])
#
#     if x_right < x_left or y_bottom < y_top:
#         return 0, 0, 0, 0, 0
#     overlap_size = (x_right - x_left) * (y_bottom - y_top)
#     return overlap_size, x_left, y_top, x_right, y_bottom


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolo-sfc-multi-ped pipeline for PIE dataset')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='input path of PIE dataset')
    parser.add_argument('-s', '--save_yolo_result', action='store_true',
                        help='save yolo detection bounding box images.')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='path for sfc input/output csv file. ')

    parser.add_argument('-m', '--mode', type = str, required=True,
                        choices=['track', 'detect'], help='yolo mode, either track or detect')

    args = parser.parse_args()

    with open('config.json', 'r') as f:
        configs = json.load(f)

    run(args, configs)
