#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 3rd March, 2025
"""

import os
import cv2
import time
import json
import glob
import zCurve
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


def calculate_morton(values):
    # Cap floating point numbers to one decimal place and convert to integers
    int_values = [int(round(value, 1) * 10) for value in values]
    value = zCurve.interlace(*int_values, dims=len(int_values))
    return value


def roi_cell_compute(roi_grid_cfg):
    attention_cells_start_cord = np.array(roi_grid_cfg['grid_left_top_coord'])
    attention_cell_width = roi_grid_cfg['width']
    attention_cell_height = roi_grid_cfg['height']
    attention_cells_end_cord = [start_piont + np.array([attention_cell_width, attention_cell_height]) for start_piont in
                                attention_cells_start_cord]

    cell_coord_all = np.concatenate([attention_cells_start_cord, attention_cells_end_cord], axis=1)

    return cell_coord_all

def run(args, config):
    pie_path = args.path
    morton_codes = []
    sfc_input_df = []
    frame_id = 0
    for rgb_frame in sorted(glob.glob(os.path.join(pie_path, '*.png'))):
        yolo_start_point = []
        yolo_end_point = []
        yolo = YOLO("yolo11x.pt")
        bgr = cv2.imread(rgb_frame)
        results = yolo.predict(bgr, verbose=False)
        for result in results:
            result = result.boxes.cpu().numpy()
            human_cls_index = np.where(result.cls == 0)
            if result.data[human_cls_index].size > 0:
                humans_xyxyc = result.data[human_cls_index].astype(np.int64)
                for i in range(len(humans_xyxyc)):
                    yolo_start_point.append([humans_xyxyc[i][0], humans_xyxyc[i][1]])
                    yolo_end_point.append([humans_xyxyc[i][2], humans_xyxyc[i][3]])
            else:
                yolo_start_point = None
                yolo_end_point = None

        if yolo_start_point is not None and yolo_end_point is not None:
            yolo_coord_all = np.concatenate((yolo_start_point, yolo_end_point), axis=1)
        else:
            yolo_coord_all = None

        roi_coord_all = roi_cell_compute(config['attention_grid'])
        roi_yolo_overlap_coord_all = find_yolo_roi_overlap(roi_coord_all, yolo_coord_all)

        # bgr_copy = bgr.copy()
        # for i in roi_coord_all:
        #     cv2.rectangle(bgr_copy, i[:2], i[2:], (0, 0, 255), 1)
        # for j in roi_yolo_overlap_coord_all:
        #      for m in j:
        #          cv2.rectangle(bgr_copy, m[1:3], m[3:5], (0,255,0), 1)
        #
        # plt.imshow(cv2.cvtColor(bgr_copy, cv2.COLOR_BGR2RGB))
        # plt.show()

        if args.save_yolo_result is True:
            bgr_copy = bgr.copy()
            for roi_coord in roi_coord_all:
                cv2.rectangle(bgr_copy, roi_coord[:2], roi_coord[2:], (0, 0, 255), 1)
            for yolo_coord in yolo_coord_all:
                cv2.rectangle(bgr_copy, yolo_coord[:2], yolo_coord[2:], (0, 255, 0), 1)

            yolo_detect_path = rgb_frame.replace('datasets', 'outputs')
            if not os.path.exists(os.path.dirname(yolo_detect_path)):
                os.makedirs(os.path.dirname(yolo_detect_path))
            cv2.imwrite(yolo_detect_path, bgr_copy)

        sfc_input = np.zeros(len(roi_coord_all))
        for i in range(len(roi_yolo_overlap_coord_all)):
            roi_yolo_overlap_coord = roi_yolo_overlap_coord_all[i]
            overlap_size_cum = 0
            valid_overlap = []
            valid_overlap.extend([i[1:] for i in roi_yolo_overlap_coord if i != [0, 0, 0, 0, 0]])
            overlap_size_cum += np.sum(np.array(roi_yolo_overlap_coord)[:, 0])
            # print(overlap_size_cum)
            # print(valid_overlap)
            repeat_count_overlap_roi = compute_repeat_count_roi_yolo_overlap_size(valid_overlap)
            if repeat_count_overlap_roi is not None:
                sfc_input[i] = overlap_size_cum - repeat_count_overlap_roi
            else:
                sfc_input[i] = overlap_size_cum

        sfc_input = sfc_input / (config['attention_grid']['width'] * config['attention_grid']['width'])
        sfc_input = [1 if num >= 1 else num for num in sfc_input]

        frame_ms_timestamp = frame_id * 33
        frame_id += 1
        sfc_input_df.append({'time_stamp_ms': frame_ms_timestamp, 'frame': rgb_frame,
                             'cell_0': sfc_input[0], 'cell_1': sfc_input[1], 'cell_2': sfc_input[2],
                             'cell_3': sfc_input[3],'cell_4': sfc_input[4],'cell_5': sfc_input[5]})

    sfc_input_df = pd.DataFrame(sfc_input_df, columns=['time_stamp_ms', 'frame', 'cell_0', 'cell_1', 'cell_2',
                                                       'cell_3', 'cell_4', 'cell_5'])
    sfc_csv_name = pie_path.split('/')[-2] + '.csv'
    sfc_csv_path = os.path.join('./outputs/pie_yolo_csv', sfc_csv_name)
    if not os.path.exists(os.path.dirname(sfc_csv_path)):
        os.makedirs(os.path.dirname(sfc_csv_path))
    sfc_input_df.to_csv(sfc_csv_path, sep=';', index=False)





def compute_repeat_count_roi_yolo_overlap_size(valid_overlap):
    """
    compute the repeat-count overlap size
    :param valid_overlap:
    :return:
    """
    repeat_count_overlap_area = 0
    # if len(valid_overlap) > 1:
    for overlap_box_pair in itertools.combinations(valid_overlap, r=2):
        # print(overlap_box_pair)
        overlap_area, _, _, _, _ = find_overlap(overlap_box_pair[0], overlap_box_pair[1])
        repeat_count_overlap_area += overlap_area
        # print(f'overlap_area:{valid_overlap_area}')
    # else:
    #     valid_overlap_area = valid_overlap[0]

    return repeat_count_overlap_area


def find_yolo_roi_overlap(roi_coord_all, yolo_coord_all):
    """
    :param roi_coord_all: 6 pre-defined ROI cell coordinates [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    :param yolo_coord_all: All YOLO pedestrian detection coordinates
                            [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    :return roi_yolo_overlap_all: overlap size-and-coordinates of all yolo_ped_bbox to each pre-defined ROI cell
    """
    roi_yolo_overlap_all = [None] * len(roi_coord_all)
    for i in range(len(roi_yolo_overlap_all)):
        roi_coord = roi_coord_all[i]
        roi_yolo_overlap = [None] * len(yolo_coord_all)
        for j in range(len(yolo_coord_all)):
            yolo_coord = yolo_coord_all[j]
            overlap_area, x_left, y_top, x_right, y_bottom = find_overlap(roi_coord, yolo_coord)
            roi_yolo_overlap[j] = [overlap_area, x_left, y_top, x_right, y_bottom]

        roi_yolo_overlap_all[i] = roi_yolo_overlap

    return roi_yolo_overlap_all


def find_overlap(cord_0, cord_1):
    """
    :param cord_0: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    :param cord_1: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    :return:
    """
    x_left = max(cord_0[0], cord_1[0])
    x_right = min(cord_0[2], cord_1[2])
    y_top = max(cord_0[1], cord_1[1])
    y_bottom = min(cord_0[3], cord_1[3])

    if x_right < x_left or y_bottom < y_top:
        return 0, 0, 0, 0, 0
    overlap_size = (x_right - x_left) * (y_bottom - y_top)
    return overlap_size, x_left, y_top, x_right, y_bottom




    #     cell_coord_all = np.concatenate([attention_cells_start_cord, attention_cells_end_cord], axis=1)
    #
    #     bgr = bgr_resize.copy()
    #
    #     input_zcurve = np.zeros(6)
    #     if yolo_coord_all is not None:
    #         for i in range(len(cell_coord_all)):
    #             cell_coord = cell_coord_all[i]
    #             bgr_overlap = cv2.rectangle(bgr, cell_coord[:2], cell_coord[2:], (0, 0, 255), 2)
    #             overlap_cum = 0
    #             for j in yolo_coord_all:
    #                 overlap_area, x_left, y_top, x_right, y_bottom = overlap(cell_coord, j)
    #                 color = list(np.random.random(size=3) * 256)
    #                 bgr_overlap = cv2.rectangle(bgr, [x_left, y_top], [x_right, y_bottom], color, 2)
    #                 overlap_cum += overlap_area
    #
    #             cell_area = config['attention_grid']['width'] * config['attention_grid']['height']
    #             if overlap_cum > 0:
    #                 input_zcurve[i] = overlap_cum / cell_area
    #                 if input_zcurve[i] >= 1:
    #                     input_zcurve[i] = 1
    #     else:
    #         for cell_coord in cell_coord_all:
    #             bgr_overlap = cv2.rectangle(bgr, cell_coord[:2], cell_coord[2:], (0, 0, 255), 2)
    #
    #     # overlap_path = rgb_frame.replace('datasets/zod_multi_ped', 'outputs/zod_multi_ped')
    #     # if not os.path.exists(os.path.dirname(overlap_path)):
    #     #     os.makedirs(os.path.dirname(overlap_path))
    #     # cv2.imwrite(overlap_path, bgr_overlap)
    #
    #     sfc_input.append({'frame': rgb_frame, 'cell_0': input_zcurve[0], 'cell_1': input_zcurve[1],
    #                       'cell_2': input_zcurve[2], 'cell_3': input_zcurve[3],
    #                       'cell_4': input_zcurve[4], 'cell_5': input_zcurve[5]})
    #
    #     morton_bgr = calculateMortonFromList_with_zCurve(input_zcurve)
    #     morton_codes.append({'frame': rgb_frame, 'morton': morton_bgr})
    #
    # morton_codes = pd.DataFrame(morton_codes, columns=['frame', 'morton'])
    # morton_codes.to_csv('../outputs/yolo_morton_zod_multi_ped.csv', sep=';', index=False)
    #
    # sfc_input = pd.DataFrame(sfc_input, columns=['frame', 'cell_0', 'cell_1', 'cell_2',
    #                                              'cell_3', 'cell_4', 'cell_5'])
    # sfc_input.to_csv('../outputs/sfc_input_zod_multi_ped.csv', sep=';', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolo-sfc-multi-ped pipeline for PIE dataset')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='input path of PIE dataset')
    parser.add_argument('-s', '--save_yolo_result', action='store_true',
                        help='save yolo detection bounding box images.')

    args = parser.parse_args()

    with open('config.json', 'r') as f:
        configs = json.load(f)

    run(args, configs)