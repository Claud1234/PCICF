#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import glob
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

from utils.attention_grid import roi_cell_compute
from tools.sfc import calculate_morton


class Yolo(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        #self.input_path = input_path
        # if self.args.mode == 'detect':
        #     self.yolo_detecter = YOLO("yolo11x.pt")
        # elif self.args.mode == 'track':
        self.yolo_tracker = YOLO("yolo11x.pt")

        self.track_dict = defaultdict(lambda: [])
        self.morton_code_df = []
        self.sfc_input_df = []
        self.frame_counter = 0

    def yolo_track(self, input_glob):
        #print(input_glob)
        for rgb_img in input_glob:
            #rgb_img = rgb_img.replace('.labels.png', '.png')
            frame_name = int(rgb_img.split('/')[-1].split('.')[0])
            result = self.yolo_tracker.track(rgb_img, persist=True, verbose=False)[0]
            result = result.boxes.cpu().numpy()
            human_cls_index = np.where(result.cls == 0)
            if result.is_track is True:
                humans_xyxy = result.xyxy[human_cls_index].astype(np.uint16)
                humans_xywh = result.xywh[human_cls_index].astype(np.uint16)
                #print(result.id)
                humans_track_id = result.id[human_cls_index].astype(np.uint16)

                assert len(humans_track_id) == len(humans_xyxy)
                assert len(humans_xyxy) == len(humans_xywh)
                # print(humans_xyxy)
                # print(humans_track_id)
                # for i in range(len(humans_xyxyc)):
                for xywh, xyxy, track_id in zip(humans_xywh, humans_xyxy, humans_track_id):
                    x_lt, y_lt, x_rb, y_rb = xyxy
                    x_m, y_m, width, height = xywh
                    track = self.track_dict[track_id]
                    track.append((x_lt, y_lt, x_rb, y_rb, x_m, y_m, width, height, frame_name))

        full_ped_cross = []
        for track_id, xyxyxywhi in self.track_dict.items():
            center_x_all = [i[4] for i in xyxyxywhi]
            min_center_x = min(center_x_all)
            max_center_x = max(center_x_all)

            if (max_center_x - min_center_x) > self.config['General']['yolo_track_threshold'] \
                    and max_center_x > self.config['General']['image_width'] / 2 > min_center_x:
                full_ped_cross.append(xyxyxywhi)

        if len(full_ped_cross) > 0:
            full_ped_cross = np.vstack(full_ped_cross)

            for rgb_img in input_glob:
                #rgb_img = rgb_img.replace('.labels.png', '.png')
                frame_name = int(rgb_img.split('/')[-1].split('.')[0])
                frame_mask = full_ped_cross[:, -1] == frame_name

                yolo_coord_all = full_ped_cross[frame_mask][:, 0:4]

                sfc_input, frame_ms_timestamp = self.yolo_roi_compute_one_frame(yolo_coord_all, rgb_img)
                sfc_input_norm = (np.array(sfc_input) > self.config['General']['sfc_bounding']).astype(int)

                frame_id = int(rgb_img.split('/')[-1].split('.')[0])
                self.sfc_input_df.append({'time_stamp_ms': frame_ms_timestamp, 'frame_id': frame_id,
                                          'cell_0': sfc_input_norm[0], 'cell_1': sfc_input_norm[1], 'cell_2': sfc_input_norm[2],
                                          'cell_3': sfc_input_norm[3], 'cell_4': sfc_input_norm[4], 'cell_5': sfc_input_norm[5]})

                morton_code = calculate_morton(sfc_input_norm)
                self.morton_code_df.append({'time_stamp_ms': frame_ms_timestamp,
                                            'frame_id': frame_id, 'morton': morton_code})
                self.frame_counter += 1

        return self.sfc_input_df, self.morton_code_df

    def yolo_roi_compute_one_frame(self, yolo_coord_all, rgb_img):
        bgr = cv2.imread(rgb_img)
        if self.args.dataset == 'pie':
            roi_coord_all = roi_cell_compute(self.config['Dataset']['pie'],
                                             self.config['General']['roi_width'],
                                             self.config['General']['roi_height'])
        elif self.args.dataset == 'betterSMIRK':
            roi_coord_all = roi_cell_compute(self.config['Dataset']['betterSMIRK'],
                                             self.config['General']['roi_width'],
                                             self.config['General']['roi_height'])
        else:
            sys.exit('A mode must be specified for yolo! (detect or track)')

        roi_yolo_overlap_coord_all = self.find_yolo_roi_overlap(roi_coord_all, yolo_coord_all)

        # bgr_copy = bgr.copy()
        # for i in roi_coord_all:
        #     cv2.rectangle(bgr_copy, i[:2], i[2:], (0, 0, 255), 1)
        # for j in roi_yolo_overlap_coord_all:
        #      for m in j:
        #          cv2.rectangle(bgr_copy, m[1:3], m[3:5], (0,255,0), 1)
        #
        # plt.imshow(cv2.cvtColor(bgr_copy, cv2.COLOR_BGR2RGB))
        # plt.show()

        if self.args.save_yolo_result is True:
            bgr_copy = bgr.copy()
            for roi_coord in roi_coord_all:
                cv2.rectangle(bgr_copy, roi_coord[:2], roi_coord[2:], (0, 0, 255), 1)
            for yolo_coord in yolo_coord_all:
                cv2.rectangle(bgr_copy, yolo_coord[:2], yolo_coord[2:], (0, 255, 0), 1)

            # yolo_detect_path = f'outputs/png_visual_{self.args.dataset}' + rgb_img.split('/')[-1]
            yolo_detect_path = rgb_img.replace('datasets', f'outputs/png_visual_{self.args.dataset}')
            if not os.path.exists(os.path.dirname(yolo_detect_path)):
                os.makedirs(os.path.dirname(yolo_detect_path))
            cv2.imwrite(yolo_detect_path, bgr_copy)

        sfc_input = np.zeros(len(roi_coord_all))
        for i in range(len(roi_yolo_overlap_coord_all)):
            roi_yolo_overlap_coord = roi_yolo_overlap_coord_all[i]
            overlap_size_cum = 0
            overlap_without_zeros = []
            overlap_without_zeros.extend([i[1:] for i in roi_yolo_overlap_coord if i != [0, 0, 0, 0, 0]])

            valid_overlap_area = self.compute_valid_overlap_area(bgr, overlap_without_zeros)
            # print(f'valid_overlap_area: {valid_overlap_area}')
            sfc_input[i] = valid_overlap_area

        sfc_input = sfc_input / (self.config['General']['roi_width'] * self.config['General']['roi_height'])
        sfc_input = [1 if num >= 1 else num for num in sfc_input]

        frame_ms_timestamp = self.frame_counter * 33

        return sfc_input, frame_ms_timestamp

    @staticmethod
    def compute_valid_overlap_area(bgr, overlap_without_zeros):
        empty_mask = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint8)
        for i in overlap_without_zeros:
            empty_mask[i[1]:i[3], i[0]:i[2]] = 1
        valid_overlap_area = np.count_nonzero(empty_mask == 1)

        return valid_overlap_area

    def find_yolo_roi_overlap(self, roi_coord_all, yolo_coord_all):
        """
        :param roi_coord_all: 6 pre-defined ROI cell coordinates[top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        :param yolo_coord_all: All YOLO pedestrian detection coordinates
                                [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        :return roi_yolo_overlap_all: overlap size-and-coordinates of all yolo_ped_bbox to each pre-defined ROI cell
        """
        roi_yolo_overlap_all = [None] * len(roi_coord_all)
        for i in range(len(roi_yolo_overlap_all)):
            roi_coord = roi_coord_all[i]
            roi_yolo_overlap = [None] * len(yolo_coord_all)
            # print(yolo_coord_all)
            for j in range(len(yolo_coord_all)):
                yolo_coord = yolo_coord_all[j]

                # print(roi_coord)
                overlap_area, x_left, y_top, x_right, y_bottom = self.find_overlap_for_two_bbox(roi_coord, yolo_coord)
                roi_yolo_overlap[j] = [overlap_area, x_left, y_top, x_right, y_bottom]

            roi_yolo_overlap_all[i] = roi_yolo_overlap

        return roi_yolo_overlap_all

    @staticmethod
    def find_overlap_for_two_bbox(cord_0, cord_1):
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





