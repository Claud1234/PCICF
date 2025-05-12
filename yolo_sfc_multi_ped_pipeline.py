#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 3rd March, 2025
"""

import os
import sys
import cv2
import yaml
import json
import glob
import zCurve
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools.yolo import Yolo
from utils import helper


def make_input_list(args, config):
    input_png_list = []
    if args.dataset == 'pie':
        input_yml = config['Dataset']['pie']['yaml_path']
        with open(input_yml, 'r') as _f:
            input_list = yaml.safe_load(_f)
        for seq in input_list:
            dir_name = seq['id']
            start_frame = seq['event_window'][0]
            end_frame = seq['event_window'][-1]
            # TODO: change this multi plus thing.
            input_png_list = ['./datasets/pie/' + dir_name + '/' + ('%03d' % i) + '.png' for i in
                              range(start_frame, end_frame + 1)]

    elif args.dataset == 'betterSMIRK':
        input_path = config['Dataset']['betterSMIRK']['input_path']
        label_glob = sorted(glob.glob(os.path.join(input_path, '*.labels.png')))
        input_png_list.append([i.replace('.labels.png', '.png') for i in label_glob])

    else:
        sys.exit("Must specify a dataset, either 'pie' or 'betterSMIRK'")

    return input_png_list


def run(input_list, args, config):
    input_png_list = input_list


    print(f'Processing the path {dir_name} of frame {start_frame} to {end_frame}...')
    yolo = Yolo(args, config)
    # input_glob = sorted(glob.glob(os.path.join(input_path, '*.png')))
    if args.mode == 'detect':
        sfc_input_df, morton_code_df = yolo.yolo_detect(input_png_list)
    elif args.mode == 'track':
        sfc_input_df, morton_code_df = yolo.yolo_track(input_png_list)
    else:
        sys.exit('A mode must be specified for yolo! (detect or track)')

    if sfc_input_df is not None:
        sfc_input_df = pd.DataFrame(sfc_input_df, columns=['time_stamp_ms', 'frame_id',
                                                           'cell_0', 'cell_1', 'cell_2',
                                                           'cell_3', 'cell_4', 'cell_5'])
        sfc_csv_name = dir_name + '_' + str(start_frame) + '_' + str(end_frame) + '_sfc_input' + '.csv'
        sfc_csv_path = os.path.join(args.output_path, sfc_csv_name)
        helper.dir_path_check(sfc_csv_path)
        sfc_input_df.to_csv(sfc_csv_path, sep=';', index=False)

    if morton_code_df is not None:
        #print(morton_code_df)
        morton_code_df = pd.DataFrame(morton_code_df, columns=['time_stamp_ms', 'frame_id', 'morton'])
        morton_csv_name = dir_name + '_' + str(start_frame) + '_' + str(end_frame) + '_morton' + '.csv'
        morton_csv_path = os.path.join(args.output_path, morton_csv_name)
        helper.dir_path_check(morton_csv_path)
        morton_code_df.to_csv(morton_csv_path, sep=';', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolo-sfc-multi-ped pipeline for PIE dataset')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['pie', 'betterSMIRK'], help='input dataset, pie or betterSMIRK')
    parser.add_argument('-s', '--save_yolo_result', action='store_true',
                        help='save yolo detection bounding box images.')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='path for sfc input/output csv file.')
    # parser.add_argument('-m', '--mode', type=str, required=True,
    #                     choices=['track', 'detect'], help='yolo mode, either track or detect')

    args = parser.parse_args()

    with open('config.json', 'r') as f:
        configs = json.load(f)

    input_png_list = make_input_list(args, configs)
    run(args, configs)
