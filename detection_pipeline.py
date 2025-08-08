#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 3rd March, 2025
"""

import os
import sys
import yaml
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from tools.yolo import Yolo
from utils import helper


def make_input_list(args, config):
    if args.dataset == 'pie':
        input_yml = os.path.join('./datasets/pie_splits/', config['Dataset']['pie']['yaml_path'])
        with open(input_yml, 'r') as _f:
            input_list = yaml.safe_load(_f)
        for seq in input_list:
            input_png_list, seq_name = helper.make_pie_png_list(seq)
            seq_name = config['Dataset']['pie']['yaml_path'].split('.')[0] + '/' + seq_name
            print(f'Processing the path {seq['id']} of frame {seq['event_window'][0]} to {seq['event_window'][-1]}...')
            run(input_png_list, args, configs, seq_name)

    elif args.dataset == 'MoreSMIRK':
        input_path = config['Dataset']['MoreSMIRK']['input_path']
        input_png_list = sorted(glob.glob(os.path.join(input_path, '*.png')))
        seq_name = 'more_smirk_' + input_path.split('/')[-1]
        # input_png_list.append([i.replace('.labels.png', '.png') for i in label_glob])
        print(f'Processing the path {seq_name}...')
        run(input_png_list, args, configs, seq_name)

    else:
        sys.exit("Must specify a dataset, either 'pie' or 'MoreSMIRK'")


def run(input_list, args, config, seq_name):
    png_list = input_list

    yolo = Yolo(args, config)
    sfc_input_df, morton_code_df = yolo.yolo_track(png_list)

    sfc_csv_name = seq_name + '_sfc_input' + '.csv'
    morton_csv_name = seq_name + '_morton' + '.csv'

    if sfc_input_df is not None:
        sfc_input_df = pd.DataFrame(sfc_input_df, columns=['time_stamp_ms', 'frame_id',
                                                           'cell_0', 'cell_1', 'cell_2',
                                                           'cell_3', 'cell_4', 'cell_5'])
        sfc_csv_path = os.path.join(args.output_path, sfc_csv_name)
        helper.dir_path_check(sfc_csv_path)
        sfc_input_df.to_csv(sfc_csv_path, sep=';', index=False)

    if morton_code_df is not None:
        morton_code_df = pd.DataFrame(morton_code_df, columns=['time_stamp_ms', 'frame_id', 'morton'])
        morton_csv_path = os.path.join(args.output_path, morton_csv_name)
        helper.dir_path_check(morton_csv_path)
        morton_code_df.to_csv(morton_csv_path, sep=';', index=False)

        df_numpy = morton_code_df.to_numpy()
        morton = [i[-1] for i in df_numpy]
        frame = [i[-2] for i in df_numpy]
        _, ax1 = plt.subplots()
        ax1.set_xlabel("Morton")
        ax1.set_ylabel("Frequency")
        ax1.set_ylim((0, 1))
        ax1.eventplot(morton, orientation='horizontal', colors="red", lineoffsets=0.5)

        ax2 = ax1.twinx()

        ax2.scatter(morton, frame, s=5, color='green')
        ax2.set_ylabel("Frame Number")
        plt.savefig(morton_csv_path.replace('.csv', '.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolo-sfc-multi-ped pipeline for PIE dataset')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['pie', 'MoreSMIRK'], help='input dataset, pie or betterSMIRK')
    parser.add_argument('-s', '--save_yolo_result', action='store_true',
                        help='save yolo detection bounding box images.')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='path for sfc input/output csv file.')

    args = parser.parse_args()

    with open('config.json', 'r') as f:
        configs = json.load(f)

    make_input_list(args, configs)
