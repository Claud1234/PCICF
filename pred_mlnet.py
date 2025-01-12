#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script to predict with the MLNet.

Created on 6th Jan. 2025
"""

import os
import cv2
import csv
import sys
import json
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import utils.helper as helper
import utils.attention_grid as attention_grid
from models.mlnet.mlnet import MLnet
from utils.helper import rgb_anno_normalize,  image_overlay


def run(backbone, config):
    device = torch.device(config['General']['device'] if torch.cuda.is_available() else "cpu")

    if backbone == 'mlnet':
        print(f'Using backbone {args.backbone}')
        model = MLnet([config['mlnet_input_size'][1], config['mlnet_input_size'][0]])
        checkpoint_path = config['mlnet_model_path']
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])

    elif backbone == 'tasednet':
        print(f'Using backbone {args.backbone}')

    elif backbone == 'transalnet':
        print(f'Using backbone {args.backbone}')

    else:
        sys.exit("A backbone must be specified!")

    model.to(device)
    model.eval()

    if config['Dataset']['name'] == "smirk":
        data_root = './datasets/smirk/'
    elif config['Dataset']['name'] == "zod":
        data_root = './datasets/zod/'
    else:
        sys.exit('Can not find the config[Dataset][name]')

    data_list = open(args.path, 'r')
    data_cam = np.array(data_list.read().splitlines())
    data_list.close()

    dataset_name = config['Dataset']['name']
    roi_value_csv_path = './outputs/' + f'{dataset_name}_roi_mean_values.csv'
    roi_mean_values_all = []

    i = 1
    for path in data_cam:
        cam_path = os.path.join(data_root, path)
        anno_path = cam_path.replace('.png', '.labels.png')
        # output_path = cam_path.replace('datasets', 'outputs')

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        anno_name = anno_path.split('/')[-1].split('.')[0]
        assert (rgb_name == anno_name), "rgb and anno input not matching"

        rgb = Image.open(cam_path).convert('RGB')
        anno = Image.open(anno_path)

        rgb_normlizer, anno_resizer = helper.rgb_anno_normalize(rgb, config, norm=True)
        rgb_norm = rgb_normlizer(rgb).to(device, non_blocking=True)
        anno_resize = anno_resizer(anno).to(device, non_blocking=True)

        rgb_resizer, _ = helper.rgb_anno_normalize(rgb, config, norm=False)
        rgb_resize = rgb_resizer(rgb)
        rgb_resize = rgb_resize.detach().numpy()
        rgb_resize = np.moveaxis(rgb_resize, 0, -1) * 255

        # print(rgb_resize.shape)
        # im = rgb_resize*255
        # plt.imshow(np.uint8(im))
        # plt.show()
        rgb_norm = rgb_norm.unsqueeze(0)
        anno_resize = anno_resize.unsqueeze(0)

        if backbone == 'mlnet':
            with torch.no_grad():
                output = model(rgb_norm)
                output = output.cpu().numpy() if output.is_cuda else output.detach().numpy()
                output = np.squeeze(output)

                print(f'Saving heatmap and computing mean ROI {i}...', end='\r', flush=True)
                #  Save grayscale and overlay
                helper.save_heatmap(output, rgb_resize, config['mlnet_input_size'], cam_path)
                roi_mean_values = attention_grid.roi_values(output, config)

        elif backbone == 'tasednet':
            return

        elif backbone == 'transalnet':
            return

        else:
            sys.exit("A backbone must be specified! ")

        roi_mean_values_all.append(np.hstack((path, roi_mean_values)))
        i += 1

    print('Saving ROI mean values to csv...')
    header = []
    with open(roi_value_csv_path, 'w') as _fd:
        writer = csv.writer(_fd)
        header.extend(f'cell_{i + 1}' for i in range(len(config['attention_grid']['grid_left_top_coord'])))
        writer.writerow(['frame'] + header)
        writer.writerows(roi_mean_values_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visual run script')
    parser.add_argument('-bb', '--backbone', required=True,
                        choices=['mlnet', 'tasednet', 'transalnet'],
                        help='Use the backbone of training, mlnet or tasednet or transalnet')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path of the text file to visualize')
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        configs = json.load(f)

    run(args.backbone, configs)