#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script to select ROI and compute mean vaules of attention in ROi

Created on 10th Jan. 2025
"""

import cv2
import sys
import numpy as np


def roi_values(heatmap, config):
    """
    select ther ROI regions defined in config.json and compute np.mean()
    :param heatmap: cpu().numpy() array which (h/8, w/8).float32 from mlnet prediction
    :param config: config.json. For getting to know the regions' coordinates.
    :return: mean values (float32) for each ROI region.
    """
    roi_left_top_all = np.array(config['attention_grid']['grid_left_top_coord']).transpose()
    roi_right_all = roi_left_top_all[0] + config['attention_grid']['width']
    roi_bottom_all = roi_left_top_all[1] + config['attention_grid']['height']
    if any(i > config['mlnet_input_size'][0] for i in roi_right_all) or any(
            i > config['mlnet_input_size'][1] for i in roi_bottom_all):
        sys.exit('The attention_grid definition is beyond the image size!')

    heatmap = cv2.resize(heatmap, [640, 480], interpolation=cv2.INTER_LINEAR)  # 1 channel
    roi_mean_values = np.zeros(len(roi_right_all))
    for i in range(len(roi_right_all)):
        roi_mean_values[i] = np.mean(heatmap[roi_left_top_all[1][i]:roi_bottom_all[i],
                                             roi_left_top_all[0][i]:roi_right_all[i]])

    return roi_mean_values


def roi_cell_compute(roi_grid_cfg):
    attention_cells_start_cord = np.array(roi_grid_cfg['grid_left_top_coord'])
    attention_cell_width = roi_grid_cfg['width']
    attention_cell_height = roi_grid_cfg['height']
    attention_cells_end_cord = [start_piont + np.array([attention_cell_width, attention_cell_height]) for
                                start_piont in
                                attention_cells_start_cord]

    cell_coord_all = np.concatenate([attention_cells_start_cord, attention_cells_end_cord], axis=1)

    return cell_coord_all

