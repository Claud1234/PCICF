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

