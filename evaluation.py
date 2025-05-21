#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 12nd May, 2025
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

better_path = './outputs/bettersmirk_results_yolo_track/evt_01_morton.csv'
# pie_path = '../outputs/pie_results_yolo_track/output_016_607_744_morton.csv'
pie_path = './outputs/pie_results_yolo_track/output_008_571_697_morton.csv'
# pie_path = '../outputs/pie_results_yolo_track/output_013_84_153_morton.csv'

better_csv = pd.read_csv(better_path, sep=';')
pie_csv = pd.read_csv(pie_path, sep=';')

better_np = better_csv.to_numpy()
pie_np = pie_csv.to_numpy()

pie_morton = pie_np[:, -1] / 10000000
better_morton = better_np[:, -1] / 10000000

pie_morton = pie_morton[pie_morton != 0]
better_morton = better_morton[better_morton != 0]

pie_morton_uni = pd.unique(pie_morton)
better_morton_uni = pd.unique(better_morton)

iou_unfiltered = sorted(list(set(pie_morton_uni).intersection(set(better_morton_uni))), key=list(pie_morton_uni).index)

iou_filter = []
better_update = list(better_morton_uni)
for item in iou_unfiltered:
    if item in better_update:
        index = better_update.index(item)
        better_update = better_update[index+1:]
        iou_filter.append(item)

print(iou_filter)
print(better_morton_uni)
print(f'the percentage of iou(pie,better) over better is {len(iou_filter)/len(better_morton_uni)}')