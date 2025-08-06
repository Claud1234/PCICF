#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 12nd May, 2025
"""
import os
import glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def morotn_similarity_check(pie_path, moresmirk_path):

    moresmirk_csv = pd.read_csv(moresmirk_path, sep=';')
    pie_csv = pd.read_csv(pie_path, sep=';')

    moresmirk_np = moresmirk_csv.to_numpy()
    pie_np = pie_csv.to_numpy()

    pie_morton = pie_np[:, -1] / 10000000
    moresmirk_morton = moresmirk_np[:, -1] / 10000000

    pie_morton = pie_morton[pie_morton != 0]
    moresmirk_morton = moresmirk_morton[moresmirk_morton != 0]

    pie_morton_uni = pd.unique(pie_morton)
    moresmirk_morton_uni = pd.unique(moresmirk_morton)

    iou_unfiltered = sorted(list(set(pie_morton_uni).intersection(set(moresmirk_morton_uni))),
                            key=list(pie_morton_uni).index)

    iou_filter = []
    moresmirk_update = list(moresmirk_morton_uni)
    for item in iou_unfiltered:
        if item in moresmirk_update:
            index = moresmirk_update.index(item)
            moresmirk_update = moresmirk_update[index+1:]
            iou_filter.append(item)

    return iou_filter, moresmirk_morton_uni
    # print(iou_filter)
    # print(moresmirk_morton_uni)
    # print(f'the percentage of iou(pie,moresmirk) over moresmirk is {len(iou_filter)/len(moresmirk_morton_uni)}')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Evaluation script for pie-MoreSMIRK matching')
    # parser.add_argument('-i', '--input', type=str, required=True, help='pie morton csv')
    #
    # args = parser.parse_args()
    for p in sorted(glob.glob('./outputs/pie_results/pie_results_event_3/*_morton.csv')):
        #result = {'event': [], 'similarity': [], 'intersection': [], 'moresmirk': []}
        event, similarity, pie_intersecion, moresmirk_uni_morton = [], [], [], []
        print(f'Pedestrian crossing classification analysis for {p.split("/")[-1].split('_morton')[0]}')
        for m in sorted(glob.glob('./datasets/MoreSMIRK/morton_codes/*')):
            intersection, moresmirk = morotn_similarity_check(p, m)
            iou = len(intersection) / len(moresmirk)
            if iou > 0.19:
                event.append(m.split('/')[-1].split('_')[1])
                similarity.append(round(iou, 4))
                pie_intersecion.append(intersection)
                moresmirk_uni_morton.append(moresmirk)
            # if m.split('/')[-1].split('_')[1] != '00' and len(event) == 0:
            #     event.append(m.split('/')[-1].split('_')[1])
            #     similarity.append(round(iou, 4))
            #     pie_intersecion.append(intersection)
            #     moresmirk_uni_morton.append(moresmirk)

        print(f'event: {event}'
              f'similarity: {similarity}'
              f'intersection: {pie_intersecion}'
              f'moresmirk: {moresmirk_uni_morton}')
        #         result.append({'event': m.split('/')[-1].split('_')[1],
        #                        'similarity': similarity,
        #                        'intersection': intersection,
        #                        'moresmirk': moresmirk})
        # print(result)
                #result.update({'similarity': m.split('/')[-1].split('_')[1],similarity})
                # print(f'The similarity with event_{m.split('/')[-1].split('_')[1]} is {similarity}')
                # print(f'intersection: {intersection}')
                # print(f'moresmirk: {moresmirk}')

