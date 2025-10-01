#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 12nd May, 2025
"""
import glob
import pandas as pd
import argparse


def morton_similarity_check(pie_path, moresmirk_path):

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
    parser = argparse.ArgumentParser(description='Evaluation script for pie-MoreSMIRK matching')
    parser.add_argument('-i', '--input', type=str, required=True, help='morton csv of pie')
    parser.add_argument('-t', '--threshold', type=float, required=True, help='threshold of iou')

    args = parser.parse_args()

    for p in sorted(glob.glob(args.input + '/*_morton.csv')):
        event, similarity, pie_intersecion, moresmirk_uni_morton = [], [], [], []
        #print(f'Pedestrian crossing classification analysis for {p.split("/")[-1].split('_morton')[0]}')
        for m in sorted(glob.glob('./datasets/MoreSMIRK/morton_codes/*')):
            intersection, moresmirk = morton_similarity_check(p, m)
            iou = len(intersection) / len(moresmirk)
            if iou >= args.threshold:
                event.append(m.split('/')[-1].split('_')[3])
                similarity.append(round(iou, 4))
                pie_intersecion.append(intersection)
                moresmirk_uni_morton.append(moresmirk)

        eval_file = open(args.input.replace('pie_results', 'pie_eval') + '.txt', 'a')
        eval_file.write(f'Pedestrian crossing classification analysis for {p.split("/")[-1].split('_morton')[0]} \n')
        event_sim_pair = [f'{event[i]}/{similarity[i]}' for i in range(len(event))]
        eval_file.write(f'Event/similarity pair: {event_sim_pair} \n')
        eval_file.write(f'intersection: {pie_intersecion} \n'
                        f'moresmirk: {moresmirk_uni_morton} \n')
        eval_file.close()
