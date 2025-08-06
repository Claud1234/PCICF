#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def make_pie_png_list(seq):
    dir_name = seq['id']
    start_frame = seq['event_window'][0]
    end_frame = seq['event_window'][-1]

    input_png_list = [os.path.join('./datasets/pie', dir_name, ('%03d' % i) + '.png')
                      for i in range(start_frame, end_frame)]
    seq_name = dir_name + '_' + str(start_frame) + '_' + str(end_frame)
    #print(input_png_list)
    return input_png_list, seq_name


def dir_path_check(full_path):
    if not os.path.exists(os.path.dirname(full_path)):
        os.makedirs(os.path.dirname(full_path))



