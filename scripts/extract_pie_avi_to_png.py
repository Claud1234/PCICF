#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script read pie_event_window_anno.yml and extract the manually-select event frames from corresponding avi file.
In yml:
- id: "output_008"
  event_window: [571, 697]
  type: right
Extract frame 571 to frame 697 saved as .png

Created on 2nd April 2025
"""
import os
import cv2
import yaml


pie_anno_path = '../datasets/pie_event_window_anno.yml'
with open(pie_anno_path) as _f:
    pie_event_window_anno = yaml.safe_load(_f)

for i in pie_event_window_anno:
    avi_path = os.path.join('/mnt/storage/pie_avi/', i['id'], (i['id'] + '.avi'))
    start_frame = i['event_window'][0]
    end_frame = i['event_window'][1]
    print(avi_path)
    cap = cv2.VideoCapture(avi_path)
    for j in range(end_frame - start_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + j)
        _, frame = cap.read()

        frame_path = os.path.join('../datasets/pie_1', i['id'], ('%03d' % (start_frame + j) + '.png'))
        if not os.path.exists(os.path.dirname(frame_path)):
            os.makedirs(os.path.dirname(frame_path))
        cv2.imwrite(frame_path, cv2.resize(frame, (640, 480)))


