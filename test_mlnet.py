#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script to test the MLNet.

Created on 29th Dec. 2024
"""

import json
import argparse

with open('config.json', 'r') as f:
    config = json.load(f)

print(config['Dataset']['transforms']['image_mean'])
