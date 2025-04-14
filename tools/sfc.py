#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import zCurve


def calculate_morton(values):
    # Cap floating point numbers to one decimal place and convert to integers
    int_values = [int(round(value, 1) * 10) for value in values]
    value = zCurve.interlace(*int_values, dims=len(int_values))
    return value
