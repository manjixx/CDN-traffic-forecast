# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math

alpha, beta = 0.5, 0.5
b_up, b_down = 25, 18.5
g_up, g_down = 2.0, 1.0
y_feature = 'thermal_sensation'


def split(data, index):
    if index == 'bmi':
        upper = b_up
        down = b_down
    else:
        upper = g_up
        down = g_down

    data1 = data[(data[index] <= down)].reset_index(drop=True)
    data2 = data[(data[index] > down) & (data[index] < upper)].reset_index(drop=True)
    data3 = data[(data[index] >= upper)].reset_index(drop=True)

    return data1, data2, data3


def weight_bmi(df):

    no = df['no'].unique()
    level1, level2, level3 = 0, 0, 0

    for n in no:
        data = df.loc[(df['no'] == n)]
        b = data['bmi'].unique()[0]

        if b <= b_down:
            level1 += 1
        elif b >= b_up:
            level3 += 1
        else:
            level2 += 1
    total = no.shape[0]
    p1 = level1 / total
    p2 = level2 / total
    p3 = level3 / total

    bmi1, bmi2, bmi3 = split(df, index='bmi')

    p4 = bmi1[(bmi1[y_feature] != 1)].shape[0] / bmi1.shape[0]
    p5 = bmi2[(bmi2[y_feature] != 1)].shape[0] / bmi2.shape[0]
    p6 = bmi3[(bmi3[y_feature] != 1)].shape[0] / bmi3.shape[0]

    w1 = alpha * p1 + beta * p4
    w2 = alpha * p2 + beta * p5
    w3 = alpha * p3 + beta * p6

    return round(w1, 2), round(w2, 2), round(w3, 2)


def weight_griffith(df):

    no = df['no'].unique()
    level1, level2, level3 = 0, 0, 0

    for n in no:
        data = df.loc[(df['no'] == n)]
        g = data['griffith'].unique()[0]
        if g <= g_down:
            level1 += 1
        elif g >= g_up:
            level3 += 1
        else:
            level2 += 1

    total = no.shape[0]
    p1 = level1 / total
    p2 = level2 / total
    p3 = level3 / total

    griffith1, griffith2, griffith3 = split(df, 'griffith')

    p4 = griffith1[(griffith1[y_feature] != 1)].shape[0] / griffith1.shape[0]
    p5 = griffith2[(griffith2[y_feature] != 1)].shape[0] / griffith2.shape[0]
    p6 = griffith3[(griffith3[y_feature] != 1)].shape[0] / griffith3.shape[0]

    w1 = alpha * p1 + beta * p4
    w2 = alpha * p2 + beta * p5
    w3 = alpha * p3 + beta * p6
    return round(w1, 2), round(w2, 2), round(w3, 2)
