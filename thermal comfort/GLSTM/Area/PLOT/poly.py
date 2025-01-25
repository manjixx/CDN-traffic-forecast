# -*- coding: utf-8 -*-
"""
@Project ：2023GRAD 
@File ：poly.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/4/14 17:37 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def summer():
    x = np.array([23, 22.2])
    y = np.array([40, 70])

    fs_left = np.polyfit(x, y, 1)
    print('f1 is :\n', fs_left)

    y1 = np.arange(40, 71, 1)
    x1 = []
    for i in np.arange(40, 71, 1):
        x1.append(round((i - fs_left[1]) / fs_left[0], 2))

    x = np.array([26, 25.2])
    y = np.array([40, 70])

    fs_right = np.polyfit(x, y, 1)
    print('f1 is :\n', fs_right)

    y2 = np.arange(40, 71, 1)
    x2 = []
    for i in np.arange(40, 71, 1):
        x2.append(round((i - fs_right[1]) / fs_right[0], 2))

    return fs_left, fs_right


def winter():
    x = np.array([19.2, 22.4])
    y = np.array([40, 70])

    fs_left = np.polyfit(x, y, 1)
    print('f1 is :\n', fs_left)

    y1 = np.arange(40, 71, 1)
    x1 = []
    for i in np.arange(40, 71, 1):
        x1.append(round((i - fs_left[1]) / fs_left[0], 2))

    x = np.array([24.8, 24])
    y = np.array([40, 70])

    fs_right = np.polyfit(x, y, 1)
    print('f1 is :\n', fs_right)

    y2 = np.arange(40, 71, 1)
    x2 = []
    for i in np.arange(40, 71, 1):
        x2.append(round((i - fs_right[1]) / fs_right[0], 2))

    return fs_left, fs_right


if __name__ == '__main__':


    season = ['winter', 'summer']

    pmv_name = ['ta', 'hr', 'season']
    pmvDf = pd.DataFrame(columns=pmv_name)
    for s in season:
        lstm = []
        pmv = []
        if s == 'winter':
            t_down, t_up = 18, 28
            h_down, h_up = 10, 41
            t_gap = 0.2
            se = 1
            fs_l, fs_r = winter()
        else:
            t_down, t_up = 20, 30
            h_down, h_up = 40, 71
            t_gap = 0.2
            se = 0
            fs_l, fs_r = summer()

        for t in np.arange(t_down, t_up, 2):
            for h in np.arange(h_down, h_up, 2):
                left = fs_l[0] * t + fs_l[1]

