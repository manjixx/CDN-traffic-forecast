# -*- coding: utf-8 -*-
"""
@Project ：2023GRAD 
@File ：data.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/4/14 16:23 
"""
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontManager
import matplotlib.ticker as ticker
import random
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    season = ['winter', 'summer']

    pmv_name = ['ta', 'hr', 'season']
    pmvDf = pd.DataFrame(columns=pmv_name)
    for s in season:
        lstm = []
        pmv = []
        if s == 'winter':
            t_down, t_up = 18, 28
            h_down, h_up = 10, 26
            t_gap = 0.2
            se = 1
        else:
            t_down, t_up = 20, 30
            h_down, h_up = 50, 81
            t_gap = 0.2
            se = 0

        temp = np.arange(t_down, t_up, t_gap)
        humid = np.arange(h_down, h_up, 1)
        print(len(temp) * len(humid))
        for h in humid:
            for t in temp:
                p = []
                p.append(t)
                p.append(h)
                p.append(se)
                pmv.append(p)

        pmvDf = pd.concat([pmvDf, pd.DataFrame(columns=pmv_name, data=pmv)], axis=0)
    pmvDf.to_csv('./pmv.csv', index=False)