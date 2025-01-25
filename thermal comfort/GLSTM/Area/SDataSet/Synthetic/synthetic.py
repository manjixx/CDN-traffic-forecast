# -*- coding: utf-8 -*-
"""
@Project ：2023GRAD 
@File ：synthetic.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/4/13 19:33 
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


def dataload(season):
    df = pd.read_csv('../../ODataSet/' + season + '.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)

    return df


if __name__ == '__main__':

    person_feature = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith',
                      'sensitivity', 'preference', 'environment', 'season']
    lstm_name = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith',
            'sensitivity', 'preference', 'environment',
            'season', 'ta', 'hr', 'va', 'date', 'time']
    season = ['winter', 'summer']

    lstmDf = pd.DataFrame(columns=lstm_name)
    pmv_name = ['season', 'ta', 'hr', 'va']
    pmvDf = pd.DataFrame(columns=pmv_name)
    for s in season:
        lstm = []
        pmv = []
        df = dataload(s)
        no = df['no'].unique().tolist()
        if s == 'winter':
            t_down, t_up = 20, 28
            h_down, h_up = 10, 26
            t_gap = 0.5
        else:
            t_down, t_up = 22, 30
            h_down, h_up = 40, 71
            t_gap = 0.5
        date = 0
        temp = np.arange(t_down, t_up, t_gap)
        humid = np.arange(h_down, h_up, 1)
        print(len(temp) * len(humid))
        for h in humid:
            for t in temp:
                for n in no:
                    data = df.loc[df['no'] == n][person_feature].drop_duplicates()
                    data = np.array(data).flatten().tolist()
                    va = 1.2 * round(random.random(), 1)
                    time = 0

                    for i in range(0, 4):
                        r = []
                        r.extend(data)
                        if i == 0:
                            r.append(t)
                            r.append(h)
                        else:
                            r.append(t + 1 * round(random.random(), 2))
                            r.append(h)
                        r.append(1.2 * round(random.random(), 1))
                        r.append(str(date))
                        r.append(str(time))
                        lstm.append(r)
                        time += 1
                p = []
                p.append(data[-1])
                p.append(t)
                p.append(h)
                p.append(va)
                pmv.append(p)
                date += 1

        lstmDf = pd.concat([lstmDf, pd.DataFrame(columns=lstm_name, data=lstm)], axis=0)
        pmvDf = pd.concat([pmvDf, pd.DataFrame(columns=pmv_name, data=pmv)], axis=0)
    print(pmvDf.shape)
    print(lstmDf.shape)
    lstmDf.to_csv('./lstm.csv', index=False)
    pmvDf.to_csv('./pmv.csv', index=False)
